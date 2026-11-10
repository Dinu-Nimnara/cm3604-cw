"""
Member 4: word-overlap pruning, efficient DeBERTa inference, and evaluation.

Run from the repository root:
  python experiments/m4_optimization/run_m4_experiments.py

Optional:
  python experiments/m4_optimization/run_m4_experiments.py --cuda
  python experiments/m4_optimization/run_m4_experiments.py --batch-sizes 1 4 8 16
  python experiments/m4_optimization/run_m4_experiments.py --pruning-thresholds 0.0 0.25 0.5

This script does NOT modify factcheck.py or Member 3's tune.py.
It uses Member 1's TextPreprocessor/split_sentences and selected preprocessing settings.
"""
import argparse
import csv
import json
import math
import platform
try:
    import resource
except ImportError:  # Windows
    resource = None
import sys
import time
from pathlib import Path

import torch  # type: ignore[import-not-found]
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from factcheck import TextPreprocessor, split_sentences

MODEL_ID = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
PREPROCESSING = {
    "lowercase": True,
    "remove_stopwords": True,
    "normalizer": "porter",
    "min_token_len": 2,
}
PRUNING_METRIC = "word_recall"


def binary_label(label):
    if label not in ("S", "NS", "IR"):
        raise ValueError(f"Unexpected label: {label!r}")
    return "S" if label == "S" else "NS"


def load_examples(labels_path, passages_path):
    lookup = {}
    with Path(passages_path).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if any(p["title"] != row["name"] for p in row["passages"]):
                raise ValueError(f"Passage title mismatch for {row['name']!r}")
            lookup[row["sent"]] = row["passages"]

    examples = []
    missing = 0
    with Path(labels_path).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for sentence in row["annotations"] or []:
                for fact in sentence["human-atomic-facts"] or []:
                    fact_text = fact["text"]
                    if fact_text not in lookup:
                        missing += 1
                        continue
                    examples.append({
                        "fact": fact_text,
                        "gold": binary_label(fact["label"]),
                        "passages": lookup[fact_text],
                    })
    if not examples:
        raise ValueError("No matching labelled facts found.")
    return examples, missing


def passage_word_recall(fact_tokens, passage_tokens):
    fact_set = set(fact_tokens)
    if not fact_set:
        return 0.0
    return len(fact_set.intersection(passage_tokens)) / len(fact_set)


def prepare_fact(example, preprocessor, pruning_threshold=None):
    """
    Convert passages into sentence-level premise/hypothesis pairs.
    Passage pruning is based on Member 1's selected word-recall preprocessing.
    The entity title is removed from both fact and passage tokens, matching Member 1.
    If every passage is pruned, retain the highest-scoring passage as a safe fallback.
    """
    passages = example["passages"]
    title_tokens = set()
    for p in passages:
        title_tokens.update(preprocessor.process_set(p.get("title", "")))

    fact_tokens = [t for t in preprocessor.process(example["fact"]) if t not in title_tokens]
    if not fact_tokens:
        fact_tokens = preprocessor.process(example["fact"])

    scored = []
    for p_index, passage in enumerate(passages):
        passage_tokens = [t for t in preprocessor.process(passage.get("text", "")) if t not in title_tokens]
        score = passage_word_recall(fact_tokens, passage_tokens)
        scored.append((score, p_index, passage))

    if pruning_threshold is None:
        retained = scored
    else:
        retained = [item for item in scored if item[0] >= pruning_threshold]
        if not retained and scored:
            retained = [max(scored, key=lambda item: (item[0], -item[1]))]

    pairs = []
    for overlap, p_index, passage in retained:
        sentences = split_sentences(passage.get("text", ""))
        if not sentences and passage.get("text", "").strip():
            sentences = [passage["text"].strip()]
        for sentence_index, sentence in enumerate(sentences):
            cleaned = sentence.strip()
            if cleaned:
                pairs.append({
                    "premise": cleaned,
                    "hypothesis": example["fact"],
                    "passage_index": p_index,
                    "sentence_index": sentence_index,
                    "overlap": overlap,
                })
    return pairs, len(passages), len(retained)


def entailment_id(model):
    """Resolve entailment class robustly from model config labels."""
    labels = getattr(model.config, "id2label", {}) or {}
    for idx, label in labels.items():
        if "entail" in str(label).lower():
            return int(idx)
    # This checkpoint's documented label order is entailment, neutral, contradiction.
    if int(model.config.num_labels) == 3:
        return 0
    raise ValueError(f"Cannot identify entailment label from model config: {labels}")


def score_pairs(pairs, tokenizer, model, device, entail_idx, batch_size, max_length):
    """Batched pair scoring; returns one entailment probability per premise/hypothesis pair."""
    scores = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            encoded = tokenizer(
                [x["premise"] for x in batch],
                [x["hypothesis"] for x in batch],
                padding=True,
                truncation="only_first",
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)[:, entail_idx]
            scores.extend(probabilities.detach().cpu().tolist())
            del encoded, logits, probabilities
    return scores


def score_dataset(examples, preprocessor, tokenizer, model, device, entail_idx,
                  batch_size, pruning_threshold, max_length):
    fact_scores = []
    total_passages = 0
    retained_passages = 0
    total_pairs = 0
    started = time.perf_counter()

    for example in examples:
        pairs, original_count, retained_count = prepare_fact(
            example, preprocessor, pruning_threshold
        )
        total_passages += original_count
        retained_passages += retained_count
        total_pairs += len(pairs)
        pair_scores = score_pairs(
            pairs, tokenizer, model, device, entail_idx, batch_size, max_length
        )
        # Coursework rule: maximum entailment probability across evidence sentences.
        fact_scores.append(max(pair_scores, default=0.0))

    elapsed = time.perf_counter() - started
    return {
        "scores": fact_scores,
        "elapsed_seconds": elapsed,
        "original_passages": total_passages,
        "retained_passages": retained_passages,
        "passage_retention": retained_passages / total_passages if total_passages else 0.0,
        "pairs_scored": total_pairs,
    }


def predict(scores, threshold):
    return ["S" if s >= threshold else "NS" for s in scores]


def calculate_metrics(gold, pred):
    tp = sum(g == p == "S" for g, p in zip(gold, pred))
    tn = sum(g == p == "NS" for g, p in zip(gold, pred))
    fp = sum(g == "NS" and p == "S" for g, p in zip(gold, pred))
    fn = sum(g == "S" and p == "NS" for g, p in zip(gold, pred))

    def div(a, b):
        return a / b if b else 0.0

    p_s, r_s = div(tp, tp + fp), div(tp, tp + fn)
    p_ns, r_ns = div(tn, tn + fn), div(tn, tn + fp)
    f1_s = div(2 * p_s * r_s, p_s + r_s)
    f1_ns = div(2 * p_ns * r_ns, p_ns + r_ns)
    return {
        "accuracy": div(tp + tn, len(gold)),
        "precision_S": p_s, "recall_S": r_s, "f1_S": f1_s,
        "precision_NS": p_ns, "recall_NS": r_ns, "f1_NS": f1_ns,
        "macro_f1": (f1_s + f1_ns) / 2,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def tune_threshold(scores, gold):
    # Exhaustive candidate thresholds; tie-break toward 0.5 for stability.
    candidates = sorted(set([0.0, 1.0000001] + list(scores)))
    rows = []
    for threshold in candidates:
        pred = predict(scores, threshold)
        m = calculate_metrics(gold, pred)
        rows.append((m["accuracy"], -abs(threshold - 0.5), -threshold, threshold, m))
    return max(rows, key=lambda x: x[:3])[3:]


def peak_rss_mb():
    # Process high-water RSS. resource is unavailable on Windows.
    if resource is not None:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value / 1024 if sys.platform.startswith("linux") else value / (1024 * 1024)
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return None


def main():
    parser = argparse.ArgumentParser()
    data_dir = ROOT / "Framework Code" / "data"
    parser.add_argument("--labels-path", default=data_dir / "dev_labeled_ChatGPT.jsonl")
    parser.add_argument("--passages-path", default=data_dir / "passages_bm25_ChatGPT_humfacts.jsonl")
    parser.add_argument("--out-dir", default="results/m4")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--pruning-thresholds", nargs="+", type=float, default=[0.0, 0.25, 0.5])
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--classification-threshold", type=float, default=0.5)
    parser.add_argument("--tune-each-threshold", action="store_true",
                        help="Also report development-tuned thresholds; these are not held-out results.")
    args = parser.parse_args()

    if args.cuda and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable. Run without --cuda.")
    if any(x < 0 or x > 1 for x in args.pruning_thresholds):
        parser.error("Pruning thresholds must be between 0 and 1.")
    if any(x < 1 for x in args.batch_sizes):
        parser.error("Batch sizes must be positive integers.")

    examples, missing = load_examples(args.labels_path, args.passages_path)
    gold = [x["gold"] for x in examples]
    preprocessor = TextPreprocessor(**PREPROCESSING)

    device = torch.device("cuda" if args.cuda else "cpu")
    print(f"Loading {MODEL_ID} on {device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device)
    entail_idx = entailment_id(model)

    experiments = []
    # E1/E2/E3/E4 comparisons: baseline batch=1 no pruning; pruning batch=1;
    # batching without pruning; and combined pruning+batching.
    selected_batch_sizes = sorted(set(args.batch_sizes))
    if 1 not in selected_batch_sizes:
        selected_batch_sizes.insert(0, 1)

    configs = []
    for batch in selected_batch_sizes:
        configs.append(("no_pruning", None, batch))
    for threshold in args.pruning_thresholds:
        if threshold > 0:
            for batch in selected_batch_sizes:
                configs.append((f"prune_{threshold:g}", threshold, batch))

    for name, pruning_threshold, batch_size in configs:
        print(f"Running {name}, batch={batch_size}", flush=True)
        result = score_dataset(
            examples, preprocessor, tokenizer, model, device, entail_idx,
            batch_size, pruning_threshold, args.max_length
        )
        fixed_pred = predict(result["scores"], args.classification_threshold)
        fixed_metrics = calculate_metrics(gold, fixed_pred)
        tuned_threshold, tuned_metrics = tune_threshold(result["scores"], gold)
        row = {
            "experiment": name,
            "pruning_threshold": pruning_threshold,
            "batch_size": batch_size,
            "classification_threshold_fixed": args.classification_threshold,
            **{f"fixed_{k}": v for k, v in fixed_metrics.items()},
            "development_tuned_threshold": tuned_threshold,
            **{f"tuned_{k}": v for k, v in tuned_metrics.items()},
            "runtime_seconds": result["elapsed_seconds"],
            "facts": len(examples),
            "original_passages": result["original_passages"],
            "retained_passages": result["retained_passages"],
            "passage_retention": result["passage_retention"],
            "passage_reduction": 1.0 - result["passage_retention"],
            "sentence_pairs_scored": result["pairs_scored"],
            "peak_rss_mb_process": peak_rss_mb(),
        }
        experiments.append(row)
        print(json.dumps(row, indent=2), flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(experiments[0]))
        writer.writeheader()
        writer.writerows(experiments)

    report = {
        "member": 4,
        "purpose": "Word-overlap passage pruning and efficient DeBERTa inference",
        "model": MODEL_ID,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "preprocessing_reused_from_member1": PREPROCESSING,
        "pruning_metric": PRUNING_METRIC,
        "classification_threshold_fixed": args.classification_threshold,
        "development_examples": len(examples),
        "missing_facts_skipped": missing,
        "evaluation_warning": (
            "All reported development-tuned metrics use the same development labels "
            "to select thresholds and are optimistic, not independent test performance."
        ),
        "results": experiments,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved: {csv_path}\nSaved: {out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
