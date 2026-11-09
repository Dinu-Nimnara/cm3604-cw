"""Cache label-independent NLI scores; tune a threshold on the permitted dev set.

Run in the repository root. The required factchecking_main.py is never edited.
Only --apply changes factcheck.py, after full-data scoring and target checks.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

MODEL_ID = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
SCHEMA_VERSION = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def inference_fingerprint(path):
    """Threshold edits do not invalidate scores; inference implementation edits do."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in {"EntailmentModel", "EntailmentFactChecker"}:
            node.body = [x for x in node.body if not (
                isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and
                    t.id == "DEFAULT_THRESHOLD" for t in x.targets))]
            nodes.append(ast.dump(node, include_attributes=False))
    if len(nodes) != 2:
        raise ValueError("Expected both Member 3 entailment classes in factcheck.py.")
    return digest(nodes)


def load_examples(labels_path, passages_path):
    """Match the driver's fact lookup and IR->NS conversion, preserving duplicates."""
    lookup = {}
    with Path(passages_path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if any(p["title"] != row["name"] for p in row["passages"]):
                raise ValueError("Passage title does not match the source name.")
            lookup[row["sent"]] = row["passages"]
    examples = []
    with Path(labels_path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            for sentence in row["annotations"] or []:
                for fact in sentence["human-atomic-facts"] or []:
                    if fact["text"] not in lookup:
                        print("Skipping missing retrieval:", fact["text"], file=sys.stderr)
                        continue
                    gold = "NS" if fact["label"] == "IR" else fact["label"]
                    if gold not in {"S", "NS"}:
                        raise ValueError("Unexpected gold label: " + repr(gold))
                    examples.append({"index":len(examples), "fact":fact["text"],
                                     "gold":gold, "passages":lookup[fact["text"]]})
    if not examples:
        raise ValueError("No evaluable facts were loaded.")
    return examples


def metrics(rows, threshold):
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Threshold must be in (0, 1].")
    matrix = [[0, 0], [0, 0]]
    for row in rows:
        score = row["score"]
        if not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid cached probability.")
        matrix[0 if row["gold"] == "S" else 1][0 if score >= threshold else 1] += 1
    total = len(rows)
    if not total:
        raise ValueError("No scores to evaluate.")
    tp, fn = matrix[0]
    fp, tn = matrix[1]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold":threshold, "correct":tp+tn, "total":total,
            "accuracy":(tp+tn)/total, "confusion_matrix":matrix,
            "supported_precision":precision, "supported_recall":recall,
            "supported_f1":2*precision*recall/(precision+recall) if precision+recall else 0.0}


def choose_threshold(rows):
    """Exhaust every distinct >= decision boundary; never split tied scores.

    Interior midpoints give the same predictions as their upper score boundary,
    with a margin against small platform-dependent floating-point changes.
    Ties prefer the threshold closest to 0.5, then the higher threshold.
    """
    values = sorted(set(float(row["score"]) for row in rows))
    candidates = {0.5, 1.0}
    bounds = sorted({0.0, 1.0, *values})
    for lo, hi in zip(bounds, bounds[1:]):
        middle = lo + (hi - lo) / 2
        if 0 < middle <= 1:
            candidates.add(middle)
    # metrics() also validates probabilities before any selection is accepted.
    results = [metrics(rows, threshold) for threshold in sorted(candidates)]
    return max(results, key=lambda r:(r["correct"], -abs(r["threshold"]-.5), r["threshold"]))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def validate_cache(cache, expected, examples):
    for key, value in expected.items():
        if cache.get(key) != value:
            raise ValueError("Cache provenance mismatch for " + key + "; use a new --cache filename.")
    rows = cache.get("records")
    if not isinstance(rows, list) or len(rows) > len(examples):
        raise ValueError("Invalid cache record count.")
    for index, row in enumerate(rows):
        example = examples[index]
        if any(row.get(key) != example[key] for key in ("index", "fact", "gold")):
            raise ValueError("Cache example order or labels changed.")
        metrics([row], .5)
    return rows


def apply_threshold(path, threshold):
    """Edit only the class's numeric default; preserve CRLF and all other bytes."""
    path = Path(path)
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    tree = ast.parse(text)
    assignments = [item for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "EntailmentFactChecker" for item in node.body
                   if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name)
                   and t.id == "DEFAULT_THRESHOLD" for t in item.targets)]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Constant):
        raise ValueError("Expected exactly one numeric DEFAULT_THRESHOLD assignment.")
    lines = raw.splitlines(keepends=True)
    node = assignments[0].value
    if node.lineno != node.end_lineno:
        raise ValueError("Expected a single-line threshold constant.")
    line_index = node.lineno-1
    line = lines[line_index]
    lines[line_index] = line[:node.col_offset] + repr(float(threshold)).encode("ascii") + line[node.end_col_offset:]
    updated = b"".join(lines)
    ast.parse(updated.decode("utf-8-sig"))
    backup = path.with_name(path.name + ".member3-" + str(time.time_ns()) + ".bak")
    backup.write_bytes(raw)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(updated)
    os.replace(temporary, path)
    return backup


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", default="data/dev_labeled_ChatGPT.jsonl")
    parser.add_argument("--passages-path", default="data/passages_bm25_ChatGPT_humfacts.jsonl")
    parser.add_argument("--cache", default="member3_scores.json")
    parser.add_argument("--report", default="member3_tuning_report.json")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--scores-only", action="store_true", help="Require complete cache; do not load DeBERTa")
    parser.add_argument("--apply", action="store_true", help="Write the selected threshold only if target is met")
    parser.add_argument("--target", type=float, default=.83)
    args = parser.parse_args(argv)
    if not 0 < args.target <= 1:
        parser.error("--target must be in (0, 1].")
    source = Path(__file__).resolve().with_name("factcheck.py")
    examples = load_examples(args.labels_path, args.passages_path)
    expected = {"schema_version":SCHEMA_VERSION, "model_id":MODEL_ID,
                "inference_fingerprint":inference_fingerprint(source),
                "dataset_fingerprint":digest(examples), "total":len(examples)}
    cache_path = Path(args.cache)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else dict(expected, records=[])
    rows = validate_cache(cache, expected, examples)
    print(f"Loaded {len(examples)} facts; {len(rows)} cached scores.", flush=True)
    if len(rows) < len(examples):
        if args.scores_only:
            parser.error("Cache is incomplete; run without --scores-only to compute missing scores.")
        import torch
        import transformers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from factcheck import EntailmentModel, EntailmentFactChecker
        if args.cuda and not torch.cuda.is_available():
            parser.error("CUDA is unavailable. Omit --cuda for your CPU installation.")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        if args.cuda:
            model.to("cuda")
        checker = EntailmentFactChecker(EntailmentModel(model, tokenizer, args.cuda), threshold=.5)
        runtime = {"python":sys.version, "torch":torch.__version__,
                                    "transformers":transformers.__version__,
                                    "device":"cuda" if args.cuda else "cpu",
                                    "checkpoint_commit":getattr(model.config,"_commit_hash",None)}
        if "runtime" in cache and cache["runtime"].get("checkpoint_commit") != runtime["checkpoint_commit"]:
            raise ValueError("Checkpoint revision changed; use a new --cache filename.")
        cache.setdefault("runtime", runtime)
        started = time.monotonic()
        for example in examples[len(rows):]:
            probability, evidence = checker.score_details(example["fact"], example["passages"])
            rows.append({"index":example["index"], "fact":example["fact"],
                         "gold":example["gold"], "score":probability, "best_evidence":evidence})
            atomic_json(cache_path, cache)
            print(f"Scored {len(rows)}/{len(examples)}; elapsed {time.monotonic()-started:.0f}s", flush=True)
    baseline = metrics(rows, .5)
    selected = choose_threshold(rows)
    report = {"evaluation":"Development set used for threshold selection; no independent test set.",
              "baseline":baseline, "selected":selected, "target":args.target,
              "target_met":selected["accuracy"] >= args.target,
              "cache":str(cache_path), **expected}
    atomic_json(args.report, report)
    print("Baseline:", json.dumps(baseline), flush=True)
    print("Selected:", json.dumps(selected), flush=True)
    print("Target met:", report["target_met"], flush=True)
    errors = [{**row, "prediction":"S" if row["score"] >= selected["threshold"] else "NS"}
              for row in rows if ("S" if row["score"] >= selected["threshold"] else "NS") != row["gold"]]
    atomic_json(Path(args.report).with_name(Path(args.report).stem + "_errors.json"), errors)
    if args.apply:
        if not report["target_met"]:
            print("Source unchanged: threshold tuning alone did not meet the target.", flush=True)
            return 2
        if inference_fingerprint(source) != expected["inference_fingerprint"]:
            raise ValueError("Source changed during scoring; refusing to write the threshold.")
        backup = apply_threshold(source, selected["threshold"])
        print("Updated only DEFAULT_THRESHOLD in factcheck.py; backup:", backup, flush=True)
        print("Confirm using: python factchecking_main.py --mode entailment", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print("Member 3 tuning error:", exc, file=sys.stderr)
        raise SystemExit(1)
