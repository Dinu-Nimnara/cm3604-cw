"""
Member 1 experiment: systematic comparison of preprocessing choices x similarity metrics x passage aggregation for
the word-overlap fact-checker.

Configurations are compared with ROC-AUC, which is threshold-free: it measures how well a score ranks Supported
facts above Not-Supported ones, so preprocessing and metrics can be chosen without touching Member 2's threshold
tuning. "Oracle accuracy" (the best accuracy over all thresholds on the same data) is reported as a diagnostic
only; it is optimistic because the threshold is fitted on the evaluation data.

Usage (from the repository root, Python 3.10 env):
    python experiments/m1_preprocessing/run_grid.py --data_dir data --out_dir results/m1

Outputs (all deterministic; the bootstrap uses a fixed seed):
    grid.csv            one row per configuration, sorted by AUC
    ablation.csv        one factor changed at a time from the best configuration
    factor_summary.csv  mean / best AUC for each level of each factor
    bootstrap.csv       95% CIs for the top configurations and paired differences vs the best
    scores_top.csv      per-fact scores for the top configurations (for Member 2 and error analysis)
    scores_all.csv      per-fact scores for every configuration (gold label: 1 = S, 0 = NS) for ROC curves etc.
    distribution.csv    score histogram by gold label for the best configuration
    summary.md          the tables above in Markdown, for the report
    run_config.json     versions, git commit, arguments
"""
import argparse
import csv
import io
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import redirect_stdout
from itertools import product

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from factcheck import AGGREGATIONS, METRICS, NORMALIZERS, TextPreprocessor, split_sentences, word_overlap_score  # noqa: E402

BASE_PREPROC = {"lowercase": True, "remove_stopwords": True, "normalizer": "porter", "min_token_len": 2}
FACTORS = ("remove_stopwords", "normalizer", "lowercase", "min_token_len", "drop_title", "metric", "aggregation")
TIMING_REPEATS = 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join(REPO_ROOT, "data"))
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "results", "m1"))
    parser.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples for confidence intervals")
    parser.add_argument("--top_k", type=int, default=10, help="configurations to bootstrap / export scores for")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_examples(data_dir):
    """ Loads the data with the unmodified readers from factchecking_main.py, so the text is exactly what the
    graded run sees. """
    import factchecking_main as fm
    with redirect_stdout(io.StringIO()):
        fact_to_passages = fm.read_passages(os.path.join(data_dir, "passages_bm25_ChatGPT_humfacts.jsonl"))
        examples = fm.read_fact_examples(os.path.join(data_dir, "dev_labeled_ChatGPT.jsonl"), fact_to_passages)
    return examples


def preprocessing_configs():
    """ Full grid over stop words x normaliser at the base settings, plus one-at-a-time ablations of lowercasing and
    minimum token length (with and without stop words, Porter). """
    configs = []
    for stop, norm in product((True, False), NORMALIZERS):
        configs.append(dict(BASE_PREPROC, remove_stopwords=stop, normalizer=norm))
    for stop in (True, False):
        configs.append(dict(BASE_PREPROC, remove_stopwords=stop, lowercase=False))
        configs.append(dict(BASE_PREPROC, remove_stopwords=stop, min_token_len=1))
    return configs


def roc_auc(scores, gold):
    """ Mann-Whitney AUC = P(score_S > score_NS) + 0.5 * P(tie), over all (S, NS) pairs. """
    pos, neg = scores[gold == 1], scores[gold == 0]
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def oracle_accuracy(scores, gold):
    """ Best accuracy of the rule 'S if score >= t' over every distinct threshold (midpoints between sorted unique
    scores, plus one below the minimum and one above the maximum). Returns (accuracy, threshold). """
    uniq = np.unique(scores)
    thresholds = np.concatenate(([uniq[0] - 1e-9], (uniq[:-1] + uniq[1:]) / 2, [uniq[-1] + 1e-9]))
    preds = scores[None, :] >= thresholds[:, None]
    acc = (preds == (gold[None, :] == 1)).mean(axis=1)
    best = int(np.argmax(acc))
    return float(acc[best]), float(thresholds[best])


def config_name(row):
    return ("stop={remove_stopwords:d} norm={normalizer} lc={lowercase:d} minlen={min_token_len} "
            "title={drop_title:d} metric={metric} agg={aggregation}").format(**row)


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows, fields, floats=()):
    lines = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    for r in rows:
        cells = [("%.4f" % r[k]) if k in floats and isinstance(r[k], float) else str(r[k]) for k in fields]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main():
    args = parse_args()
    run_start = time.perf_counter()
    os.makedirs(args.out_dir, exist_ok=True)
    examples = load_examples(args.data_dir)
    gold = np.array([1 if ex.label == "S" else 0 for ex in examples])  # IR counts as NS, as in factchecking_main
    print("Loaded %d examples (%d S, %d NS)" % (len(examples), gold.sum(), len(gold) - gold.sum()))

    unique_texts = set()
    for ex in examples:
        unique_texts.add(ex.fact)
        for p in ex.passages:
            unique_texts.add(p["text"])
            unique_texts.add(p["title"])
            unique_texts.update(split_sentences(p["text"]))

    rows, all_scores = [], {}
    for preproc in preprocessing_configs():
        # Preprocessing cost = median over TIMING_REPEATS cold-cache passes over every unique text (a single
        # measurement is too noisy at ~0.1 s). Building the preprocessor (e.g. loading spaCy) is not timed.
        times = []
        for _ in range(TIMING_REPEATS):
            pre = TextPreprocessor(**preproc)
            start = time.perf_counter()
            for text in unique_texts:
                pre.process(text)
            times.append(time.perf_counter() - start)
        preprocess_s = float(np.median(times))
        for metric, aggregation, drop_title in product(METRICS, AGGREGATIONS, (False, True)):
            start = time.perf_counter()
            scores = np.array([word_overlap_score(ex.fact, ex.passages, pre, metric, aggregation, drop_title)
                               for ex in examples])
            score_s = time.perf_counter() - start
            acc, thr = oracle_accuracy(scores, gold)
            row = dict(preproc, drop_title=drop_title, metric=metric, aggregation=aggregation,
                       auc=roc_auc(scores, gold), oracle_acc=acc, oracle_threshold=thr,
                       mean_S=float(scores[gold == 1].mean()), mean_NS=float(scores[gold == 0].mean()),
                       preprocess_s=round(preprocess_s, 3), score_s=round(score_s, 3))
            row["config"] = config_name(row)
            rows.append(row)
            all_scores[row["config"]] = scores
        print("  %-60s preprocessing %.2fs" % (pre.config(), preprocess_s))

    rows.sort(key=lambda r: (-r["auc"], -r["oracle_acc"], r["config"]))
    fields = ["config", "auc", "oracle_acc", "oracle_threshold", "mean_S", "mean_NS", "preprocess_s", "score_s",
              "remove_stopwords", "normalizer", "lowercase", "min_token_len", "drop_title", "metric", "aggregation"]
    write_csv(os.path.join(args.out_dir, "grid.csv"), rows, fields)
    best = rows[0]
    by_name = {r["config"]: r for r in rows}

    # One-factor-at-a-time ablation around the best configuration.
    levels = {"remove_stopwords": (True, False), "normalizer": NORMALIZERS, "lowercase": (True, False),
              "min_token_len": (1, 2), "drop_title": (False, True), "metric": METRICS, "aggregation": AGGREGATIONS}
    ablation = []
    for factor in FACTORS:
        for level in levels[factor]:
            variant = dict(best, **{factor: level})
            name = config_name(variant)
            if name in by_name:
                r = by_name[name]
                ablation.append({"factor": factor, "level": level, "config": name, "auc": r["auc"],
                                 "delta_auc": r["auc"] - best["auc"], "oracle_acc": r["oracle_acc"],
                                 "is_best": name == best["config"]})
    write_csv(os.path.join(args.out_dir, "ablation.csv"), ablation,
              ["factor", "level", "auc", "delta_auc", "oracle_acc", "is_best", "config"])

    # Marginal effect of each factor level over the grid. Each level is averaged over the same set of other settings:
    # the lowercase / min_token_len ablations were only run with Porter, so they are compared on Porter configs.
    factor_summary = []
    for factor in FACTORS:
        for level in levels[factor]:
            subset = [r for r in rows if r[factor] == level]
            if factor in ("lowercase", "min_token_len"):
                other = "min_token_len" if factor == "lowercase" else "lowercase"
                subset = [r for r in subset if r["normalizer"] == "porter" and r[other] == BASE_PREPROC[other]]
            else:
                subset = [r for r in subset if r["lowercase"] == BASE_PREPROC["lowercase"]
                          and r["min_token_len"] == BASE_PREPROC["min_token_len"]]
            if subset:
                factor_summary.append({"factor": factor, "level": level, "n_configs": len(subset),
                                       "mean_auc": float(np.mean([r["auc"] for r in subset])),
                                       "best_auc": max(r["auc"] for r in subset),
                                       "best_oracle_acc": max(r["oracle_acc"] for r in subset)})
    write_csv(os.path.join(args.out_dir, "factor_summary.csv"), factor_summary,
              ["factor", "level", "n_configs", "mean_auc", "best_auc", "best_oracle_acc"])

    # Best configuration per metric (each metric at its own best preprocessing / aggregation).
    best_per_metric = [next(r for r in rows if r["metric"] == m) for m in METRICS]
    best_per_metric.sort(key=lambda r: -r["auc"])

    # Paired bootstrap over facts: 95% CIs, and whether the top configurations, the best configuration of every
    # metric and every one-factor ablation differ from the best (a difference CI that excludes 0 is significant).
    rng = np.random.default_rng(args.seed)
    top = rows[:args.top_k]
    compare = list(top)
    for r in best_per_metric + [by_name[a["config"]] for a in ablation]:
        if r not in compare:
            compare.append(r)
    n = len(gold)
    boot = {r["config"]: [] for r in compare}
    for _ in range(args.bootstrap):
        idx = rng.integers(0, n, n)
        g = gold[idx]
        if g.min() == g.max():
            continue
        for r in compare:
            boot[r["config"]].append(roc_auc(all_scores[r["config"]][idx], g))
    best_boot = np.array(boot[best["config"]])
    bootstrap_rows = []
    for r in compare:
        b = np.array(boot[r["config"]])
        d = b - best_boot
        bootstrap_rows.append({"config": r["config"], "auc": r["auc"],
                               "auc_ci_low": float(np.percentile(b, 2.5)), "auc_ci_high": float(np.percentile(b, 97.5)),
                               "diff_vs_best_ci_low": float(np.percentile(d, 2.5)),
                               "diff_vs_best_ci_high": float(np.percentile(d, 97.5)),
                               "p_best_better": float((d < 0).mean())})
    write_csv(os.path.join(args.out_dir, "bootstrap.csv"), bootstrap_rows,
              ["config", "auc", "auc_ci_low", "auc_ci_high", "diff_vs_best_ci_low", "diff_vs_best_ci_high",
               "p_best_better"])

    # Per-fact scores for the top configurations.
    score_rows = []
    for i, ex in enumerate(examples):
        row = {"fact": ex.fact, "label": ex.label, "gold": "S" if gold[i] else "NS"}
        for k, r in enumerate(top[:5]):
            row["score_top%d" % (k + 1)] = round(float(all_scores[r["config"]][i]), 6)
        score_rows.append(row)
    write_csv(os.path.join(args.out_dir, "scores_top.csv"), score_rows,
              ["fact", "label", "gold"] + ["score_top%d" % (k + 1) for k in range(min(5, len(top)))])

    # Per-fact scores of EVERY configuration (one column per config, in grid.csv order), so ROC curves, ablations and
    # any other comparison can be redrawn later without recomputing scores.
    with open(os.path.join(args.out_dir, "scores_all.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fact_id", "fact", "label", "gold"] + [r["config"] for r in rows])
        for i, ex in enumerate(examples):
            writer.writerow([i, ex.fact, ex.label, int(gold[i])]
                            + ["%.6f" % all_scores[r["config"]][i] for r in rows])

    # Score distribution of the best configuration (10 equal-width bins on [0, 1]).
    best_scores = all_scores[best["config"]]
    edges = np.linspace(0, 1, 11)
    bins = np.clip(np.digitize(best_scores, edges[1:-1]), 0, 9)
    distribution = [{"bin": "[%.1f, %.1f%s" % (edges[b], edges[b + 1], "]" if b == 9 else ")"),
                     "n_S": int(((bins == b) & (gold == 1)).sum()), "n_NS": int(((bins == b) & (gold == 0)).sum())}
                    for b in range(10)]
    write_csv(os.path.join(args.out_dir, "distribution.csv"), distribution, ["bin", "n_S", "n_NS"])

    import nltk
    import spacy
    run_config = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": git_commit(),
                  "python": platform.python_version(), "numpy": np.__version__, "nltk": nltk.__version__,
                  "spacy": spacy.__version__, "args": vars(args), "n_examples": int(n), "n_S": int(gold.sum()),
                  "n_configs": len(rows), "best_config": best["config"],
                  "total_runtime_s": round(time.perf_counter() - run_start, 1)}
    with open(os.path.join(args.out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    summary = ["# Member 1 grid results", "",
               "Generated by `experiments/m1_preprocessing/run_grid.py` at %s (commit %s), %d examples (%d S / %d NS),"
               " %d configurations." % (run_config["timestamp"], run_config["git_commit"], n, gold.sum(),
                                        n - gold.sum(), len(rows)),
               "", "AUC is threshold-free. Oracle accuracy uses the best threshold on the same data (optimistic).",
               "", "## Top configurations", "",
               md_table(rows[:15], ["config", "auc", "oracle_acc", "oracle_threshold", "mean_S", "mean_NS"],
                        floats=("auc", "oracle_acc", "oracle_threshold", "mean_S", "mean_NS")),
               "", "## Best configuration per metric", "",
               md_table(best_per_metric, ["metric", "config", "auc", "oracle_acc"], floats=("auc", "oracle_acc")),
               "", "## Ablation around the best configuration", "",
               md_table(ablation, ["factor", "level", "auc", "delta_auc", "oracle_acc"],
                        floats=("auc", "delta_auc", "oracle_acc")),
               "", "## Factor levels over the grid", "",
               md_table(factor_summary, ["factor", "level", "n_configs", "mean_auc", "best_auc", "best_oracle_acc"],
                        floats=("mean_auc", "best_auc", "best_oracle_acc")),
               "", "## Bootstrap (%d resamples, seed %d)" % (args.bootstrap, args.seed), "",
               md_table(bootstrap_rows, ["config", "auc", "auc_ci_low", "auc_ci_high", "diff_vs_best_ci_low",
                                         "diff_vs_best_ci_high", "p_best_better"],
                        floats=("auc", "auc_ci_low", "auc_ci_high", "diff_vs_best_ci_low", "diff_vs_best_ci_high",
                                "p_best_better")),
               "", "## Score distribution of the best configuration", "",
               md_table(distribution, ["bin", "n_S", "n_NS"]),
               "", "## Preprocessing time (all unique texts, cold cache, median of %d passes)" % TIMING_REPEATS, "",
               md_table(sorted({(r["remove_stopwords"], r["normalizer"], r["lowercase"], r["min_token_len"]):
                                r for r in rows}.values(), key=lambda r: r["preprocess_s"]),
                        ["remove_stopwords", "normalizer", "lowercase", "min_token_len", "preprocess_s"])]
    with open(os.path.join(args.out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")

    print("\nBest: %s  AUC=%.4f  oracle acc=%.4f @ %.4f" % (best["config"], best["auc"], best["oracle_acc"],
                                                            best["oracle_threshold"]))
    print("Wrote results to " + args.out_dir)


if __name__ == "__main__":
    main()
