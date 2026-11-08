# Member 2: binary classification, threshold tuning and evaluation

## Status

Implemented against repository commit `075d977912df24bf5870e9ea16e70fa10aec59d1` and the uploaded DL_CW.pdf.
Member 1's preprocessing, recall score, maximum-passage aggregation and title removal are preserved.
The supplied driver and requirements.txt are unchanged. No changes have been pushed to GitHub.

**Verified on the uploaded coursework data:** 221 facts (112 S, 93 NS, 16 IR mapped to NS).
The selected full-precision threshold is `0.5634920634920635`, now set as the classifier default.
Development-tuned accuracy is **184/221 = 83.2579%**; threshold-only five-fold CV pooled accuracy is
**178/221 = 80.5430%**. Both exceed 75%, although the limitations below still apply.
The unmodified `python factchecking_main.py --mode word_overlap` command completed successfully
and independently printed the same confusion matrix and accuracy.
Verification used Python 3.10.22 and every pinned requirement, with torch 2.5.0+cpu and torchvision 0.20.0+cpu.
See `results/m2/environment.txt` for installed versions. No spaCy language-model download is needed
for the selected Porter configuration.

## Files and integration

- `factcheck.py`: only the Member 2 constructor/default threshold and `predict()` in
  `WordRecallThresholdFactChecker` are changed. It retains Member 1's `score()` interface.
- `experiments/m2_classification/evaluation.py`: threshold search, binary metrics, stratified CV.
- `experiments/m2_classification/run_evaluation.py`: dataset reader, scoring and reproducible result exports.
- `experiments/m2_classification/test_m2.py` and `test_integration.py`: ten automated tests.
- `results/m2/`: verified metrics, predictions, errors, threshold sweep and execution logs.

Copy the updated `factcheck.py` and the whole `experiments/m2_classification` directory into your
existing project. If teammates have changed factcheck.py meanwhile, merge only the
`WordRecallThresholdFactChecker` changes using `member2.patch`; do not overwrite their newer work.
The patch is based on the commit above. From the repository root, check it with
`git apply --check member2.patch`, then apply with `git apply member2.patch`.

## Classification rule

For Member 1's maximum passage recall score r, predict S when r > 0 and r >= t;
otherwise predict NS. Equality is explicitly included. The r > 0 condition prevents an empty
fact, missing passages or zero overlap from becoming supported when t = 0.
NS and IR are negative labels, consistent with the supplied driver. S is the positive label.
The score is lexical similarity, not a calibrated probability of factual correctness.

## Run in the coursework environment

Use Python 3.10 and the unchanged requirements.txt, as the coursework specifies.
Place the original coursework files at:

```
data/dev_labeled_ChatGPT.jsonl
data/passages_bm25_ChatGPT_humfacts.jsonl
```

From the repository root:

```bash
python -m unittest discover -s experiments/m2_classification -v
python experiments/m2_classification/run_evaluation.py --data_dir data --out_dir results/m2 --folds 5 --seed 0
```

The verified selected threshold is already installed in the supplied factcheck.py.
For future retuning, read `results/m2/summary.md`. Copy the full-precision `selected_threshold` from
`results/m2/evaluation.json` into `WordRecallThresholdFactChecker.DEFAULT_THRESHOLD`.
Then rerun evaluation so its default-threshold metrics match the selected-threshold metrics, and run:

```bash
python factchecking_main.py --mode word_overlap
```

This last step is necessary because the unmodified driver creates the checker with no arguments;
it does not automatically read the generated evaluation.json. The code deliberately does not fit
thresholds during inference or read labels inside predict().

## Threshold search and evaluation design

1. Compute Member 1's score once per fact, with the preprocessing configuration fixed.
2. Sort distinct positive scores. Evaluate a midpoint in every gap, t=0 (all nonzero
   scores supported), and a threshold just above 1 (all unsupported). These cover every
   decision pattern attainable under the classification rule, including duplicate-score ties.
3. Maximize development accuracy, as requested by the coursework. If several thresholds tie,
   choose the one closest to 0.5, then the lower value. This is deterministic.
4. Separately shuffle each binary class using seed 0 into five stratified folds.
   Tune on four folds and predict only the held-out fold; pool the held-out predictions.
   Each class must contain at least five examples, or use a smaller `--folds` value.
5. Report development-tuned and threshold-CV results separately, plus the current default
   and both always-S and always-NS baselines. Do not label the development result as test accuracy.

Confusion matrices use gold rows and prediction columns, both ordered [S, NS]:
`[[TP, FN], [FP, TN]]`. Precision = TP/(TP+FP), recall = TP/(TP+FN),
F1 = 2TP/(2TP+FP+FN), accuracy = (TP+TN)/N. Per-class metrics with zero denominators
are defined as 0. Macro-F1 averages the S and NS F1 values.

CV here validates **threshold selection only**. Member 1 chose preprocessing using the whole
development set, so these scores are not a fully independent evaluation of the complete pipeline.
Facts concerning the same biography may share passages across folds. Unseen-biography performance
would require grouped validation and preprocessing selection within each training split.
The brief explicitly permits tuning and evaluation on its single development set.

## Outputs

- `evaluation.json`: selected threshold, confusion matrix, per-class metrics, macro-F1,
  baselines, fold thresholds/metrics, pooled CV, dataset SHA-256 hashes and configuration.
- `threshold_sweep.csv`: sensitivity across every attainable threshold decision.
- `predictions.csv`: full-precision scores, original and binary labels, tuned/default/CV predictions.
- `errors.jsonl`: all development-tuned lexical errors, including evidence passages for inspection.
- `summary.md`: generated numerical summary and limitations.

The errors are **word-overlap errors**. Coursework Part 3 requires errors from the entailment
model, so do not submit these as the required entailment error analysis.

## Verification completed here

Ten tests passed under Python 3.10.22. Eight unit tests cover confusion-matrix orientation,
IR mapping, zero denominators, threshold boundaries, exhaustive threshold decision comparisons,
ties, deterministic CV and input validation. Two integration tests exercise the real Member 1
scoring pipeline, check empty inputs, compare our dataset reader with the supplied driver's reader,
and verify that the fitted threshold is installed and produces the expected evaluation accuracy.
The unmodified coursework word-overlap command also passed on all 221 examples.
The driver and requirements.txt are byte-identical to the base Git commit.

## Other members' work

The repository's entailment methods still contain `Not implemented` placeholders.
`python factchecking_main.py --mode entailment` will require Members 3 and 4 to finish their work.
Your Member 2 changes do not complete the whole group's coursework or its Part 3 PDF.
