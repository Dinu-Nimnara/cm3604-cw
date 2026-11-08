# Member 2 verified results

Data: 221 facts, 112 Supported and 109 Not Supported (including 16 IR).
Score: Member 1's Porter-preprocessed recall, maximum over passages, page-title tokens removed.
Rule: predict S iff score > 0 and score >= 0.5634920634920635.

| Evaluation | Accuracy | Correct / total |
|---|---:|---:|
| Development-tuned threshold | 83.2579% | 184 / 221 |
| Threshold-only stratified five-fold CV, seed 0 | 80.5430% | 178 / 221 |
| Always S | 50.6787% | 112 / 221 |
| Always NS | 49.3213% | 109 / 221 |

| Gold label / predicted label | S | NS |
|---|---:|---:|
| S | 86 | 26 |
| NS (including IR) | 11 | 98 |

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| S | 0.8866 | 0.7679 | 0.8230 |
| NS | 0.7903 | 0.8991 | 0.8412 |

Macro-F1: 0.8321. There are 11 false positives and 26 false negatives.
The coursework word-overlap target of 75% is met on its permitted development evaluation.

Threshold search covers every attainable decision pattern and optimizes accuracy.
CV tunes the threshold independently on each training partition; the final threshold uses all development data.
Development accuracy is not held-out test accuracy. CV assesses threshold selection only: Member 1's
preprocessing was selected on the full development set. Facts from a shared biography can occur across folds.

Reproducibility: Python 3.10.22, pinned coursework dependencies, CPU. Ten automated tests pass.
The original coursework command succeeds without editing factchecking_main.py.
See the JSON/CSV exports and execution logs alongside this file for complete results and data hashes.
No claim is made here about entailment performance, which is the other members' work.
