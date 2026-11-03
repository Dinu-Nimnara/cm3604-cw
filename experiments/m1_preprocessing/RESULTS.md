# Member 1 Results — Preprocessing & Similarity (Part 1)

All numbers below come from the final run of `run_grid.py` (3 Nov 2026, 11:59, in the main project folder): Python 3.10.20, nltk 3.9.1,
numpy 2.0.2, spaCy 3.8.2, base commit `282527d` plus uncommitted Member 1 changes. The AUC, accuracy and
bootstrap numbers are **byte-identical** across repeated runs with different Python hash seeds. The full tables
are in `results/m1/summary.md`, which is git-ignored, so regenerate them with the commands at the bottom. The
figures are in `report/figures/m1_*.png`, drawn by `m1_analysis.ipynb`.

Data: 221 facts, 112 S and 109 NS (93 NS + 16 IR mapped to NS). Grid: 12 preprocessing configs × 6 metrics ×
3 aggregations × 2 title options = **432 configurations**.

**How configurations are compared:** ROC-AUC, which is threshold-free, so choosing preprocessing does not
depend on Member 2's threshold. "Oracle acc" is the best accuracy over all thresholds *on the same data*. It
is a diagnostic only, and it is optimistic.

## Chosen configuration (now the defaults in `WordRecallThresholdFactChecker`)

| Setting | Value |
|---|---|
| lowercase | yes |
| tokenisation | regex `[a-z0-9]+` after cleaning (`<s>` tags, accents, possessives, apostrophes, digit commas) |
| stop words | NLTK English list, hard-coded, negations kept |
| normaliser | Porter stemmer (applied after stop-word removal) |
| min token length | 2 (digits always kept) |
| entity-name (title) tokens | removed from the fact and passages |
| metric | recall = \|F ∩ P\| / \|F\| |
| aggregation | max over passages |
| **AUC** | **0.8975** (bootstrap 95% CI 0.850–0.937) |
| oracle acc | 0.8326 at threshold 0.5635 |
| mean score S / NS | 0.789 / 0.267 |

## Evidence for each decision (one factor changed from the chosen config)

"Δ CI" is the paired-bootstrap 95% CI of AUC(variant) − AUC(chosen), from 2000 resamples with seed 0. An
interval entirely below 0 means the chosen setting is significantly better.

| Decision | Alternative | AUC | Δ AUC | Δ CI | Conclusion |
|---|---|---|---|---|---|
| metric = recall | precision | 0.7585 | −0.139 | [−0.199, −0.086] | significantly worse |
| | Jaccard | 0.7672 | −0.130 | [−0.188, −0.080] | significantly worse |
| | F1 | 0.7672 | −0.130 | [−0.188, −0.079] | significantly worse |
| | TF-IDF cosine | 0.8596 | −0.038 | [−0.077, −0.004] | significantly worse |
| | IDF-weighted recall | 0.8955 | −0.002 | [−0.008, 0.004] | equivalent |
| normaliser = Porter | none | 0.8718 | −0.026 | [−0.045, −0.010] | significantly worse |
| | spaCy lemma | 0.8861 | −0.011 | [−0.029, 0.001] | worse, borderline; also far slower (final run: 2.9 s vs 0.084 s; about 35–65× across runs) |
| | Snowball | 0.8975 | 0 | — | identical scores on this data |
| drop title = yes | keep name | 0.8794 | −0.018 | [−0.042, 0.006] | worse in 93% of resamples, not significant |
| stop words removed | kept | 0.8861 | −0.011 | [−0.038, 0.016] | worse, not significant |
| aggregation = max | concat | 0.8869 | −0.011 | [−0.024, 0.001] | worse, borderline |
| | sentence max | 0.8913 | −0.006 | [−0.027, 0.016] | equivalent |
| lowercase | cased | 0.8950 | −0.003 | — | equivalent |
| min token len 2 | 1 | 0.8953 | −0.002 | [−0.007, 0.000] | equivalent |

Grid-wide means (each level averaged over the same other settings) point the same way. Mean AUC with stop
words removed is 0.791 vs 0.715 with them kept. By metric: recall 0.865, IDF-recall 0.873, TF-IDF 0.800,
Jaccard 0.677, F1 0.677, precision 0.629.

Best configuration of each metric (each at its own best preprocessing and aggregation): recall 0.8975,
IDF-weighted recall 0.8957, TF-IDF cosine 0.8844 (sentence max), F1 0.8050, Jaccard 0.8040, precision 0.7885.
At the oracle threshold, the chosen configuration makes 11 false positives and 26 false negatives (184/221
correct).

## Interpretation (for the report)

1. **Asymmetric metrics win.** Facts have about 7 words and passages about 875 characters. Precision, Jaccard
   and F1 divide by the passage size, so their scores mostly measure passage length. Jaccard ≤ |F|/|P|. Recall
   divides only by the fact. That explains the gap of about 0.13 AUC, which is the largest effect in the grid.
2. **TF-IDF cosine is second-tier.** Unmatched passage words lengthen ‖P‖, so even a fully supported fact
   scores below 1. It does best with sentence-level aggregation (0.884), where the passage vectors are short,
   which supports the same explanation.
3. **IDF-weighted recall matches recall plus name-dropping.** IDF down-weights the entity name automatically,
   because it appears in most sentences. The hand-made `drop_title` rule and the principled IDF weighting reach
   the same AUC. We keep plain recall because it is simpler and easier to explain at a threshold.
4. **Stemming helps significantly; lemmatisation does not beat it.** Inflection mismatches (`studied` /
   `studies`) are frequent. spaCy is slower and makes surprising choices, such as lemmatising `born → bear`.
5. **Removing the name helps.** Without it, "X is a singer" (NS) gets credit for the name tokens. Remaining
   noise is small (bootstrap), so this is a justified but not statistically conclusive choice.
6. **Differences among the top ~10 configurations are within bootstrap noise.** Their CIs on the difference
   include 0. Report this honestly. The chosen config is the simplest member of a group of equivalent ones.

## Score distribution of the chosen config (for Member 2)

| bin | S | NS |
|---|---|---|
| [0.0, 0.1) | 4 | 39 |
| [0.1, 0.2) | 0 | 3 |
| [0.2, 0.3) | 2 | 24 |
| [0.3, 0.4) | 1 | 9 |
| [0.4, 0.5) | 2 | 5 |
| [0.5, 0.6) | 22 | 19 |
| [0.6, 0.7) | 10 | 3 |
| [0.7, 0.8) | 7 | 2 |
| [0.8, 0.9) | 4 | 0 |
| [0.9, 1.0] | 60 | 5 |

Recall takes only the values k/n for a fact with n content words, so scores are discrete. The ambiguous mass
is at 0.5 (half the words matched, 22 S vs 19 NS). An oracle threshold of 0.5635 means "S iff more than half
matched". **The threshold is Member 2's decision**, and it should be validated with k-fold rather than taken
from this oracle number.

## Clear errors of the chosen config (for Part 3 / Member 4)

Content words shown are after preprocessing and name removal.
- **NS scored 1.0 (words present, claim not supported):** "Marianne McAndrew is an American." [american];
  "Trivisonno worked at WKNR." [work, wknr]; "He served as a sniper." [serv, sniper]; "Kerser is from Sydney."
  [sydney]; "Cleveland is in Ohio." (IR) [cleveland, ohio]. Bag-of-words cannot check the relation between the
  words, and IR facts can be lexically "supported".
- **S scored 0.0 (supported, but different wording):** "Florencia Bertotti is an Argentinian." [argentinian]
  (the passages say "Argentina"/"Argentine"); "Hibo Wardere is based in the United Kingdom." [base, unit,
  kingdom] (the passages say "England"/"London"); "Haagensen later transitioned to television." [later, transit,
  televis]. These are paraphrase or derivational mismatches that stemming cannot bridge.
- **S scored 0.0 (evidence missing from the retrieved text):** "They divorced in 1964." [divorc, 1964]. Neither
  "divorc…" nor "1964" occurs in any of its passages, and the subject is a pronoun. No lexical method can
  recover this; it is a retrieval/annotation issue worth noting in Part 3.

## Limitations

- The configuration and threshold are selected on the same 221 facts as they are evaluated on (the spec has
  no separate test set). AUC-based selection avoids threshold overfitting, but some selection bias remains.
- Bag-of-words ignores word order, negation scope and relations, and cannot match synonyms or paraphrases.
  These are the motivation for Part 2 (entailment).

## Reproduce

```bash
python -m unittest discover -s experiments/m1_preprocessing -v
python experiments/m1_preprocessing/run_grid.py --data_dir data --out_dir results/m1
```
```bash
jupyter nbconvert --to notebook --execute --inplace experiments/m1_preprocessing/m1_analysis.ipynb
```
The run is deterministic, and the bootstrap uses `--seed 0`. It runs on CPU with no model downloads beyond
`en_core_web_sm`. The final run took 43.2 s (each preprocessing setting is timed 3 times). Earlier runs took
29.7–162.8 s depending on CPU throttling; the results were identical every time.
