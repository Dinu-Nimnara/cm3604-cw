# CM3604 Deep Learning — Fact-Checking Coursework

Group coursework for CM3604 (Deep Learning). We build and evaluate fact-checkers
that decide whether a short factual claim is **Supported (S)** or **Not Supported
(NS)** against retrieved Wikipedia passages, using (1) lexical word overlap and
(2) a pre-trained textual-entailment model (DeBERTa-v3).

Based on the FActScore dataset and setup.

---

## Requirements

- **Python 3.10** (required by the brief — do not use 3.11/3.12/3.13)
- The exact package versions in `requirements.txt`

The two graded commands must run without errors:

```bash
python factchecking_main.py --mode word_overlap
python factchecking_main.py --mode entailment
```

---

## Setup

Any tool that gives you Python 3.10 works (conda or venv). The brief only
requires the version and the pinned packages, not a specific tool.

### Option A — conda

```bash
conda create -n cm3604 python=3.10 -y
conda activate cm3604
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Option B — venv (needs Python 3.10 installed first)

```bash
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1      # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Verify the environment

```bash
python -c "import torch, numpy, spacy, nltk, transformers; print(torch.__version__, numpy.__version__)"
# expected: 2.5.0+cpu 2.0.2
```

### Test the pipeline

```bash
python factchecking_main.py --mode random
```

Expected: reads 221 examples, prints a confusion matrix and accuracy near 50%.

---

## Data

The dataset is **not committed** to the repo. Place these two files in `data/`:

```
data/
├── dev_labeled_ChatGPT.jsonl
└── passages_bm25_ChatGPT_humfacts.jsonl
```

These are the default paths the driver expects. Dataset stats: 221 facts
(112 Supported, 109 Not Supported), ~5 passages per fact.

---

## Project structure

```
cm3604-cw/
├── factchecking_main.py     # given — DO NOT EDIT
├── factcheck.py             # the graded file — we implement here
├── requirements.txt         # given, pinned versions
├── README.md
├── data/                    # dataset (not committed)
├── experiments/             # experiment scripts (not submitted)
│   ├── m1_preprocessing/
│   ├── m2_threshold/
│   ├── m3_entailment/
│   └── m4_pruning/
├── results/                 # tables, plots, cached scores (not committed)
├── error_analysis/          # Part 3 working files
└── report/                  # PDF write-up drafts
```

Only `factcheck.py` and the final PDF are submitted. `factcheck.py` must work
standalone — it must never import from `experiments/`.

---

## Tasks and ownership

| Part | Weight | Target | Owner |
|------|--------|--------|-------|
| 1. Word overlap | 40% | ≥75% accuracy | Member 1 + Member 2 |
| 2. Textual entailment | 40% | ≥83% accuracy | Member 3 + Member 4 |
| 3. Error analysis | 20% | 10 FP + 10 FN, 2–4 categories | Member 4 / group |

| Member | Responsibility |
|--------|----------------|
| Member 1 | Word preprocessing + similarity metric implementation |
| Member 2 | Binary classification + threshold tuning + evaluation |
| Member 3 | DeBERTa inference + entailment-based classification |
| Member 4 | Word-overlap pruning + DeBERTa optimization and evaluation |

---

## Constraints (read before coding)

- **Use only packages in `requirements.txt`.** No scikit-learn — implement
  TF-IDF / Jaccard / recall manually.
- **No runtime downloads** in `factcheck.py`. Hardcode the stopword list, use
  `PorterStemmer` (no download needed), and provide a regex fallback for
  sentence splitting.
- `factchecking_main.py` builds the checkers with **no constructor arguments** —
  hardcode tuned thresholds and settings as class constants.
- In `EntailmentModel.check_entailment`, remove the `raise` before the
  `del`/`gc.collect()` cleanup, and confirm label order via
  `model.config.id2label`.
- Code must run on **CPU without out-of-memory errors** (an explicit grading
  criterion). Use `torch.no_grad()`, batch pairs, cap pairs per fact.

---

## GenAI policy

This assessment is **category ❸ (Editing only)**. GenAI may be used to proofread
or polish wording, not to generate ideas, code, experiments, or analysis. If
used for editing, add the acknowledgement statement from the brief.

---

## Deadline

**11 November 2026, 23:59** (48-hour automatic extension available to 13 Nov —
emergency buffer only). Internal target: finish by 10 Nov.

Submit separately: the PDF write-up and `factcheck.py`.