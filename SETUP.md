# Team Setup Guide — CM3604 Fact-Checking

How to get the project running on your machine, check that it works, and use what is already implemented.
The README has the project overview; this file is the hands-on part.

**Current status:** Part 1 preprocessing and similarity scoring (Member 1) is done. `predict()` for word overlap
(Member 2) and everything in Part 2 (Members 3 and 4) are still to do.

---

## 1. Environment (once per machine)

The brief requires **Python 3.10** and the **exact** versions in `requirements.txt`. Do not use 3.11+.

```bash
conda create -n cm3604 python=3.10 -y
```
```bash
conda activate cm3604
```
```bash
pip install -r requirements.txt
```
```bash
python -m spacy download en_core_web_sm
```

Check it:
```bash
python -c "import sys, torch, numpy, nltk, spacy, transformers; print(sys.version.split()[0], torch.__version__, numpy.__version__, nltk.__version__, spacy.__version__, transformers.__version__)"
```
Expected: `3.10.x 2.5.0(+cpu) 2.0.2 3.9.1 3.8.2 4.45.2`.

> **Windows tip:** if `python --version` shows anything other than 3.10 after `conda activate cm3604`, another
> Python is first on your PATH. Call the environment's interpreter directly, e.g.
> `C:\Users\<you>\.conda\envs\cm3604\python.exe`.

## 2. Data (not in git)

The dataset is git-ignored and must not be redistributed. Put these two files in `data/` at the repository root:

```
data/dev_labeled_ChatGPT.jsonl
data/passages_bm25_ChatGPT_humfacts.jsonl
```

Those are the default paths of `factchecking_main.py`. To keep the data elsewhere, pass `--labels_path` and
`--passages_path` to `factchecking_main.py`, or `--data_dir` to the Member 1 experiment script.

## 3. Check that everything works

Run these from the repository root.

**1. The pipeline runs.** This is the random baseline; you should see 221 examples and accuracy around 50%:
```bash
python factchecking_main.py --mode random
```

**2. The Member 1 unit tests pass.** Expect `Ran 42 tests … OK`:
```bash
python -m unittest discover -s experiments/m1_preprocessing -v
```

**3. The graded command.** `python factchecking_main.py --mode word_overlap` currently stops with
`Exception: Implement me` until Member 2 adds `predict()`. That is expected.

---

## 4. What is already implemented in `factcheck.py` (Member 1) and how to use it

All of it is hand-written, using no scikit-learn and no runtime downloads.

| Name | What it does | Who needs it |
|---|---|---|
| `WordRecallThresholdFactChecker.score(fact, passages)` | word-overlap support score in **[0, 1]**; higher = more supported | **Member 2** (threshold), **Member 4** (pruning) |
| `word_overlap_score(fact, passages, preprocessor, metric, aggregation, drop_title)` | the same score with every option exposed (6 metrics, 3 aggregations) | experiments |
| `TextPreprocessor(...).process(text)` | clean → lowercase → tokenise → remove stop words → Porter stem; cached | anyone who needs tokens |
| `split_sentences(passage_text)` | splits passages on the `<s>…</s>` markers, then on sentence punctuation (regex only, no download) | **Member 3** (sentence-level entailment) |
| `clean_text`, `tokenize` | the first two preprocessing steps on their own | — |

**Chosen settings** (class constants on `WordRecallThresholdFactChecker`): recall metric, Porter stemming, stop
words removed (negations kept), lowercase, tokens of at least 2 letters (numbers always kept), the person's
name removed, and the max over passages. They were chosen by ROC-AUC (0.8975) out of 432 configurations; see
`experiments/m1_preprocessing/RESULTS.md`.

**Contract for Member 2:**
- `score()` is deterministic and returns 0.0 when there are no passages or the fact is empty.
- `predict()` should return `"S"` when `self.score(fact, passages) >= THRESHOLD`, otherwise `"NS"`.
  `THRESHOLD` must be a class constant, because `factchecking_main.py` builds the checker with no arguments.
- Recall takes discrete values (k/n). The most ambiguous facts sit at exactly 0.5, so the threshold choice there
  matters. The value 0.5635 in RESULTS.md is the best threshold *on the same data* (optimistic). Please
  validate your own, e.g. with k-fold.

**For Member 4 (pruning):** create one `WordRecallThresholdFactChecker()` and reuse it inside
`EntailmentFactChecker`. Its cache makes repeated passages free. Facts with a very low `score()` can skip
DeBERTa.

---

## 5. Reproducing the Member 1 experiment and figures (optional)

**Experiment.** Writes CSVs and `summary.md` to `results/m1/` (git-ignored); takes under a minute on a laptop
CPU:
```bash
python experiments/m1_preprocessing/run_grid.py --data_dir data --out_dir results/m1
```

**Figures.** The notebook needs plotting tools that are **not** part of the graded requirements. Install them
with `requirements.txt` as a *constraint*, so no graded package version can change:
```bash
pip install -r experiments/m1_preprocessing/requirements-analysis.txt -c requirements.txt
```
Register a Jupyter kernel for this environment. The notebook expects a kernel named `cm3604`:
```bash
python -m ipykernel install --user --name cm3604 --display-name "Python 3.10 (cm3604)"
```
Run the notebook headless; it rewrites `report/figures/m1_*.png`:
```bash
python -m jupyter nbconvert --to notebook --execute --inplace experiments/m1_preprocessing/m1_analysis.ipynb
```
Or open it with `python -m jupyter notebook`, choose the kernel "Python 3.10 (cm3604)", and run
Kernel → Restart & Run All. If your data is not in `data/`, set the environment variable `CM3604_DATA_DIR` first.

---

## 6. Folder map

| Path | In git? | Contents |
|---|---|---|
| `factcheck.py` | yes | **the graded file** (everyone implements here) |
| `factchecking_main.py` | yes | given driver: **do not edit** |
| `requirements.txt` | yes | pinned versions: **do not edit** |
| `data/` | no | dataset (see section 2) |
| `experiments/m1_preprocessing/` | yes | Member 1 experiment, tests, analysis notebook, `RESULTS.md` |
| `experiments/m2_threshold/`, `m3_entailment/`, `m4_pruning/` | (to create) | per-member experiment scripts, as in the README |
| `results/` | no | experiment outputs; regenerate with the scripts |
| `report/figures/` | yes | figures for the write-up (`*.png` here is tracked) |

`factcheck.py` must never import from `experiments/`. Experiments import from `factcheck`, not the other way round.

## 7. Rules from the brief (quick checklist)

- Python 3.10 and only the packages in `requirements.txt`; **no scikit-learn**. Implement TF-IDF and similar
  metrics by hand.
- Do **not** modify `factchecking_main.py`. Before committing, check that `git diff factchecking_main.py` is
  empty; even stray blank lines count.
- No downloads at runtime inside `factcheck.py`. Hard-code lists, and use a regex for sentence splitting.
- Settings and thresholds are class constants (no constructor arguments).
- Entailment must run on CPU without running out of memory: `torch.no_grad()`, batching, and `del` / `gc.collect()`.
- Both graded commands must run without errors before submission:
  `python factchecking_main.py --mode word_overlap` and `python factchecking_main.py --mode entailment`.

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: nltk` (or similar) | wrong Python | activate `cm3604`, or call its `python.exe` directly |
| Notebook: `numpy.dtype size changed, may indicate binary incompatibility` | the Jupyter kernel is another environment's Python | register the `cm3604` kernel (section 5) and select it |
| `jupyter: command not found` | the env's `Scripts` folder is not on PATH | use `python -m jupyter …` |
| `FileNotFoundError … data/...jsonl` | dataset not in `data/` | see section 2 |
| `RuntimeWarning: Proactor event loop …` when running nbconvert | harmless Windows/zmq warning | ignore |
| `Exception: Implement me` in `--mode word_overlap` | `predict()` not written yet | Member 2's task |

## 9. Running on Google Colab (optional)

Colab ships a newer Python than 3.10, so treat Colab results as a convenience, not the graded environment.
For an exact match, create a Python 3.10 virtual environment inside Colab and run the scripts with its
interpreter. Upload `data/` to Drive and point at it with `--data_dir`. Colab's disk is wiped when the session
ends, so copy `results/` back to Drive. A Colab GPU (`--cuda`) mainly helps Part 2 (DeBERTa). Before submitting,
always confirm both graded commands locally on Python 3.10.

## 10. Git workflow

- Work on your own branch (e.g. `member2/threshold`), and merge into `main` when your part runs and its tests pass.
- Never commit `data/`, `results/`, virtual environments or personal notes.
- Pull `main` before you start working, so you build on the latest `factcheck.py`.
