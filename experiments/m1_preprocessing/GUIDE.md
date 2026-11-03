# Member 1 Guide — Preprocessing & Similarity Metrics (Part 1, Word Overlap)

Working notes for Member 1. Not submitted. Everything marked **Expectation** is a
hypothesis to test — none of it is a result until your script has produced it.

---

## 0. What the repository and data actually contain (checked 2 Nov 2026)

| Fact | Consequence for you |
|---|---|
| `factcheck.py` → `WordRecallThresholdFactChecker.predict` raises `"Implement me"`; no `__init__`. | You add preprocessing + scoring here. `factchecking_main.py` calls `WordRecallThresholdFactChecker()` with **no args**, so every setting must be a class constant. |
| 221 examples: 112 S, 93 NS, 16 IR (IR → NS in `predict_two_classes`, so 112 vs 109). | Always-S baseline = 112/221 = **50.7 %**. Target ≥ 75 %. |
| Passages per fact: 2–5 (98 facts have 3, 66 have 2, 38 have 5, 19 have 4). README says "~5"; it is fewer. | Aggregation over a *variable* number of passages must be well-defined. |
| Passage length 23–1295 chars (mean ≈ 875). Fact length 4–14 words (mean ≈ 6.8). | Strong length asymmetry → precision/Jaccard are dominated by passage length (see §3). |
| **551 of 692 passages contain `<s>…</s>` tags.** | If you don't strip them, a regex tokenizer emits a spurious token `s` in every tagged passage. They are also free sentence boundaries. |
| Passage text starts with the page title, e.g. `"<s>Lanny Flaherty Lanny Flaherty (born …"`; every passage's `title` equals the fact's entity (asserted in `read_passages`). | Entity-name tokens match for almost every fact, S or NS → they inflate overlap without discriminating. Candidate preprocessing option: drop title tokens. |
| Facts contain dates/numbers: "born on November 21, 1942" (labelled **NS**). | Tokenizer must keep digits; numbers are among the most discriminative tokens. |
| Env `cm3604`: Python 3.10.20, nltk 3.9.1, numpy 2.0.2, spaCy 3.8.2, rouge-score installed. | Use `C:\Users\LENOVO\.conda\envs\cm3604\python.exe`. The base `python` is 3.14 — wrong. |
| `data/` is git-ignored, so it is not in a fresh clone. | Copy the two `.jsonl` files into `data/`, or pass `--data_dir` (scripts) / `--labels_path --passages_path` (`factchecking_main.py`). |
| `.gitignore` ignores `results/` and `*.csv`. | Results stay local. Share them via the report or a drive, or un-ignore a `results/m1/summary/` folder **as a group decision**. |
| Main checkout has an empty, untracked `experiments/preprocessing/`; README says `experiments/m1_preprocessing/`. | Follow the README (this folder). Delete the stray folder. |
| `pytest` is not in `requirements.txt`. | Write tests with stdlib `unittest`. |

---

## 1. Implementation plan (in order) mapped to the Grade A rubric

Grade A wording (page 8): *"well-justified preprocessing decisions (tokenization,
stemming/lemmatization, stop-word handling) supported by experiments or literature.
Correctly implements one or more appropriate similarity/scoring methods … with clear
mathematical understanding. Systematically tunes preprocessing choices … Demonstrates
critical evaluation of alternative methods, provides comprehensive experimental analysis
… Implementation is accurate, efficient, and reproducible."*

| # | Step | Where | Rubric phrase it earns |
|---|---|---|---|
| 1 | `clean_text`: strip `<s>` tags, Unicode normalise, possessives | `factcheck.py` | "technically sound", "tokenization" |
| 2 | `tokenize`: lowercase + regex alphanumeric tokens | `factcheck.py` | "tokenization" |
| 3 | Hard-coded `STOPWORDS` frozenset + removal | `factcheck.py` | "stop-word handling" |
| 4 | Normaliser: none / Porter / Snowball / spaCy lemma | `factcheck.py` | "stemming/lemmatization" |
| 5 | `TextPreprocessor` class that combines 1–4 by config, with caching | `factcheck.py` | "efficient" |
| 6 | Unit tests for 1–5 | `experiments/m1_preprocessing/test_m1.py` | "accurate" |
| 7 | Set metrics: recall, precision, F1, Jaccard | `factcheck.py` | "one or more … similarity methods" |
| 8 | TF-IDF: IDF, vectors, cosine (manual) | `factcheck.py` | "TF-IDF cosine … mathematical understanding" |
| 9 | (Optional, strong) IDF-weighted recall | `factcheck.py` | "critical evaluation of alternatives" |
| 10 | `score_fact(fact, passages)` + aggregation (max / concat / sentence-max) | `factcheck.py` | the interface Member 2 thresholds |
| 11 | Unit tests for 7–10, hand-computed values, cross-check vs `rouge_score` | `test_m1.py` | "accurate", "mathematical understanding" |
| 12 | Experiment grid: preprocessing × metric × aggregation → AUC + best accuracy | `experiments/m1_preprocessing/run_grid.py` | "systematically tunes", "comprehensive experimental analysis" |
| 13 | Hand the best config to Member 2; write report section + tables | report | "justifies all implementation decisions", "reproducible" |

Not yours: choosing the final threshold, k-fold threshold validation, the final accuracy
claim (Member 2); pruning (Member 4); entailment (Member 3).

---

## 2. Text preprocessing

The pipeline is a function `text → list[str]`. **Order matters:**

```
raw text
  → clean_text      (strip <s>, </s>; NFKD-normalise accents; drop possessive 's)
  → lowercase
  → tokenize        (regex [a-z0-9]+)
  → remove stopwords   ← BEFORE stemming
  → stem / lemmatise
  → (optional) drop title tokens
```

Why stopwords before stemming: Porter maps `was → wa` and `has → ha` (checked in the
`cm3604` env). If you stem first, `wa` no longer matches `was` in the stopword list and leaks
through as a content word.

### 2.1 Cleaning
- **What:** remove markup and normalise characters before tokenizing.
- **Why:** `<s>` would become token `s`. `"Beyoncé"` vs `"Beyonce"` should match. `"McAndrew's"`
  must match `"McAndrew"`. Porter alone gives `mcandrew's → mcandrew'`, which does not match.
- **Pseudocode:**
  ```
  clean_text(t):
      t = replace regex </?s> with " "
      t = unicodedata.normalize("NFKD", t); drop combining marks   # é → e
      t = replace regex ['’]s\b with ""                            # possessive
      return t
  ```
- **Edge cases:** `None` or `""` → `""`. A curly apostrophe `’` also appears in Wikipedia text.

### 2.2 Tokenization and lowercasing
- **Recommended:** `re.findall(r"[a-z0-9]+", text.lower())`. This also handles punctuation:
  every non-alphanumeric character becomes a boundary.
- **Effects:** `"(born July 27, 1942)" → born, july, 27, 1942`. `"singer-songwriter" → singer, songwriter`.
  `"U.S." → u, s`. Single letters are noise, so consider dropping tokens of length 1 that are not digits.
- **Alternatives to mention:** `str.split()` keeps punctuation glued on (`"actor."` ≠ `"actor"`), so it
  is clearly worse. A spaCy tokenizer is linguistically better but slower and gives the same output on
  these simple facts. NLTK `word_tokenize` needs the `punkt` download, which is banned at runtime.
- **Lowercasing trade-off (viva):** it merges `"Actor"`/`"actor"` (good) but turns the country `"US"`
  into the stopword `"us"` (bad, but rare). Bag-of-words cannot use case reliably anyway.

### 2.3 Stop-word removal
- **What:** drop function words (`is, an, the, was, in, of …`).
- **Why:** facts are short, e.g. "Lanny Flaherty is an actor." has 5 tokens and 2 are stopwords. Those
  stopwords appear in every passage, so they push recall toward 1 for every fact and shrink the S/NS
  gap. Removing them makes the score depend on content words.
- **How:** hard-code a `frozenset` (README: no `nltk.download`). Copy NLTK's English list (≈179
  words) into the file. **Decision to justify:** keep negations (`not, no, nor, never`) *out* of the
  stopword set, because they flip meaning. Facts here are almost all affirmative, so the measured effect
  will be small. Say that honestly.
- **Edge case:** a fact that is all stopwords after filtering (e.g. "He was there.") becomes empty.
  Fallback: if filtering leaves nothing, use the unfiltered tokens.

### 2.4 Stemming vs lemmatization

| Option | Example (checked) | Pros | Cons |
|---|---|---|---|
| none | studies ≠ studied | exact | misses inflections |
| Porter (`nltk.stem.PorterStemmer`) | studies, studied → `studi`; resides → `resid` | no download, fast with memo | non-words; `1960s → 1960` |
| Snowball (`SnowballStemmer("english")`) | same here, keeps `1960s` | slightly gentler Porter2 | still crude |
| spaCy lemma (`en_core_web_sm`) | `sang → sing`, `studied → study`, `universities → university` (stemmers leave `sang`) | real words, handles irregulars | slow (whole pipeline); context-dependent; needs the **original-case** text for POS tagging and returns proper nouns capitalised (`Flaherty`), so lowercase *after* lemmatizing |

Note: neither approach merges `actor`/`actress` (both stemmers keep them distinct, checked). If you
see that in the errors, it is an argument about the **limits of bag-of-words** and belongs in the
report.

**Efficiency:** Porter is slow per call, so memoise per word with a dict. For spaCy, use
`nlp(text, disable=["parser","ner"])` and cache per passage. Passages repeat across facts of the
same entity, so caching per passage text pays off.

### 2.5 Title (entity-name) tokens
- **Idea:** `passages[i]["title"]` is the entity. Preprocess it with the same pipeline and remove
  those tokens from the **fact** before scoring.
- **Why:** "Marianne McAndrew is a singer." (NS) and "Marianne McAndrew is an actress." (S) both get
  2/3 of their content tokens matched from the name alone. Removing the name leaves `{singer}` vs
  `{actress}`, which is the actual claim.
- **Risk:** an empty fact (use the §2.3 fallback). Some facts are about other entities ("Cleveland is
  in Ohio.") and are unaffected.
- **Alternative:** TF-IDF handles this automatically, because the name appears in nearly every sentence
  and gets low IDF. Comparing these two approaches is good "critical evaluation" material.

### 2.6 Configurations to compare, and why

| Factor | Levels | Question it answers |
|---|---|---|
| stopwords | off / on | Do function words dilute the signal? |
| normaliser | none / porter / snowball / lemma | Does merging inflections help, and is lemmatization worth its cost? |
| drop_title | off / on | Is entity-name overlap uninformative noise? |
| lowercase | on (fixed); one ablation with off | Sanity check only |

That gives 2 × 4 × 2 = 16 preprocessing configs. Each is crossed with the metrics in §3 and the
aggregations in §4.

---

## 3. Similarity metrics (manual, no scikit-learn)

Notation: F = fact token **set**, P = passage token set. |·| = size.

Running example (hand-checkable; values verified in the `cm3604` env):
```
Fact : "Lanny Flaherty is an actor."
P1   : "<s>Lanny Flaherty (born July 27, 1942) is an American actor.</s>"
P2   : "<s>Flaherty resides in New York City.</s>"
After cleaning, lowercase, stopwords, Porter:
F  = {lanni, flaherti, actor}
P1 = {lanni, flaherti, born, juli, 27, 1942, american, actor}
P2 = {flaherti, resid, new, york, citi}
```

### 3.1 Recall (fact coverage), the primary metric
$$\text{recall}(F,P) = \frac{|F \cap P|}{|F|}$$
- **Meaning:** what fraction of the claim's words the evidence contains. The class name
  `WordRecallThresholdFactChecker` says this is the intended baseline.
- **Example:** P1 = 3/3 = **1.0**; P2 = 1/3 = **0.333**.
- **Why it suits this task:** it is asymmetric and normalised by the short side, so it does not
  depend on passage length.
- **Weakness:** if the words are all present somewhere, it scores 1 even when the relation differs
  ("born in Cleveland" vs "died in Cleveland" with "born" elsewhere in the passage).
- **Edge:** |F| = 0 → return 0.0 (see viva Q6).

### 3.2 Precision
$$\text{precision}(F,P) = \frac{|F\cap P|}{|P|}$$
- Example: P1 = 3/8 = 0.375; P2 = 1/5 = 0.2.
- **Problem:** |P| ≈ 100+ tokens for real passages, so precision ≈ 0.01–0.05 and varies mainly
  with passage length. **Expectation:** weak alone. Include it so you can *show* that, which counts
  as critical evaluation.

### 3.3 F1
$$F_1 = \frac{2\,\text{prec}\cdot\text{rec}}{\text{prec}+\text{rec}}\quad(0 \text{ if both } 0)$$
- Harmonic mean, so it is dominated by the smaller value, which is precision here. **Expectation:**
  it behaves like precision. Example P1: 2·0.375·1/(1.375) = 0.545.

### 3.4 Jaccard
$$J(F,P)=\frac{|F\cap P|}{|F\cup P|}$$
- Example: P1 = 3/8 = **0.375** (F ⊂ P1, so the union = P1); P2 = 1/7 = 0.143.
- **Bound (viva):** J ≤ |F|/|P| when |F| ≤ |P|. With a 4-token fact and a 120-token passage,
  J ≤ 0.033 even for perfect support. Jaccard is symmetric, but this task is asymmetric (a short
  claim against long evidence). **Expectation:** poor, unless used at sentence level.
- **Identity:** J ≤ min(precision, recall). Use it as a unit-test property.

### 3.5 TF-IDF cosine
Term frequency is a count (repeats matter here, unlike sets):
$$\text{tf}(w,d)=\text{count of } w \text{ in } d$$
Smoothed IDF over a document collection D of size N (df = number of docs containing w):
$$\text{idf}(w)=\ln\frac{1+N}{1+\text{df}(w)}+1$$
Vector and cosine:
$$v_d[w]=\text{tf}(w,d)\cdot\text{idf}(w),\qquad \cos(u,v)=\frac{\sum_w u_w v_w}{\lVert u\rVert\,\lVert v\rVert}$$

**Key design decision: which documents define IDF?** `predict` sees only one fact's passages, and
the constructor takes no arguments.

| Choice | Pros | Cons |
|---|---|---|
| (a) the fact's 2–5 passages | trivial | N tiny, IDF almost flat |
| **(b) the sentences of the fact's passages** (recommended) | N ≈ 10–40; the entity name is in most sentences → low IDF (automatic down-weighting) | still local |
| (c) global IDF over the whole passages file | most "correct" IDF | `factcheck.py` would have to read the data file or hard-code a vocabulary; fragile |

Do not include the fact as a document. Smoothing (+1) keeps fact words that are absent from every
passage finite: df = 0 → idf = ln(N+1) + 1.

**Example (docs = {P1, P2}, N = 2):** `flaherti` df = 2 → idf = ln(3/3)+1 = **1.000**; every other
term has df = 1 → idf = ln(3/2)+1 = **1.4055**.
- fact vector: lanni 1.4055, flaherti 1, actor 1.4055 → ‖f‖ = √(2·1.9754+1) = 2.2252
- P1: 7 terms at 1.4055 plus flaherti 1 → ‖P1‖ = √(7·1.9754+1) = 3.8507
- dot = 2·1.9754 + 1 = 4.9508 → cos = 4.9508 / (2.2252·3.8507) = **0.5778**
- cos(F, P2) = **0.1506**

Note that cosine stays below 1 for a perfectly supported fact, because extra passage words lengthen
‖P‖. That is why sentence-level max (§4) may help TF-IDF more than it helps recall.

**Implementation:** use sparse dicts (`collections.Counter`), not numpy arrays over a vocabulary.
Vectors have about 10–150 non-zeros, so a dict dot-product iterating over the *shorter* vector is
O(min(|u|,|v|)).

**Edge cases:** a zero vector (empty text) makes the denominator 0 → return 0.0. Never divide by
zero.

### 3.6 IDF-weighted recall (optional, strong)
$$\text{wrecall}(F,P)=\frac{\sum_{w\in F\cap P}\text{idf}(w)}{\sum_{w\in F}\text{idf}(w)}$$
It combines the length-independence of recall with the TF-IDF insight that rare words (dates,
places) matter more than the ubiquitous entity name. It is a principled alternative to the
`drop_title` hack, so compare the two directly.

### 3.7 ROUGE-1 / BLEU
`rouge-score` is in `requirements.txt`. ROUGE-1 recall is multiset unigram recall, so it equals your
set recall when there are no repeated tokens. **Use it as a test oracle**, not in `factcheck.py`.
Watch the argument order: `scorer.score(target, prediction)`, so recall is measured against
`target`. To get *fact* coverage, pass the fact as target. With the passage as target (checked),
`recall = 0.5`. BLEU is a precision metric with a brevity penalty, designed for comparing a short
hypothesis with references of similar length, so it is a poor fit here. One sentence on that in the
report is enough.

---

## 4. Aggregating over multiple passages

A fact has 2–5 passages, and the classifier needs one number.

| Aggregation | Definition | Note |
|---|---|---|
| **max** over passages | max_i s(F, P_i) | Evidence usually sits in one place; matches the spec's "max over" wording for entailment |
| concat | s(F, ∪_i P_i) | Recall can only go up (more NS facts score high); cosine is diluted |
| sentence-max | max over sentences of all passages | Most precise. Split on `</s>` (551/692 passages have tags), else a regex fallback on `[.!?]\s+` |

For recall: concat ≥ max ≥ sentence-max, always (a subset argument, and a good viva point). The
question is which one *separates* S from NS best. That is an experiment, not something to argue
from first principles.

`split_sentences` is also needed by Member 3/4 for entailment. **Agree on one shared function**
rather than two versions.

---

## 5. Code structure in `factcheck.py` and the contract with Member 2

```python
# ---------- Member 1: preprocessing ----------
STOPWORDS: frozenset[str]                 # hard-coded
_TAG_RE, _TOKEN_RE, _SENT_RE              # compiled once at import

def clean_text(text: str) -> str
def tokenize(text: str, lowercase: bool = True) -> List[str]
def split_sentences(passage_text: str) -> List[str]      # shared with M3/M4

class TextPreprocessor:
    def __init__(self, lowercase=True, remove_stopwords=True,
                 normalizer="porter",        # "none"|"porter"|"snowball"|"lemma"
                 min_token_len=1)
    def process(self, text: str) -> List[str]           # cached per text
    def process_set(self, text: str) -> frozenset[str]

# ---------- Member 1: similarity ----------
def overlap_recall(fact: set, passage: set) -> float
def overlap_precision(fact: set, passage: set) -> float
def overlap_f1(fact: set, passage: set) -> float
def jaccard(a: set, b: set) -> float
def compute_idf(docs: List[List[str]]) -> Dict[str, float]
def tfidf_vector(tokens: List[str], idf: Dict[str, float], default_idf: float) -> Dict[str, float]
def cosine(u: Dict[str, float], v: Dict[str, float]) -> float
def idf_weighted_recall(fact: set, passage: set, idf) -> float

def word_overlap_score(fact: str, passages: List[dict], pre: TextPreprocessor,
                       metric="recall", aggregation="max", drop_title=False) -> float
    # returns a float in [0, 1]; returns 0.0 if passages is empty or the fact is empty

# ---------- Member 2 (not you) ----------
class WordRecallThresholdFactChecker(FactChecker):
    METRIC = "recall"; AGGREGATION = "max"; PREPROC = dict(...)   # M1 fills these
    THRESHOLD = ...                                                # M2 fills this
    def __init__(self): self.pre = TextPreprocessor(**self.PREPROC)
    def score(self, fact, passages) -> float: return word_overlap_score(...)
    def predict(self, fact, passages): return "S" if self.score(...) >= self.THRESHOLD else "NS"
```

**Contract to agree with Member 2 (write it in the group chat):**
1. `score(fact, passages) -> float in [0, 1]`, deterministic, higher means more supported.
2. Empty or missing passages and empty facts → `0.0` (so the result is NS).
3. M1 owns the `PREPROC/METRIC/AGGREGATION` constants. M2 owns `THRESHOLD` and `predict`.
4. M4 reuses `word_overlap_score` for pruning, so do not rename it after it is agreed.
5. `factcheck.py` never imports from `experiments/`. Experiments import from `factcheck`.

---

## 6. Experiments (`experiments/m1_preprocessing/`)

Files:
- `run_grid.py`: loads data by reusing `read_passages` and `read_fact_examples` from
  `factchecking_main` (importing is fine; just don't edit the file). It computes scores for every
  (preproc × metric × aggregation) combination and writes results.
- `test_m1.py`: unit tests (§7).

**How to compare configs without doing Member 2's job:** use a threshold-free measure.

- **ROC-AUC (Mann–Whitney), manual:** AUC = P(s_S > s_NS) + ½·P(s_S = s_NS) over all
  S×NS pairs. 112 × 109 = 12,208 pairs, which is trivial to compute. It measures how well the score
  *ranks* supported facts above unsupported ones, independent of any threshold.
- **Best-threshold accuracy (diagnostic):** sweep thresholds at midpoints between sorted unique
  scores. Label this "oracle threshold on the same data". It is optimistic, and the spec admits there
  is no separate test set. Member 2 should report the honest version (k-fold threshold selection).
- **Ablations:** change one factor at a time from the best config, and report the Δ AUC for each
  factor. This is the "analysis of alternatives" table.
- **Score distributions:** a histogram of the score for S vs NS for the top 2–3 configs. This
  justifies the threshold region for Member 2.
- **Timing:** wall-clock time per config (lemma vs Porter) supports the "efficient" rubric phrase.

**Efficiency in the grid:** preprocess every unique passage once per preprocessing config, then
reuse the tokens across all metrics and aggregations. Optionally cache them to
`results/cache/m1_tokens_<config>.pkl`.

**Reproducibility:** the output path is `results/m1/grid_<timestamp>.csv` with columns
`preproc, metric, aggregation, auc, best_acc, best_threshold, runtime_s, n_examples, git_commit`.
Also write a `run_config.json` (Python version, package versions, git commit). The whole pipeline is
deterministic: no randomness, no seeds needed. Say so in the report.

Run with:
```bash
"C:/Users/LENOVO/.conda/envs/cm3604/python.exe" experiments/m1_preprocessing/run_grid.py --data_dir data
```

---

## 7. Testing (`unittest`, stdlib)

| Test | Input | Expected |
|---|---|---|
| tags stripped | `"<s>Hi there.</s>"` | tokens `["hi","there"]`, no `"s"` |
| punctuation | `"(born July 27, 1942)"` | `born, july, 27, 1942` |
| possessive | `"McAndrew's"` | `mcandrew` |
| accent | `"Beyoncé"` | `beyonce` |
| stop-then-stem order | `"was"` with stopwords + Porter | removed (not `wa`) |
| empty / None | `""` | `[]`, no exception |
| all-stopword fact | `"He was there."` | fallback to unfiltered tokens |
| recall example | §3 F, P1 / P2 | 1.0 / 0.3333 |
| Jaccard example | F, P1 / P2 | 0.375 / 0.142857 |
| TF-IDF example | F vs P1 / P2, docs = {P1, P2} | 0.5778 / 0.1506 (±1e-4) |
| repeated words | fact `"born born Ohio"` | set metrics same as `"born Ohio"`; tf counts 2 |
| empty passages list | `passages=[]` | `0.0` |
| empty passage text | `[{"title":"X","text":""}]` | `0.0`, no ZeroDivisionError |
| zero vector cosine | `{}` vs `{a:1}` | `0.0` |
| properties | random small sets | 0 ≤ s ≤ 1; recall(F,F) = 1; J symmetric; J ≤ min(p, r); cos(v,v) = 1; order-invariant |
| oracle | recall vs `rouge_score` rouge1 recall (fact as target, no stopword removal, no repeats) | equal |
| aggregation order | any fact | recall(concat) ≥ recall(max) ≥ recall(sentence-max) |
| end-to-end | `python factchecking_main.py --mode word_overlap` | runs, prints a confusion matrix |

Run with:
```bash
"C:/Users/LENOVO/.conda/envs/cm3604/python.exe" -m unittest discover -s experiments/m1_preprocessing -v
```

---

## 8. Likely viva questions

1. **Why bag-of-words at all?** It is a cheap, transparent baseline. Supported facts should share
   content words with their evidence. It ignores word order and relations, which is why Part 2 exists.
2. **Why recall rather than Jaccard?** The fact is short and the passage is long. Jaccard and
   precision are capped by |F|/|P|, so they mostly measure passage length. Recall is normalised by
   the fact only. (Then show your AUC table.)
3. **Why remove stopwords before stemming?** Porter turns `was → wa`, which then escapes the
   stopword list.
4. **Stemming vs lemmatization?** Stemming is rule-based suffix stripping: fast, no model, and it
   produces non-words. Lemmatization maps to dictionary forms using POS: it handles `sang → sing`,
   but it is slower and needs spaCy. Choose using AUC and runtime evidence.
5. **What does IDF do here?** It down-weights words that appear in many sentences, mainly the entity
   name and generic biography words, and up-weights rare words like dates and places, which carry the
   claim.
6. **Empty fact → why 0 and not 1?** Mathematically ∅ ⊆ P gives recall 1 (vacuous truth). But with
   no content there is no evidence, and predicting S without evidence is unsafe. The fallback to
   unfiltered tokens makes this case rare.
7. **Why smooth IDF?** To avoid division by zero / log(∞) for fact words absent from all passages,
   and to keep ubiquitous words at a weight of 1 rather than 0.
8. **Why max over passages?** Support usually lives in one passage or sentence. Concatenation lets
   words scattered across unrelated passages add up to false support (an FP mechanism).
9. **Isn't tuning on the dev set overfitting?** Yes, a bit. The spec allows it. We use a
   threshold-free AUC to choose preprocessing, and Member 2 reports k-fold threshold accuracy.
10. **Complexity?** Preprocessing is O(total characters). Set metrics are O(|F| + |P|) with hashing.
    Cosine is O(min nnz). Caching makes repeated passages free.
11. **Failure modes?** Numbers matching wrongly (a year mentioned elsewhere), negation, relation
    mismatch (born vs died in X), paraphrase (actor/actress, "performer"). These feed Part 3.

---

## 9. Status

Implemented: the Member 1 section of `factcheck.py`, plus `WordRecallThresholdFactChecker.score()` and its
config constants. Tests: `test_m1.py` and `test_evaluation.py` (42 tests). Experiment: `run_grid.py`. Figures:
`m1_analysis.ipynb`. Team setup: `SETUP.md` in the repository root. Measured results and the
justification for each choice: `RESULTS.md`. The **Expectations** in §3–§4 above were hypotheses; see
`RESULTS.md` for what the data actually showed. `predict()` and the threshold remain Member 2's.
