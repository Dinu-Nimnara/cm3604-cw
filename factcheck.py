# factcheck.py

import torch
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import spacy
import gc
import math
import re
import unicodedata
from collections import Counter
from nltk.stem import PorterStemmer, SnowballStemmer


########################################################################################################################
# Part 1 (Member 1): text preprocessing and word-overlap similarity metrics.
# Everything here is implemented from scratch (no scikit-learn) and needs no runtime downloads.
########################################################################################################################

# NLTK's English stop-word list, hard-coded so nothing has to be downloaded at runtime. Contraction fragments are
# stored in their apostrophe-free form because clean_text() deletes apostrophes ("don't" -> "dont").
_NLTK_ENGLISH_STOPWORDS = frozenset("""
i me my myself we our ours ourselves you youre youve youll youd your yours yourself yourselves he him his himself
she shes her hers herself it its itself they them their theirs themselves what which who whom this that thatll these
those am is are was were be been being have has had having do does did doing a an the and but if or because as until
while of at by for with about against between into through during before after above below to from up down in out on
off over under again further then once here there when where why how all any both each few more most other some such
no nor not only own same so than too very s t can will just don dont should shouldve now d ll m o re ve y ain aren
arent couldn couldnt didn didnt doesn doesnt hadn hadnt hasn hasnt haven havent isn isnt ma mightn mightnt mustn
mustnt needn neednt shan shant shouldn shouldnt wasn wasnt weren werent won wont wouldn wouldnt
""".split())

# Negations flip the meaning of a claim, so they are kept as content words rather than discarded.
NEGATION_WORDS = frozenset("""
no nor not never don dont ain aren arent couldn couldnt didn didnt doesn doesnt hadn hadnt hasn hasnt haven havent isn
isnt mightn mightnt mustn mustnt needn neednt shan shant shouldn shouldnt wasn wasnt weren werent won wont wouldn
wouldnt
""".split())

STOPWORDS = _NLTK_ENGLISH_STOPWORDS - NEGATION_WORDS

# Compiled once at import time.
_SENT_TAG_RE = re.compile(r"</?s>")                        # <s> ... </s> markers in the retrieved passages
_POSSESSIVE_RE = re.compile(r"['’]s\b")               # McAndrew's -> McAndrew
_APOSTROPHE_RE = re.compile(r"['’]")                  # O'Brien -> OBrien, don't -> dont
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")         # 1,000,000 -> 1000000
_TOKEN_RE_LOWER = re.compile(r"[a-z0-9]+")
_TOKEN_RE_CASED = re.compile(r"[A-Za-z0-9]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def clean_text(text: Optional[str]) -> str:
    """
    Normalises raw passage/fact text before tokenisation: removes the <s>/</s> markers, strips accents
    (Beyoncé -> Beyonce), drops possessive 's, deletes remaining apostrophes and joins digit groups (1,000 -> 1000).
    :param text: raw string (None is treated as empty)
    :return: cleaned string, still cased and still containing punctuation
    """
    if not text:
        return ""
    text = _SENT_TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _POSSESSIVE_RE.sub("", text)
    text = _APOSTROPHE_RE.sub("", text)
    text = _THOUSANDS_RE.sub("", text)
    return text


def tokenize(text: Optional[str], lowercase: bool = True) -> List[str]:
    """
    Cleans text and splits it into alphanumeric tokens; every other character (punctuation, hyphens, dashes,
    brackets, whitespace) acts as a boundary. "(born July 27, 1942)" -> ["born", "july", "27", "1942"].
    :param text: raw string
    :param lowercase: whether to lowercase before tokenising
    :return: list of tokens in their original order (duplicates kept)
    """
    text = clean_text(text)
    if lowercase:
        return _TOKEN_RE_LOWER.findall(text.lower())
    return _TOKEN_RE_CASED.findall(text)


def split_sentences(passage_text: Optional[str]) -> List[str]:
    """
    Splits a passage into sentences. The <s>...</s> markers are used as boundaries first (they often wrap several
    sentences), then each chunk is split with a regex on sentence-final punctuation followed by a capital/digit.
    Needs no downloaded model.
    :param passage_text: raw passage text
    :return: list of non-empty sentence strings (tags removed)
    """
    if not passage_text:
        return []
    sentences = []
    for chunk in _SENT_TAG_RE.split(passage_text):
        chunk = chunk.strip()
        if chunk:
            sentences.extend(s.strip() for s in _SENT_SPLIT_RE.split(chunk) if s.strip())
    return sentences


NORMALIZERS = ("none", "porter", "snowball", "lemma")


class TextPreprocessor(object):
    """
    Configurable text -> token pipeline: clean -> (lowercase) -> tokenise -> remove stop words -> stem/lemmatise.
    Stop words are removed BEFORE stemming because Porter maps "was" -> "wa", which would then escape the list.
    Results are cached per input string, since the same passages are shared by every fact about one person.
    """

    def __init__(self, lowercase: bool = True, remove_stopwords: bool = True, normalizer: str = "porter",
                 min_token_len: int = 2):
        """
        :param lowercase: lowercase text before matching
        :param remove_stopwords: drop STOPWORDS (negations are kept)
        :param normalizer: "none", "porter", "snowball" or "lemma" (spaCy en_core_web_sm)
        :param min_token_len: drop alphabetic tokens shorter than this (stray letters such as the "s" of "U.S.");
        numbers are always kept because dates and years are highly informative for these facts
        """
        if normalizer not in NORMALIZERS:
            raise ValueError("normalizer must be one of " + repr(NORMALIZERS))
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.normalizer = normalizer
        self.min_token_len = min_token_len
        self._stem_cache: Dict[str, str] = {}
        self._cache: Dict[Tuple[str, bool], Tuple[str, ...]] = {}
        self._stemmer = None
        self._nlp = None
        if normalizer == "porter":
            self._stemmer = PorterStemmer()
        elif normalizer == "snowball":
            self._stemmer = SnowballStemmer("english")
        elif normalizer == "lemma":
            # The lemmatiser only needs the tagger and attribute ruler; the parser and NER are disabled for speed.
            self._nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def config(self) -> dict:
        return {"lowercase": self.lowercase, "remove_stopwords": self.remove_stopwords,
                "normalizer": self.normalizer, "min_token_len": self.min_token_len}

    def _keep(self, token: str, filter_stopwords: bool) -> bool:
        if len(token) < self.min_token_len and not token.isdigit():
            return False
        return not (filter_stopwords and token.lower() in STOPWORDS)

    def _stem(self, token: str) -> str:
        stem = self._stem_cache.get(token)
        if stem is None:
            # NLTK stemmers lowercase their output; keep the case if lowercasing was switched off.
            stem = self._stemmer.stem(token, to_lowercase=False) if self.normalizer == "porter" \
                else self._stemmer.stem(token)
            self._stem_cache[token] = stem
        return stem

    def _run(self, text: str, filter_stopwords: bool) -> List[str]:
        if self._nlp is not None:
            # spaCy needs the original casing for POS tagging, so lowercase after lemmatising. A lemma can contain
            # punctuation ("New-York"), so it is re-tokenised with the same regex as the other normalizers.
            tokens = []
            for tok in self._nlp(clean_text(text)):
                if filter_stopwords and tok.text.lower() in STOPWORDS:
                    continue
                for piece in tokenize(tok.lemma_, self.lowercase):
                    if self._keep(piece, filter_stopwords):
                        tokens.append(piece)
            return tokens
        tokens = [t for t in tokenize(text, self.lowercase) if self._keep(t, filter_stopwords)]
        if self._stemmer is not None:
            tokens = [self._stem(t) for t in tokens]
        return tokens

    def process(self, text: Optional[str]) -> List[str]:
        """
        :param text: raw text
        :return: token list (duplicates kept, needed for term frequencies). If stop-word removal would leave
        nothing (e.g. "He was there."), the unfiltered tokens are returned so the text is not silently emptied.
        """
        text = text or ""
        key = (text, self.remove_stopwords)
        cached = self._cache.get(key)
        if cached is None:
            tokens = self._run(text, self.remove_stopwords)
            if not tokens and self.remove_stopwords:
                tokens = self._run(text, False)
            cached = tuple(tokens)
            self._cache[key] = cached
        return list(cached)

    def process_set(self, text: Optional[str]) -> frozenset:
        return frozenset(self.process(text))


# ---------------------------------------------- similarity metrics -----------------------------------------------
# Set-based metrics take token sets; F = fact tokens, P = passage tokens. All return a float in [0, 1] and return
# 0.0 instead of dividing by zero, so empty inputs count as "no evidence" (-> Not Supported).

def overlap_recall(fact: Iterable[str], passage: Iterable[str]) -> float:
    """ |F ∩ P| / |F| : the fraction of the fact's words that the passage contains. """
    fact, passage = set(fact), set(passage)
    return len(fact & passage) / len(fact) if fact else 0.0


def overlap_precision(fact: Iterable[str], passage: Iterable[str]) -> float:
    """ |F ∩ P| / |P| : the fraction of the passage's words that appear in the fact. """
    fact, passage = set(fact), set(passage)
    return len(fact & passage) / len(passage) if passage else 0.0


def overlap_f1(fact: Iterable[str], passage: Iterable[str]) -> float:
    """ Harmonic mean of overlap precision and recall. """
    p, r = overlap_precision(fact, passage), overlap_recall(fact, passage)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """ |A ∩ B| / |A ∪ B| (symmetric). """
    a, b = set(a), set(b)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def compute_idf(docs: Sequence[Iterable[str]]) -> Tuple[Dict[str, float], float]:
    """
    Smoothed inverse document frequency: idf(w) = ln((1 + N) / (1 + df(w))) + 1, where N = number of documents and
    df(w) = number of documents containing w. The +1 terms keep words found in every document at weight 1 and give
    words found in no document the finite weight ln(1 + N) + 1.
    :param docs: list of token lists/sets (one per document)
    :return: (idf dict for the words seen, idf value to use for unseen words)
    """
    n_docs = len(docs)
    df = Counter(w for doc in docs for w in set(doc))
    idf = {w: math.log((1 + n_docs) / (1 + d)) + 1 for w, d in df.items()}
    return idf, math.log(1 + n_docs) + 1


def tfidf_vector(tokens: Iterable[str], idf: Dict[str, float], unseen_idf: float) -> Dict[str, float]:
    """ Sparse TF-IDF vector {word: count(word) * idf(word)}, with raw counts as term frequency. """
    return {w: c * idf.get(w, unseen_idf) for w, c in Counter(tokens).items()}


def cosine(u: Dict[str, float], v: Dict[str, float]) -> float:
    """ Cosine similarity of two sparse vectors; 0.0 if either is the zero vector. """
    if len(u) > len(v):
        u, v = v, u
    # math.fsum is exactly rounded, so the result does not depend on iteration order (reproducible across runs).
    dot = math.fsum(x * v.get(w, 0.0) for w, x in u.items())
    norm_u = math.sqrt(math.fsum(x * x for x in u.values()))
    norm_v = math.sqrt(math.fsum(x * x for x in v.values()))
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return dot / (norm_u * norm_v)


def tfidf_cosine(fact: Iterable[str], passage: Iterable[str], idf: Dict[str, float], unseen_idf: float) -> float:
    return cosine(tfidf_vector(fact, idf, unseen_idf), tfidf_vector(passage, idf, unseen_idf))


def idf_weighted_recall(fact: Iterable[str], passage: Iterable[str], idf: Dict[str, float],
                        unseen_idf: float) -> float:
    """ Recall where each fact word counts with weight idf(w): rare words (dates, places) matter more than words
    that occur in every sentence (the person's name). """
    fact, passage = set(fact), set(passage)
    # Set iteration order changes between Python runs (hash randomisation); math.fsum is exactly rounded, so the
    # sums - and therefore ties between facts - are identical in every run.
    total = math.fsum(idf.get(w, unseen_idf) for w in fact)
    if total == 0.0:
        return 0.0
    return math.fsum(idf.get(w, unseen_idf) for w in fact & passage) / total


SET_METRICS = {"recall": overlap_recall, "precision": overlap_precision, "f1": overlap_f1, "jaccard": jaccard}
IDF_METRICS = {"tfidf_cosine": tfidf_cosine, "idf_recall": idf_weighted_recall}
METRICS = tuple(SET_METRICS) + tuple(IDF_METRICS)
AGGREGATIONS = ("max", "concat", "sentence_max")


def word_overlap_score(fact: str, passages: List[dict], preprocessor: TextPreprocessor, metric: str = "recall",
                       aggregation: str = "max", drop_title: bool = False) -> float:
    """
    Real-valued bag-of-words support score of a fact against its retrieved passages (higher = more supported).
    :param fact: the fact string
    :param passages: list of {"title", "text"} dicts (may be empty)
    :param preprocessor: TextPreprocessor applied identically to the fact, passages and titles
    :param metric: one of METRICS
    :param aggregation: "max" = best single passage, "concat" = all passages as one bag of words,
    "sentence_max" = best single sentence of any passage
    :param drop_title: remove the page-title (entity name) tokens before scoring, because the name appears in every
    passage and therefore matches for supported and unsupported facts alike
    :return: score in [0, 1]; 0.0 if there are no passages or nothing to compare
    """
    if metric not in METRICS:
        raise ValueError("metric must be one of " + repr(METRICS))
    if aggregation not in AGGREGATIONS:
        raise ValueError("aggregation must be one of " + repr(AGGREGATIONS))
    if not passages:
        return 0.0

    fact_tokens = preprocessor.process(fact)
    texts = [p.get("text") or "" for p in passages]
    sentence_tokens = [preprocessor.process(s) for t in texts for s in split_sentences(t)]
    if aggregation == "max":
        units = [preprocessor.process(t) for t in texts]
    elif aggregation == "concat":
        units = [[w for t in texts for w in preprocessor.process(t)]]
    else:
        units = sentence_tokens

    if drop_title:
        title_set = set()
        for p in passages:
            title_set |= preprocessor.process_set(p.get("title"))
        kept = [w for w in fact_tokens if w not in title_set]
        if kept:  # a fact consisting only of the name keeps its tokens rather than becoming empty
            fact_tokens = kept
            units = [[w for w in unit if w not in title_set] for unit in units]
            sentence_tokens = [[w for w in s if w not in title_set] for s in sentence_tokens]

    if not fact_tokens:
        return 0.0
    if metric in SET_METRICS:
        fn = SET_METRICS[metric]
        return max((fn(fact_tokens, unit) for unit in units), default=0.0)
    # IDF is always estimated over the sentences of this fact's passages (N is typically 10-40), independently of
    # the aggregation, so all aggregations share the same word weights.
    idf, unseen_idf = compute_idf(sentence_tokens)
    fn = IDF_METRICS[metric]
    return max((fn(fact_tokens, unit, idf, unseen_idf) for unit in units), default=0.0)


class FactExample(object):
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            if self.cuda:
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.

        raise Exception("Not implemented")

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        gc.collect()

        # return something


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    # Member 1: preprocessing / similarity configuration with the highest ROC-AUC (0.8975) of the 432 configurations
    # compared by experiments/m1_preprocessing/run_grid.py. factchecking_main.py constructs this class with no
    # arguments, so the settings are class constants.
    PREPROCESSING = {"lowercase": True, "remove_stopwords": True, "normalizer": "porter", "min_token_len": 2}
    METRIC = "recall"
    AGGREGATION = "max"
    DROP_TITLE = True

    # Fitted on the supplied development set: 184/221 correct (83.26%).
    # Reproduce with experiments/m2_classification/run_evaluation.py.
    # Development tuning is permitted by the brief; it is not held-out test accuracy.
    DEFAULT_THRESHOLD = 0.5634920634920635

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite")
        self.threshold = float(threshold)
        self.preprocessor = TextPreprocessor(**self.PREPROCESSING)

    def score(self, fact: str, passages: List[dict]) -> float:
        """ Word-overlap support score in [0, 1] (Member 1); predict() thresholds it (Member 2). """
        return word_overlap_score(fact, passages, self.preprocessor, self.METRIC, self.AGGREGATION, self.DROP_TITLE)

    def predict(self, fact: str, passages: List[dict]) -> str:
        # Equality is supported; tuning uses exactly the same comparison.
        # A zero score means no lexical evidence, even at a threshold of zero.
        value = self.score(fact, passages)
        return "S" if value > 0.0 and value >= self.threshold else "NS"


class EntailmentFactChecker(FactChecker):
    def __init__(self, ent_model):
        self.ent_model = ent_model

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")


# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations
