"""
Unit tests for Member 1's preprocessing and similarity code in factcheck.py.

Run from the repository root:
    python -m unittest discover -s experiments/m1_preprocessing -v
"""
import math
import os
import random
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from factcheck import (AGGREGATIONS, METRICS, STOPWORDS, TextPreprocessor, WordRecallThresholdFactChecker,  # noqa: E402
                       clean_text, compute_idf, cosine, idf_weighted_recall, jaccard, overlap_f1,
                       overlap_precision, overlap_recall, split_sentences, tfidf_cosine, tfidf_vector, tokenize,
                       word_overlap_score)

# Running example for these tests. All expected values were worked out by hand: after preprocessing,
# F = {lanni, flaherti, actor}; recall(F, P1) = 3/3, recall(F, P2) = 1/3, Jaccard(F, P1) = 3/8, Jaccard(F, P2) = 1/7.
FACT = "Lanny Flaherty is an actor."
P1 = "<s>Lanny Flaherty (born July 27, 1942) is an American actor.</s>"
P2 = "<s>Flaherty resides in New York City.</s>"
PASSAGES = [{"title": "Lanny Flaherty", "text": P1}, {"title": "Lanny Flaherty", "text": P2}]
F_SET = {"lanni", "flaherti", "actor"}
P1_SET = {"lanni", "flaherti", "born", "juli", "27", "1942", "american", "actor"}
P2_SET = {"flaherti", "resid", "new", "york", "citi"}


class TestCleaningAndTokenisation(unittest.TestCase):
    def test_sentence_tags_removed(self):
        self.assertEqual(tokenize("<s>Hi there.</s>"), ["hi", "there"])
        self.assertNotIn("s", tokenize(P1))

    def test_punctuation_is_a_boundary(self):
        self.assertEqual(tokenize("(born July 27, 1942)"), ["born", "july", "27", "1942"])
        self.assertEqual(tokenize("singer-songwriter"), ["singer", "songwriter"])
        self.assertEqual(tokenize("1942–2010"), ["1942", "2010"])  # en dash, common in Wikipedia dates

    def test_possessive_and_apostrophes(self):
        self.assertEqual(tokenize("McAndrew's"), ["mcandrew"])
        self.assertEqual(tokenize("McAndrew’s"), ["mcandrew"])
        self.assertEqual(tokenize("O'Brien"), ["obrien"])

    def test_accents_removed(self):
        self.assertEqual(tokenize("Beyoncé Knowles"), ["beyonce", "knowles"])

    def test_thousands_separator(self):
        self.assertEqual(tokenize("1,000,000 copies"), ["1000000", "copies"])
        self.assertEqual(tokenize("21, 1942"), ["21", "1942"])  # a comma followed by a space is not a separator

    def test_lowercase_switch(self):
        self.assertEqual(tokenize("Lanny Flaherty", lowercase=False), ["Lanny", "Flaherty"])

    def test_empty_and_none(self):
        self.assertEqual(clean_text(None), "")
        self.assertEqual(tokenize(""), [])
        self.assertEqual(TextPreprocessor().process(None), [])
        self.assertEqual(split_sentences(""), [])

    def test_split_sentences(self):
        text = "<s>Career. He has given 3 roles. He attended U. of Miss.</s><s>Personal life. He lives in Ohio.</s>"
        self.assertEqual(split_sentences(text), ["Career.", "He has given 3 roles.", "He attended U. of Miss.",
                                                 "Personal life.", "He lives in Ohio."])
        self.assertEqual(split_sentences("No tags here. Second one."), ["No tags here.", "Second one."])


class TestTextPreprocessor(unittest.TestCase):
    def test_running_example(self):
        pre = TextPreprocessor()
        self.assertEqual(set(pre.process(FACT)), F_SET)
        self.assertEqual(set(pre.process(P1)), P1_SET)
        self.assertEqual(set(pre.process(P2)), P2_SET)

    def test_stopwords_removed_before_stemming(self):
        # Porter maps "was" -> "wa"; if stemming ran first, "wa" would leak through the stop-word filter.
        tokens = TextPreprocessor(normalizer="porter").process("Smith was a singer")
        self.assertNotIn("wa", tokens)
        self.assertEqual(tokens, ["smith", "singer"])

    def test_negations_kept(self):
        self.assertNotIn("not", STOPWORDS)
        self.assertIn("not", TextPreprocessor(normalizer="none").process("He was not married"))

    def test_all_stopword_fallback(self):
        self.assertEqual(TextPreprocessor(normalizer="none").process("He was there."), ["he", "was", "there"])

    def test_short_tokens_dropped_but_digits_kept(self):
        self.assertEqual(TextPreprocessor(normalizer="none").process("U.S. army, 3 wars"), ["army", "3", "wars"])

    def test_repeated_words_kept_in_list(self):
        self.assertEqual(TextPreprocessor().process("born born Ohio"), ["born", "born", "ohio"])

    def test_normalizers(self):
        self.assertEqual(TextPreprocessor(normalizer="none").process("studies studied"), ["studies", "studied"])
        self.assertEqual(TextPreprocessor(normalizer="porter").process("studies studied"), ["studi", "studi"])
        self.assertEqual(TextPreprocessor(normalizer="snowball").process("studies studied"), ["studi", "studi"])
        self.assertEqual(TextPreprocessor(normalizer="lemma").process("She sang and studied"), ["sing", "study"])
        with self.assertRaises(ValueError):
            TextPreprocessor(normalizer="wordnet")

    def test_cache_returns_copies(self):
        pre = TextPreprocessor()
        first = pre.process(FACT)
        first.append("mutated")
        self.assertEqual(set(pre.process(FACT)), F_SET)


class TestSetMetrics(unittest.TestCase):
    def test_running_example_values(self):
        self.assertAlmostEqual(overlap_recall(F_SET, P1_SET), 1.0)
        self.assertAlmostEqual(overlap_recall(F_SET, P2_SET), 1 / 3)
        self.assertAlmostEqual(overlap_precision(F_SET, P1_SET), 3 / 8)
        self.assertAlmostEqual(overlap_precision(F_SET, P2_SET), 1 / 5)
        self.assertAlmostEqual(jaccard(F_SET, P1_SET), 3 / 8)
        self.assertAlmostEqual(jaccard(F_SET, P2_SET), 1 / 7)
        self.assertAlmostEqual(overlap_f1(F_SET, P1_SET), 2 * 0.375 * 1.0 / 1.375)

    def test_empty_inputs(self):
        for fn in (overlap_recall, overlap_precision, overlap_f1, jaccard):
            self.assertEqual(fn(set(), set()), 0.0)
            self.assertEqual(fn(set(), {"a"}), 0.0)
            self.assertEqual(fn({"a"}, set()), 0.0)

    def test_repeated_words_do_not_change_set_metrics(self):
        self.assertEqual(overlap_recall(["born", "born", "ohio"], ["born"]), overlap_recall(["born", "ohio"], ["born"]))
        self.assertEqual(jaccard(["a", "a", "b"], ["a"]), jaccard(["a", "b"], ["a"]))

    def test_properties_on_random_sets(self):
        rng = random.Random(0)
        vocab = [str(i) for i in range(12)]
        for _ in range(500):
            a = set(rng.sample(vocab, rng.randint(1, 8)))
            b = set(rng.sample(vocab, rng.randint(1, 8)))
            r, p, j = overlap_recall(a, b), overlap_precision(a, b), jaccard(a, b)
            for value in (r, p, j, overlap_f1(a, b)):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
            self.assertAlmostEqual(j, jaccard(b, a))                  # Jaccard is symmetric
            self.assertLessEqual(j, min(p, r) + 1e-12)                # |A∪B| >= max(|A|, |B|)
            self.assertAlmostEqual(r, overlap_precision(b, a))        # recall(F,P) = precision(P,F)
            self.assertEqual(overlap_recall(a, a), 1.0)
            self.assertEqual(r == 1.0, a <= b)                        # recall is 1 exactly when F ⊆ P

    def test_matches_rouge1_recall(self):
        # Independent oracle: rouge_score's ROUGE-1 recall is multiset unigram recall against the *target*, so with
        # the fact as target, no stop-word removal, no stemming and no repeated fact words it must equal our recall.
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)
        pre = TextPreprocessor(remove_stopwords=False, normalizer="none", min_token_len=1)
        pairs = [(FACT, P1), (FACT, P2), ("Marianne McAndrew was born in Cleveland Ohio.",
                                          "McAndrew was born in Cleveland and moved to Ohio in 1960.")]
        for fact, passage in pairs:
            expected = scorer.score(fact, passage)["rouge1"].recall
            self.assertAlmostEqual(overlap_recall(pre.process(fact), pre.process(passage)), expected)


class TestIdfMetrics(unittest.TestCase):
    def setUp(self):
        self.idf, self.unseen = compute_idf([list(P1_SET), list(P2_SET)])

    def test_idf_values(self):
        self.assertAlmostEqual(self.idf["flaherti"], 1.0)              # in both documents: ln(3/3) + 1
        self.assertAlmostEqual(self.idf["actor"], math.log(1.5) + 1)   # in one document:  ln(3/2) + 1
        self.assertAlmostEqual(self.unseen, math.log(3) + 1)           # in no document:   ln(3/1) + 1

    def test_tfidf_cosine_running_example(self):
        self.assertAlmostEqual(tfidf_cosine(F_SET, P1_SET, self.idf, self.unseen), 0.5778, places=4)
        self.assertAlmostEqual(tfidf_cosine(F_SET, P2_SET, self.idf, self.unseen), 0.1506, places=4)

    def test_term_frequency_counts_repeats(self):
        vec = tfidf_vector(["actor", "actor", "flaherti"], self.idf, self.unseen)
        self.assertAlmostEqual(vec["actor"], 2 * self.idf["actor"])
        self.assertAlmostEqual(vec["flaherti"], 1.0)

    def test_cosine_edge_cases(self):
        self.assertEqual(cosine({}, {"a": 1.0}), 0.0)
        self.assertEqual(cosine({}, {}), 0.0)
        self.assertAlmostEqual(cosine({"a": 2.0, "b": 1.0}, {"a": 2.0, "b": 1.0}), 1.0)
        self.assertAlmostEqual(cosine({"a": 1.0}, {"b": 1.0}), 0.0)
        self.assertAlmostEqual(cosine({"a": 1.0, "b": 2.0}, {"a": 3.0}), cosine({"a": 3.0}, {"a": 1.0, "b": 2.0}))

    def test_idf_sums_independent_of_order(self):
        # Regression test: plain sum() over a set gave run-dependent rounding (hash randomisation), which changed AUC
        # ties between runs. With math.fsum the result must not depend on the order of the words.
        idf = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 1e-17, "e": 1.7}
        words = list(idf)
        rng = random.Random(0)
        reference = idf_weighted_recall(words, ["a", "c", "d"], idf, 1.0)
        for _ in range(50):
            rng.shuffle(words)
            self.assertEqual(idf_weighted_recall(words, ["a", "c", "d"], idf, 1.0), reference)

    def test_idf_weighted_recall(self):
        # Only "flaherti" (idf 1) matches P2; the unmatched words each weigh ln(1.5) + 1.
        w = math.log(1.5) + 1
        self.assertAlmostEqual(idf_weighted_recall(F_SET, P2_SET, self.idf, self.unseen), 1 / (1 + 2 * w))
        self.assertAlmostEqual(idf_weighted_recall(F_SET, P1_SET, self.idf, self.unseen), 1.0)
        self.assertEqual(idf_weighted_recall(set(), P1_SET, self.idf, self.unseen), 0.0)


class TestWordOverlapScore(unittest.TestCase):
    def setUp(self):
        self.pre = TextPreprocessor()

    def test_running_example(self):
        self.assertAlmostEqual(word_overlap_score(FACT, PASSAGES, self.pre, "recall", "max"), 1.0)
        self.assertAlmostEqual(word_overlap_score(FACT, PASSAGES, self.pre, "tfidf_cosine", "max"), 0.5778, places=4)
        self.assertAlmostEqual(word_overlap_score(FACT, PASSAGES[1:], self.pre, "recall", "max"), 1 / 3)

    def test_missing_or_empty_passages(self):
        for metric in METRICS:
            for aggregation in AGGREGATIONS:
                self.assertEqual(word_overlap_score(FACT, [], self.pre, metric, aggregation), 0.0)
                self.assertEqual(word_overlap_score(FACT, [{"title": "X", "text": ""}], self.pre, metric,
                                                    aggregation), 0.0)
                self.assertEqual(word_overlap_score(FACT, [{"title": "X", "text": None}], self.pre, metric,
                                                    aggregation), 0.0)

    def test_empty_fact(self):
        for metric in METRICS:
            self.assertEqual(word_overlap_score("", PASSAGES, self.pre, metric), 0.0)
            self.assertEqual(word_overlap_score("...", PASSAGES, self.pre, metric), 0.0)

    def test_scores_in_unit_interval(self):
        for metric in METRICS:
            for aggregation in AGGREGATIONS:
                for drop_title in (False, True):
                    s = word_overlap_score(FACT, PASSAGES, self.pre, metric, aggregation, drop_title)
                    self.assertGreaterEqual(s, 0.0)
                    self.assertLessEqual(s, 1.0 + 1e-12)

    def test_aggregation_ordering_for_recall(self):
        # A sentence is a subset of its passage, which is a subset of the concatenation.
        facts = [FACT, "Lanny Flaherty lives in New York.", "Lanny Flaherty was born in 1942 in Texas."]
        for fact in facts:
            concat = word_overlap_score(fact, PASSAGES, self.pre, "recall", "concat")
            best_passage = word_overlap_score(fact, PASSAGES, self.pre, "recall", "max")
            best_sentence = word_overlap_score(fact, PASSAGES, self.pre, "recall", "sentence_max")
            self.assertGreaterEqual(concat, best_passage)
            self.assertGreaterEqual(best_passage, best_sentence)

    def test_drop_title(self):
        # Without the name, "Lanny Flaherty is a singer." shares nothing with the passages.
        fact = "Lanny Flaherty is a singer."
        self.assertAlmostEqual(word_overlap_score(fact, PASSAGES, self.pre, "recall", "max", False), 2 / 3)
        self.assertEqual(word_overlap_score(fact, PASSAGES, self.pre, "recall", "max", True), 0.0)
        # A fact made only of the name keeps its tokens instead of becoming empty.
        self.assertEqual(word_overlap_score("Lanny Flaherty.", PASSAGES, self.pre, "recall", "max", True), 1.0)

    def test_word_order_invariance(self):
        a = word_overlap_score("Flaherty actor Lanny", PASSAGES, self.pre, "tfidf_cosine", "max")
        b = word_overlap_score("Lanny Flaherty actor", PASSAGES, self.pre, "tfidf_cosine", "max")
        self.assertAlmostEqual(a, b)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            word_overlap_score(FACT, PASSAGES, self.pre, metric="bleu")
        with self.assertRaises(ValueError):
            word_overlap_score(FACT, PASSAGES, self.pre, aggregation="mean")

    def test_checker_builds_without_arguments(self):
        checker = WordRecallThresholdFactChecker()
        s = checker.score(FACT, PASSAGES)
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)


if __name__ == "__main__":
    unittest.main()
