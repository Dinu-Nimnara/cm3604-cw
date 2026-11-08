"""Integration checks requiring the coursework dependencies and uploaded dataset."""
import contextlib
import io
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class IntegrationTests(unittest.TestCase):
    def test_real_classifier_and_edge_cases(self):
        from factcheck import WordRecallThresholdFactChecker
        c = WordRecallThresholdFactChecker()
        evidence = [{'title': 'Ada', 'text': 'Ada studied mathematics in London.'}]
        self.assertEqual(c.predict('Ada studied mathematics.', evidence), 'S')
        self.assertEqual(c.predict('Ada played football.', evidence), 'NS')
        self.assertEqual(c.predict('', evidence), 'NS')
        self.assertEqual(c.predict('Ada studied mathematics.', []), 'NS')

    @unittest.skipUnless((ROOT/'data/dev_labeled_ChatGPT.jsonl').exists(), 'coursework data absent')
    def test_driver_reader_parity_and_default_threshold(self):
        import factchecking_main as fm
        from factcheck import WordRecallThresholdFactChecker
        from run_evaluation import load_examples
        from evaluation import metrics, tune_threshold
        ours, missing = load_examples(ROOT/'data')
        with contextlib.redirect_stdout(io.StringIO()):
            official = fm.read_fact_examples(str(ROOT/'data/dev_labeled_ChatGPT.jsonl'),
                fm.read_passages(str(ROOT/'data/passages_bm25_ChatGPT_humfacts.jsonl')))
        self.assertEqual([(e.fact, e.label, e.passages) for e in ours],
                         [(e.fact, e.label, e.passages) for e in official])
        checker = WordRecallThresholdFactChecker()
        scores = [checker.score(e.fact, e.passages) for e in ours]
        best = tune_threshold(scores, [e.label for e in ours])
        actual = metrics([e.label for e in ours], [checker.predict(e.fact, e.passages) for e in ours])
        self.assertEqual(best['threshold'], checker.threshold)
        self.assertEqual(best['accuracy'], actual['accuracy'])


if __name__ == '__main__':
    unittest.main()
