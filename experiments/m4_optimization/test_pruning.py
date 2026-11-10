import unittest
from run_m4_experiments import passage_word_recall

class PruningTests(unittest.TestCase):
    def test_full_overlap(self):
        self.assertEqual(passage_word_recall(["founded", "1998"], ["company", "founded", "1998"]), 1.0)

    def test_partial_overlap(self):
        self.assertEqual(passage_word_recall(["founded", "1998"], ["founded"]), 0.5)

    def test_empty_fact(self):
        self.assertEqual(passage_word_recall([], ["anything"]), 0.0)

if __name__ == "__main__":
    unittest.main()
