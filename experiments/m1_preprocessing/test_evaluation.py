"""
Unit tests for the evaluation helpers in run_grid.py (ROC-AUC and oracle accuracy), so the numbers used to choose
the configuration are themselves verified.

Run from the repository root:
    python -m unittest discover -s experiments/m1_preprocessing -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_grid import oracle_accuracy, roc_auc  # noqa: E402


class TestRocAuc(unittest.TestCase):
    def test_perfect_reversed_and_constant(self):
        gold = np.array([1, 1, 0, 0])
        self.assertEqual(roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), gold), 1.0)   # every S above every NS
        self.assertEqual(roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), gold), 0.0)   # every S below every NS
        self.assertEqual(roc_auc(np.array([0.5, 0.5, 0.5, 0.5]), gold), 0.5)   # all ties count as half

    def test_hand_computed(self):
        # S scores {0.9, 0.4}, NS scores {0.6, 0.4}. Pairs: (0.9>0.6) 1, (0.9>0.4) 1, (0.4<0.6) 0, (0.4=0.4) 0.5
        # -> AUC = 2.5 / 4 = 0.625
        self.assertAlmostEqual(roc_auc(np.array([0.9, 0.4, 0.6, 0.4]), np.array([1, 1, 0, 0])), 0.625)

    def test_invariant_to_monotone_transform(self):
        rng = np.random.default_rng(0)
        scores, gold = rng.random(50), rng.integers(0, 2, 50)
        self.assertAlmostEqual(roc_auc(scores, gold), roc_auc(scores ** 3 + 7, gold))


class TestOracleAccuracy(unittest.TestCase):
    def test_separable(self):
        acc, thr = oracle_accuracy(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0]))
        self.assertEqual(acc, 1.0)
        self.assertTrue(0.2 < thr <= 0.8)

    def test_hand_computed(self):
        # Scores 0.1(S) 0.3(NS) 0.6(S) 0.7(NS) 0.9(S). Accuracy of "S if score >= t" for every distinct cut:
        # t<0.1 -> 3/5, t=0.2 -> 2/5, t=0.45 -> 3/5, t=0.65 -> 2/5, t=0.8 -> 3/5, t>0.9 -> 2/5. The best is 3/5.
        acc, _ = oracle_accuracy(np.array([0.1, 0.3, 0.6, 0.7, 0.9]), np.array([1, 0, 1, 0, 1]))
        self.assertAlmostEqual(acc, 0.6)

    def test_never_below_majority_class(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            scores, gold = rng.random(40), rng.integers(0, 2, 40)
            acc, _ = oracle_accuracy(scores, gold)
            # The thresholds below the minimum / above the maximum give "all S" / "all NS".
            self.assertGreaterEqual(acc + 1e-12, max(gold.mean(), 1 - gold.mean()))


if __name__ == "__main__":
    unittest.main()
