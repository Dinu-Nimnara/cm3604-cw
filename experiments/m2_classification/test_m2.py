import ast
import itertools
import math
from pathlib import Path
import random
import unittest
from evaluation import (binary_label, cross_validate, metrics, predict_score,
                        threshold_sweep, tune_threshold)


class EvaluationTests(unittest.TestCase):
    def test_confusion_and_ir(self):
        r = metrics(['S', 'S', 'NS', 'IR'], ['S', 'NS', 'S', 'NS'])
        self.assertEqual(r['confusion_matrix'], [[1, 1], [1, 1]])
        self.assertEqual(r['accuracy'], .5)
        self.assertEqual(r['S']['f1'], .5)
        self.assertEqual(binary_label('IR'), 'NS')

    def test_zero_denominators(self):
        self.assertEqual(metrics(['NS'], ['NS'])['S']['f1'], 0)
        self.assertEqual(metrics(['S'], ['NS'])['NS']['precision'], 0)

    def test_boundaries(self):
        self.assertEqual(predict_score(.5, .5), 'S')
        self.assertEqual(predict_score(.499, .5), 'NS')
        self.assertEqual(predict_score(0, 0), 'NS')
        self.assertEqual(predict_score(1, math.nextafter(1, math.inf)), 'NS')

    def test_search_matches_exhaustive_decisions(self):
        rng = random.Random(4)
        for _ in range(100):
            scores = [rng.choice([0, .1, .5, .8, 1]) for _ in range(15)]
            gold = [rng.choice(['S', 'NS']) for _ in scores]
            exact = max(metrics(gold, [predict_score(s, t) for s in scores])['accuracy']
                        for t in [0, .1, .5, .8, 1, 1.1])
            self.assertEqual(tune_threshold(scores, gold)['accuracy'], exact)

    def test_ties_and_all_zero(self):
        self.assertEqual(tune_threshold([0, 0], ['S', 'NS'])['accuracy'], .5)
        self.assertEqual(len(threshold_sweep([.5, .5], ['S', 'NS'])), 2)

    def test_cv_held_out_labels_not_used_for_threshold(self):
        scores = [i/19 for i in range(20)]
        labels = ['NS']*10 + ['S']*10
        r = cross_validate(scores, labels, 5, 7)
        self.assertEqual(r, cross_validate(scores, labels, 5, 7))
        self.assertEqual(len(r['predictions']), 20)
        for row in r['folds']:
            train = [i for i, fold in enumerate(r['assignments']) if fold != row['fold']]
            best = tune_threshold([scores[i] for i in train], [labels[i] for i in train])
            self.assertEqual(row['threshold'], best['threshold'])
        self.assertEqual(sum(x['n'] for x in r['folds']), 20)

    def test_invalid_input(self):
        for scores, gold in [([], []), ([float('nan')], ['S']), ([2], ['NS']), ([.5], ['X'])]:
            with self.assertRaises(ValueError):
                tune_threshold(scores, gold)
        with self.assertRaises(ValueError):
            cross_validate([.2, .8], ['NS', 'S'])

    def test_actual_classifier_class_wiring_without_optional_dependencies(self):
        # Execute the real class AST with a fake scoring dependency. This checks
        # Member 2's constructor/predict integration, NOT Member 1's NLP pipeline.
        path = Path(__file__).resolve().parents[2]/'factcheck.py'
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                    and n.name == 'WordRecallThresholdFactChecker')
        calls = []
        def score(fact, passages, *args):
            calls.append((fact, passages))
            return float(fact)
        env = dict(math=math, FactChecker=object, TextPreprocessor=lambda **kw: object(),
                   word_overlap_score=score, List=list)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
        cls = env['WordRecallThresholdFactChecker']
        checker = cls(.5)
        self.assertEqual(checker.predict('.5', []), 'S')
        self.assertEqual(checker.predict('.4', []), 'NS')
        self.assertEqual(cls(0).predict('0', []), 'NS')
        self.assertEqual(len(calls), 3)
        for t in [float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                cls(t)


if __name__ == '__main__':
    unittest.main()
