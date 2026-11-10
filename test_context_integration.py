"""Structural and evidence parity checks; no DeBERTa download required."""
import ast
import math
from pathlib import Path
import re
from types import SimpleNamespace
import unittest

from experiment_member3_evidence import iter_candidates


def load_checker():
    tree = ast.parse(Path(__file__).with_name('factcheck.py').read_text(encoding='utf-8-sig'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == 'EntailmentFactChecker')
    scope = {'FactChecker': object, 'List': list, 're': re, 'math': math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'checker-test', 'exec'), scope)
    checker = scope['EntailmentFactChecker'].__new__(scope['EntailmentFactChecker'])
    checker.nlp = lambda text: SimpleNamespace(sents=[SimpleNamespace(text=s)
        for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()])
    checker.threshold = checker.DEFAULT_THRESHOLD
    return checker


class ContextTests(unittest.TestCase):
    def test_expected_threshold(self):
        self.assertEqual(load_checker().threshold, .436032310128212)

    def test_exact_experiment_candidate_order(self):
        checker = load_checker()
        passages = [{'text': '<s>Alice arrived. She did not sing.{{cite web|url=x}}</s>'
                     '<s>He starred in "A Title. " Then left.</s>'},
                    {'text': 'Bob arrived. He left.'}]
        self.assertEqual(list(checker.iter_evidence(passages)),
                         list(iter_candidates(passages, checker.nlp, 'context')))
        self.assertNotIn('Then left. Bob arrived.', list(checker.iter_evidence(passages)))

    def test_predict_uses_max_and_selected_threshold(self):
        checker = load_checker()
        checker.ent_model = SimpleNamespace(check_entailment=lambda evidence, fact:
            .8 if evidence == 'Alice arrived. She sang.' else .1)
        self.assertEqual(checker.score_details('Alice sang.', [{'text':'Alice arrived. She sang.'}]),
                         (.8, 'Alice arrived. She sang.'))
        self.assertEqual(checker.predict('Alice sang.', [{'text':'Alice arrived. She sang.'}]), 'S')
        self.assertEqual(checker.predict('Alice sang.', [{'text':'Nothing relevant.'}]), 'NS')


if __name__ == '__main__':
    unittest.main()
