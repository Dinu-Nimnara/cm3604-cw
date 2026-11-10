import unittest
from types import SimpleNamespace
from experiment_member3_evidence import clean_text, sentence_units, iter_candidates


def nlp(text):
    return SimpleNamespace(sents=[SimpleNamespace(text=s) for s in text.split('||')])


class EvidenceTests(unittest.TestCase):
    def test_preserves_fact_and_negation(self):
        self.assertEqual(clean_text('He did not sing in 1964.{{cite web|url=x}}'),
                         'He did not sing in 1964.')

    def test_unclosed_citation_does_not_erase_prose(self):
        self.assertIn('He married in 1971.', clean_text(
            '{{cite news|title=x He married in 1971.'))

    def test_nested_templates(self):
        self.assertEqual(clean_text('A {{cite web|date={{date|2020}}}} B'), 'A B')

    def test_links_and_entities(self):
        self.assertEqual(clean_text('[[Donald Featherstone|Featherstone]] &amp; X'),
                         'Featherstone & X')

    def test_quote_repair_preserves_title(self):
        self.assertEqual(sentence_units('He starred in "Save Me.||" Later he left.', nlp, True),
                         ['He starred in "Save Me. " Later he left.'])

    def test_context_stays_inside_segment(self):
        passages = [{'text':'Alice arrived.||She sang.</s><s>Bob left.'},
                    {'text':'Carol returned.'}]
        result = list(iter_candidates(passages, nlp, 'context'))
        self.assertIn('Alice arrived. She sang.', result)
        self.assertNotIn('She sang. Bob left.', result)
        self.assertNotIn('Bob left. Carol returned.', result)

    def test_clean_mode_does_not_add_context(self):
        self.assertEqual(list(iter_candidates([{'text':'A||B'}], nlp, 'clean')), ['A','B'])

    def test_deduplication(self):
        self.assertEqual(list(iter_candidates([{'text':'A</s><s>A'}], nlp, 'split')), ['A'])


if __name__ == '__main__':
    unittest.main()
