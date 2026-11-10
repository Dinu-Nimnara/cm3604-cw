"""Isolated evidence experiments. Run beside factcheck.py and tune_member3.py.

Never edits either source file. Context mode is exploratory and changes the
evidence unit to include adjacent sentences; confirm coursework acceptability.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re


def clean_text(text):
    # Remove only CLOSED citation templates. An unfinished citation may contain
    # real prose after it, so do not delete everything following its opening.
    ranges, stack = [], []
    for match in re.finditer(r"\{\{|\}\}", text):
        if match.group() == "{{":
            stack.append(match.start())
        elif stack:
            start = stack.pop()
            if re.match(r"\{\{\s*(?:cite\b|citation\b)", text[start:], re.I):
                ranges.append((start, match.end()))
    for start, end in sorted(ranges, reverse=True):
        # Nested removals would invalidate offsets: remove outermost ranges only.
        if not any(a <= start and end <= b and (a, b) != (start, end)
                   for a, b in ranges):
            text = text[:start] + " " + text[end:]
    text = re.sub(r"<ref\b[^>]*>.*?</ref\s*>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[(?:https?://\S+)\s+([^\]]+)\]", r"\1", text)
    return " ".join(html.unescape(text).split())


def sentence_units(text, nlp, repair_quotes=False):
    sentences = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    if not repair_quotes:
        return sentences
    # Merge boundaries that occur while inside an ASCII/curly double quote.
    # An unclosed quotation is only merged to the end of this retrieval segment.
    result, pending, quote = [], [], None
    for sentence in sentences:
        pending.append(sentence)
        for char in sentence:
            if char == '"':
                quote = None if quote == '"' else ('"' if quote is None else quote)
            elif char == '\u201c' and quote is None:
                quote = '\u201d'
            elif char == '\u201d' and quote == '\u201d':
                quote = None
        if quote is None:
            result.append(" ".join(pending))
            pending = []
    if pending:
        result.append(" ".join(pending))
    return result


def iter_candidates(passages, nlp, mode):
    seen = set()
    for passage in passages:
        for segment in re.split(r"</?s\s*>", passage['text'], flags=re.I):
            cleaned = clean_text(segment)
            if not cleaned:
                continue
            sentences = sentence_units(cleaned, nlp, mode in {'split', 'context'})
            for index, sentence in enumerate(sentences):
                candidates = [sentence]
                if mode == 'context' and index:
                    candidates.append(sentences[index - 1] + ' ' + sentence)
                for evidence in candidates:
                    if evidence not in seen:
                        seen.add(evidence)
                        yield evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['clean', 'split', 'context'])
    parser.add_argument('--baseline', default='results/m3/baseline_scores.json')
    parser.add_argument('--output-dir', default='results/m3')
    parser.add_argument('--scores-only', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    import tune_member3 as tuner
    baseline = json.loads(Path(args.baseline).read_text(encoding='utf-8-sig'))
    examples = tuner.load_examples('data/dev_labeled_ChatGPT.jsonl',
                                   'data/passages_bm25_ChatGPT_humfacts.jsonl')
    expected = dict(baseline)
    expected['dataset_fingerprint'] = tuner.digest(examples)
    expected['total'] = len(examples)
    baseline_rows = tuner.validate_cache(baseline, expected, examples)
    if len(baseline_rows) != len(examples):
        raise ValueError('Baseline cache must contain every fact.')

    # Include this runner's entire source AND selected mode in the fingerprint,
    # so an old cache cannot silently score a new experiment implementation.
    original_fingerprint = tuner.inference_fingerprint
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    tuner.inference_fingerprint = lambda path: tuner.digest({
        'core': original_fingerprint(path), 'runner': runner_hash, 'mode': args.mode})

    if not args.scores_only:
        import factcheck
        factcheck.EntailmentFactChecker.iter_evidence = (
            lambda self, passages: iter_candidates(passages, self.nlp, args.mode))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / ('evidence_' + args.mode)
    cache = prefix.with_name(prefix.name + '_scores.json')
    report = prefix.with_name(prefix.name + '_report.json')
    cli = ['--cache', str(cache), '--report', str(report)]
    if args.cuda:
        cli.append('--cuda')
    if args.scores_only:
        cli.append('--scores-only')
    result = tuner.main(cli)  # --apply deliberately unavailable
    if result:
        return result
    candidate_cache = json.loads(cache.read_text(encoding='utf-8'))
    rows = candidate_cache['records']
    report_data = json.loads(report.read_text(encoding='utf-8'))
    original_threshold = 0.3308477997779846
    old = tuner.metrics(baseline_rows, original_threshold)
    fixed = tuner.metrics(rows, original_threshold)
    tuned = report_data['selected']
    changes = []
    for before, after in zip(baseline_rows, rows):
        old_prediction = 'S' if before['score'] >= original_threshold else 'NS'
        new_prediction = 'S' if after['score'] >= tuned['threshold'] else 'NS'
        if old_prediction != new_prediction:
            changes.append({'index': before['index'], 'fact': before['fact'],
                            'gold': before['gold'], 'before': old_prediction,
                            'after': new_prediction, 'before_score': before['score'],
                            'after_score': after['score'],
                            'best_evidence_after': after['best_evidence']})
    report_data['experiment'] = {
        'mode': args.mode, 'runner_sha256': runner_hash,
        'context_requires_coursework_review': args.mode == 'context',
        'original_implementation': old, 'candidate_at_original_threshold': fixed,
        'net_correct_change_after_retuning': tuned['correct'] - old['correct'],
        'changed_predictions_after_retuning': changes,
        'source_files_modified': False,
        'limitation': 'Development labels used for tuning; not held-out accuracy.'}
    tuner.atomic_json(report, report_data)
    print('Original:', old['correct'], '/', old['total'])
    print('Candidate at original threshold:', fixed['correct'], '/', fixed['total'])
    print('Candidate after retuning:', tuned['correct'], '/', tuned['total'])
    print('Net correct change:', tuned['correct'] - old['correct'])
    print('Saved:', report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
