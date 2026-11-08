"""Run from the repository root; see README.md for methodology and limitations."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.m2_classification.evaluation import (binary_label, cross_validate,
    metrics, predict_score, threshold_sweep, tune_threshold)


def load_examples(data_dir):
    # Match the supplied driver's ordering, missing-fact filtering and labels.
    # Reading directly avoids importing transformers for a lexical experiment.
    from factcheck import FactExample
    passages = {}
    with (data_dir/'passages_bm25_ChatGPT_humfacts.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if any(p['title'] != row['name'] for p in row['passages']):
                raise ValueError('Passage title does not match name')
            passages[row['sent']] = row['passages']
    examples, missing = [], 0
    with (data_dir/'dev_labeled_ChatGPT.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            for sent in row['annotations'] or []:
                for fact in sent['human-atomic-facts'] or []:
                    binary_label(fact['label'])
                    if fact['text'] in passages:
                        examples.append(FactExample(fact['text'], passages[fact['text']], fact['label']))
                    else:
                        missing += 1
    if not examples:
        raise ValueError('No matched examples')
    return examples, missing


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', type=Path, default=ROOT/'data')
    parser.add_argument('--out_dir', type=Path, default=ROOT/'results/m2')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    for name in ('dev_labeled_ChatGPT.jsonl', 'passages_bm25_ChatGPT_humfacts.jsonl'):
        if not (args.data_dir/name).is_file():
            parser.error('Missing coursework data file: ' + str(args.data_dir/name))
    from factcheck import WordRecallThresholdFactChecker
    examples, missing = load_examples(args.data_dir)
    checker = WordRecallThresholdFactChecker()
    scores = [checker.score(e.fact, e.passages) for e in examples]
    labels = [e.label for e in examples]
    best = tune_threshold(scores, labels)
    cv = cross_validate(scores, labels, args.folds, args.seed)
    predictions = [predict_score(s, best['threshold']) for s in scores]
    default_pred = [checker.predict(e.fact, e.passages) for e in examples]
    baseline = {label: metrics(labels, [label]*len(labels)) for label in ('S', 'NS')}
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unavailable'
    report = dict(selected_threshold=best['threshold'], development_tuned=best,
                  cross_validation=cv, default_threshold=checker.threshold,
                  default_metrics=metrics(labels, default_pred), baselines=baseline,
                  target_met_on_development=best['accuracy'] >= .75,
                  missing_facts_skipped=missing, python=platform.python_version(),
                  commit=commit, seed=args.seed, folds=args.folds,
                  preprocessing=checker.PREPROCESSING, metric=checker.METRIC,
                  aggregation=checker.AGGREGATION, drop_title=checker.DROP_TITLE,
                  data_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(args.data_dir.glob('*.jsonl'))},
                  limitations=['Development-tuned accuracy reuses threshold-fitting data.',
                    'CV validates threshold selection only: Member 1 selected preprocessing on the full development set.',
                    'Fact-level folds may share biography passages; this is not unseen-biography evaluation.'])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir/'evaluation.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    sweep = threshold_sweep(scores, labels)
    write_csv(args.out_dir/'threshold_sweep.csv', [dict(threshold=r['threshold'], accuracy=r['accuracy'],
              tp=r['tp'], tn=r['tn'], fp=r['fp'], fn=r['fn'], macro_f1=r['macro_f1']) for r in sweep])
    rows = [dict(fact_id=i, fact=e.fact, original_label=e.label, gold=binary_label(e.label),
            score=scores[i], tuned_prediction=predictions[i], default_prediction=default_pred[i],
            cv_prediction=cv['predictions'][i], cv_fold=cv['assignments'][i],
            error_type=('FP' if predictions[i]=='S' else 'FN') if predictions[i]!=binary_label(e.label) else '')
            for i, e in enumerate(examples)]
    write_csv(args.out_dir/'predictions.csv', rows)
    with (args.out_dir/'errors.jsonl').open('w', encoding='utf-8') as f:
        for row, ex in zip(rows, examples):
            if row['error_type']:
                f.write(json.dumps(dict(row, passages=ex.passages), ensure_ascii=False)+'\n')
    summary = ('# Member 2 evaluation\n\n'
        f"Selected threshold: {best['threshold']!r}. Rule: S iff score > 0 and score >= threshold.\n\n"
        f"Development-tuned accuracy: {best['accuracy']:.4%} ({best['tp']+best['tn']}/{len(labels)}).\n\n"
        f"Threshold-only {args.folds}-fold CV pooled accuracy: {cv['pooled']['accuracy']:.4%}.\n\n"
        f"Current default threshold accuracy: {report['default_metrics']['accuracy']:.4%}.\n\n"
        'Confusion matrix rows=gold, columns=prediction, both ordered S, NS: '
        f"{best['confusion_matrix']}.\n\n"
        'Use evaluation.json for per-class precision/recall/F1, baselines and fold results. '
        'Copy selected_threshold into WordRecallThresholdFactChecker.DEFAULT_THRESHOLD to use it with '
        'the unmodified driver. Keep its full precision.\n\n'
        + '\n'.join('- '+x for x in report['limitations'])+'\n')
    (args.out_dir/'summary.md').write_text(summary, encoding='utf-8')
    print(summary)


if __name__ == '__main__':
    main()
