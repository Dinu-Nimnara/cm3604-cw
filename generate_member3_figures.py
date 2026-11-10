"""Generate report figures from saved M3 results. No model inference or imports.

Requires matplotlib and numpy only; use a separate plotting environment.
Run this file from the repository root or supply --root PATH.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def validate_rows(cache):
    rows = cache['records']
    if len(rows) != cache['total'] or not rows:
        raise ValueError('Incomplete score cache.')
    if len({r['index'] for r in rows}) != len(rows):
        raise ValueError('Duplicate record indices.')
    for row in rows:
        if row['gold'] not in {'S', 'NS'}:
            raise ValueError('Expected binary S/NS labels.')
        if not isinstance(row['score'], (int, float)) or not math.isfinite(row['score']) or not 0 <= row['score'] <= 1:
            raise ValueError('Invalid entailment probability.')
    return rows


def matrix(rows, threshold):
    result = np.zeros((2, 2), dtype=int)
    for row in rows:
        result[0 if row['gold'] == 'S' else 1,
               0 if row['score'] >= threshold else 1] += 1
    return result


def accuracy(rows, threshold):
    return float(np.trace(matrix(rows, threshold))) / len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, help='Override output directory')
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / 'results/m3'
    paths = [source/'baseline_scores.json', source/'evidence_context_scores.json',
             source/'evidence_context_report.json']
    baseline, cache, report = map(read_json, paths)
    old, rows = validate_rows(baseline), validate_rows(cache)
    if baseline['dataset_fingerprint'] != cache['dataset_fingerprint'] or baseline['model_id'] != cache['model_id']:
        raise ValueError('Baseline and context use different data or models.')
    if [(r['index'], r['fact'], r['gold']) for r in old] != [(r['index'], r['fact'], r['gold']) for r in rows]:
        raise ValueError('Facts, ordering or gold labels differ.')
    for key in ['total', 'model_id', 'dataset_fingerprint', 'inference_fingerprint']:
        if cache[key] != report[key]:
            raise ValueError('Context report/cache mismatch: ' + key)
    threshold = float(report['selected']['threshold'])
    if not 0 < threshold <= 1:
        raise ValueError('Invalid selected threshold.')
    cm = matrix(rows, threshold)
    correct = int(np.trace(cm))
    if cm.tolist() != report['selected']['confusion_matrix'] or correct != report['selected']['correct']:
        raise ValueError('Saved context metrics disagree with per-fact scores.')
    if not math.isclose(correct/len(rows), report['selected']['accuracy'], abs_tol=1e-12):
        raise ValueError('Saved accuracy disagrees with counts.')
    original_threshold = float(report['experiment']['original_implementation']['threshold'])
    if matrix(old, original_threshold).tolist() != report['experiment']['original_implementation']['confusion_matrix']:
        raise ValueError('Saved original metrics disagree with baseline cache.')

    # The integration is confirmed separately by the actual normal-driver log.
    driver = source / 'context_driver_console.txt'
    driver_verified = False
    if driver.exists():
        raw = driver.read_bytes()
        text = raw.decode('utf-16') if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else raw.decode('utf-8-sig')
        matches = re.findall(r'Accuracy:\s*(\d+)\s*/\s*(\d+)\s*=', text)
        if not matches or tuple(map(int, matches[-1])) != (correct, len(rows)):
            raise ValueError('Driver log is missing or disagrees with score accuracy.')
        driver_verified = True
        paths.append(driver)

    output = args.output.resolve() if args.output else root/'report/figures'
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 14,
                         'axes.spines.top': False, 'axes.spines.right': False})
    generated = []

    def save(fig, name):
        path = output/name
        fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=.2, facecolor='white')
        plt.close(fig)
        generated.append(name)
        print('Saved:', path)

    # Rows = actual, columns = predicted; Supported is the positive class.
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    ax.imshow(cm, cmap='Blues', vmin=0, vmax=int(cm.sum(axis=1).max()))
    labels = ['Supported (S)', 'Not Supported (NS)']
    ax.set_xticks([0, 1], labels); ax.set_yticks([0, 1], labels)
    ax.set_xlabel('Predicted label', labelpad=12); ax.set_ylabel('Actual label', labelpad=12)
    names = [['True positive', 'False negative'], ['False positive', 'True negative']]
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i,j] > cm.sum(axis=1).max()/2 else '#17324d'
            ax.text(j, i-.07, str(cm[i,j]), ha='center', va='center', fontsize=28, fontweight='bold', color=color)
            ax.text(j, i+.17, names[i][j], ha='center', va='center', fontsize=10, color=color)
    ax.set_xticks([-.5, .5, 1.5], minor=True); ax.set_yticks([-.5, .5, 1.5], minor=True)
    ax.grid(which='minor', color='white', linewidth=3); ax.tick_params(which='both', length=0)
    ax.set_title(f'Member 3: DeBERTa confusion matrix\nDevelopment accuracy: {correct/len(rows):.2%} ({correct}/{len(rows)})', pad=18)
    fig.text(.5, .015, f'Context evidence · Threshold: {threshold:.9f}', ha='center', fontsize=9)
    fig.subplots_adjust(left=.29, right=.97, bottom=.16, top=.83)
    save(fig, 'm3_confusion_matrix.png')

    settings = [('Sentence\nthreshold 0.5', old, .5),
                ('Sentence\ntuned threshold', old, original_threshold),
                ('Context\noriginal threshold', rows, original_threshold),
                ('Context\nretuned threshold', rows, threshold)]
    counts = [int(np.trace(matrix(r,t))) for _,r,t in settings]
    values = [100*n/len(rows) for n in counts]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    bars = ax.bar([s[0] for s in settings], values, color=['#94a3b8','#64748b','#60a5fa','#1d4ed8'], width=.62)
    for bar, value, count in zip(bars, values, counts):
        ax.text(bar.get_x()+bar.get_width()/2, value+2.0, f'{value:.2f}%\n{count}/{len(rows)}', ha='center', va='bottom', fontsize=10)
    ax.axhline(83, color='#b45309', linestyle='--', label='Coursework target: 83%')
    ax.set_ylim(0, 100); ax.set_ylabel('Development accuracy (%)')
    ax.set_title('Member 3: evidence and threshold comparison')
    ax.legend(loc='lower right'); ax.set_axisbelow(True); ax.grid(axis='y', alpha=.2)
    fig.subplots_adjust(bottom=.2)
    save(fig, 'm3_accuracy_comparison.png')

    # Exhaust score boundaries and include the exact chosen threshold.
    unique = sorted({0., 1., threshold, *[float(r['score']) for r in rows]})
    thresholds = sorted(set(unique + [float(np.nextafter(s, 1.)) for s in unique if s < 1.]))
    acc = [100*accuracy(rows, t) for t in thresholds]
    fig, ax = plt.subplots(figsize=(8.4, 5.3))
    ax.step(thresholds, acc, where='post', color='#1d4ed8', linewidth=1.8)
    ax.axhline(83, color='#b45309', linestyle='--', label='83% target')
    ax.axvline(threshold, color='#475569', linestyle=':', label=f'Selected threshold: {threshold:.6f}')
    ax.scatter([threshold], [100*correct/len(rows)], color='#1d4ed8', zorder=5)
    ax.set(xlim=(0,1), ylim=(0,100), xlabel='Entailment threshold', ylabel='Development accuracy (%)',
           title='Member 3: threshold selection for context evidence')
    ax.legend(loc='lower left'); ax.grid(alpha=.2)
    fig.text(.5,.005,'Threshold selected on this development set; this is not held-out test accuracy.', ha='center', fontsize=9)
    save(fig, 'm3_threshold_accuracy.png')

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.4), sharex=True)
    for ax, label, color in zip(axes, ['S','NS'], ['#2563eb','#d97706']):
        scores = [r['score'] for r in rows if r['gold']==label]
        ax.hist(scores, bins=np.linspace(0,1,21), color=color, edgecolor='white')
        ax.axvline(threshold, color='#111827', linestyle='--', label=f'Threshold: {threshold:.6f}')
        ax.set_ylabel('Number of facts'); ax.set_title(f'Gold label: {label} (n={len(scores)})', fontsize=11)
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    axes[1].set_xlabel('Maximum entailment probability per fact')
    fig.suptitle('Member 3: context-score distributions on development data', fontsize=13)
    fig.subplots_adjust(hspace=.38, top=.88)
    save(fig, 'm3_score_distributions.png')

    metadata = {'evaluation': 'Development data used for threshold selection; not independent test data.',
                'correct': correct, 'total': len(rows), 'accuracy': correct/len(rows),
                'threshold': threshold, 'confusion_matrix_order': 'rows actual S,NS; columns predicted S,NS',
                'confusion_matrix': cm.tolist(), 'normal_driver_log_accuracy_verified': driver_verified,
                'model_inference_performed': False, 'figures': generated,
                'inputs': [{'filename': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}
    (output/'m3_figure_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print(f'Finished: {len(generated)} PNG figures plus metadata. No model inference performed.')


if __name__ == '__main__':
    main()
