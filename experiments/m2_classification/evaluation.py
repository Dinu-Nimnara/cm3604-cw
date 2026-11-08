"""Dependency-free binary evaluation; S is positive and IR maps to NS."""
import math
import random


def binary_label(label):
    if label not in ('S', 'NS', 'IR'):
        raise ValueError('Unknown label: ' + repr(label))
    return 'S' if label == 'S' else 'NS'


def validate(scores, labels):
    if not scores or len(scores) != len(labels):
        raise ValueError('Expected nonempty, equally sized scores and labels')
    if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError('Scores must be finite and in [0, 1]')
    return [binary_label(y) for y in labels]


def predict_score(score, threshold):
    return 'S' if score > 0 and score >= threshold else 'NS'


def metrics(gold, predicted):
    if not gold or len(gold) != len(predicted):
        raise ValueError('Expected nonempty aligned predictions')
    gold = [binary_label(y) for y in gold]
    if any(y not in ('S', 'NS') for y in predicted):
        raise ValueError('Predictions must be S or NS')
    tp = sum(g == p == 'S' for g, p in zip(gold, predicted))
    tn = sum(g == p == 'NS' for g, p in zip(gold, predicted))
    fp = sum(g == 'NS' and p == 'S' for g, p in zip(gold, predicted))
    fn = sum(g == 'S' and p == 'NS' for g, p in zip(gold, predicted))
    def ratio(a, b):
        return a / b if b else 0.0
    def per_class(a, b, c):
        return dict(precision=ratio(a, a+b), recall=ratio(a, a+c), f1=ratio(2*a, 2*a+b+c))
    s, ns = per_class(tp, fp, fn), per_class(tn, fn, fp)
    return dict(n=len(gold), accuracy=(tp+tn)/len(gold), tp=tp, tn=tn, fp=fp, fn=fn,
                confusion_matrix=[[tp, fn], [fp, tn]], label_order=['S', 'NS'],
                S=s, NS=ns, macro_f1=(s['f1']+ns['f1'])/2)


def threshold_sweep(scores, labels):
    gold = validate(scores, labels)
    # Sorted unique positive scores cover every possible attainable decision pattern.
    # Midpoints avoid choosing a threshold at a discontinuity; 0 and >1 cover extremes.
    unique = sorted(set(s for s in scores if s > 0))
    candidates = [0.0] + [(a+b)/2 for a, b in zip(unique, unique[1:])]
    candidates.append(math.nextafter(1.0, math.inf))
    return [dict(threshold=t, **metrics(gold, [predict_score(s, t) for s in scores])) for t in candidates]


def tune_threshold(scores, labels):
    # Optimize accuracy only. On ties prefer the threshold closest to 0.5,
    # then the smaller value. No evaluation-fold labels enter this decision.
    rows = threshold_sweep(scores, labels)
    return min(rows, key=lambda r: (-r['accuracy'], abs(r['threshold']-.5), r['threshold']))


def cross_validate(scores, labels, folds=5, seed=0):
    gold = validate(scores, labels)
    groups = [[i for i, y in enumerate(gold) if y == label] for label in ('S', 'NS')]
    if folds < 2 or any(len(g) < folds for g in groups):
        raise ValueError('Each class must have at least folds examples; folds must be >=2')
    rng = random.Random(seed)
    test_folds = [[] for _ in range(folds)]
    for group in groups:
        rng.shuffle(group)
        for j, idx in enumerate(group):
            test_folds[j % folds].append(idx)
    predictions, assignments, rows = [None]*len(gold), [None]*len(gold), []
    for fold, test in enumerate(test_folds):
        test_set = set(test)
        train = [i for i in range(len(gold)) if i not in test_set]
        tuned = tune_threshold([scores[i] for i in train], [gold[i] for i in train])
        pred = [predict_score(scores[i], tuned['threshold']) for i in test]
        for i, p in zip(test, pred):
            predictions[i], assignments[i] = p, fold
        rows.append(dict(fold=fold, threshold=tuned['threshold'], train_n=len(train),
                         **metrics([gold[i] for i in test], pred)))
    return dict(pooled=metrics(gold, predictions), folds=rows,
                predictions=predictions, assignments=assignments)
