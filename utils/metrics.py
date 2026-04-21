# src/utils/metrics.py
"""
Common evaluation metrics helper functions.
Provides:
  - accuracy
  - precision/recall/f1 for binary and multi-class (macro/micro)
  - top-k accuracy for ranking outputs
  - MRR (mean reciprocal rank)
"""

from typing import List, Iterable, Sequence, Tuple, Optional
from collections import Counter
import math


def accuracy(preds: Sequence, labels: Sequence) -> float:
    """Compute simple accuracy."""
    assert len(preds) == len(labels)
    if not preds:
        return 0.0
    correct = sum(1 for p, y in zip(preds, labels) if p == y)
    return correct / len(preds)


def precision_recall_f1(preds: Sequence, labels: Sequence, average: str = "macro") -> Tuple[float, float, float]:
    """
    Compute precision, recall, f1 for multi-class classification.

    Args:
      preds: predicted labels
      labels: ground truth labels
      average: 'macro' or 'micro' supported

    Returns:
      (precision, recall, f1)
    """
    assert len(preds) == len(labels)
    if not preds:
        return 0.0, 0.0, 0.0

    labels_set = set(labels) | set(preds)
    label_list = list(labels_set)

    # per-class counts
    tp = {lab: 0 for lab in label_list}
    fp = {lab: 0 for lab in label_list}
    fn = {lab: 0 for lab in label_list}

    for p, y in zip(preds, labels):
        if p == y:
            tp[y] += 1
        else:
            fp[p] += 1
            fn[y] += 1

    if average == "micro":
        sum_tp = sum(tp.values())
        sum_fp = sum(fp.values())
        sum_fn = sum(fn.values())
        prec = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
        rec = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1
    elif average == "macro":
        precs, recs, f1s = [], [], []
        for lab in label_list:
            p_ = tp[lab] / (tp[lab] + fp[lab]) if (tp[lab] + fp[lab]) > 0 else 0.0
            r_ = tp[lab] / (tp[lab] + fn[lab]) if (tp[lab] + fn[lab]) > 0 else 0.0
            f1_ = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) > 0 else 0.0
            precs.append(p_)
            recs.append(r_)
            f1s.append(f1_)
        return sum(precs) / len(precs), sum(recs) / len(recs), sum(f1s) / len(f1s)
    else:
        raise ValueError("average must be 'macro' or 'micro'")


def top_k_accuracy(ranked_preds: Sequence[Sequence], golds: Sequence, k: int = 1) -> float:
    """
    Compute top-k accuracy given ranked predictions for each example.

    ranked_preds: iterable of sequences/lists of predictions ordered by score
    golds: iterable of gold labels
    """
    assert len(ranked_preds) == len(golds)
    if not ranked_preds:
        return 0.0
    hits = 0
    for preds, gold in zip(ranked_preds, golds):
        topk = list(preds)[:k]
        if gold in topk:
            hits += 1
    return hits / len(ranked_preds)


def mean_reciprocal_rank(ranked_preds: Sequence[Sequence], golds: Sequence) -> float:
    """
    Compute MRR over multiple examples.
    For each example, search ranked_preds[i] for golds[i] and compute reciprocal rank.
    """
    assert len(ranked_preds) == len(golds)
    if not ranked_preds:
        return 0.0
    rr_sum = 0.0
    for preds, gold in zip(ranked_preds, golds):
        found = False
        for idx, p in enumerate(preds, start=1):
            if p == gold:
                rr_sum += 1.0 / idx
                found = True
                break
        if not found:
            rr_sum += 0.0
    return rr_sum / len(ranked_preds)


def classification_report(preds: Sequence, labels: Sequence) -> dict:
    """
    Return a small classification report dictionary: per-label precision/recall/f1/support,
    plus macro averages.
    """
    assert len(preds) == len(labels)
    label_set = sorted(set(labels) | set(preds))
    report = {}
    precs, recs, f1s, supports = [], [], [], []
    for lab in label_set:
        lab_preds = [p == lab for p in preds]
        lab_labels = [y == lab for y in labels]
        tp = sum(1 for lp, ly in zip(lab_preds, lab_labels) if lp and ly)
        fp = sum(1 for lp, ly in zip(lab_preds, lab_labels) if lp and not ly)
        fn = sum(1 for lp, ly in zip(lab_preds, lab_labels) if not lp and ly)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        sup = sum(1 for y in labels if y == lab)
        report[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": sup}
        precs.append(prec); recs.append(rec); f1s.append(f1); supports.append(sup)
    # macro avg
    report["macro_avg"] = {
        "precision": sum(precs) / len(precs) if precs else 0.0,
        "recall": sum(recs) / len(recs) if recs else 0.0,
        "f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "support": sum(supports),
    }
    return report
