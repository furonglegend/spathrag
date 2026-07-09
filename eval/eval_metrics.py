# src/eval/eval_metrics.py
"""
Evaluation metric utilities.
Includes:
  - exact_match: exact string match
  - f1_score: token-level F1
  - mrr: mean reciprocal rank over ranked answers
  - path_coverage: fraction of gold paths recovered by candidate list
"""

import math
from typing import List, Iterable, Tuple
from collections import Counter


def normalize_answer(s: str) -> str:
    """Lower, strip, remove redundant whitespace. Keep simple normalization."""
    return " ".join(s.lower().strip().split())


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def mean_reciprocal_rank(ranked_list: List[str], golds: Iterable[str]) -> float:
    """
    Compute MRR for a single query given ranked_list of predictions and set/list of gold answers.
    """
    golds = set([normalize_answer(g) for g in golds])
    for i, pred in enumerate(ranked_list, start=1):
        if normalize_answer(pred) in golds:
            return 1.0 / i
    return 0.0


def compute_mrr(all_ranked: List[List[str]], all_golds: List[Iterable[str]]) -> float:
    """
    Compute MRR across multiple queries.
    all_ranked: list of ranked prediction lists per query
    all_golds: list of gold answer iterables per query
    """
    rr = [mean_reciprocal_rank(ranked, gold) for ranked, gold in zip(all_ranked, all_golds)]
    return sum(rr) / len(rr) if rr else 0.0


def path_coverage(candidate_paths: List[List[str]], gold_paths: List[List[str]]) -> float:
    """
    Fraction of gold paths that appear in candidate_paths (exact path match).
    If there are duplicates, treat them as unique by content.
    """
    cand_set = set(tuple(p) for p in candidate_paths)
    gold_set = set(tuple(p) for p in gold_paths)
    if not gold_set:
        return 1.0
    recovered = sum(1 for g in gold_set if g in cand_set)
    return recovered / len(gold_set)
