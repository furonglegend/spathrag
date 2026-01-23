# src/eval/diagnostics.py
"""
Diagnostic utilities for model introspection:
  - attention_mass: compute mass of attention on injected tokens given attention tensors
  - causal_ablation: compare model outputs with and without injection to estimate causal effect
  - coverage_stats: produce simple coverage statistics for candidate paths
"""

from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import torch


def attention_mass(attentions: torch.Tensor, injected_token_indices: List[int]) -> float:
    """
    Compute the fraction of attention mass directed at injected tokens.
    attentions: shape [num_layers, num_heads, seq_len, seq_len] or [num_heads, seq_len, seq_len]
    injected_token_indices: list of token positions that are injected/associated with paths
    Returns scalar mass in [0,1].
    """
    att = attentions.detach().cpu().numpy()
    if att.ndim == 4:
        # average over layers and heads
        att_mean = att.mean(axis=(0,1))  # [seq_len, seq_len]
    elif att.ndim == 3:
        att_mean = att.mean(axis=0)  # [seq_len, seq_len]
    else:
        raise ValueError("attentions must be 3D or 4D tensor")

    # assume we measure attention from all query tokens to injected tokens:
    seq_len = att_mean.shape[0]
    injected_mask = np.zeros(seq_len, dtype=float)
    for idx in injected_token_indices:
        if 0 <= idx < seq_len:
            injected_mask[idx] = 1.0

    # attention from every token to injected positions
    mass = (att_mean * injected_mask[None, :]).sum()
    # normalize by total attention mass
    total = att_mean.sum()
    return float(mass / total) if total > 0 else 0.0


def causal_ablation(generate_fn, query: str, injected_kv: Optional[Any], compare_tokens: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Simple causal ablation wrapper.
    generate_fn: callable(query, injected_kv) -> text or dict with 'answer'
    Returns dict with:
      - baseline: output without injection
      - with_injection: output with injection
      - effect: textual/structural difference (simple)
    """
    baseline_out = generate_fn(query, None)
    with_out = generate_fn(query, injected_kv)
    baseline_text = baseline_out if isinstance(baseline_out, str) else baseline_out.get("answer", "")
    with_text = with_out if isinstance(with_out, str) else with_out.get("answer", "")

    # very simple effect: whether answer changed and token-level diff length
    baseline_tokens = baseline_text.split()
    with_tokens = with_text.split()
    effect = {
        "changed": baseline_text.strip() != with_text.strip(),
        "baseline_len": len(baseline_tokens),
        "with_len": len(with_tokens),
        "baseline": baseline_text,
        "with": with_text,
    }
    return {"baseline_out": baseline_out, "with_out": with_out, "effect": effect}


def coverage_stats(candidate_paths: List[List[str]]) -> Dict[str, Any]:
    """
    Simple coverage stats: distribution of path lengths, most frequent nodes, number of unique paths.
    """
    lengths = [len(p) - 1 for p in candidate_paths if isinstance(p, (list, tuple)) and len(p) > 0]
    unique_paths = len({tuple(p) for p in candidate_paths})
    node_counter = {}
    for p in candidate_paths:
        for n in p:
            node_counter[n] = node_counter.get(n, 0) + 1
    top_nodes = sorted(node_counter.items(), key=lambda x: x[1], reverse=True)[:10]
    stats = {
        "num_paths": len(candidate_paths),
        "unique_paths": unique_paths,
        "lengths": {"min": min(lengths) if lengths else 0, "max": max(lengths) if lengths else 0, "avg": (sum(lengths)/len(lengths)) if lengths else 0},
        "top_nodes": top_nodes,
    }
    return stats
