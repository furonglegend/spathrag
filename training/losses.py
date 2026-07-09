# src/training/losses.py
"""
Loss functions used across training stages.
Provides:
  - info_nce_loss: contrastive loss commonly used for path vs. query matching
  - verifier_loss: binary cross entropy for verifier
  - answer_loss: typical sequence/answer loss (cross-entropy)
  - align_loss: alignment loss (MSE) between injected latents and target latents
"""

from typing import Optional
import torch
import torch.nn.functional as F


def info_nce_loss(query_emb: torch.Tensor, positive_emb: torch.Tensor, negatives: Optional[torch.Tensor] = None, temperature: float = 0.07) -> torch.Tensor:
    """
    Compute InfoNCE loss.
    query_emb: [B, D]
    positive_emb: [B, D]
    negatives: [B, N, D] or None
    Returns scalar loss.
    """
    # normalize
    q = F.normalize(query_emb, p=2, dim=-1)
    p = F.normalize(positive_emb, p=2, dim=-1)
    pos_logits = (q * p).sum(dim=-1) / temperature  # [B]

    if negatives is None:
        # simple negative sampling using other elements in the batch
        logits = torch.matmul(q, p.t()) / temperature  # [B, B]
        labels = torch.arange(q.size(0), device=q.device)
        loss = F.cross_entropy(logits, labels)
        return loss
    else:
        # negatives shape [B, N, D] -> reshape
        neg = negatives.view(negatives.size(0), -1, negatives.size(-1))
        neg_flat = neg.transpose(0,1).contiguous().view(-1, neg.size(-1))  # not ideal but simple
        # compute logits with positives and negatives
        # concat positive then negatives
        positive_logits = pos_logits.unsqueeze(1)  # [B,1]
        # compute q vs negatives
        q_expand = q.unsqueeze(1)  # [B,1,D]
        neg_logits = torch.matmul(q_expand, neg.transpose(-2, -1)).squeeze(1) / temperature
        logits = torch.cat([positive_logits, neg_logits], dim=1)
        labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
        loss = F.cross_entropy(logits, labels)
        return loss


def verifier_loss(pred_scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy loss for verifier.
    pred_scores: [B] logits or probabilities
    targets: [B] in {0,1}
    """
    if pred_scores.dim() == 2 and pred_scores.size(1) == 1:
        pred_scores = pred_scores.squeeze(1)
    loss = F.binary_cross_entropy_with_logits(pred_scores, targets.float())
    return loss


def answer_loss(logits: torch.Tensor, target_ids: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    Standard token-level cross entropy for autoregressive models.
    logits: [B, T, V]
    target_ids: [B, T]
    """
    # flatten
    b, t, v = logits.size()
    loss = F.cross_entropy(logits.view(-1, v), target_ids.view(-1), ignore_index=ignore_index)
    return loss


def align_loss(injected_latents: torch.Tensor, target_latents: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error between projected injection latents and target latents (or path-encoded latents).
    Both tensors expected to be same shape.
    """
    return F.mse_loss(injected_latents, target_latents)
