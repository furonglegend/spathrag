# src/models/scorer.py
"""
Path scorer module (s_theta).
Implements:
  - a small MLP scorer that maps path latents -> scalar score
  - a utility to obtain soft top-k weights using Gumbel-Softmax
All comments in English.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Scorer(nn.Module):
    """
    Simple MLP scorer for candidate path latents.
    Expected input: tensor of shape [batch, num_paths, latent_dim]
    Output: tensor of shape [batch, num_paths] of scalar scores (logits).
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # MLP: latent -> hidden -> score
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # output scalar per path
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        latents: [batch, num_paths, latent_dim] or [num_paths, latent_dim] or [batch, latent_dim]
        Returns:
          scores: [batch, num_paths] logits (float)
        """
        if latents.dim() == 2:
            # [num_paths, latent_dim] -> treat as batch=1
            latents = latents.unsqueeze(0)
        if latents.dim() == 3:
            b, n, d = latents.size()
            x = latents.view(b * n, d)
            out = self.net(x).view(b, n)
            return out
        elif latents.dim() == 1:
            # [latent_dim] -> single path
            out = self.net(latents.unsqueeze(0)).squeeze(0).unsqueeze(0)  # shape [1,1]
            return out
        else:
            raise ValueError(f"Unsupported latents shape: {latents.shape}")


def gumbel_softmax_topk(logits: torch.Tensor, k: int = 1, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
    """
    Compute a differentiable top-k selection using repeated Gumbel-Softmax.
    Args:
      logits: [batch, num_items]
      k: number of selections per batch item
      temperature: temperature for Gumbel-Softmax
      hard: whether to produce hard one-hot samples (with straight-through)
    Returns:
      weights: [batch, num_items] non-negative weights that sum to k (soft selection)
    Notes:
      - This routine draws k independent Gumbel-softmax samples and sums them. This is
        a simple (biased) trick to get a differentiable 'top-k' proxy.
      - For large k this may produce duplicates; if duplicates are undesired, use other
        schemes (e.g., sparsemax or differentiable top-k algorithms).
    """
    if logits.dim() != 2:
        raise ValueError("logits must be 2D (batch, num_items)")

    batch, num_items = logits.shape
    weights = torch.zeros_like(logits)

    for i in range(k):
        # sample Gumbel noise
        gumbels = -torch.empty_like(logits).exponential_().log()  # sample Gumbel(0,1)
        y = (logits + gumbels) / temperature
        y_soft = F.softmax(y, dim=-1)  # shape [batch, num_items]
        if hard:
            # straight-through one-hot
            _, idx = y_soft.max(dim=-1)  # [batch]
            y_hard = torch.zeros_like(y_soft).scatter_(-1, idx.unsqueeze(-1), 1.0)
            y = (y_hard - y_soft).detach() + y_soft
            weights = weights + y
        else:
            weights = weights + y_soft

    return weights  # sum to k across last dim (approximately)


if __name__ == "__main__":
    # quick sanity check
    scorer = Scorer(latent_dim=32)
    dummy = torch.randn(2, 5, 32)
    scores = scorer(dummy)  # [2,5]
    print("scores shape:", scores.shape)
    weights = gumbel_softmax_topk(scores, k=2, temperature=0.5, hard=False)
    print("weights shape:", weights.shape, "sum per batch:", weights.sum(dim=-1))
