# src/models/verifier.py
"""
Verifier module (v_eta).
Implements a lightweight binary classifier that judges whether a candidate path
is plausible given its latent representation.
All comments in English.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Verifier(nn.Module):
    """
    Simple verifier MLP.
    Input: path latents [batch, num_paths, latent_dim]
    Output: logits [batch, num_paths] (for BCEWithLogitsLoss)
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        latents: [batch, num_paths, latent_dim] or [num_paths, latent_dim]
        Returns:
          logits: [batch, num_paths]
        """
        if latents.dim() == 2:
            latents = latents.unsqueeze(0)  # [1, num_paths, latent_dim]
        if latents.dim() != 3:
            raise ValueError("latents must be [batch, num_paths, latent_dim]")

        b, n, d = latents.size()
        x = latents.view(b * n, d)
        logits = self.net(x).view(b, n)
        return logits  # use BCEWithLogitsLoss with these logits


if __name__ == "__main__":
    v = Verifier(latent_dim=32)
    dummy = torch.randn(2, 4, 32)
    out = v(dummy)
    print("verifier output shape:", out.shape)
