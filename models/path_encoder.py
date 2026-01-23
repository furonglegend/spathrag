# src/models/path_encoder.py
"""
PathEncoder: encode a path (sequence of node ids and optional relation labels)
into a fixed-size latent vector.

Features:
  - node embedding table (learnable)
  - optional relation embedding pooling
  - configurable pooling strategies: mean, max, attention
  - returns torch.Tensor latent vector for each path

This is a modular, trainable encoder suitable for contrastive pretraining.
"""

from typing import Iterable, List, Any, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PathEncoder(nn.Module):
    """
    PathEncoder with node embeddings + optional relation embeddings.
    """

    def __init__(self, vocab_size: int = 10000, embed_dim: int = 128, max_path_len: int = 10, use_relations: bool = True, rel_vocab_size: int = 256, pooling: str = "mean"):
        """
        Args:
          vocab_size: number of nodes in the node embedding table (use large or map nodes to indices)
          embed_dim: embedding dimension for node embeddings (and relation embeddings)
          max_path_len: maximum supported path length (for positional enc)
          use_relations: whether to embed relation labels
          rel_vocab_size: number of relation types
          pooling: 'mean', 'max', or 'attention'
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_path_len = max_path_len
        self.use_relations = use_relations
        self.pooling = pooling

        self.node_embed = nn.Embedding(vocab_size, embed_dim)
        if use_relations:
            self.rel_embed = nn.Embedding(rel_vocab_size, embed_dim)
        else:
            self.rel_embed = None

        # optional small transformer-style self-attention for path composition
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True) if pooling == "attention" else None

        # positional encodings for path positions (sinusoidal)
        pe = torch.zeros(max_path_len, embed_dim)
        position = torch.arange(0, max_path_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_enc", pe)  # [max_path_len, embed_dim]

        # small MLP to map pooled embedding to final latent
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, paths: List[List[Any]], node_to_idx: Optional[Dict[Any, int]] = None, rels: Optional[List[List[Any]]] = None) -> torch.Tensor:
        """
        Encode a batch of paths.
        Args:
          paths: list of paths; each path is a list of node identifiers (or integers if node_to_idx is None)
          node_to_idx: optional mapping from node id to integer index in embedding table
          rels: optional list of relation sequences aligned with paths (each relation sequence has length len(path)-1)
        Returns:
          latents: torch.Tensor [batch, embed_dim]
        """
        device = next(self.parameters()).device
        batch = len(paths)
        max_len = max(len(p) for p in paths)
        max_len = min(max_len, self.max_path_len)

        # build index tensor with padding index 0
        idxs = torch.zeros(batch, max_len, dtype=torch.long, device=device)
        rel_idxs = None
        if self.use_relations and rels is not None:
            rel_idxs = torch.zeros(batch, max_len - 1, dtype=torch.long, device=device)

        for i, p in enumerate(paths):
            trunc = p[:max_len]
            for j, node in enumerate(trunc):
                if node_to_idx is None:
                    if isinstance(node, int):
                        idxs[i, j] = node
                    else:
                        # fallback hash mapping: use python hash to produce index (deterministic but may collide)
                        idxs[i, j] = abs(hash(node)) % self.vocab_size
                else:
                    idxs[i, j] = node_to_idx.get(node, 0)
            if self.use_relations and rels is not None:
                rseq = rels[i][:max_len - 1]
                for j, r in enumerate(rseq):
                    if isinstance(r, int):
                        rel_idxs[i, j] = r
                    else:
                        rel_idxs[i, j] = abs(hash(r)) % (self.rel_embed.num_embeddings)

        # node embeddings
        node_emb = self.node_embed(idxs)  # [B, L, D]
        # add positional enc
        pos = self.pos_enc[:node_emb.size(1), :].unsqueeze(0)  # [1, L, D]
        node_emb = node_emb + pos

        if self.use_relations and rels is not None and self.rel_embed is not None:
            # expand relation embeddings to node positions by interleaving or summing
            # simple strategy: for node at position j (except first), add rel_emb[j-1]
            rel_emb = self.rel_embed(rel_idxs)  # [B, L-1, D]
            # pad one zero at front then add
            rel_pad = torch.zeros(batch, 1, self.embed_dim, device=device)
            rel_full = torch.cat([rel_pad, rel_emb], dim=1)  # [B, L, D]
            node_emb = node_emb + rel_full

        # pooling
        if self.pooling == "mean":
            mask = (idxs != 0).float().unsqueeze(-1)
            summed = (node_emb * mask).sum(dim=1)  # [B, D]
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / denom
        elif self.pooling == "max":
            # set padding positions to large negative to ignore
            mask = (idxs != 0).unsqueeze(-1)
            large_neg = -1e9
            emb_masked = node_emb.masked_fill(~mask, large_neg)
            pooled, _ = emb_masked.max(dim=1)
            # replace -1e9 when all were masked
            pooled = pooled.where((pooled > large_neg/2).any(dim=1, keepdim=True), torch.zeros_like(pooled))
        elif self.pooling == "attention":
            # self-attention across tokens
            attn_out, _ = self.attn(node_emb, node_emb, node_emb)  # [B, L, D]
            # mean pool attn outputs (or use cls token)
            pooled = attn_out.mean(dim=1)
        else:
            raise ValueError("unknown pooling type")

        latents = self.proj(pooled)  # [B, D]
        return latents


if __name__ == "__main__":
    # demo
    enc = PathEncoder(vocab_size=5000, embed_dim=64, pooling="mean")
    paths = [["A","B","C"], ["X","Y"]]
    lat = enc(paths)
    print("latent shape:", lat.shape)
