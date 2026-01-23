# src/llm_integration/injection.py
"""
Projection utilities for path latents -> injection artifacts.

This module provides two complementary projection functions:
  1) project_path_latents_to_kv: produce (k, v) tensors per attention head/layer
     intended for advanced cross-attention injection into transformer internals.
     Output format is a dict: {"k": tensor, "v": tensor} with shapes:
       k, v: [num_layers, batch, num_heads, prefix_len, head_dim]
     This is a general-format helper; integrating it into a specific model requires
     adapting to the model's attention internals.

  2) project_path_latents_to_prefix_embeddings: produce dense embeddings that can be
     prepended to the token embedding sequence and passed to HuggingFace generate() via
     inputs_embeds. This approach works for causal LMs and is portable.

Both functions accept latents of shape [batch, num_paths, latent_dim] or [num_paths, latent_dim].
"""

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import math


def _ensure_batch_latents(latents: torch.Tensor) -> torch.Tensor:
    """
    Accept latents in shapes [num_paths, latent_dim], [batch, num_paths, latent_dim], or [latent_dim]
    and return [batch, num_paths, latent_dim].
    """
    if latents.dim() == 1:
        # single latent vector -> batch=1, num_paths=1
        return latents.unsqueeze(0).unsqueeze(0)
    if latents.dim() == 2:
        # [num_paths, latent_dim] -> batch=1
        return latents.unsqueeze(0)
    if latents.dim() == 3:
        return latents
    raise ValueError(f"Unsupported latent shape: {latents.shape}")


def project_path_latents_to_kv(
    latents: torch.Tensor,
    num_layers: int = 6,
    num_heads: int = 8,
    head_dim: int = 64,
    prefix_len_per_path: int = 1,
    device: Optional[torch.device] = None,
    projection_bias: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Project path latents to key/value tensors for cross-attention injection.

    Args:
      latents: [batch, num_paths, latent_dim] or [num_paths, latent_dim]
      num_layers, num_heads, head_dim: target attention configuration
      prefix_len_per_path: how many prefix tokens to reserve per path (=> prefix_len = num_paths * prefix_len_per_path)
      device: optional target device
    Returns:
      {"k": k_tensor, "v": v_tensor}
      k/v shapes: [num_layers, batch, num_heads, prefix_len, head_dim]
    """
    latents = _ensure_batch_latents(latents)
    b, num_paths, latent_dim = latents.shape
    device = device or latents.device

    prefix_len = num_paths * prefix_len_per_path
    total_dim = num_heads * head_dim

    # We'll create a small projection MLP that maps latent_dim -> total_dim*2 (for k and v)
    # For simplicity we create one linear layer per layer (no parameter sharing).
    # Because this function is stateless, we use a deterministic linear projection via a fixed matrix
    # seeded by latent_dim and target dims. In a trainable setting, replace this by an nn.Module.
    # Create projection matrices deterministically (but random-like) using torch.randn with fixed seed.
    rng = torch.Generator(device=device)
    rng.manual_seed((latent_dim + total_dim + num_layers) & 0xFFFFFFFF)

    # Build projection weights: shape [num_layers, latent_dim, total_dim*2]
    weight = torch.randn(num_layers, latent_dim, total_dim * 2, generator=rng, device=device) / math.sqrt(latent_dim)
    bias = torch.randn(num_layers, total_dim * 2, generator=rng, device=device) * 0.01 if projection_bias else torch.zeros(num_layers, total_dim * 2, device=device)

    # Compute per-layer projections
    # latents_flat: [b * num_paths, latent_dim]
    latents_flat = latents.view(b * num_paths, latent_dim)  # [B*num_paths, D]
    # result per layer -> list length num_layers of [B*num_paths, total_dim*2]
    proj_per_layer = []
    for l in range(num_layers):
        w = weight[l]  # [latent_dim, total_dim*2]
        bvec = bias[l]  # [total_dim*2]
        out = latents_flat @ w + bvec  # [B*num_paths, total_dim*2]
        proj_per_layer.append(out)

    # Stack and reshape -> [num_layers, b, num_paths, total_dim*2]
    stacked = torch.stack(proj_per_layer, dim=0).view(num_layers, b, num_paths, total_dim * 2)

    # Now split into k and v and reshape into heads/prefix_len
    # First, we may want prefix_len_per_path >1, so for each path produce that many tokens by simple linear upsampling.
    # For simplicity, we repeat each path projection prefix_len_per_path times.
    stacked = stacked.unsqueeze(3)  # [num_layers, b, num_paths, 1, total_dim*2]
    stacked = stacked.repeat(1, 1, 1, prefix_len_per_path, 1)  # [num_layers, b, num_paths, prefix_len_per_path, total_dim*2]
    stacked = stacked.view(num_layers, b, prefix_len, total_dim * 2)  # [num_layers, b, prefix_len, total_dim*2]

    k_all = stacked[..., :total_dim]  # [num_layers, b, prefix_len, total_dim]
    v_all = stacked[..., total_dim:]  # [num_layers, b, prefix_len, total_dim]

    # reshape to [num_layers, b, num_heads, prefix_len, head_dim]
    k = k_all.view(num_layers, b, num_heads, prefix_len, head_dim).contiguous()
    v = v_all.view(num_layers, b, num_heads, prefix_len, head_dim).contiguous()

    return {"k": k.to(device), "v": v.to(device)}


def project_path_latents_to_prefix_embeddings(
    latents: torch.Tensor,
    embed_dim: int,
    num_prefix_tokens_per_path: int = 1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Project path latents into prefix token embeddings that can be prepended to model input embeddings.

    Args:
      latents: [batch, num_paths, latent_dim] or [num_paths, latent_dim]
      embed_dim: target embedding dimension matching model.get_input_embeddings().embedding_dim
      num_prefix_tokens_per_path: how many tokens to reserve per path
    Returns:
      prefix_embeddings: [batch, prefix_len, embed_dim] where prefix_len = num_paths * num_prefix_tokens_per_path
    Notes:
      - This function is stateless and uses a deterministic projection. Replace with a trainable nn.Module
        for end-to-end learning (e.g., a small MLP).
    """
    latents = _ensure_batch_latents(latents)
    b, num_paths, latent_dim = latents.shape
    device = device or latents.device
    prefix_len = num_paths * num_prefix_tokens_per_path

    # simple deterministic projection using a fixed linear map seeded by dims
    rng = torch.Generator(device=device)
    rng.manual_seed((latent_dim + embed_dim + num_paths) & 0xFFFFFFFF)

    weight = torch.randn(latent_dim, embed_dim, generator=rng, device=device) / max(1.0, latent_dim ** 0.5)
    bias = torch.randn(embed_dim, generator=rng, device=device) * 0.01

    # project each path latent to embed_dim
    proj = latents @ weight + bias  # [b, num_paths, embed_dim]

    # if num_prefix_tokens_per_path > 1 we tile and optionally apply small positional modulation
    if num_prefix_tokens_per_path == 1:
        prefix = proj  # [b, num_paths, embed_dim]
    else:
        # repeat each path vector num_prefix_tokens_per_path times and add tiny sinusoidal offsets
        prefix = proj.unsqueeze(2).repeat(1, 1, num_prefix_tokens_per_path, 1)  # [b, num_paths, k, embed_dim]
        prefix = prefix.view(b, prefix_len, embed_dim)

        # positional modulation
        pos = torch.arange(prefix_len, device=device).float().unsqueeze(-1)  # [prefix_len,1]
        pos = (pos / max(1.0, prefix_len))  # normalized
        prefix = prefix + 0.01 * torch.sin(pos)

    if prefix.dim() == 3:
        # ensure final shape [b, prefix_len, embed_dim]
        if prefix.shape[1] != prefix_len:
            prefix = prefix.view(b, prefix_len, embed_dim)
    else:
        prefix = prefix.view(b, prefix_len, embed_dim)

    return prefix.to(device)


# quick dry run when executed directly
if __name__ == "__main__":
    import torch
    lat = torch.randn(2, 4, 128)  # batch=2, num_paths=4, latent_dim=128
    kv = project_path_latents_to_kv(lat, num_layers=4, num_heads=4, head_dim=16, prefix_len_per_path=1)
    print("k shape", kv["k"].shape, "v shape", kv["v"].shape)
    pref = project_path_latents_to_prefix_embeddings(lat, embed_dim=512, num_prefix_tokens_per_path=2)
    print("prefix shape", pref.shape)
