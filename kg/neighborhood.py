# src/kg/neighborhood.py
"""
Neighborhood utilities for KG.
Provides:
  - expand_neighborhood: expand a set of seed nodes within a KG by hop count and optional filtering
  - knn_seed_expansion: expand seeds by nearest neighbors using node embeddings (cosine similarity)
  - degree_based_expansion: pick high-degree neighbors to prioritize hubs
"""

from typing import Iterable, List, Set, Optional, Dict, Any, Tuple
import math
import numpy as np

try:
    import torch
except Exception:
    torch = None


def expand_neighborhood(kg, seeds: Iterable[Any], hops: int = 1, direction: str = "both", max_nodes: Optional[int] = None) -> Set[Any]:
    """
    Breadth-first expansion from seeds for given number of hops.
    kg: KGStore or networkx.Graph-like object with successors/predecessors
    seeds: iterable of seed node IDs
    hops: number of hops to expand
    direction: 'in', 'out', 'both'
    max_nodes: optional cap on total nodes to return
    Returns a set of node IDs in the expanded neighborhood (including seeds).
    """
    frontier = set(seeds)
    visited = set(seeds)
    for _ in range(max(0, hops)):
        next_frontier = set()
        for node in frontier:
            try:
                if direction in ("out", "both"):
                    next_frontier.update(kg._graph.successors(node) if hasattr(kg, "_graph") else kg.successors(node))
                if direction in ("in", "both"):
                    next_frontier.update(kg._graph.predecessors(node) if hasattr(kg, "_graph") else kg.predecessors(node))
            except Exception:
                # fallback to key-based neighbor queries
                try:
                    next_frontier.update(kg.neighbors(node, direction=direction))
                except Exception:
                    continue
        next_frontier = next_frontier - visited
        visited |= next_frontier
        frontier = next_frontier
        if max_nodes is not None and len(visited) >= max_nodes:
            break
    if max_nodes is not None and len(visited) > max_nodes:
        # truncate deterministically (convert to list and slice)
        visited = set(list(visited)[:max_nodes])
    return visited


def knn_seed_expansion(kg, seed_embeddings: Dict[Any, Any], top_k: int = 5, exclude: Optional[Iterable[Any]] = None) -> List[Tuple[Any, float]]:
    """
    Expand by nearest neighbors in embedding space.
    seed_embeddings: dict of node -> embedding vector (numpy or torch tensors)
    top_k: number of nearest neighbors to return
    exclude: optional iterable of nodes to exclude from results
    Returns a list of (node_id, similarity) sorted by descending similarity.
    """
    exclude_set = set(exclude) if exclude is not None else set()
    nodes = list(seed_embeddings.keys())
    if not nodes:
        return []
    # convert to numpy matrix
    first = seed_embeddings[nodes[0]]
    if torch is not None and (hasattr(first, "detach") or torch.is_tensor(first)):
        # convert tensors to numpy
        mat = np.stack([ (v.detach().cpu().numpy() if hasattr(v, "detach") else np.array(v)) for v in [seed_embeddings[n] for n in nodes] ])
    else:
        mat = np.stack([ np.array(seed_embeddings[n]) for n in nodes ])
    # normalize
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    mat_norm = mat / norms
    # compute similarity to centroid of seeds
    centroid = mat_norm.mean(axis=0, keepdims=True)  # [1, D]
    sims = (mat_norm @ centroid.T).squeeze(-1)  # [N]
    # sort nodes by similarity
    idx = np.argsort(-sims)
    results = []
    for i in idx:
        name = nodes[i]
        if name in exclude_set:
            continue
        results.append((name, float(sims[i])))
        if len(results) >= top_k:
            break
    return results


def degree_based_expansion(kg, seeds: Iterable[Any], top_k: int = 10, direction: str = "both") -> List[Any]:
    """
    Expand seeds by selecting top_k neighbors with the highest degree (hub nodes).
    Returns a list of node IDs.
    """
    candidates = set()
    for s in seeds:
        try:
            if direction in ("out", "both"):
                candidates.update(kg._graph.successors(s) if hasattr(kg, "_graph") else kg.neighbors(s, direction="out"))
            if direction in ("in", "both"):
                candidates.update(kg._graph.predecessors(s) if hasattr(kg, "_graph") else kg.neighbors(s, direction="in"))
        except Exception:
            try:
                candidates.update(kg.neighbors(s, direction=direction))
            except Exception:
                continue

    # compute degrees
    degs = []
    for n in candidates:
        try:
            deg = kg._graph.degree(n) if hasattr(kg, "_graph") else len(kg.neighbors(n))
        except Exception:
            deg = 0
        degs.append((n, deg))
    degs_sorted = sorted(degs, key=lambda x: x[1], reverse=True)
    return [n for n, _ in degs_sorted[:top_k]]


if __name__ == "__main__":
    # small demo
    from src.kg.kg_store import KGStore
    kg = KGStore()
    kg.add_edge("A", "B", relation="r1")
    kg.add_edge("B", "C", relation="r2")
    kg.add_edge("A", "D", relation="r3")
    print("expand from A hops=1:", expand_neighborhood(kg, ["A"], hops=1))
    # embeddings demo
    emb = {"A": [1.0, 0.0], "B": [0.9, 0.1], "C": [0.0, 1.0], "D": [0.95,0.05]}
    print("knn expansion centroid:", knn_seed_expansion(kg, emb, top_k=2, exclude=["A"]))
    print("degree expansion:", degree_based_expansion(kg, ["A"], top_k=2))
