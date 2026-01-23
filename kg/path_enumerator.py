# src/kg/path_enumerator.py
"""
PathEnumerator: enumerates candidate paths in a KG.
Supports multiple strategies:
  - k_shortest (Yen's algorithm via networkx.shortest_simple_paths with weights)
  - beam_search (heuristic BFS with beam width using node scoring)
  - random_walks (stochastic sampling with restart)
This module returns lists of node-id sequences (paths).
"""

from typing import List, Iterable, Any, Optional
import networkx as nx
import random
import heapq
import math


class PathEnumerator:
    """
    PathEnumerator orchestrates different path search strategies.
    """

    def __init__(self, graph: Optional[nx.Graph] = None):
        """
        graph: optional networkx graph to operate on. If not provided,
               the enumerator methods accept a graph argument per-call.
        """
        self.graph = graph

    # ----------------- k-shortest via networkx -----------------
    def enumerate_k_shortest(self, graph: Optional[nx.Graph], source: Any, target: Any, k: int = 5, weight: Optional[str] = None) -> List[List[Any]]:
        """
        Use networkx.shortest_simple_paths to generate k shortest simple paths.
        Returns up to k paths (each path is a list of node ids).
        """
        if graph is None:
            graph = self.graph
        if graph is None:
            raise ValueError("graph must be provided")

        paths = []
        try:
            gen = nx.shortest_simple_paths(graph, source, target, weight=weight)
            for i, p in enumerate(gen):
                if i >= k:
                    break
                paths.append(p)
        except nx.NetworkXNoPath:
            return []
        return paths

    # ----------------- beam search -----------------
    def enumerate_beam(self, graph: Optional[nx.Graph], source: Any, target: Any, beam_width: int = 4, max_steps: int = 10, score_fn=None) -> List[List[Any]]:
        """
        Beam search from source towards target.
        score_fn(node, path) -> score scalar for prioritization (higher better).
        If score_fn is None, prefer shorter paths (negative length).
        Returns unique paths that reach target.
        """
        if graph is None:
            graph = self.graph
        if graph is None:
            raise ValueError("graph must be provided")

        if score_fn is None:
            def score_fn(node, path):
                return -len(path)

        # each beam entry: (score, path)
        beam = [ (score_fn(source, [source]), [source]) ]
        completed = []
        for step in range(max_steps):
            candidates = []
            for score, path in beam:
                last = path[-1]
                for nbr in graph.successors(last) if graph.is_directed() else graph.neighbors(last):
                    if nbr in path:  # avoid cycles
                        continue
                    new_path = path + [nbr]
                    s = score_fn(nbr, new_path)
                    candidates.append( (s, new_path) )
            if not candidates:
                break
            # keep top beam_width candidates
            candidates.sort(key=lambda x: x[0], reverse=True)
            beam = candidates[:beam_width]
            # check for target in beam
            for s, p in beam:
                if p[-1] == target:
                    completed.append(p)
            # optionally stop early if found some
            if completed:
                break
        # deduplicate completed paths while preserving order
        seen = set()
        uniq = []
        for p in completed:
            tup = tuple(p)
            if tup not in seen:
                uniq.append(p)
                seen.add(tup)
        return uniq

    # ----------------- random walk enumerator -----------------
    def sample_random_walks(self, graph: Optional[nx.Graph], start_nodes: Iterable[Any], num_walks: int = 10, walk_length: int = 5, restart_prob: float = 0.1) -> List[List[Any]]:
        """
        Sample random walks starting from any of the start_nodes.
        Returns a list of node sequences.
        """
        if graph is None:
            graph = self.graph
        if graph is None:
            raise ValueError("graph must be provided")

        walks = []
        nodes = list(start_nodes)
        if not nodes:
            return walks
        for _ in range(num_walks):
            cur = random.choice(nodes)
            path = [cur]
            for _ in range(walk_length - 1):
                # possible neighbors
                nbrs = list(graph.successors(cur) if graph.is_directed() else graph.neighbors(cur))
                if not nbrs:
                    break
                if random.random() < restart_prob:
                    cur = random.choice(nodes)
                    path.append(cur)
                    continue
                cur = random.choice(nbrs)
                path.append(cur)
            walks.append(path)
        return walks

    # ----------------- unified enumerate API -----------------
    def enumerate(self, graph: Optional[nx.Graph] = None, seeds: Optional[Iterable[Any]] = None, source: Optional[Any] = None, target: Optional[Any] = None, max_paths: int = 10, method: str = "k_shortest", **kwargs) -> List[List[Any]]:
        """
        Unified enumeration API.
        - If method == 'k_shortest', requires source and target.
        - If method == 'beam', requires source and target or will use seed->any endpoint.
        - If method == 'random_walk', uses seeds as start nodes.
        Additional kwargs are forwarded to the underlying method.
        """
        graph = graph or self.graph
        if graph is None:
            raise ValueError("graph must be provided")

        method = method.lower()
        if method in ("k_shortest", "yen", "shortest"):
            if source is None or target is None:
                raise ValueError("k_shortest requires source and target")
            return self.enumerate_k_shortest(graph, source, target, k=max_paths, weight=kwargs.get("weight"))
        elif method in ("beam", "beam_search"):
            if source is None or target is None:
                # try to pick a random source from seeds
                if seeds is not None:
                    source = next(iter(seeds))
                else:
                    raise ValueError("beam search requires source and target or seeds")
            return self.enumerate_beam(graph, source, target, beam_width=kwargs.get("beam_width", 4), max_steps=kwargs.get("max_steps", 10), score_fn=kwargs.get("score_fn"))
        elif method in ("random_walk", "rw"):
            start_nodes = seeds or ([source] if source is not None else [])
            return self.sample_random_walks(graph, start_nodes, num_walks=max_paths, walk_length=kwargs.get("walk_length", 6), restart_prob=kwargs.get("restart_prob", 0.1))
        else:
            raise ValueError(f"Unknown enumeration method: {method}")


if __name__ == "__main__":
    # small demo
    g = nx.DiGraph()
    edges = [("A","B"), ("B","C"), ("A","C"), ("C","D"), ("B","D"), ("A","D")]
    g.add_edges_from(edges)
    pe = PathEnumerator(g)
    print("k_shortest A->D:", pe.enumerate(graph=g, source="A", target="D", max_paths=3, method="k_shortest"))
    print("beam A->D:", pe.enumerate(graph=g, source="A", target="D", max_paths=3, method="beam", beam_width=3))
    print("random walks from A,B:", pe.enumerate(graph=g, seeds=["A","B"], max_paths=5, method="random_walk"))
