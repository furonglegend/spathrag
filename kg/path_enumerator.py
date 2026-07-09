# src/kg/path_enumerator.py
"""
PathEnumerator: enumerates candidate paths in a KG.
Supports multiple strategies:
  - k_shortest (Yen's algorithm via networkx.shortest_simple_paths with weights)
  - beam_search (heuristic BFS with beam width using node scoring)
  - random_walks (stochastic sampling with restart)
This module returns lists of node-id sequences (paths).
"""

from typing import Callable, Dict, List, Iterable, Any, Optional, Tuple
import networkx as nx
import random
import re


RELATION_ALIASES = {
    "directed_by": {"direct", "directed", "director", "filmmaker"},
    "starred_actors": {"actor", "actors", "acted", "starred", "cast", "appeared", "performed"},
    "born_in": {"born", "birth", "birthplace", "country", "nationality"},
    "genre": {"genre", "type", "category"},
    "spouse": {"spouse", "wife", "husband", "married"},
    "produced": {"produce", "produced", "producer"},
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "of", "on", "or", "the", "to", "was", "were", "what",
    "which", "who", "whom", "whose",
}


def _query_terms(query: Optional[str]) -> set:
    if not query:
        return set()
    return {
        token.rstrip("s")
        for token in re.findall(r"[a-z0-9]+", query.lower().replace("_", " "))
        if len(token) > 1 and token not in STOPWORDS
    }


def _relation_terms(relation: Optional[str]) -> set:
    if not relation:
        return set()
    relation_text = str(relation).lower()
    terms = {
        token.rstrip("s")
        for token in re.findall(r"[a-z0-9]+", relation_text.replace("_", " "))
        if len(token) > 1 and token not in STOPWORDS
    }
    terms.update(RELATION_ALIASES.get(relation_text, set()))
    return {token.rstrip("s") for token in terms}


def _label_terms(label: Any) -> set:
    return _relation_terms(str(label))


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
    def enumerate_k_shortest(self, graph: Optional[nx.Graph], source: Any, target: Any, k: int = 5, weight: Optional[Any] = None) -> List[List[Any]]:
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
        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError):
            return []
        return paths

    # ----------------- beam search -----------------
    def enumerate_beam(self, graph: Optional[nx.Graph], source: Any, target: Optional[Any] = None, beam_width: int = 4, max_steps: int = 10, score_fn=None) -> List[List[Any]]:
        """
        Beam search from source towards target. If target is None, return the
        best partial paths reached during expansion.
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
        if source not in graph:
            return []

        beam = [(score_fn(source, [source]), [source])]
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
            # check for target in beam, or keep useful partial paths when no
            # explicit target is known at inference time.
            for s, p in beam:
                if target is None and len(p) > 1:
                    completed.append(p)
                elif p[-1] == target:
                    completed.append(p)
            # optionally stop early if found some
            if target is not None and completed:
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
    def sample_random_walks(self, graph: Optional[nx.Graph], start_nodes: Iterable[Any], num_walks: int = 10, walk_length: int = 5, restart_prob: float = 0.1, random_seed: int = 13) -> List[List[Any]]:
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
        rng = random.Random(random_seed)
        for _ in range(num_walks):
            cur = rng.choice(nodes)
            path = [cur]
            for _ in range(walk_length - 1):
                # possible neighbors
                nbrs = list(graph.successors(cur) if graph.is_directed() else graph.neighbors(cur))
                if not nbrs:
                    break
                if rng.random() < restart_prob:
                    cur = rng.choice(nodes)
                    path.append(cur)
                    continue
                cur = rng.choice(nbrs)
                path.append(cur)
            walks.append(path)
        return walks

    # ----------------- semantic weighted hybrid search -----------------
    def edge_weight_fn(self, query: Optional[str] = None) -> Callable[[Any, Any, Dict[str, Any]], float]:
        """
        Build a NetworkX-compatible edge weight function. Relations whose
        words overlap the question receive lower cost, matching the paper's
        semantic-aware shortest-path objective in a lightweight way.
        """
        terms = _query_terms(query)

        def weight(_u: Any, _v: Any, data: Dict[str, Any]) -> float:
            base = float(data.get("weight", 1.0))
            rel_terms = _relation_terms(data.get("relation"))
            overlap = len(terms & rel_terms)
            if overlap:
                base -= min(0.75, 0.35 * overlap)
            return max(0.05, base)

        return weight

    def path_relations(self, graph: nx.Graph, path: List[Any]) -> List[str]:
        relations = []
        for src, dst in zip(path, path[1:]):
            data = graph.get_edge_data(src, dst, default={}) or {}
            relations.append(str(data.get("relation", "")))
        return relations

    def score_path(self, graph: nx.Graph, path: List[Any], query: Optional[str] = None) -> float:
        """
        Higher scores are better. The heuristic rewards relation/query overlap,
        answer-label overlap, and short paths. It is deterministic and small
        enough for synthetic/local KGQA experiments.
        """
        if len(path) < 2:
            return 0.0
        terms = _query_terms(query)
        relations = self.path_relations(graph, path)
        rel_overlap = sum(len(terms & _relation_terms(rel)) for rel in relations)
        end_overlap = len(terms & _label_terms(path[-1]))
        relation_prior = sum(float((graph.get_edge_data(u, v, default={}) or {}).get("prior", 0.0)) for u, v in zip(path, path[1:]))
        length_penalty = 1.0 / max(1, len(path) - 1)
        return (2.0 * rel_overlap) + (0.25 * end_overlap) + relation_prior + length_penalty

    def _reachable_targets(self, graph: nx.Graph, seed: Any, query: Optional[str], max_length: int, limit: int) -> List[Any]:
        if seed not in graph:
            return []
        try:
            lengths = nx.single_source_shortest_path_length(graph, seed, cutoff=max_length)
        except nx.NetworkXError:
            return []
        targets = [node for node, dist in lengths.items() if node != seed and dist > 0]

        def target_score(node: Any) -> Tuple[float, str]:
            try:
                paths = self.enumerate_k_shortest(graph, seed, node, k=1, weight=self.edge_weight_fn(query))
            except Exception:
                paths = []
            score = self.score_path(graph, paths[0], query) if paths else 0.0
            return (-score, str(node))

        return sorted(targets, key=target_score)[:limit]

    def enumerate_hybrid(
        self,
        graph: Optional[nx.Graph],
        seeds: Iterable[Any],
        query: Optional[str] = None,
        max_paths: int = 10,
        max_length: int = 4,
        beam_width: int = 4,
        random_walks: int = 8,
        random_seed: int = 13,
    ) -> List[List[Any]]:
        """
        Hybrid candidate generator inspired by the paper:
          1. semantic weighted k-shortest paths to reachable candidates,
          2. semantic beam expansion when the answer target is unknown,
          3. constrained random walks for diversity.
        """
        graph = graph or self.graph
        if graph is None:
            raise ValueError("graph must be provided")

        seed_list = [seed for seed in (seeds or []) if seed in graph]
        if not seed_list:
            return []

        candidates: List[List[Any]] = []
        target_limit = max(max_paths * 3, 12)
        weight = self.edge_weight_fn(query)

        for seed in seed_list:
            targets = self._reachable_targets(graph, seed, query, max_length=max_length, limit=target_limit)
            for target in targets:
                candidates.extend(self.enumerate_k_shortest(graph, seed, target, k=2, weight=weight))

            def beam_score(node, path):
                return self.score_path(graph, path, query)

            candidates.extend(
                self.enumerate_beam(
                    graph,
                    seed,
                    target=None,
                    beam_width=beam_width,
                    max_steps=max_length,
                    score_fn=beam_score,
                )
            )

        candidates.extend(
            self.sample_random_walks(
                graph,
                seed_list,
                num_walks=random_walks,
                walk_length=max_length + 1,
                restart_prob=0.15,
                random_seed=random_seed,
            )
        )

        dedup: Dict[Tuple[Any, ...], List[Any]] = {}
        for path in candidates:
            if len(path) < 2 or len(path) > max_length + 1:
                continue
            if len(set(path)) != len(path):
                continue
            dedup.setdefault(tuple(path), path)

        ranked = sorted(
            dedup.values(),
            key=lambda p: (self.score_path(graph, p, query), -len(p), " ".join(map(str, p))),
            reverse=True,
        )
        return ranked[:max_paths]

    # ----------------- unified enumerate API -----------------
    def enumerate(self, graph: Optional[nx.Graph] = None, seeds: Optional[Iterable[Any]] = None, source: Optional[Any] = None, target: Optional[Any] = None, max_paths: int = 10, method: str = "k_shortest", query: Optional[str] = None, **kwargs) -> List[List[Any]]:
        """
        Unified enumeration API.
        - If method == 'k_shortest', requires source and target.
        - If method == 'beam', requires source and target or will use seed->any endpoint.
        - If method == 'random_walk', uses seeds as start nodes.
        - If method == 'hybrid', uses semantic weighted k-shortest, beam,
          and random-walk proposals from seeds.
        Additional kwargs are forwarded to the underlying method.
        """
        graph = graph or self.graph
        if graph is None:
            raise ValueError("graph must be provided")

        method = method.lower()
        if method in ("hybrid", "s_path", "spath", "semantic"):
            start_nodes = seeds or ([source] if source is not None else [])
            return self.enumerate_hybrid(
                graph=graph,
                seeds=start_nodes,
                query=query,
                max_paths=max_paths,
                max_length=kwargs.get("max_length", kwargs.get("walk_length", 4)),
                beam_width=kwargs.get("beam_width", 4),
                random_walks=kwargs.get("random_walks", max_paths),
                random_seed=kwargs.get("random_seed", 13),
            )
        elif method in ("k_shortest", "yen", "shortest"):
            if source is None or target is None:
                raise ValueError("k_shortest requires source and target")
            return self.enumerate_k_shortest(graph, source, target, k=max_paths, weight=kwargs.get("weight", self.edge_weight_fn(query) if query else None))
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
            return self.sample_random_walks(graph, start_nodes, num_walks=max_paths, walk_length=kwargs.get("walk_length", 6), restart_prob=kwargs.get("restart_prob", 0.1), random_seed=kwargs.get("random_seed", 13))
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
