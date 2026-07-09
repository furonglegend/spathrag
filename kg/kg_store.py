# src/kg/kg_store.py
"""
KGStore: lightweight Knowledge Graph store backed by networkx.
This class provides basic KG operations:
  - load/save triples from/to TSV/CSV
  - add/remove edges
  - neighbor queries
  - return subgraphs
  - optional attachment of node / relation embeddings

This is intentionally dependency-light (only networkx, csv, torch optional).
It is meant as a local, in-memory KG store for development and testing.
"""

from typing import Iterable, Tuple, Optional, Dict, Any, List
import networkx as nx
import csv
import os

try:
    import torch
except Exception:
    torch = None


class KGStore:
    """
    Simple directed KG store.
    Nodes are arbitrary hashable objects (strings recommended).
    Edges have optional 'relation' attribute to store relation types.
    """

    def __init__(self, directed: bool = True):
        self._directed = directed
        self._graph = nx.DiGraph() if directed else nx.Graph()
        # optional embeddings: dict[node_id] -> numpy/torch vector
        self.node_embeddings: Dict[Any, Any] = {}
        # optional relation embeddings: dict[rel] -> vector
        self.relation_embeddings: Dict[Any, Any] = {}

    # ---------- I/O ----------
    def load_triples(self, path: str, delimiter: str = "\t", header: bool = False):
        """
        Load triples file with columns (subject, relation, object).
        Supports TSV or CSV depending on delimiter.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter=delimiter)
            if header:
                next(reader, None)
            for row in reader:
                if len(row) < 3:
                    continue
                s, r, o = row[0].strip(), row[1].strip(), row[2].strip()
                self.add_edge(s, o, relation=r)

    def save_triples(self, path: str, delimiter: str = "\t"):
        """
        Save current edges to a triples file (subject, relation, object).
        """
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, delimiter=delimiter)
            for u, v, data in self._graph.edges(data=True):
                rel = data.get("relation", "")
                writer.writerow([u, rel, v])

    # ---------- Basic graph ops ----------
    def add_node(self, node_id: Any, **attrs):
        """Add a node with optional attributes."""
        self._graph.add_node(node_id, **attrs)

    def add_edge(self, src: Any, dst: Any, relation: Optional[str] = None, **attrs):
        """Add a directed edge with optional relation type and attributes."""
        if relation is not None:
            attrs = dict(attrs)
            attrs["relation"] = relation
        self._graph.add_edge(src, dst, **attrs)

    def remove_edge(self, src: Any, dst: Any):
        """Remove an edge if it exists."""
        if self._graph.has_edge(src, dst):
            self._graph.remove_edge(src, dst)

    def has_edge(self, src: Any, dst: Any) -> bool:
        return self._graph.has_edge(src, dst)

    def neighbors(self, node_id: Any, direction: str = "out") -> List[Any]:
        """
        Return neighbors of a node.
        direction: 'out' for successors, 'in' for predecessors, 'both' for all neighbors.
        """
        if direction == "out":
            return list(self._graph.successors(node_id))
        elif direction == "in":
            return list(self._graph.predecessors(node_id))
        elif direction == "both":
            return list(set(self._graph.successors(node_id)) | set(self._graph.predecessors(node_id)))
        else:
            raise ValueError("direction must be 'out', 'in', or 'both'")

    def get_edge_relation(self, src: Any, dst: Any) -> Optional[str]:
        data = self._graph.get_edge_data(src, dst, default=None)
        if data:
            return data.get("relation")
        return None

    def nodes(self) -> Iterable[Any]:
        return self._graph.nodes()

    def edges(self) -> Iterable[Tuple[Any, Any]]:
        return self._graph.edges()

    def number_of_nodes(self) -> int:
        return self._graph.number_of_nodes()

    def number_of_edges(self) -> int:
        return self._graph.number_of_edges()

    def subgraph(self, nodes: Iterable[Any]):
        """
        Return an induced subgraph containing the provided nodes.
        """
        return self._graph.subgraph(nodes).copy()

    def to_networkx(self) -> nx.Graph:
        """Return the underlying networkx graph (a copy)."""
        return self._graph.copy()

    # ---------- Embedding helpers ----------
    def set_node_embedding(self, node_id: Any, vector):
        """Attach an embedding to a node (numpy or torch tensor)."""
        self.node_embeddings[node_id] = vector

    def get_node_embedding(self, node_id: Any):
        return self.node_embeddings.get(node_id)

    def set_relation_embedding(self, rel: Any, vector):
        self.relation_embeddings[rel] = vector

    def get_relation_embedding(self, rel: Any):
        return self.relation_embeddings.get(rel)

    # ---------- Utility ----------
    def sample_random_node(self):
        """Return a random node from the graph."""
        import random
        nodes = list(self._graph.nodes())
        if not nodes:
            return None
        return random.choice(nodes)

    def ego_subgraph(self, seeds: Iterable[Any], hops: int = 1, direction: str = "both"):
        """
        Build an ego subgraph expanding seeds by `hops` steps.
        direction: 'in', 'out', or 'both'
        """
        curr = set(seeds)
        seen = set(curr)
        for _ in range(hops):
            nxt = set()
            for n in curr:
                if direction in ("out", "both"):
                    nxt.update(self._graph.successors(n))
                if direction in ("in", "both"):
                    nxt.update(self._graph.predecessors(n))
            nxt = nxt - seen
            seen |= nxt
            curr = nxt
        return self._graph.subgraph(seen).copy()

    def clear(self):
        """Remove all nodes and edges."""
        self._graph.clear()
        self.node_embeddings.clear()
        self.relation_embeddings.clear()


if __name__ == "__main__":
    # small demo
    kg = KGStore()
    kg.add_edge("A", "B", relation="rel1")
    kg.add_edge("B", "C", relation="rel2")
    kg.add_edge("A", "C", relation="rel3")
    print("Nodes:", list(kg.nodes()))
    print("Neighbors of A:", kg.neighbors("A"))
    sub = kg.ego_subgraph(["A"], hops=1)
    print("Ego sub nodes:", list(sub.nodes()))
