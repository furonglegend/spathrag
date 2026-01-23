# src/models/gnn_encoder.py
"""
Lightweight relation-aware GNN implemented with pure PyTorch (no external GNN libs).
This Graph Neural Network:
  - accepts a networkx graph or KGStore
  - builds adjacency lists and relation index mapping
  - performs T rounds of message passing where messages incorporate relation embeddings
  - supports returning node embeddings as a torch.Tensor aligned to a node->index map

This is a minimal implementation suitable for small graphs and prototyping.
For large-scale experiments replace with DGL/PyG implementations for speed.
"""

from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx


class RelationAwareGNN(nn.Module):
    """
    A simple relation-aware GNN.
    Node features -> aggregated neighbor messages -> updated node features.

    Message function:
      m_v = mean_{(u->v, rel)} ( linear_node(h_u) + linear_rel(r) )
    Update:
      h_v' = ReLU( LayerNorm( h_v + m_v ) )
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_relations: int = 32, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.dropout = dropout

        # input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        # relation embedding table (learnable)
        self.rel_emb = nn.Embedding(num_relations, hidden_dim)
        # per-layer neighbor linear
        self.neigh_linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        # self-update linear
        self.self_linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout_layer = nn.Dropout(dropout)

        # placeholder relation-to-index mapping; can be set at runtime
        self.rel2idx: Dict[Any, int] = {}

    def set_relation_mapping(self, rel2idx: Dict[Any, int]):
        self.rel2idx = rel2idx
        # ensure embedding table large enough
        max_idx = max(rel2idx.values()) if rel2idx else -1
        if max_idx >= self.rel_emb.num_embeddings:
            # expand embedding table
            new_num = max_idx + 1
            old = self.rel_emb.weight.data
            self.rel_emb = nn.Embedding(new_num, self.rel_emb.embedding_dim)
            with torch.no_grad():
                self.rel_emb.weight[:old.size(0)].copy_(old)

    @staticmethod
    def build_index_map(graph: nx.Graph) -> Tuple[Dict[Any, int], List[Any]]:
        """
        Build node->index mapping and return list of nodes.
        """
        nodes = list(graph.nodes())
        idx_map = {n: i for i, n in enumerate(nodes)}
        return idx_map, nodes

    def forward(self, graph: nx.Graph, node_features: Dict[Any, torch.Tensor]) -> Tuple[Dict[Any, torch.Tensor], torch.Tensor]:
        """
        Run the GNN and return:
          - mapping node_id -> embedding tensor
          - matrix of embeddings [num_nodes, hidden_dim] aligned with internal index
        Args:
          graph: networkx graph (nodes must match keys in node_features)
          node_features: dict node_id -> feature tensor of shape [in_dim]
        """
        idx_map, nodes = self.build_index_map(graph)
        n = len(nodes)
        device = next(self.parameters()).device

        # build feature matrix
        feat = torch.zeros(n, self.in_dim, device=device)
        for i, node in enumerate(nodes):
            v = node_features.get(node)
            if v is None:
                # zero vector if missing
                continue
            # convert to tensor if necessary
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v, dtype=torch.float32, device=device)
            feat[i] = v.to(device)

        h = self.input_proj(feat)  # [n, hidden_dim]

        # build adjacency list with relation indices for message passing
        # edges: (u, v, rel_idx)
        edges = []
        for u, v, data in graph.edges(data=True):
            rel = data.get("relation", 0)
            rel_idx = self.rel2idx.get(rel, 0) if self.rel2idx else 0
            edges.append((idx_map[u], idx_map[v], rel_idx))

        # organize incoming neighbors per node
        incoming = [[] for _ in range(n)]
        for u_idx, v_idx, r_idx in edges:
            incoming[v_idx].append((u_idx, r_idx))

        # propagate for num_layers
        for layer in range(self.num_layers):
            neigh_msgs = torch.zeros_like(h)
            for v_idx in range(n):
                neigh = incoming[v_idx]
                if not neigh:
                    continue
                msgs = []
                for u_idx, r_idx in neigh:
                    rel_vec = self.rel_emb(torch.tensor(r_idx, device=device)).unsqueeze(0)  # [1, hidden]
                    node_vec = h[u_idx].unsqueeze(0)  # [1, hidden]
                    msgs.append(node_vec + rel_vec)  # [1, hidden]
                msgs = torch.cat(msgs, dim=0)  # [num_neighbors, hidden]
                agg = msgs.mean(dim=0)  # mean aggregator
                neigh_msgs[v_idx] = self.neigh_linears[layer](agg)
            self_vecs = self.self_linears[layer](h)
            h = F.relu(self.layer_norms[layer](self_vecs + neigh_msgs))
            h = self.dropout_layer(h)

        # map back to node_id -> tensor
        node_embs = {node: h[idx_map[node]] for node in nodes}
        return node_embs, h


if __name__ == "__main__":
    # small demo
    kg = nx.DiGraph()
    kg.add_edge("A", "B", relation="r1")
    kg.add_edge("B", "C", relation="r2")
    kg.add_edge("A", "C", relation="r3")
    # create dummy node features (in_dim=4)
    node_feats = {"A": [1.0,0.0,0.0,0.0], "B":[0.0,1.0,0.0,0.0], "C":[0.0,0.0,1.0,0.0]}
    gnn = RelationAwareGNN(in_dim=4, hidden_dim=16, num_relations=8, num_layers=2)
    # set relation mapping
    gnn.set_relation_mapping({"r1":0, "r2":1, "r3":2})
    node_embs, mat = gnn(kg, node_feats)
    print("node_embs keys:", list(node_embs.keys()))
    print("embedding matrix shape:", mat.shape)
