# notebooks/visualize_paths.py
# English comments only.
# Small utility to visualize enumerated KG paths and simple diagnostics.
# Usage as a script: python notebooks/visualize_paths.py results.json
# results.json is expected to be a small JSON containing a list of paths or a dict with "paths" key.

import json
import sys
from collections import Counter

import networkx as nx
import matplotlib.pyplot as plt

def load_results(path):
    """Load results JSON produced by inference runner."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def plot_path_length_distribution(paths, outpath=None):
    """Plot distribution of path lengths."""
    lengths = [len(p) - 1 if isinstance(p, list) else 0 for p in paths]  # assume list of node ids
    cnt = Counter(lengths)
    xs = sorted(cnt.keys())
    ys = [cnt[x] for x in xs]

    plt.figure(figsize=(6,4))
    plt.bar(xs, ys)
    plt.xlabel("Path length (edges)")
    plt.ylabel("Count")
    plt.title("Path length distribution")
    if outpath:
        plt.savefig(outpath, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

def draw_sample_graph(paths, outpath=None, max_nodes=50):
    """Build a small graph from paths and draw it with networkx."""
    G = nx.DiGraph()
    for p in paths:
        # path expected as list of node ids (e.g., ["A","B","C"])
        if not isinstance(p, list) or len(p) < 2:
            continue
        for u, v in zip(p[:-1], p[1:]):
            G.add_edge(u, v)

    # optionally limit nodes for plotting
    if G.number_of_nodes() > max_nodes:
        # pick top nodes by degree
        deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
        top_nodes = set([n for n,_ in deg[:max_nodes]])
        G = G.subgraph(top_nodes).copy()

    plt.figure(figsize=(8,6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=300, arrowsize=12)
    plt.title("Extracted subgraph from paths")
    if outpath:
        plt.savefig(outpath, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python notebooks/visualize_paths.py <results.json> [out_prefix]")
        sys.exit(1)
    results_path = sys.argv[1]
    out_prefix = sys.argv[2] if len(sys.argv) > 2 else None

    data = load_results(results_path)
    # support either {"paths": [...]} or just [...]
    paths = data.get("paths", data) if isinstance(data, dict) else data

    plot_path_length_distribution(paths, outpath=(out_prefix + "_length_dist.png") if out_prefix else None)
    draw_sample_graph(paths, outpath=(out_prefix + "_subgraph.png") if out_prefix else None)

if __name__ == "__main__":
    main()
