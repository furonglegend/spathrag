# src/inference/s_path_rag_runner.py
"""
Iterative retrieval-reasoning runner (Algorithm 1 skeleton).
This module orchestrates:
  - path enumeration
  - path scoring (scorer)
  - path encoding -> projection -> injection into LLM
  - call to LLM wrapper to produce answer + diagnostic
  - mapping diagnostics to graph edits via mapper_pi
  - updating the KG store / subgraph and repeating

"""

import logging
import time
from typing import List, Dict, Any, Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Try to import project components; fallback to stubs
try:
    from src.kg.path_enumerator import PathEnumerator
except Exception:
    PathEnumerator = None

try:
    from src.models.path_encoder import PathEncoder
except Exception:
    PathEncoder = None

try:
    from src.models.scorer import Scorer
except Exception:
    Scorer = None

try:
    from src.llm_integration.llm_wrapper import LLMWrapper
except Exception:
    LLMWrapper = None

try:
    from src.models.mapper_pi import MapperPi
except Exception:
    MapperPi = None

try:
    from src.kg.kg_store import KGStore
except Exception:
    KGStore = None

try:
    from src.llm_integration.injection import project_path_latents_to_kv
except Exception:
    project_path_latents_to_kv = None


class SPathRAGRunner:
    """
    High-level runner class for the retrieval-augmented iterative loop.
    """

    def __init__(self, device: str = "cpu", config: Optional[Dict[str, Any]] = None):
        self.device = device
        self.config = config or {}
        # initialize components or placeholders
        self.enumerator = PathEnumerator() if PathEnumerator is not None else None
        self.path_encoder = PathEncoder() if PathEncoder is not None else None
        self.scorer = Scorer() if Scorer is not None else None
        self.llm = LLMWrapper() if LLMWrapper is not None else None
        self.mapper = MapperPi() if MapperPi is not None else None
        self.kg = KGStore() if KGStore is not None else None

        # defaults
        self.max_iterations = self.config.get("max_iterations", 3)
        self.top_k = self.config.get("top_k", 5)

    def enumerate_and_score(self, query: str, seed_nodes: List[str]) -> List[Dict[str, Any]]:
        """
        Enumerate candidate paths and score them.
        Returns list of dicts: [{"path": [...], "score": float, "latent": Tensor or np.array}, ...]
        """
        if self.enumerator is None:
            # fallback: return trivial paths composed from seed_nodes
            logger.warning("PathEnumerator not available; using fallback simple paths")
            paths = [[seed_nodes[0], seed_nodes[0]]]  # trivial self-loop path
            return [{"path": paths[0], "score": 1.0, "latent": np.zeros((32,))}]
        # call enumerator to produce candidate paths
        candidates = self.enumerator.enumerate(query=query, seeds=seed_nodes, max_paths=100)
        scored = []
        for p in candidates:
            # compute path encoding and scoring if modules exist
            latent = None
            score = 1.0
            if self.path_encoder is not None:
                latent = self.path_encoder.encode_path(p)  # user API expected
            if self.scorer is not None:
                score = float(self.scorer.score(p, latent))
            scored.append({"path": p, "score": score, "latent": latent})
        # sort by score desc
        scored = sorted(scored, key=lambda x: x["score"], reverse=True)
        return scored

    def project_and_inject(self, latents: List[Any]):
        """
        Project path latents into key/value tensors for LLM injection.
        Returns (k, v) or any object expected by the llm.inject API.
        """
        if project_path_latents_to_kv is None:
            logger.warning("Injection projection not available; returning None")
            return None
        # stack latents into tensor
        # expected shape: [num_paths, latent_dim] or [batch, num_paths, latent_dim]
        try:
            latent_stack = torch.tensor(np.stack([np.asarray(x) for x in latents]), dtype=torch.float32).to(self.device)
            kv = project_path_latents_to_kv(latent_stack)
            return kv
        except Exception as e:
            logger.exception("Projection failed: %s", e)
            return None

    def call_llm(self, query: str, injected_kv: Any):
        """
        Call the LLM wrapper with the query and injected kv (if available).
        Expects returned dict: {"answer": str, "diagnostic": str, "meta": {...}}
        """
        if self.llm is None:
            # fallback: echo query as answer with a basic diagnostic
            return {"answer": f"ANSWER ECHO: {query}", "diagnostic": "no_llm", "meta": {}}
        return self.llm.generate_with_injection(query=query, kv=injected_kv, top_k=self.top_k)

    def map_diagnostic_to_edits(self, diagnostic: str):
        """
        Map diagnostic text to graph edits via mapper_pi.
        Expected returned edits: list of dicts: {"op": "add_edge"/"remove_edge", "edge": (u,v), ...}
        """
        if self.mapper is None:
            logger.warning("MapperPi not available; no graph edit performed")
            return []
        return self.mapper.map(diagnostic)

    def update_kg(self, edits: List[Dict[str, Any]]):
        """
        Apply edits to the KG store.
        """
        if self.kg is None:
            logger.warning("KGStore not available; skipping KG update")
            return
        for e in edits:
            op = e.get("op")
            if op == "add_edge":
                self.kg.add_edge(e["edge"][0], e["edge"][1], **e.get("attrs", {}))
            elif op == "remove_edge":
                self.kg.remove_edge(e["edge"][0], e["edge"][1])
            else:
                logger.warning("Unknown edit op: %s", op)

    def run(self, query: str, seed_nodes: List[str]):
        """
        Execute the iterative loop until convergence or max_iterations.
        Returns the final answer and a trace of iterations.
        """
        trace = []
        for iteration in range(self.max_iterations):
            logger.info(f"Iteration {iteration+1}/{self.max_iterations} for query: {query}")
            start = time.time()

            scored = self.enumerate_and_score(query, seed_nodes)
            top_candidates = scored[: self.top_k]
            latents = [c["latent"] if c["latent"] is not None else np.zeros((32,)) for c in top_candidates]

            injected_kv = self.project_and_inject(latents)
            llm_out = self.call_llm(query, injected_kv)

            diagnostic = llm_out.get("diagnostic", "")
            edits = self.map_diagnostic_to_edits(diagnostic)
            self.update_kg(edits)

            trace.append({
                "iteration": iteration,
                "candidates": top_candidates,
                "llm_out": llm_out,
                "applied_edits": edits,
                "time_s": time.time() - start,
            })

            # termination heuristic: if LLM signals done or no edits produced
            if llm_out.get("meta", {}).get("done", False) or not edits:
                logger.info("Termination condition met (done or no edits).")
                break

        final_answer = trace[-1]["llm_out"]["answer"] if trace else ""
        return final_answer, trace


# simple CLI
if __name__ == "__main__":
    runner = SPathRAGRunner(device="cpu", config={"max_iterations": 3, "top_k": 5})
    answer, trace = runner.run(query="Who directed Inception?", seed_nodes=["Leonardo DiCaprio"])
    print("Final answer:", answer)
    print("Trace length:", len(trace))
