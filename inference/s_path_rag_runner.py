"""
Runnable S-Path-RAG inference loop.

This module implements the paper's core loop in a lightweight form:
entity seeds -> semantic hybrid path enumeration -> path scoring/verifying
-> compact latent placeholder -> local or optional LLM generation -> diagnostic
mapper -> graph edit. Neural modules remain optional; the default path is fully
local and deterministic so the project runs without torch or model downloads.
"""

import logging
import os
import time
from typing import List, Dict, Any, Optional

import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    from kg.path_enumerator import PathEnumerator
    from kg.kg_store import KGStore
    from data.entity_linker import EntityLinker
    from llm_integration.llm_wrapper import LLMWrapper
    from models.mapper_pi import MapperPi
except Exception:
    from src.kg.path_enumerator import PathEnumerator
    from src.kg.kg_store import KGStore
    from src.data.entity_linker import EntityLinker
    from src.llm_integration.llm_wrapper import LLMWrapper
    from src.models.mapper_pi import MapperPi

try:
    from models.path_encoder import PathEncoder
except Exception:
    PathEncoder = None

try:
    from models.scorer import Scorer
except Exception:
    Scorer = None

try:
    from models.verifier import Verifier
except Exception:
    Verifier = None

try:
    from llm_integration.injection import project_path_latents_to_prefix_embeddings
except Exception:
    project_path_latents_to_prefix_embeddings = None


logger = logging.getLogger(__name__)


BUILTIN_SYNTHETIC_TRIPLES = [
    ("Inception", "directed_by", "Christopher_Nolan"),
    ("Inception", "starred_actors", "Leonardo_DiCaprio"),
    ("Inception", "genre", "Science_Fiction"),
    ("Christopher_Nolan", "born_in", "United_Kingdom"),
    ("Christopher_Nolan", "spouse", "Emma_Thomas"),
    ("Emma_Thomas", "produced", "Inception"),
    ("The_Dark_Knight", "directed_by", "Christopher_Nolan"),
    ("The_Dark_Knight", "starred_actors", "Christian_Bale"),
    ("Christian_Bale", "born_in", "United_Kingdom"),
    ("Titanic", "directed_by", "James_Cameron"),
    ("Titanic", "starred_actors", "Leonardo_DiCaprio"),
    ("James_Cameron", "born_in", "Canada"),
    ("Avatar", "directed_by", "James_Cameron"),
    ("Avatar", "genre", "Science_Fiction"),
]


def _stable_hash_vector(parts: List[str], dim: int = 32) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for part in parts:
        token = str(part)
        idx = abs(hash(token)) % dim
        sign = 1.0 if (abs(hash("+" + token)) % 2 == 0) else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class SPathRAGRunner:
    """
    High-level runner for the iterative Neural-Socratic Graph Dialogue loop.
    """

    def __init__(self, device: str = "cpu", config: Optional[Dict[str, Any]] = None):
        self.device = device
        self.config = config or {}
        self.max_iterations = int(self.config.get("max_iterations", 3))
        self.top_k = int(self.config.get("top_k", 5))
        self.max_path_length = int(self.config.get("max_path_length", 4))
        self.enumeration_method = self.config.get("enumeration_method", "hybrid")
        self.beam_width = int(self.config.get("beam_width", 4))

        self.kg = KGStore()
        self._load_kg(self.config.get("kg_file"))
        self.enumerator = PathEnumerator(self.kg.to_networkx())
        self.entity_linker = self._build_entity_linker()

        self.path_encoder = PathEncoder() if PathEncoder is not None else None
        self.scorer = Scorer() if Scorer is not None else None
        self.verifier = Verifier() if Verifier is not None else None
        self.mapper = MapperPi()
        self.llm = LLMWrapper(
            model_name_or_path=self.config.get("model_name_or_path", "gpt2"),
            device=device,
            mode=self.config.get("llm_mode", "heuristic"),
        )

    def _load_kg(self, kg_file: Optional[str]) -> None:
        if kg_file and os.path.exists(kg_file):
            self.kg.load_triples(kg_file, delimiter="\t", header=False)
            return
        for src, rel, dst in BUILTIN_SYNTHETIC_TRIPLES:
            self.kg.add_edge(src, dst, relation=rel)

    def _build_entity_linker(self) -> EntityLinker:
        linker = EntityLinker(case_sensitive=False, fuzzy_threshold=0.72)
        mapping = {}
        for node in self.kg.nodes():
            surface = str(node).replace("_", " ")
            mapping[str(node)] = [surface, str(node)]
        linker.index_entities(mapping)
        return linker

    def infer_seed_nodes(self, query: str, provided: Optional[List[str]] = None) -> List[str]:
        seeds = [seed for seed in (provided or []) if seed in set(self.kg.nodes())]
        if seeds:
            return seeds
        linked = self.entity_linker.link(query, top_k=3)
        return [item["entity"] for item in linked]

    def _path_relations(self, path: List[str]) -> List[str]:
        return [self.kg.get_edge_relation(src, dst) or "" for src, dst in zip(path, path[1:])]

    def _encode_path(self, path: List[str], relations: List[str]) -> Any:
        if self.path_encoder is not None and hasattr(self.path_encoder, "encode_path"):
            try:
                return self.path_encoder.encode_path(path, relations=relations)
            except Exception:
                logger.debug("Neural path encoder failed; using hash latent", exc_info=True)
        return _stable_hash_vector(list(path) + list(relations), dim=32)

    def _score_with_optional_model(self, path: List[str], latent: Any, query: str) -> float:
        graph = self.kg.to_networkx()
        base = self.enumerator.score_path(graph, path, query)
        if self.scorer is not None and hasattr(self.scorer, "score"):
            try:
                return float(base + self.scorer.score(path, latent))
            except Exception:
                logger.debug("Neural scorer failed; using heuristic score", exc_info=True)
        return float(base)

    def enumerate_and_score(self, query: str, seed_nodes: List[str]) -> List[Dict[str, Any]]:
        graph = self.kg.to_networkx()
        self.enumerator.graph = graph
        candidates = self.enumerator.enumerate(
            graph=graph,
            seeds=seed_nodes,
            query=query,
            method=self.enumeration_method,
            max_paths=max(self.top_k * 4, 12),
            max_length=self.max_path_length,
            beam_width=self.beam_width,
        )

        scored = []
        for path in candidates:
            relations = self._path_relations(path)
            latent = self._encode_path(path, relations)
            score = self._score_with_optional_model(path, latent, query)
            scored.append({"path": path, "relations": relations, "score": score, "latent": latent})

        scored.sort(key=lambda row: (row["score"], -len(row["path"])), reverse=True)
        return scored

    def project_and_inject(self, latents: List[Any]) -> Optional[Any]:
        if torch is None or project_path_latents_to_prefix_embeddings is None or not latents:
            return None
        try:
            latent_stack = torch.tensor(np.stack([np.asarray(x) for x in latents]), dtype=torch.float32)
            embed_dim = getattr(self.llm, "embed_dim", None)
            if not embed_dim:
                return None
            return project_path_latents_to_prefix_embeddings(latent_stack, embed_dim=embed_dim)
        except Exception:
            logger.debug("Path latent projection failed; continuing without injection", exc_info=True)
            return None

    def call_llm(self, query: str, injected: Any, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        public_records = [self._public_candidate(candidate) for candidate in candidates]
        paths = [candidate["path"] for candidate in candidates]
        return self.llm.generate_with_injection(
            query=query,
            kv_or_prefix=injected,
            paths=paths,
            path_records=public_records,
            top_k=self.top_k,
        )

    def map_diagnostic_to_edits(self, diagnostic: str) -> List[Dict[str, Any]]:
        if diagnostic.strip().lower().startswith("done"):
            return []
        return self.mapper.map(diagnostic)

    def update_kg(self, edits: List[Dict[str, Any]]) -> None:
        for edit in edits:
            op = edit.get("op")
            if op == "add_edge":
                src, dst = edit["edge"]
                attrs = edit.get("attrs", {})
                relation = attrs.pop("relation", attrs.pop("rel", None)) if attrs else None
                self.kg.add_edge(src, dst, relation=relation, **attrs)
            elif op == "remove_edge":
                src, dst = edit["edge"]
                self.kg.remove_edge(src, dst)

    @staticmethod
    def _public_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "path": list(candidate.get("path", [])),
            "relations": list(candidate.get("relations", [])),
            "score": float(candidate.get("score", 0.0)),
        }

    def run(self, query: str, seed_nodes: Optional[List[str]] = None):
        seeds = self.infer_seed_nodes(query, seed_nodes)
        trace = []

        for iteration in range(self.max_iterations):
            start = time.time()
            scored = self.enumerate_and_score(query, seeds)
            top_candidates = scored[: self.top_k]
            latents = [candidate["latent"] for candidate in top_candidates]
            injected = self.project_and_inject(latents)
            llm_out = self.call_llm(query, injected, top_candidates)

            diagnostic = llm_out.get("diagnostic", "")
            edits = self.map_diagnostic_to_edits(diagnostic)
            self.update_kg(edits)

            trace.append(
                {
                    "iteration": iteration,
                    "seed_nodes": list(seeds),
                    "candidates": [self._public_candidate(candidate) for candidate in top_candidates],
                    "llm_out": llm_out,
                    "applied_edits": edits,
                    "time_s": time.time() - start,
                }
            )

            if llm_out.get("meta", {}).get("done", False) or not edits:
                break

        final_answer = trace[-1]["llm_out"]["answer"] if trace else ""
        return final_answer, trace


if __name__ == "__main__":
    runner = SPathRAGRunner(config={"max_iterations": 2, "top_k": 3})
    answer, run_trace = runner.run("Who directed Inception?", ["Inception"])
    print("Final answer:", answer)
    print("Trace length:", len(run_trace))
