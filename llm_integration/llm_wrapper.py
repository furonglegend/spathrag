"""
Lightweight LLM wrapper for S-Path-RAG.

The original paper injects a compact mixture of path latents into a frozen LLM.
That path is still supported when optional neural dependencies are installed,
but the default mode is a deterministic local heuristic so the repository can
run without downloading model weights.
"""

from typing import Optional, Any, Dict, List
import re

try:
    import torch
except Exception:
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    from llm_integration.injection import project_path_latents_to_prefix_embeddings
except Exception:
    try:
        from src.llm_integration.injection import project_path_latents_to_prefix_embeddings
    except Exception:
        project_path_latents_to_prefix_embeddings = None


def _pretty_entity(entity: Any) -> str:
    return str(entity).replace("_", " ").strip()


def _pretty_relation(relation: Optional[str]) -> str:
    if not relation:
        return "related to"
    return str(relation).replace("_", " ").strip()


class LLMWrapper:
    """
    Wrapper class with three modes:
      - heuristic: no external model, answers from top retrieved paths.
      - prompt: HuggingFace text prompt with path verbalization.
      - prefix: HuggingFace inputs_embeds prefix injection.
    """

    def __init__(self, model_name_or_path: str = "gpt2", device: str = "cpu", mode: str = "heuristic"):
        self.model_name_or_path = model_name_or_path
        self.mode = (mode or "heuristic").lower()
        self.device_name = device
        self.model = None
        self.tokenizer = None
        self.embed_dim = None
        self.default_gen_kwargs = {
            "max_new_tokens": 64,
            "do_sample": False,
            "num_return_sequences": 1,
        }

        if self.mode in ("heuristic", "none", "local"):
            self.mode = "heuristic"
            return

        if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None:
            self.mode = "heuristic"
            return

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
        self.embed_dim = self.model.get_input_embeddings().weight.shape[1]

    def _build_prompt_with_paths(self, query: str, paths: Optional[List[List[str]]] = None) -> str:
        if not paths:
            return query
        lines = ["[PATHS]"]
        for path in paths:
            lines.append(" -> ".join(map(_pretty_entity, path)))
        lines.append("[QUERY]")
        lines.append(query)
        return "\n".join(lines)

    def _heuristic_answer(self, query: str, path_records: Optional[List[Dict[str, Any]]] = None, paths: Optional[List[List[str]]] = None) -> Dict[str, Any]:
        records = path_records or []
        if records:
            best = records[0]
            path = best.get("path", [])
            relations = best.get("relations", [])
            confidence = float(best.get("score", 0.0))
        else:
            path = paths[0] if paths else []
            relations = []
            confidence = 0.0

        if len(path) >= 2:
            answer_entity = _pretty_entity(path[-1])
            evidence_bits = []
            for idx, (src, dst) in enumerate(zip(path, path[1:])):
                rel = relations[idx] if idx < len(relations) else "related_to"
                evidence_bits.append(f"{_pretty_entity(src)} --{_pretty_relation(rel)}--> {_pretty_entity(dst)}")
            evidence = "; ".join(evidence_bits)
            answer = f"{answer_entity}. Evidence path: {evidence}"
            diagnostic = "done: supported_by_top_path"
        else:
            answer = f"No supported KG path was found for: {query}"
            diagnostic = "expand seeds"
            confidence = 0.0

        return {
            "answer": answer,
            "diagnostic": diagnostic,
            "meta": {"mode": "heuristic", "confidence": confidence, "done": bool(path)},
        }

    def generate_with_injection(
        self,
        query: str,
        kv_or_prefix: Optional[Any] = None,
        paths: Optional[List[List[str]]] = None,
        path_records: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 5,
        **gen_kwargs,
    ) -> Dict[str, Any]:
        if self.mode == "heuristic" or self.model is None or self.tokenizer is None:
            return self._heuristic_answer(query, path_records=path_records, paths=paths)

        gen_kwargs_combined = dict(self.default_gen_kwargs)
        gen_kwargs_combined.update(gen_kwargs or {})

        if self.mode == "prompt" or kv_or_prefix is None:
            prompt = self._build_prompt_with_paths(query, paths)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, **gen_kwargs_combined)
            answer = self.tokenizer.decode(out[0], skip_special_tokens=True)
            return {"answer": answer, "diagnostic": "prompt_mode", "meta": {"mode": "prompt"}}

        if isinstance(kv_or_prefix, torch.Tensor):
            prefix_embeddings = kv_or_prefix.to(self.device)
            if prefix_embeddings.dim() != 3:
                raise ValueError("prefix_embeddings must be [batch, prefix_len, embed_dim]")
            if prefix_embeddings.size(0) != 1:
                raise NotImplementedError("Only batch=1 is supported for prefix generation")

            tokenized = self.tokenizer(query, return_tensors="pt", truncation=True).to(self.device)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            input_embeds = self.model.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([prefix_embeddings, input_embeds], dim=1)
            prefix_mask = torch.ones((1, prefix_embeddings.size(1)), dtype=attention_mask.dtype, device=attention_mask.device)
            new_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            with torch.no_grad():
                out = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=new_attention_mask,
                    **gen_kwargs_combined,
                )
            answer = self.tokenizer.decode(out[0], skip_special_tokens=True)
            return {"answer": answer, "diagnostic": "prefix_mode", "meta": {"mode": "prefix", "prefix_len": prefix_embeddings.size(1)}}

        return self._heuristic_answer(query, path_records=path_records, paths=paths)


if __name__ == "__main__":
    wrapper = LLMWrapper(mode="heuristic")
    out = wrapper.generate_with_injection(
        "Who directed Inception?",
        paths=[["Inception", "Christopher_Nolan"]],
        path_records=[{"path": ["Inception", "Christopher_Nolan"], "relations": ["directed_by"], "score": 3.0}],
    )
    print(out["answer"])
