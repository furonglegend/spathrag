# src/llm_integration/llm_wrapper.py
"""
Lightweight LLM wrapper that supports two modes:
  - prompt_mode: simply concatenate textual path info into the prompt (safe fallback)
  - prefix_injection_mode: prepend learned or projected prefix embeddings to model inputs
This wrapper is implemented for HuggingFace causal LMs (AutoModelForCausalLM).
It uses `inputs_embeds` to pass concatenated prefix embeddings + token embeddings to model.generate().

Important notes:
  - prefix injection is compatible with causal LMs that accept inputs_embeds in generate().
  - For encoder-decoder models or custom cross-attention injection, you will need to adapt the integration.
"""

from typing import Optional, Any, Dict, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, StoppingCriteriaList

# import injection helpers
try:
    from src.llm_integration.injection import project_path_latents_to_prefix_embeddings
except Exception:
    project_path_latents_to_prefix_embeddings = None


class LLMWrapper:
    """
    LLM wrapper class.
    Args:
      model_name_or_path: HuggingFace model id or local path.
      device: 'cpu' or 'cuda'
      mode: 'prompt' or 'prefix' (prefix uses embedding-level injection)
    """

    def __init__(self, model_name_or_path: str = "gpt2", device: str = "cpu", mode: str = "prefix"):
        self.model_name_or_path = model_name_or_path
        self.device = torch.device(device)
        self.mode = mode.lower()
        # load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        # ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # some small models might not have model parallel weights etc.
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()

        # embedding dimension for prefix projection
        self.embed_dim = self.model.get_input_embeddings().weight.shape[1]

        # default generation kwargs
        self.default_gen_kwargs = {
            "max_new_tokens": 64,
            "do_sample": True,
            "top_p": 0.95,
            "temperature": 0.8,
            "num_return_sequences": 1,
        }

    def _build_prompt_with_paths(self, query: str, paths: Optional[List[List[str]]] = None) -> str:
        """
        Build a single textual prompt by concatenating a short structured summary
        of the candidate paths. This serves as a fallback when embedding injection
        is not available or disabled.
        """
        if not paths:
            return query
        lines = ["[PATHS]"]
        for p in paths:
            # join nodes using -> for readability
            lines.append(" -> ".join(map(str, p)))
        lines.append("[QUERY]")
        lines.append(query)
        return "\n".join(lines)

    def generate_with_injection(self, query: str, kv_or_prefix: Optional[Any] = None, paths: Optional[List[List[str]]] = None, top_k: int = 5, **gen_kwargs) -> Dict[str, Any]:
        """
        Generate an answer for the query with optional injection.
        Args:
          query: input query string
          kv_or_prefix: either a dict returned by project_path_latents_to_kv or a tensor of prefix embeddings
                        This wrapper prefers prefix-embedding tensors produced by project_path_latents_to_prefix_embeddings.
          paths: optional list of candidate paths (used to create textual prompt fallback)
          top_k: used for candidate truncation (not used inside wrapper)
          gen_kwargs: override generation kwargs
        Returns:
          dict: {"answer": str, "diagnostic": str, "meta": {...}}
        """
        gen_kwargs_combined = dict(self.default_gen_kwargs)
        gen_kwargs_combined.update(gen_kwargs or {})

        if self.mode == "prompt" or kv_or_prefix is None:
            # fallback: produce textual prompt by inserting path info
            prompt = self._build_prompt_with_paths(query, paths)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, **gen_kwargs_combined)
            answer = self.tokenizer.decode(out[0], skip_special_tokens=True)
            return {"answer": answer, "diagnostic": "prompt_mode", "meta": {"mode": "prompt"}}

        # attempting prefix-injection mode
        if isinstance(kv_or_prefix, torch.Tensor):
            prefix_embeddings = kv_or_prefix.to(self.device)  # expected [batch, prefix_len, embed_dim]
            # only support batch size 1 for now in generation mode; else handle multiple sequences.
            if prefix_embeddings.dim() != 3:
                raise ValueError("prefix_embeddings must be [batch, prefix_len, embed_dim]")
            batch = prefix_embeddings.size(0)
            if batch != 1:
                # for now, support only single example generation
                raise NotImplementedError("Currently only batch=1 supported for prefix injection generate path")

            # tokenize query and obtain input embeddings
            tokenized = self.tokenizer(query, return_tensors="pt", truncation=True).to(self.device)
            input_ids = tokenized["input_ids"]  # [1, seq_len]
            attention_mask = tokenized["attention_mask"]  # [1, seq_len]
            # get token embeddings from model
            input_embeds = self.model.get_input_embeddings()(input_ids)  # [1, seq_len, embed_dim]
            # concatenate prefix embeddings before input embeddings
            inputs_embeds = torch.cat([prefix_embeddings, input_embeds], dim=1)  # [1, prefix_len + seq_len, embed_dim]
            # adjust attention mask
            prefix_mask = torch.ones((1, prefix_embeddings.size(1)), dtype=attention_mask.dtype, device=attention_mask.device)
            new_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # [1, new_seq_len]

            # Use generate with inputs_embeds
            generate_inputs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": new_attention_mask,
                **gen_kwargs_combined
            }
            with torch.no_grad():
                out = self.model.generate(**generate_inputs)
            answer = self.tokenizer.decode(out[0], skip_special_tokens=True)
            return {"answer": answer, "diagnostic": "prefix_mode", "meta": {"mode": "prefix", "prefix_len": prefix_embeddings.size(1)}}

        # kv_or_prefix is something else (e.g., dict containing 'k'/'v'), fallback to prompt
        prompt = self._build_prompt_with_paths(query, paths)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs_combined)
        answer = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return {"answer": answer, "diagnostic": "fallback_prompt", "meta": {"mode": "fallback"}}


# quick demo when run directly (CPU)
if __name__ == "__main__":
    wrapper = LLMWrapper(model_name_or_path="gpt2", device="cpu", mode="prefix")
    # small example latents (batch=1, num_paths=3, latent_dim=128)
    import torch
    lat = torch.randn(1, 3, 128)
    # project to prefix embeddings using injection helper if available
    if project_path_latents_to_prefix_embeddings is not None:
        prefix = project_path_latents_to_prefix_embeddings(lat, embed_dim=wrapper.embed_dim, num_prefix_tokens_per_path=1)
        out = wrapper.generate_with_injection("Who directed the movie Inception?", kv_or_prefix=prefix, paths=[["Inception","director","Christopher_Nolan"]])
        print("answer snippet:", out["answer"][:200])
    else:
        out = wrapper.generate_with_injection("Who directed the movie Inception?", kv_or_prefix=None, paths=[["Inception","director","Christopher_Nolan"]])
        print("answer snippet:", out["answer"][:200])
