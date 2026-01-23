# src/data/entity_linker.py
"""
Simple entity linker utilities.

Provides:
  - EntityLinker class that supports:
      * exact matching
      * case-insensitive matching
      * substring matching
      * fuzzy matching using difflib (SequenceMatcher)
  - Link result format: list of dicts with keys ('entity', 'span', 'score', 'match_type')

This implementation is lightweight and dependency-free (uses Python stdlib only).
For production use replace with a specialized EL system or integrate external APIs.
"""

import re
from typing import Iterable, List, Dict, Tuple, Optional
from difflib import SequenceMatcher


class EntityLinker:
    """
    EntityLinker holds a catalog of candidate entity surface forms (strings -> entity_ids).
    You can index by multiple surface forms per entity.
    """

    def __init__(self, case_sensitive: bool = False, fuzzy_threshold: float = 0.7):
        """
        Args:
          case_sensitive: whether to treat surface forms case-sensitively
          fuzzy_threshold: minimal ratio for difflib matching to consider a fuzzy match
        """
        self.case_sensitive = case_sensitive
        self.fuzzy_threshold = fuzzy_threshold
        # mapping surface_form -> list of entity_ids (allow aliasing)
        self.surface2entities = {}
        # set of all surface forms for iteration
        self.surfaces = set()

    def index_entities(self, mapping: Dict[str, Iterable[str]]):
        """
        Index entities provided as a mapping: entity_id -> iterable of surface_forms.
        Example: {"Q123": ["New York", "NYC"], ...}
        """
        for ent_id, surfaces in mapping.items():
            for s in surfaces:
                key = s if self.case_sensitive else s.lower()
                self.surface2entities.setdefault(key, []).append(ent_id)
                self.surfaces.add(key)

    def _normalize_text(self, text: str) -> str:
        return text if self.case_sensitive else text.lower()

    def link(self, text: str, top_k: int = 5, span_window: int = 1000) -> List[Dict]:
        """
        Link entities in a piece of text.

        Returns a list of matches with dict fields:
          - 'entity': entity id
          - 'surface': matched surface form
          - 'span': (start, end) indices in original text
          - 'score': float in [0,1]
          - 'match_type': one of ('exact', 'substring', 'fuzzy')
        The results are sorted by score descending.
        """
        text_norm = self._normalize_text(text)
        results = []

        # exact and substring matches (fast)
        for surf in self.surfaces:
            # skip if surface longer than text
            if len(surf) > len(text_norm):
                continue
            idx = text_norm.find(surf)
            if idx != -1:
                ents = self.surface2entities.get(surf, [])
                for ent in ents:
                    results.append({
                        "entity": ent,
                        "surface": surf,
                        "span": (idx, idx + len(surf)),
                        "score": 1.0,
                        "match_type": "exact" if text_norm[idx:idx+len(surf)] == surf else "substring"
                    })

        # fuzzy matching fallback: compare candidate surfaces to sliding windows or tokens
        # prepare token windows up to length of longest surface
        max_len = max((len(s) for s in self.surfaces), default=0)
        if max_len == 0:
            return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

        # consider candidate substrings by token windows (simple heuristic)
        tokens = re.finditer(r"\S+", text)
        token_spans = [m.span() for m in tokens]
        windows = []
        for i in range(len(token_spans)):
            start = token_spans[i][0]
            # accumulate tokens until window length exceeds max_len or until next token beyond length
            j = i
            while j < len(token_spans) and (token_spans[j][1] - start) <= max_len:
                end = token_spans[j][1]
                windows.append((start, end, text[start:end]))
                j += 1

        # evaluate fuzzy similarity between candidate surfaces and windows
        for start, end, substr in windows:
            substr_norm = self._normalize_text(substr)
            for surf in self.surfaces:
                # quick length filter
                if abs(len(surf) - len(substr_norm)) > max(2, len(surf) * 0.5):
                    continue
                ratio = SequenceMatcher(None, surf, substr_norm).ratio()
                if ratio >= self.fuzzy_threshold:
                    ents = self.surface2entities.get(surf, [])
                    for ent in ents:
                        results.append({
                            "entity": ent,
                            "surface": surf,
                            "span": (start, end),
                            "score": float(ratio),
                            "match_type": "fuzzy"
                        })

        # deduplicate by (entity, span) keeping highest score
        dedup = {}
        for r in results:
            key = (r["entity"], r["span"])
            if key not in dedup or r["score"] > dedup[key]["score"]:
                dedup[key] = r

        sorted_results = sorted(dedup.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:top_k]


# quick example demo
if __name__ == "__main__":
    linker = EntityLinker(case_sensitive=False, fuzzy_threshold=0.6)
    mapping = {
        "Q1": ["New York", "NYC", "New York City"],
        "Q2": ["Los Angeles", "LA"],
        "Q3": ["San Francisco", "SF"],
    }
    linker.index_entities(mapping)
    text = "I went to nyc and then to San Fran."
    matches = linker.link(text, top_k=10)
    for m in matches:
        print(m)
