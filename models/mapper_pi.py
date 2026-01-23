# src/models/mapper_pi.py
"""
MapperPi: diagnostic -> graph edits mapper.
This file provides:
  - a rule-based MapperPi that extracts simple add/remove edge commands from
    diagnostic textual messages using regex.
  - a placeholder LearnedMapper class that can be implemented if you want to
    train a mapping model.
All comments in English.
"""

import re
from typing import List, Dict, Any, Optional


class MapperPi:
    """
    Rule-based mapper.
    Parses diagnostic strings and returns a list of edits.
    Each edit is a dict, e.g., {"op": "add_edge", "edge": ("EntityA","EntityB"), "attrs": {...}}
    """

    # simple regex patterns for commands like "add edge A->B" or "remove A -> B"
    ADD_EDGE_RE = re.compile(r"(?:add|insert|create)\s+(?:edge|relation)\s+([^\s:,\->]+)\s*[-:>]+\s*([^\s,;]+)", re.IGNORECASE)
    REMOVE_EDGE_RE = re.compile(r"(?:remove|delete|drop)\s+(?:edge|relation)\s+([^\s:,\->]+)\s*[-:>]+\s*([^\s,;]+)", re.IGNORECASE)
    ADD_EDGE_VERBOSE_RE = re.compile(r"add edge from\s+([^\s,]+)\s+to\s+([^\s,]+)", re.IGNORECASE)
    REMOVE_EDGE_VERBOSE_RE = re.compile(r"remove edge from\s+([^\s,]+)\s+to\s+([^\s,]+)", re.IGNORECASE)

    def __init__(self, allow_unknown: bool = False):
        """
        allow_unknown: if True, mapper will attempt to extract pairs of capitalized tokens
                     even if patterns don't match strictly.
        """
        self.allow_unknown = allow_unknown

    def map(self, diagnostic: str) -> List[Dict[str, Any]]:
        """
        Map diagnostic text to a list of edits.
        """
        edits = []
        if not diagnostic or not diagnostic.strip():
            return edits

        text = diagnostic.strip()

        # apply strict patterns first
        for m in self.ADD_EDGE_RE.finditer(text):
            u, v = m.group(1).strip(), m.group(2).strip()
            edits.append({"op": "add_edge", "edge": (u, v), "attrs": {}})

        for m in self.REMOVE_EDGE_RE.finditer(text):
            u, v = m.group(1).strip(), m.group(2).strip()
            edits.append({"op": "remove_edge", "edge": (u, v)})

        for m in self.ADD_EDGE_VERBOSE_RE.finditer(text):
            u, v = m.group(1).strip(), m.group(2).strip()
            edits.append({"op": "add_edge", "edge": (u, v), "attrs": {}})

        for m in self.REMOVE_EDGE_VERBOSE_RE.finditer(text):
            u, v = m.group(1).strip(), m.group(2).strip()
            edits.append({"op": "remove_edge", "edge": (u, v)})

        # fallback heuristics: look for patterns like "A -> B" or "A ->B"
        arrow_pairs = re.findall(r"([A-Za-z0-9_]+)\s*[-:>]+\s*([A-Za-z0-9_]+)", text)
        for u, v in arrow_pairs:
            # avoid duplicating already captured pairs
            if not any(e for e in edits if tuple(e.get("edge", ())) == (u, v)):
                edits.append({"op": "add_edge", "edge": (u, v), "attrs": {}, "inferred": True})

        # optional loose heuristics
        if self.allow_unknown and not edits:
            # pick capitalized token pairs as probable entities
            tokens = re.findall(r"[A-Z][a-zA-Z0-9_]{1,}", text)
            if len(tokens) >= 2:
                # make a chain of edges
                for i in range(len(tokens) - 1):
                    edits.append({"op": "add_edge", "edge": (tokens[i], tokens[i + 1]), "attrs": {}, "inferred_loose": True})

        return edits


class LearnedMapper:
    """
    Placeholder for a learned mapper API.
    You can implement this class to load a classifier (e.g., seq2seq or text-to-structured)
    that maps diagnostics -> edits. For now this is an API stub.
    """

    def __init__(self, model=None):
        self.model = model

    def map(self, diagnostic: str) -> List[Dict[str, Any]]:
        """
        Convert diagnostic to edits using self.model. Must return same format as MapperPi.map.
        """
        if self.model is None:
            return []
        # Example pseudo-code:
        # structured = self.model.predict(diagnostic)
        # return structured_to_edits(structured)
        return []


if __name__ == "__main__":
    mp = MapperPi()
    txts = [
        "Add relation PersonA->PersonB because it's missing",
        "Please remove edge Foo->Bar",
        "We should add edge from Alice to Bob.",
        "Candidate mapping: X->Y and Z->W"
    ]
    for t in txts:
        print("input:", t)
        print("edits:", mp.map(t))
