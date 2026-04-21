# src/utils/io.py
"""
I/O helper utilities for common file formats.
Includes JSON, YAML, pickle, and simple text line readers/writers.
"""

import json
import pickle
import os
from typing import Any, Iterable, List, Optional

try:
    import yaml
except Exception:
    yaml = None  # YAML optional; functions will raise helpful error if used


def ensure_dir(path: str):
    """Ensure directory exists for a file or a directory path."""
    dirpath = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)


# -------- JSON ----------
def read_json(path: str, encoding: str = "utf-8") -> Any:
    """Read and return JSON object from file."""
    with open(path, "r", encoding=encoding) as fh:
        return json.load(fh)


def write_json(obj: Any, path: str, indent: int = 2, encoding: str = "utf-8"):
    """Write object to JSON file (creates parent dirs)."""
    ensure_dir(path)
    with open(path, "w", encoding=encoding) as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False)


# -------- YAML ----------
def read_yaml(path: str) -> Any:
    """Read YAML file. Requires PyYAML."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to read YAML files. Install pyyaml.")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def write_yaml(obj: Any, path: str):
    """Write object to YAML file. Requires PyYAML."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to write YAML files. Install pyyaml.")
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(obj, fh, default_flow_style=False, allow_unicode=True)


# -------- Pickle ----------
def read_pickle(path: str) -> Any:
    """Read object from a pickle file."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


def write_pickle(obj: Any, path: str, protocol: int = pickle.HIGHEST_PROTOCOL):
    """Write object to a pickle file (creates parent dirs)."""
    ensure_dir(path)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=protocol)


# -------- plain text lines ----------
def read_lines(path: str, strip: bool = True, encoding: str = "utf-8") -> List[str]:
    """Read lines from a text file into a list."""
    with open(path, "r", encoding=encoding) as fh:
        lines = fh.readlines()
    if strip:
        return [l.rstrip("\n\r") for l in lines]
    return lines


def write_lines(lines: Iterable[str], path: str, encoding: str = "utf-8"):
    """Write an iterable of lines to a file (creates parent dirs)."""
    ensure_dir(path)
    with open(path, "w", encoding=encoding) as fh:
        for line in lines:
            fh.write(line.rstrip("\n") + "\n")
