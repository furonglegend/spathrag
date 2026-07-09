# src/data/dataset.py
"""
Dataset utilities for S-Path-RAG experiments.

Provides:
  - QADataset: a simple PyTorch Dataset for question-answer pairs with optional gold paths
  - collate_fn: batching function to prepare tensors/dicts for training or inference
  - DataLoader factory helper
"""

from typing import List, Dict, Any, Optional, Iterable, Tuple
import random
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except Exception:
    torch = None

    class Dataset:
        """Tiny fallback matching the parts of torch Dataset used here."""

        def __iter__(self):
            for idx in range(len(self)):
                yield self[idx]

    class DataLoader:
        """List-based fallback used when PyTorch is not installed."""

        def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=None):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.collate_fn = collate_fn or (lambda x: x)

        def __iter__(self):
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = [self.dataset[i] for i in indices[start:start + self.batch_size]]
                yield self.collate_fn(batch)


class QADataset(Dataset):
    """
    Simple QA dataset storing entries as dictionaries with keys:
      - 'query': str
      - 'answer': str (optional)
      - 'seed_entities': list of str (optional) : seeds to start path enumeration
      - 'gold_paths': list of paths (optional) where each path is list of node ids
      - 'meta': optional dict

    This class keeps data in memory; for large datasets replace with streaming reader.
    """

    def __init__(self, examples: Optional[List[Dict[str, Any]]] = None):
        self.examples = examples or []

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]

    def add(self, example: Dict[str, Any]):
        """Append a single example dict."""
        self.examples.append(example)

    def extend(self, examples: Iterable[Dict[str, Any]]):
        """Extend dataset with multiple examples."""
        self.examples.extend(list(examples))

    @classmethod
    def from_jsonl(cls, path: str, line_parser=None):
        """
        Load examples from a JSONL file (one JSON object per line).
        Optional line_parser can convert raw dict to expected format.
        """
        try:
            from utils.io import read_lines
        except Exception:
            from src.utils.io import read_lines
        import json
        lines = read_lines(path)
        items = []
        for ln in lines:
            if not ln.strip():
                continue
            obj = json.loads(ln)
            if line_parser:
                obj = line_parser(obj)
            items.append(obj)
        return cls(items)


def default_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function that converts a list of example dicts into a batch dict.
    This function keeps strings as is and batches lists into list-of-lists.
    Customize this function to produce tensors required by your model.
    """
    batch_out: Dict[str, Any] = {}
    # collect keys
    keys = set().union(*(b.keys() for b in batch))
    for k in keys:
        vals = [b.get(k) for b in batch]
        # if all are lists, keep list-of-lists
        if all(isinstance(v, list) for v in vals if v is not None):
            batch_out[k] = vals
        else:
            batch_out[k] = vals
    return batch_out


def make_dataloader(dataset: QADataset, batch_size: int = 8, shuffle: bool = True, num_workers: int = 0, collate_fn=default_collate_fn) -> DataLoader:
    """
    Convenience wrapper to create a PyTorch DataLoader for QADataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)


# quick demo
if __name__ == "__main__":
    ds = QADataset()
    ds.add({"query": "Who directed Inception?", "answer": "Christopher Nolan", "seed_entities": ["Inception"]})
    ds.add({"query": "Who starred in Titanic?", "answer": "Leonardo DiCaprio", "seed_entities": ["Titanic"]})
    loader = make_dataloader(ds, batch_size=2, shuffle=False)
    for batch in loader:
        print("Batch keys:", list(batch.keys()))
        print("Queries:", batch["query"])
