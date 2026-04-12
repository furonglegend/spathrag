# S-Path-RAG (Prototype)

A research-oriented, modular codebase skeleton for **S-Path-RAG**, a semantic-aware shortest-path Retrieval-Augmented Generation framework for multi-hop Knowledge Graph Question Answering (KGQA).

This repository contains a complete engineering scaffold (data, KG utilities, enumerators, encoders, training pipeline, LLM integration and evaluation utilities) intended for prototyping and research experiments. The provided modules are lightweight, well-documented, and easy to replace with production-grade components (DGL/PyG, Neo4j, Faiss, etc.).


## Key features (prototype)
- In-memory `KGStore` with simple I/O and neighborhood utilities.
- `PathEnumerator` supporting k-shortest, beam search and random-walk sampling.
- Relation-aware `RelationAwareGNN` (small, PyTorch-based prototype).
- `PathEncoder` with node/relation embeddings and pooling options.
- `Scorer`, `Verifier`, and rule-based `MapperPi`.
- LLM wrapper supporting prompt fallback and prefix-embedding injection (HuggingFace).
- Staged training driver (pretrain / scorer / joint finetune) and evaluation runner.
- Lightweight utilities and example scripts to speed up experimentation.

## Installation

Clone the repository and install Python dependencies:

```bash
git clone <repo-url> s-path-rag
cd s-path-rag

# create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# install required packages
pip install -r requirements.txt

# optionally install in editable/development mode
pip install -e .
```
## Acknowledge
This project code is only used for principle explanation. The code is only used as an early version.
