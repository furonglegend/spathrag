# S-Path-RAG Reconstruction Notes

This project reconstructs a runnable Python prototype for the paper
"S-Path-RAG: Semantic-Aware Shortest-Path Retrieval Augmented Generation for Multi-Hop Knowledge Graph Question Answering".
The original `README.md` is preserved unchanged. The current version runs locally by default without requiring `torch`,
`transformers`, or online model downloads. It uses a lightweight NetworkX + NumPy + synthetic KGQA setup to reproduce the
main paper workflow. Optional neural dependencies can still be installed later to extend the trainable path encoder,
scorer, verifier, and HuggingFace LLM prefix-injection path.

## Public Data Search Results

The LaTeX paper states that source code is available at `https://github.com/furonglegend/spathrag`. I found that repository
as a public GitHub project with a `Camera-ready code for S-Path-RAG` release. The repository page did not show WebQSP, CWQ,
or MetaQA data files bundled directly with the code. The paper arXiv page is `https://arxiv.org/abs/2603.23512`, and its
abstract matches the local LaTeX source.

The benchmarks mentioned in the paper can be obtained separately:

- WebQSP: Microsoft Download Center provides `WebQSP.zip`. The page says it contains 4,737 questions with full SPARQL
  semantic parses. Link: `https://www.microsoft.com/en-us/download/details.aspx?id=52763`
- MetaQA: the official `yuyuz/MetaQA` GitHub repository provides a Google Drive download entry and describes 1-hop,
  2-hop, and 3-hop question-answer files plus a `kb.txt` triple knowledge base. Link: `https://github.com/yuyuz/MetaQA`
- OGB WikiKG2: the OGB documentation describes `ogbl-wikikg2` as containing 2,500,604 entities and 17,137,181 edges,
  available through `ogb>=1.2.4`. Link: `https://ogb.stanford.edu/docs/linkprop/#ogbl-wikikg2`

Because these datasets are large and have different licensing and download mechanisms, this reconstruction does not
automatically download external datasets. Instead, it includes a small synthetic KGQA dataset so the project can run,
test, and explain its evidence paths immediately on a local machine.

## Implemented Paper Components

- `kg/kg_store.py`: in-memory knowledge graph with TSV triple loading, neighbor queries, and graph edits.
- `kg/path_enumerator.py`: a lightweight reconstruction of hybrid candidate generation, including semantic weighted
  k-shortest search, beam expansion, and constrained random walks. Edges whose relation names overlap the query receive
  lower traversal cost, which prioritizes semantically relevant evidence paths.
- `inference/s_path_rag_runner.py`: a runnable Neural-Socratic Graph Dialogue loop with seed linking, path enumeration,
  path scoring, latent placeholders, answer generation, diagnostic mapping, graph updates, and trace output.
- `llm_integration/llm_wrapper.py`: a local heuristic answer generator by default, producing answers and evidence paths
  from the highest-scoring retrieved path. With optional neural dependencies, the project can switch to prompt or prefix
  modes.
- `models/path_encoder.py` and `models/scorer.py`: the torch-based versions are retained, with `encode_path()` and
  `score()` convenience APIs added for the runner.
- `training/trainer.py`: runs a dry-run when torch is unavailable instead of crashing. When torch is installed, it can
  execute the lightweight placeholder training flow.
- `data/raw/synthetic_kg.tsv` and `data/processed/*.jsonl`: a small synthetic KGQA dataset covering single-hop and
  multi-hop evidence paths.
- `tests/test_runnable_core.py`: regression tests for hybrid path retrieval, runner inference, and default CLI eval.

## Installation And Running

```powershell
cd E:\pyproject\BigDataAssignmentTwo\temp\spathrag\spathrag-main\spathrag-main
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the default evaluation:

```powershell
python main.py --mode eval --config configs/default.yaml
```

The output is written to:

```text
logs/eval_results.jsonl
```

Run the tests:

```powershell
python -m pytest
```

Run the training dry-run:

```powershell
python main.py --mode train --config configs/default.yaml
```

## Real Data Format

The knowledge graph uses TSV triples:

```text
subject<TAB>relation<TAB>object
```

The QA data uses JSONL:

```json
{"query":"Who directed Inception?","answer":"Christopher Nolan","seed_entities":["Inception"],"gold_paths":[["Inception","Christopher_Nolan"]]}
```

Then update `configs/default.yaml`:

```yaml
data:
  kg_file: "data/raw/your_kg.tsv"
  train_file: "data/processed/train.jsonl"
  dev_file: "data/processed/dev.jsonl"
  test_file: "data/processed/test.jsonl"
```

## Limitations

This version is a runnable reconstruction of the paper mechanism, not a full reproduction of the LLaMA2-7B and A100-scale
experiments reported in the paper. It prioritizes correct project structure, runnable default commands, interpretable path
retrieval, and clear extension points for WebQSP, CWQ, MetaQA, or OGB WikiKG2.
