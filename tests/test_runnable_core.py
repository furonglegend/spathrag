import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_hybrid_path_enumerator_finds_semantic_answer_path():
    from kg.kg_store import KGStore
    from kg.path_enumerator import PathEnumerator

    kg = KGStore()
    kg.add_edge("Inception", "Christopher_Nolan", relation="directed_by")
    kg.add_edge("Christopher_Nolan", "United_Kingdom", relation="born_in")
    kg.add_edge("Inception", "Leonardo_DiCaprio", relation="starred_actors")

    enumerator = PathEnumerator(kg.to_networkx())
    paths = enumerator.enumerate(
        seeds=["Inception"],
        query="Who directed Inception?",
        method="hybrid",
        max_paths=5,
        max_length=3,
    )

    assert ["Inception", "Christopher_Nolan"] in paths


def test_runner_answers_from_synthetic_kg_without_neural_dependencies():
    from inference.s_path_rag_runner import SPathRAGRunner

    runner = SPathRAGRunner(
        config={
            "kg_file": "data/raw/synthetic_kg.tsv",
            "max_iterations": 2,
            "top_k": 3,
            "max_path_length": 4,
            "llm_mode": "heuristic",
        }
    )
    answer, trace = runner.run("Who directed Inception?", ["Inception"])

    assert "Christopher Nolan" in answer
    assert trace
    assert trace[0]["candidates"]
    assert trace[0]["candidates"][0]["path"][0] == "Inception"


def test_runner_prefers_direct_actor_path_over_stopword_relation_noise():
    from inference.s_path_rag_runner import SPathRAGRunner

    runner = SPathRAGRunner(config={"kg_file": "data/raw/synthetic_kg.tsv", "llm_mode": "heuristic"})
    answer, trace = runner.run("Who acted in The Dark Knight?", ["The_Dark_Knight"])

    assert "Christian Bale" in answer
    assert trace[0]["candidates"][0]["path"] == ["The_Dark_Knight", "Christian_Bale"]


def test_default_eval_cli_writes_jsonl_results():
    output_path = PROJECT_ROOT / "logs" / "eval_results.jsonl"
    if output_path.exists():
        output_path.unlink()

    result = subprocess.run(
        [sys.executable, "main.py", "--mode", "eval", "--config", "configs/default.yaml"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert output_path.exists()
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line]
    assert rows
    assert any("Christopher Nolan" in row["pred"] for row in rows)
