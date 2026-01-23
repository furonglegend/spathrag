# src/main.py
"""
Main entrypoint for experiments.
Supports modes:
  - train : run staged training (calls src.training.trainer.main)
  - eval  : run evaluation on a checkpoint (uses SPathRAGRunner for inference)
  - infer : single-query interactive inference
All comments are in English.
"""

import argparse
import os
import sys
import json
from typing import Optional

# Try to use PyYAML for config loading
try:
    import yaml
except Exception:
    yaml = None

# project utilities (previously provided)
try:
    from src.utils.logging import get_logger
except Exception:
    # fallback to std logging
    import logging
    def get_logger(name="s_path_rag", log_file=None, level=logging.INFO, console=True, **_):
        logger = logging.getLogger(name)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(ch)
        logger.setLevel(level)
        return logger

logger = get_logger("main", log_file="logs/main.log", level=20)

# import trainer and runner if available
try:
    from src.training.trainer import main as trainer_main
except Exception:
    trainer_main = None

try:
    from src.inference.s_path_rag_runner import SPathRAGRunner
except Exception:
    SPathRAGRunner = None


def load_yaml(path: str) -> dict:
    """Load YAML file into a Python dict. Requires PyYAML."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to read YAML configuration files. Install pyyaml.")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def merge_configs(base: dict, override: Optional[dict]) -> dict:
    """Shallow merge of two dicts (override keys in base)."""
    if not override:
        return base
    out = dict(base)
    out.update(override)
    return out


def ensure_dirs_from_config(cfg: dict):
    """Create directories referenced in configuration if needed."""
    # logging dir
    log_dir = cfg.get("logging", {}).get("log_dir")
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    # checkpoint dir
    chk = cfg.get("checkpoint", {}).get("save_dir") or cfg.get("checkpoint", {}).get("save_dir")
    if chk:
        os.makedirs(chk, exist_ok=True)
    # data dirs
    data = cfg.get("data", {})
    for p in ("raw_dir", "processed_dir", "vocab_dir"):
        if p in data and data[p]:
            os.makedirs(data[p], exist_ok=True)


def run_train(config_path: Optional[str]):
    """Run staged training using trainer_main."""
    if trainer_main is None:
        logger.error("Trainer module not available (src.training.trainer). Cannot run training.")
        return
    logger.info("Starting training using config: %s", config_path)
    trainer_main(config_path)


def run_eval(config: dict, checkpoint: Optional[str]):
    """Run evaluation / inference loop over test set using SPathRAGRunner."""
    if SPathRAGRunner is None:
        logger.error("SPathRAGRunner not available (src.inference.s_path_rag_runner). Cannot run eval.")
        return

    device = config.get("training", {}).get("device", config.get("model", {}).get("device", "cpu"))
    runner_cfg = {
        "max_iterations": config.get("runner", {}).get("max_iterations", 3),
        "top_k": config.get("runner", {}).get("top_k", 5),
    }
    runner = SPathRAGRunner(device=device, config=runner_cfg)
    test_file = config.get("data", {}).get("test_file")
    if not test_file or not os.path.exists(test_file):
        logger.error("Test file not found: %s", test_file)
        return

    # load test examples (assume jsonl)
    examples = []
    with open(test_file, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError:
                continue
            examples.append(obj)

    logger.info("Loaded %d test examples", len(examples))

    results = []
    for i, ex in enumerate(examples):
        query = ex.get("query") or ex.get("question")
        seeds = ex.get("seed_entities") or ex.get("seed_nodes") or []
        answer, trace = runner.run(query=query, seed_nodes=seeds)
        results.append({"query": query, "pred": answer, "trace": trace})
        if (i + 1) % 10 == 0:
            logger.info("Processed %d/%d examples", i + 1, len(examples))

    out_path = config.get("logging", {}).get("log_dir", "logs") + "/eval_results.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Evaluation results written to %s", out_path)


def run_infer_interactive(config: dict):
    """Interactive inference from the command line."""
    if SPathRAGRunner is None:
        logger.error("SPathRAGRunner not available. Cannot run interactive inference.")
        return
    device = config.get("training", {}).get("device", "cpu")
    runner_cfg = {
        "max_iterations": config.get("runner", {}).get("max_iterations", 3),
        "top_k": config.get("runner", {}).get("top_k", 5),
    }
    runner = SPathRAGRunner(device=device, config=runner_cfg)
    print("Interactive inference. Type 'exit' or 'quit' to stop.")
    while True:
        query = input("QUERY> ").strip()
        if not query or query.lower() in ("exit", "quit"):
            break
        seeds_line = input("SEED NODES (comma separated, optional)> ").strip()
        seeds = [s.strip() for s in seeds_line.split(",")] if seeds_line else []
        answer, trace = runner.run(query=query, seed_nodes=seeds)
        print("ANSWER:\n", answer)
        print("TRACE LENGTH:", len(trace))


def parse_args():
    parser = argparse.ArgumentParser(description="S-Path-RAG experiment entrypoint")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "infer"], default="train", help="Mode to run")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path for eval/infer")
    parser.add_argument("--override", type=str, default=None, help="JSON string to override config keys (shallow merge)")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config
    if not os.path.exists(config_path):
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    # load config
    config = {}
    try:
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            config = load_yaml(config_path)
        else:
            # try JSON
            with open(config_path, "r", encoding="utf-8") as fh:
                config = json.load(fh)
    except Exception as e:
        logger.exception("Failed to load config: %s", e)
        sys.exit(1)

    # optional override JSON string
    if args.override:
        try:
            override = json.loads(args.override)
            config = merge_configs(config, override)
        except Exception:
            logger.warning("Failed to parse override JSON; ignoring")

    ensure_dirs_from_config(config)

    if args.mode == "train":
        run_train(config_path)
    elif args.mode == "eval":
        run_eval(config, args.checkpoint)
    elif args.mode == "infer":
        run_infer_interactive(config)
    else:
        logger.error("Unknown mode: %s", args.mode)


if __name__ == "__main__":
    main()
