# lab/run.py
"""
LLM-PromptLab runner
- Loads a YAML config
- Prepares dataset & prompt(s)
- Calls a model backend (OpenAI or HF)
- Computes simple metrics
- Writes predictions, metrics.json, and summary.md
- Prints lightweight progress logs

Example:
  python -m lab.run --config configs/cls_agnews/openai_0shot.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import yaml

from lab.data import load_dataset_split
from lab.prompts import build_prompt
from lab.models.openai_backend import OpenAIBackend
from lab.models.hf_backend import HFBackend
from lab.eval.metrics import compute_metrics
from lab.eval.reporting import write_summary


# -------------------------
# Utilities
# -------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    try:
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
    except Exception:
        random.seed(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -------------------------
# Backend registry
# -------------------------
BACKENDS = {
    "openai": OpenAIBackend,
    "hf": HFBackend,
}


# -------------------------
# Core run
# -------------------------
def run_experiment(cfg: Dict) -> None:
    # 0) seed & folders
    run_name: str = cfg.get("run_name", f"run_{int(time.time())}")
    seed: Optional[int] = cfg.get("seed")
    set_seed(seed)

    run_dir = Path("runs") / run_name
    ensure_dir(run_dir)

    # 1) dataset
    log(f"[run] config: {run_name}")
    log("[run] loading dataset...")
    ds = load_dataset_split(cfg["dataset"])
    log(f"[run] dataset loaded: {len(ds)} samples")

    # 2) backend
    model_cfg: Dict = cfg.get("model", {})
    backend_name = model_cfg.get("backend", "openai")
    if backend_name not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend_name}. Available: {list(BACKENDS)}")

    backend = BACKENDS[backend_name](model_cfg)
    log(f"[run] backend: {backend_name} | model: {model_cfg.get('model_name', '<default>')}")

    # 3) labels / prompt settings
    labels = cfg.get("labels")
    prompt_cfg = cfg.get("prompt", {})
    dataset_cfg = cfg.get("dataset", {})

    # 4) loop
    predictions: List[str] = []
    references: List[str] = []
    n_total = len(ds)

    start = time.time()
    for i, ex in enumerate(ds):
        if i % max(1, n_total // 20 or 1) == 0:
            log(f"[run] progress {i}/{n_total}")

        prompt = build_prompt(prompt_cfg, ex, labels=labels, dataset_cfg=dataset_cfg)

        out = backend.generate(
            prompt,
            max_tokens=model_cfg.get("max_tokens", 64),
            temperature=model_cfg.get("temperature", 0.0),
        )
        predictions.append((out or "").strip())

        # collect reference if available
        if "label_field" in dataset_cfg:
            ref = ex.get(dataset_cfg["label_field"])
            # map int -> label name if labels provided
            if isinstance(ref, int) and labels:
                ref = labels[ref]
            references.append(str(ref))

    # 5) write predictions
    pred_path = run_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for pred, ex in zip(predictions, ds):
            f.write(json.dumps({"prediction": pred, "input": ex}, ensure_ascii=False) + "\n")
    log(f"[run] wrote: {pred_path}")

    # 6) metrics
    metrics: Dict = {}
    if references:
        metrics = compute_metrics(
            task=cfg.get("task", "classification"),
            predictions=predictions,
            references=references,
            labels=labels,
        )
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    log(f"[run] wrote: {metrics_path}")

    # 7) summary
    write_summary(run_dir, cfg, metrics)
    log(f"[run] wrote: {run_dir / 'summary.md'}")

    elapsed = time.time() - start
    log(f"[run] done in {elapsed:.1f}s | n={n_total}")


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run an LLM-PromptLab experiment")
    ap.add_argument("--config", required=True, help="Path to a YAML config")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_experiment(cfg)


if __name__ == "__main__":
    main()
