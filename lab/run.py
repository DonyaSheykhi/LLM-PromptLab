# lab/run.py
"""
LLM-PromptLab runner

- Loads a YAML config
- Prepares dataset & prompts
- Calls a model backend (OpenAI or HF)
- Computes metrics
- Writes predictions.jsonl, metrics.json, summary.md under runs/<run_name>/
- Prints lightweight progress logs

Example:
  python -m lab.run --config configs/cls_agnews/openai_0shot.yaml
  python -m lab.run --config configs/cls_agnews/hf_tiny.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Project modules
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
    run_name: str = cfg.get("run_name") or f"run_{int(time.time())}"
    seed: Optional[int] = cfg.get("seed")
    set_seed(seed)

    run_dir = Path("runs") / run_name
    ensure_dir(run_dir)

    # breadcrumb (helps confirm execution)
    ensure_dir(Path("runs/_boot"))
    Path("runs/_boot/started.txt").write_text(
        f"runner started for {run_name}\n", encoding="utf-8"
    )
    log(f"[run] start: {run_name}")

    # 1) dataset
    if "dataset" not in cfg:
        raise ValueError("Config must include 'dataset' block.")
    dataset_cfg: Dict = cfg["dataset"]

    # Required dataset fields used by prompts/metrics (defensive defaults)
    dataset_cfg.setdefault("text_field", "text")         # input text key
    dataset_cfg.setdefault("label_field", "label")       # reference label key if any

    log("[run] loading dataset...")
    ds = load_dataset_split(dataset_cfg)
    n_total = len(ds) if hasattr(ds, "__len__") else sum(1 for _ in ds)
    log(f"[run] dataset loaded: {n_total} samples")

    # 2) backend
    model_cfg: Dict = cfg.get("model", {})
    backend_name = (model_cfg.get("backend") or "openai").lower()
    if backend_name not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend_name}'. Available: {list(BACKENDS.keys())}")

    backend = BACKENDS[backend_name](model_cfg)
    log(f"[run] backend: {backend_name} | model: {model_cfg.get('model_name', '<default>')}")

    # 3) labels / prompt settings
    labels: Optional[List[str]] = cfg.get("labels")
    prompt_cfg: Dict = cfg.get("prompt", {})
    batch_size: int = int(cfg.get("batch_size", 1))

    # 4) inference loop
    predictions: List[str] = []
    references: List[str] = []

    # Convert ds to list if it was a generator
    if not hasattr(ds, "__len__"):
        ds = list(ds)
        n_total = len(ds)

    start = time.time()
    for i, ex in enumerate(ds):
        # progress
        if i % max(1, n_total // 20 or 1) == 0:
            log(f"[run] progress {i}/{n_total}")

        # build prompt
        prompt = build_prompt(prompt_cfg, ex, labels=labels, dataset_cfg=dataset_cfg)

        # generate
        out = backend.generate(
            prompt,
            max_tokens=int(model_cfg.get("max_tokens", 64)),
            temperature=float(model_cfg.get("temperature", 0.0)),
        )
        predictions.append((out or "").strip())

        # collect gold label if present
        ref_key = dataset_cfg.get("label_field", "label")
        if ref_key in ex:
            ref_val = ex[ref_key]
            # map int id -> label string if labels provided
            if isinstance(ref_val, int) and labels:
                ref_val = labels[ref_val]
            references.append(str(ref_val))

        # simple cooperative batching (no real batch API; keeps parity with older code)
        if batch_size > 1 and (i + 1) % batch_size == 0:
            time.sleep(0.01)

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
