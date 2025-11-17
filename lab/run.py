# lab/run.py
"""
LLM-PromptLab runner

Usage:
  python -m lab.run --config configs/cls_agnews/openai_0shot.yaml
  python -m lab.run --config configs/cls_agnews/hf_tiny_fast.yaml

What it does:
- Load YAML config
- Load dataset split
- Build prompts and call backend (OpenAI/HF)
- Map raw outputs to allowed labels (optional) for stable scoring
- Compute metrics (+ optional latency/cost)
- Write predictions.jsonl, metrics.json, summary.md
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import yaml

# Project modules
from lab.data import load_dataset_split
from lab.prompts import build_prompt
from lab.models.openai_backend import OpenAIBackend
from lab.models.hf_backend import HFBackend
from lab.eval.metrics import compute_metrics
from lab.eval.reporting import write_summary
from lab.util import map_to_choice  # heuristics to coerce free text -> one of labels

# Optional cost tracking (safe if file not present)
try:
    from lab.cost import cost_usd  # returns USD float given model & token usage
except Exception:
    cost_usd = None  # type: ignore


# -------------------------
# Small utilities
# -------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def p95(values: List[float]) -> Optional[float]:
    if not values:
        return None
    vs = sorted(values)
    idx = max(0, min(len(vs) - 1, int(0.95 * len(vs)) - 1))
    return round(vs[idx], 4)


# -------------------------
# Backends registry
# -------------------------
BACKENDS = {
    "openai": OpenAIBackend,
    "hf": HFBackend,
}


# -------------------------
# Core run
# -------------------------
def run_experiment(cfg: Dict[str, Any]) -> None:
    # 1) run metadata & dirs
    run_name: str = cfg.get("run_name") or f"run_{int(time.time())}"
    run_dir = Path("runs") / run_name
    ensure_dir(run_dir)
    ensure_dir(Path("runs/_boot"))
    Path("runs/_boot/started.txt").write_text(f"runner started for {run_name}\n", encoding="utf-8")
    log(f"[run] start: {run_name}")

    # 2) dataset
    if "dataset" not in cfg:
        raise ValueError("Config must include a 'dataset' block.")
    dataset_cfg: Dict[str, Any] = dict(cfg["dataset"])
    dataset_cfg.setdefault("text_field", "text")
    dataset_cfg.setdefault("label_field", "label")

    log("[run] loading dataset...")
    ds = load_dataset_split(dataset_cfg)
    if not hasattr(ds, "__len__"):
        ds = list(ds)
    n_total = len(ds)
    log(f"[run] dataset loaded: {n_total} samples")

    # 3) backend
    model_cfg: Dict[str, Any] = cfg.get("model", {})
    backend_name = (model_cfg.get("backend") or "openai").lower()
    if backend_name not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend_name}'. Available: {list(BACKENDS.keys())}")
    backend = BACKENDS[backend_name](model_cfg)
    log(f"[run] backend: {backend_name} | model: {model_cfg.get('model_name', '<default>')}")

    # 4) prompt & labels
    labels: Optional[List[str]] = cfg.get("labels")
    prompt_cfg: Dict[str, Any] = cfg.get("prompt", {})
    task: str = cfg.get("task", "classification")

    # 5) inference loop
    predictions: List[str] = []
    references: List[str] = []
    latencies: List[float] = []
    total_cost: float = 0.0

    for i, ex in enumerate(ds):
        # progress
        if i % max(1, n_total // 20 or 1) == 0:
            log(f"[run] progress {i}/{n_total}")

        # build prompt
        prompt = build_prompt(prompt_cfg, ex, labels=labels, dataset_cfg=dataset_cfg)

        # call backend
        gen = backend.generate(
            prompt,
            max_tokens=int(model_cfg.get("max_tokens", 64)),
            temperature=float(model_cfg.get("temperature", 0.0)),
            labels=labels if isinstance(backend, HFBackend) else None,  # add this
        )

        # unify return
        if isinstance(gen, tuple):
            out_text, usage, secs = gen  # type: ignore
            if isinstance(secs, (int, float)):
                latencies.append(float(secs))
            if cost_usd and isinstance(usage, dict) and usage.get("model"):
                total_cost += cost_usd(
                    usage.get("model"),
                    usage.get("prompt_tokens"),
                    usage.get("completion_tokens"),
                )
        else:
            out_text = gen  # type: ignore

        # map to valid label (if labels provided) to avoid free-text mismatches
        if labels:
            out_text = map_to_choice(out_text, labels)

        predictions.append(str(out_text))

        # gold reference if available
        ref_key = dataset_cfg.get("label_field", "label")
        if ref_key in ex:
            ref_val = ex[ref_key]
            if isinstance(ref_val, int) and labels:
                ref_val = labels[ref_val]
            references.append(str(ref_val))

    # 6) write predictions.jsonl
    pred_path = run_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for pred, ex in zip(predictions, ds):
            f.write(json.dumps({"prediction": pred, "input": ex}, ensure_ascii=False) + "\n")
    log(f"[run] wrote: {pred_path}")

    # 7) metrics.json
    metrics: Dict[str, Any] = {}
    if references:
        metrics = compute_metrics(
            task=task,
            predictions=predictions,
            references=references,
            labels=labels,
        )

    # add latency/cost if collected
    if latencies:
        metrics["latency_mean_s"] = round(sum(latencies) / len(latencies), 4)
        p = p95(latencies)
        if p is not None:
            metrics["latency_p95_s"] = p
    if total_cost:
        metrics["cost_usd_total"] = round(total_cost, 6)

    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    log(f"[run] wrote: {metrics_path}")

    # 8) summary.md
    write_summary(run_dir, cfg, metrics)
    log(f"[run] wrote: {run_dir / 'summary.md'}")

    log(f"[run] done | n={n_total}")


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
