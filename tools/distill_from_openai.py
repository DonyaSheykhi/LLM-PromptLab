# tools/distill_from_openai.py
"""
Distill labels (and optional rationales) from an OpenAI model, then
train a compact student model on the distilled dataset.

Usage:

# 1) Label with OpenAI (writes runs/agnews_distill/labels.jsonl)
python tools/distill_from_openai.py --phase label \
  --openai_model gpt-4o-mini --split train[:2000] --out runs/agnews_distill \
  --with_rationale false --max_tokens 20 --temperature 0.0

# 2) Train student on distilled labels (writes runs/agnews_student_distilled/)
python tools/distill_from_openai.py --phase train \
  --distill_dir runs/agnews_distill --student_model distilbert-base-uncased \
  --epochs 2 --batch 16 --lr 2e-5 --out runs/agnews_student_distilled
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score

# HF/Transformers
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer
)

# OpenAI
from openai import OpenAI


LABELS = ["World", "Sports", "Business", "Sci/Tech"]
LABEL_SET = {x.lower(): x for x in LABELS}  # for normalization


def backoff_sleep(attempt: int, base: float = 0.5) -> None:
    # simple exponential backoff
    time.sleep(base * (2 ** attempt))


def normalize_label(text: str) -> str:
    t = (text or "").strip().lower()
    # accept exact or start-with
    for k, v in LABEL_SET.items():
        if t == k or t.startswith(k):
            return v
    # common shortcuts
    if t in {"sci", "science", "tech", "sci/tech", "scitech"}:
        return "Sci/Tech"
    return ""


def prompt_for(text: str) -> str:
    options = ", ".join(LABELS)
    return (
        "You are a precise news topic classifier.\n"
        f"Choose ONE category from: {options}\n"
        "Return only the category name.\n"
        "---\n"
        f'News: "{text}"\n'
        "Answer:"
    )


def label_with_openai(
    split: str,
    out_dir: Path,
    openai_model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 20,
    with_rationale: bool = False,
    limit: int | None = None,
) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)

    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.jsonl"

    ds = load_dataset("ag_news")
    if ":" in split:
        subset = ds["train"].select(range(len(ds["train"])))  # will slice below
        subset = ds[split.split(":")[0]][split.split(":")[1]]
    else:
        if split.startswith("train"):
            subset = ds["train"]
        elif split.startswith("test"):
            subset = ds["test"]
        else:
            subset = ds["train"]

    if limit:
        subset = subset.select(range(min(limit, len(subset))))

    written = 0
    with labels_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(subset):
            text = ex["text"]
            p = prompt_for(text)

            # Call with simple retry/backoff
            last_err = None
            for attempt in range(6):
                try:
                    resp = client.chat.completions.create(
                        model=openai_model,
                        messages=[{"role": "user", "content": p}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    out = (resp.choices[0].message.content or "").strip()
                    label = normalize_label(out)
                    if not label:
                        # If model returned rationale or extra words, try to extract first token
                        first = out.splitlines()[0].strip().split()[0]
                        label = normalize_label(first)
                    if not label:
                        # give up with empty label; skip
                        break

                    rec = {
                        "i": i,
                        "text": text,
                        "label": label,
                    }
                    if with_rationale:
                        rec["raw"] = out
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                    break
                except Exception as e:
                    last_err = e
                    backoff_sleep(attempt)
            # optional: progress log
            if (i + 1) % 100 == 0:
                print(f"[label] processed {i+1}/{len(subset)}")

    print(f"[label] wrote {written} records to {labels_path}")


def load_distilled_dataset(distill_dir: Path) -> Dataset
