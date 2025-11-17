# lab/models/hf_backend.py
from __future__ import annotations

import time
from typing import Optional, Tuple, Dict, Any, List

from .base import BaseBackend

class HFBackend(BaseBackend):
    """
    Hugging Face backend that:
    - uses zero-shot classification pipeline when labels are provided
    - otherwise falls back to text-generation pipeline
    Returns (text, usage_dict, elapsed_seconds)
    """
    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)
        self.model_name = model_cfg.get("model_name", "valhalla/distilbart-mnli-12-3")
        self.temperature = float(model_cfg.get("temperature", 0.7))
        self.max_tokens = int(model_cfg.get("max_tokens", 32))

        # Lazy imports to avoid heavy init if unused
        from transformers import pipeline

        # We’ll construct pipelines on first call based on whether labels are passed
        self._pipelines: Dict[str, Any] = {}
        self._pipeline_factory = pipeline

    def _get_zero_shot(self):
        if "zs" not in self._pipelines:
            self._pipelines["zs"] = self._pipeline_factory(
                "zero-shot-classification",
                model=self.model_name,
                device=-1,  # CPU
            )
        return self._pipelines["zs"]

    def _get_generator(self):
        if "gen" not in self._pipelines:
            self._pipelines["gen"] = self._pipeline_factory(
                "text-generation",
                model=self.model_name,
                device=-1,  # CPU
            )
        return self._pipelines["gen"]

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        labels: Optional[List[str]] = None,  # <- we allow labels passthrough if caller provides
    ) -> Tuple[str, Dict[str, Any], float]:
        t0 = time.perf_counter()
        usage: Dict[str, Any] = {"model": self.model_name}

        if labels:
            # ZERO-SHOT CLASSIFICATION
            pipe = self._get_zero_shot()
            res = pipe(prompt, candidate_labels=labels, multi_label=False)
            # res contains 'labels' sorted by score; pick the top
            text = res["labels"][0] if res and "labels" in res else ""
        else:
            # TEXT GENERATION (fallback)
            pipe = self._get_generator()
            gen = pipe(
                prompt,
                do_sample=(float(temperature or self.temperature) > 0),
                temperature=float(temperature or self.temperature),
                max_new_tokens=int(max_tokens or self.max_tokens),
                num_return_sequences=1,
            )
            text = gen[0]["generated_text"] if gen else ""

        secs = time.perf_counter() - t0
        return (text or "").strip(), usage, secs
