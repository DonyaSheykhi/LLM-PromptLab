# lab/models/openai_backend.py
from __future__ import annotations

import os
import time
import random
from typing import Optional, Tuple, Dict, Any

from openai import OpenAI
from lab.latency import timer
from .base import BaseBackend


class OpenAIBackend(BaseBackend):
    """
    Minimal OpenAI chat-completions backend.

    Config shape:
    model:
      backend: openai
      model_name: gpt-4o-mini
      temperature: 0.0
      max_tokens: 20
      max_attempts: 6
      base_sleep: 0.5
    """

    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

        org_id = os.getenv("OPENAI_ORG_ID") or None
        self.client = OpenAI(api_key=api_key, organization=org_id)

        self.model_name: str = model_cfg.get("model_name", "gpt-4o-mini")
        self.default_temperature: float = float(model_cfg.get("temperature", 0.0))
        self.default_max_tokens: int = int(model_cfg.get("max_tokens", 64))

        self.max_attempts: int = int(model_cfg.get("max_attempts", 6))
        self.base_sleep: float = float(model_cfg.get("base_sleep", 0.5))

    def _compute_backoff(self, attempt: int) -> float:
        """
        Exponential backoff with ±10% jitter.
        attempt: 0,1,2,...
        """
        base = self.base_sleep * (2 ** attempt)
        jitter = base * 0.1 * (2 * random.random() - 1.0)
        return max(0.0, base + jitter)

    def _request_once(self, messages, *, max_tokens: int, temperature: float):
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, Dict[str, Any], float]:
        """
        Returns: (text, usage_dict, elapsed_seconds)
        usage_dict keys: prompt_tokens, completion_tokens, total_tokens, model
        """
        max_tokens = int(max_tokens if max_tokens is not None else self.default_max_tokens)
        temperature = float(temperature if temperature is not None else self.default_temperature)

        messages = [{"role": "user", "content": prompt}]

        last_err: Optional[Exception] = None
        with timer() as elapsed:
            for attempt in range(self.max_attempts):
                try:
                    resp = self._request_once(messages, max_tokens=max_tokens, temperature=temperature)
                    text = (resp.choices[0].message.content or "").strip() if resp.choices else ""
                    usage = getattr(resp, "usage", None)
                    usage_dict = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "total_tokens": getattr(usage, "total_tokens", None),
                        "model": self.model_name,
                    }
                    return text, usage_dict, elapsed()
                except Exception as e:
                    last_err = e
                    if attempt >= self.max_attempts - 1:
                        raise
                    time.sleep(self._compute_backoff(attempt))

        # Should never hit this (we raise on last attempt), keep signature safe:
        return "", {"model": self.model_name}, elapsed()
