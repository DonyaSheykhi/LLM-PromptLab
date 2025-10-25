# lab/models/openai_backend.py
"""
OpenAI chat-completions backend for LLM-PromptLab.

- Reads API key from the environment: OPENAI_API_KEY (and optionally OPENAI_ORG_ID)
- Pulls model settings from the provided config (model_name, temperature, max_tokens)
- Adds simple exponential backoff for transient / rate-limit errors
- Returns the assistant's text content (stripped)

Expected minimal config shape:
model:
  backend: openai
  model_name: gpt-4o-mini
  temperature: 0.0
  max_tokens: 20
"""

from __future__ import annotations

import os
import time
from typing import Optional

from openai import OpenAI
from .base import BaseBackend


class OpenAIBackend(BaseBackend):
    def __init__(self, model_cfg: dict):
        """
        Args:
            model_cfg: dict with keys like:
              - model_name (str)
              - temperature (float)
              - max_tokens (int)
        """
        super().__init__(model_cfg)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Please set it in your environment."
            )

        # Organization is optional; only set if present.
        org_id = os.getenv("OPENAI_ORG_ID") or None

        # Initialize the client
        self.client = OpenAI(api_key=api_key, organization=org_id)

        # Defaults from config (can still be overridden per-call)
        self.model_name: str = model_cfg.get("model_name", "gpt-4o-mini")
        self.default_temperature: float = float(model_cfg.get("temperature", 0.0))
        self.default_max_tokens: int = int(model_cfg.get("max_tokens", 64))

        # Retry/backoff settings
        self.max_attempts: int = int(model_cfg.get("max_attempts", 6))  # 0..5
        self.base_sleep: float = float(model_cfg.get("base_sleep", 0.5))  # seconds

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a single-turn prompt using Chat Completions.

        Args:
            prompt: The user content to send.
            max_tokens: Optional override of max tokens.
            temperature: Optional override of temperature.

        Returns:
            Model's text response (stripped). Empty string on no content.
        """
        max_tokens = int(max_tokens if max_tokens is not None else self.default_max_tokens)
        temperature = float(temperature if temperature is not None else self.default_temperature)

        # Build a minimal single-message chat
        messages = [
            {"role": "user", "content": prompt}
        ]

        last_err: Optional[Exception] = None
        for attempt in range(self.max_attempts):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = resp.choices[0].message.content if resp.choices else ""
                return (content or "").strip()

            except Exception as e:
                # On the final attempt, bubble up the error.
                last_err = e
                if attempt >= self.max_attempts - 1:
                    raise

                # Exponential backoff with jitter
                sleep_s = self._compute_backoff(attempt)
                time.sleep(sleep_s)

        # Should not reach here due to raise above; return empty as a safeguard.
        return ""

    def _compute_backoff(self, attempt: int) -> float:
        """
        Exponential backoff with light jitter.
        attempt: 0,1,2,... -> sleep around base_sleep * 2^attempt (+/- 10%)
        """
        import random

        base = self.base_sleep * (2 ** attempt)
        jitter = base * 0.1 * (2 * random.random() - 1.0)  # +/-10%
        return max(0.0, base + jitter)
