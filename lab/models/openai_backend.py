import os
import time
from openai import OpenAI
from .base import BaseBackend

for attempt in range(6):
    try:
        resp=self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role":"user","content":prompt}],
            max_tokens=max_tokens, temperature=temperature,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        if attempt == 5: raise
        time.sleep(2 ** attempt / 2.0)

class OpenAIBackend(BaseBackend):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_cfg.get("model_name", "gpt-4o-mini")

    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        if attempt == 5:
            raise
        time.sleep(2 ** attempt / 2.0)  # retry with backoff
