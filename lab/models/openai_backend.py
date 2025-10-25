import os
from openai import OpenAI
from .base import BaseBackend

class OpenAIBackend(BaseBackend):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_cfg.get("model_name", "gpt-4o-mini")
