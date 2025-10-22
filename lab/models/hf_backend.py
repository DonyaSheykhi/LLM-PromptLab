from transformers import pipeline
from .base import BaseBackend

class HFBackend(BaseBackend):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        model_name = model_cfg.get("model_name", "mistralai/Mixtral-8x7B-Instruct")
        self.pipe = pipeline("text-generation", model=model_name, trust_remote_code=True, device_map="auto")
    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0) -> str:
        out = self.pipe(prompt, max_new_tokens=max_tokens, do_sample=(temperature>0), temperature=temperature)
        txt = out[0]["generated_text"]
        return txt[len(prompt):].strip() if txt.startswith(prompt) else txt.strip()
