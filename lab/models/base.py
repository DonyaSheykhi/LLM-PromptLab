class BaseBackend:
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0) -> str:
        raise NotImplementedError
