# lab/cost.py
from typing import Optional, Dict

# $ per 1M tokens (example; adjust to your provider’s current pricing)
PRICES = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},  # $/1M tokens
    "gpt-4o":      {"input": 5.00, "output": 15.0},
}

def cost_usd(model: str, prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> float:
    p = PRICES.get(model, {"input": 0.0, "output": 0.0})
    pt = (prompt_tokens or 0) / 1_000_000 * p["input"]
    ct = (completion_tokens or 0) / 1_000_000 * p["output"]
    return round(pt + ct, 6)
