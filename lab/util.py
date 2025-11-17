# lab/util.py
import re
from typing import List

def map_to_choice(text: str, labels: List[str]) -> str:
    t = (text or "").lower()

    # exact label hit
    for lab in labels:
        if re.search(rf"\b{re.escape(lab.lower())}\b", t):
            return lab

    # simple stance heuristics
    if any(k in t for k in ["support", "praise", "favor", "approve", "endorse", "pro "]):
        return "pro"
    if any(k in t for k in ["oppose", "critic", "against", "condemn", "protest"]):
        return "against"

    return "neutral"
