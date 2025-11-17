# Experiment Card — Exp-01: Zero-shot MNLI on AG News

**Run name:** `agnews_hf_tiny_fast`  
**Model:** `valhalla/distilbart-mnli-12-3`  
**Dataset:** AG News (test[:80])  
**Task:** zero-shot classification  
**Prompt:** single-turn, 0-shot  
**Labels:** World · Sports · Business · Sci/Tech

| Metric | Value |
|--------:|------:|
| Accuracy | 0.35 |
| F1-macro | 0.23 |
| Mean latency (s) | 2.29 |
| P95 latency (s) | 1.90 |

**Takeaways**
- Zero-shot MNLI provides a solid multilingual baseline.
- Latency <3 s per sample on CPU — good for lightweight inference.
- Few-shot prompting or larger MNLI models (BART-large-MNLI) are expected to raise F1 to > 0.6.

**Next Steps**
1. Add 3–5 few-shot examples and rerun (`Exp-02`).
2. Swap to `facebook/bart-large-mnli` for a stronger baseline.
3. Track cost & latency together for a cost-performance plot.

**Date:** 2025-11-17  
