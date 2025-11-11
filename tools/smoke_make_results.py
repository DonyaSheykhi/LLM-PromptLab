# tools/smoke_make_results.py
from pathlib import Path
run_dir = Path("runs/agnews_hf_tiny_fast")
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir/"predictions.jsonl").write_text("", encoding="utf-8")
(run_dir/"metrics.json").write_text('{"accuracy": 0.87, "f1_macro": 0.86}', encoding="utf-8")
(run_dir/"summary.md").write_text("""# agnews_hf_tiny_fast

- accuracy: 0.87
- f1_macro: 0.86
- samples: 40
""", encoding="utf-8")
print("Wrote demo results to", run_dir)
