import pathlib

def write_summary(run_dir, cfg, metrics: dict):
    p = pathlib.Path(run_dir) / "summary.md"
    lines = [f"# {cfg['run_name']}"] + [f"- {k}: {v}" for k, v in metrics.items()]
    p.write_text("\n".join(lines), encoding="utf-8")
