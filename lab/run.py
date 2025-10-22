import argparse, yaml, json, pathlib
from lab.data import load_dataset_split
from lab.prompts import build_prompt
from lab.models.openai_backend import OpenAIBackend
from lab.models.hf_backend import HFBackend
from lab.eval.metrics import compute_metrics
from lab.eval.reporting import write_summary

BACKENDS = {"openai": OpenAIBackend, "hf": HFBackend}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_dir = pathlib.Path("runs") / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset_split(cfg["dataset"])
    backend = BACKENDS[cfg["model"]["backend"]](cfg["model"])
    labels = cfg.get("labels")
    predictions, references = [], []

    for ex in ds:
        prompt = build_prompt(cfg["prompt"], ex, labels=labels, dataset_cfg=cfg["dataset"])
        out = backend.generate(prompt, max_tokens=cfg["model"].get("max_tokens", 64),
                               temperature=cfg["model"].get("temperature", 0.0))
        predictions.append(out.strip())
        if "label_field" in cfg["dataset"]:
            labelf = cfg["dataset"]["label_field"]
            ref = ex.get(labelf)
            if isinstance(ref, int) and labels:
                ref = labels[ref]
            references.append(str(ref))

    with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for pred, ex in zip(predictions, ds):
            f.write(json.dumps({"prediction": pred, "input": ex}) + "\\n")

    metrics = {}
    if references:
        metrics = compute_metrics(task=cfg["task"], predictions=predictions, references=references, labels=labels)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    write_summary(run_dir, cfg, metrics)

if __name__ == "__main__":
    main()
