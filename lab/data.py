from datasets import load_dataset

def load_dataset_split(cfg):
    if "name" in cfg:
        ds = load_dataset(cfg["name"], split=cfg.get("split", "test"))
        text_field = cfg.get("text_field") or cfg.get("question_field") or "text"
        label_field = cfg.get("label_field")
        out = []
        for ex in ds:
            item = {"text": ex.get(text_field, ex.get("text", ""))}
            if label_field is not None:
                item[label_field] = ex.get(label_field)
            out.append(item)
        return out
    elif "path" in cfg:
        import json
        out = []
        with open(cfg["path"], "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
        return out
    else:
        raise ValueError("Dataset config must have 'name' or 'path'.")
