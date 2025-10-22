def build_prompt(prompt_cfg, example, labels=None, dataset_cfg=None):
    system = prompt_cfg.get("system", "").strip()
    user_template = prompt_cfg.get("user_template", "").strip()
    text_key = dataset_cfg.get("text_field", "text") if dataset_cfg else "text"
    text = example.get(text_key) or example.get("text", "")
    labels_str = ", ".join(labels) if labels else ""
    user = user_template.format(text=text, labels=labels_str, schema=prompt_cfg.get("schema", []), question=example.get("text", ""))
    return f"System: {system}\n\nUser: {user}" if system else user
