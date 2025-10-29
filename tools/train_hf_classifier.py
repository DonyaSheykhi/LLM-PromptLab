# tools/train_hf_classifier.py
import argparse, os, numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from sklearn.metrics import accuracy_score, f1_score

LABELS = ["World","Sports","Business","Sci/Tech"]

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--out", default="runs/agnews_finetuned")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    ds = load_dataset("ag_news")
    # use title + description if present; HF ag_news has "text" + "label"
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def tok(ex):
        return tokenizer(ex["text"], truncation=True, max_length=args.max_len)

    ds_tok = ds.map(tok, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer)

    id2label = {i: L for i, L in enumerate(LABELS)}
    label2id = {L: i for i, L in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    args_out = os.path.abspath(args.out)
    os.makedirs(args_out, exist_ok=True)

    targs = TrainingArguments(
        output_dir=args_out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["test"],  # simple: evaluate on test after each epoch
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    print("Final eval:", eval_metrics)

    # save HF model + tokenizer
    trainer.save_model(args_out)
    tokenizer.save_pretrained(args_out)

    # write a tiny summary
    with open(os.path.join(args_out, "summary.txt"), "w", encoding="utf-8") as f:
        for k, v in eval_metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved model + tokenizer to: {args_out}")

if __name__ == "__main__":
    main()
