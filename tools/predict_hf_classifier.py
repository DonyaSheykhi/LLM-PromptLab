# tools/predict_hf_classifier.py
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, numpy as np

LABELS = ["World","Sports","Business","Sci/Tech"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--text", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    inputs = tok(args.text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = mdl(**inputs).logits
    probs = torch.softmax(logits, dim=-1).numpy()[0]
    pred = int(np.argmax(probs))
    print(f"Prediction: {LABELS[pred]} | probs={probs.round(3).tolist()}")

if __name__ == "__main__":
    main()
