from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(task, predictions, references, labels=None):
    if task == "classification":
        return {
            "accuracy": accuracy_score(references, predictions),
            "f1_macro": f1_score(references, predictions, average="macro", zero_division=0),
        }
    return {"n_predictions": len(predictions)}
