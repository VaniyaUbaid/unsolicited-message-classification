from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


METRIC_COLUMNS = [
    "Model",
    "Dataset",
    "Experiment",
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
]


def load_comparison(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation results were not found: {path}")
    comparison = pd.read_csv(path)
    missing = set(METRIC_COLUMNS) - set(comparison.columns)
    if missing:
        raise ValueError(
            "Evaluation results are missing columns: "
            + ", ".join(sorted(missing))
        )
    for metric in ("Accuracy", "Precision", "Recall", "F1 Score"):
        comparison[metric] = pd.to_numeric(comparison[metric], errors="raise")
    return comparison


def evaluate_test_models(
    models: dict[str, Any], test_path: Path
) -> dict[str, dict[str, Any]]:
    test_data = pd.read_csv(test_path, keep_default_na=False)
    if not {"message", "label"}.issubset(test_data.columns):
        raise ValueError("Test data must contain 'message' and 'label' columns.")
    actual = test_data["label"].astype(int)
    if not set(actual.unique()).issubset({0, 1}):
        raise ValueError("Test labels must use 0=Ham and 1=Spam.")

    results = {}
    for name, model in models.items():
        predicted = model.predict(test_data["message"].astype(str))
        item: dict[str, Any] = {
            "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]),
        }
        classifier = model.named_steps["classifier"]
        if hasattr(classifier, "predict_proba"):
            scores = model.predict_proba(test_data["message"].astype(str))[:, 1]
        elif hasattr(classifier, "decision_function"):
            scores = model.decision_function(test_data["message"].astype(str))
            if list(classifier.classes_).index(1) == 0:
                scores = -scores
        else:
            scores = None
        if scores is not None:
            false_positive_rate, true_positive_rate, _ = roc_curve(actual, scores)
            item["roc"] = {
                "fpr": false_positive_rate,
                "tpr": true_positive_rate,
                "auc": float(roc_auc_score(actual, scores)),
            }
        results[name] = item
    return results
