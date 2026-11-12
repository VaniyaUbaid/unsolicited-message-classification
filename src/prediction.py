from pathlib import Path
from typing import Any

import joblib


MODEL_FILES = {
    "Baseline": {
        "Naive Bayes": "naive_bayes.joblib",
        "Logistic Regression": "logistic_regression.joblib",
        "SVM": "svm.joblib",
    },
    "Balanced": {
        "Naive Bayes": "naive_bayes_balanced.joblib",
        "Logistic Regression": "logistic_regression_balanced.joblib",
        "SVM": "svm_balanced.joblib",
    },
}


def load_models(model_dir: Path) -> dict[str, dict[str, Any]]:
    """Load available saved pipelines without fitting or replacing them."""
    loaded: dict[str, dict[str, Any]] = {}
    for experiment, model_files in MODEL_FILES.items():
        loaded[experiment] = {}
        for name, filename in model_files.items():
            path = model_dir / filename
            if path.is_file():
                loaded[experiment][name] = joblib.load(path)
    return loaded


def label_name(label: Any) -> str:
    value = int(label)
    if value not in (0, 1):
        raise ValueError(f"Unsupported model label: {value}")
    return "Spam" if value == 1 else "Ham"


def predict_message(model: Any, message: str) -> int:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Enter a non-empty message to classify.")
    return int(model.predict([message])[0])


def predict_models(
    models: dict[str, Any], message: str
) -> dict[str, str]:
    if not models:
        raise ValueError("No models are available for this experiment.")
    return {
        name: label_name(predict_message(model, message))
        for name, model in models.items()
    }


def majority_prediction(predictions: dict[str, str]) -> str | None:
    if not predictions:
        raise ValueError("At least one prediction is required.")
    spam_votes = sum(label == "Spam" for label in predictions.values())
    if spam_votes * 2 == len(predictions):
        return None
    return "Spam" if spam_votes > len(predictions) / 2 else "Ham"
