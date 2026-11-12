import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.batch_analysis import classify_csv, read_message_csv
from src.database import (
    clear_history,
    delete_prediction,
    get_history,
    save_prediction,
)
from src.evaluation import evaluate_test_models, load_comparison
from src.explainability import explain_message
from src.prediction import MODEL_FILES, load_models, majority_prediction, predict_models
from src.robustness import message_variations, test_robustness
from src.suspicious_patterns import analyze_suspicious_patterns


ROOT = Path(__file__).resolve().parents[1]


class MessageGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = load_models(ROOT / "models")

    def test_all_saved_models_load_and_predict(self):
        self.assertEqual(set(self.models), set(MODEL_FILES))
        for experiment in MODEL_FILES:
            self.assertEqual(
                set(self.models[experiment]), set(MODEL_FILES[experiment])
            )
            labels = predict_models(
                self.models[experiment],
                "Congratulations, you have won a free prize!",
            )
            self.assertEqual(set(labels), set(MODEL_FILES[experiment]))
            self.assertTrue(set(labels.values()).issubset({"Spam", "Ham"}))

    def test_majority_prediction(self):
        self.assertEqual(
            majority_prediction(
                {"Naive Bayes": "Spam", "Logistic Regression": "Spam", "SVM": "Ham"}
            ),
            "Spam",
        )
        with self.assertRaises(ValueError):
            predict_models(self.models["Baseline"], " ")
        self.assertIsNone(
            majority_prediction({"Naive Bayes": "Spam", "SVM": "Ham"})
        )

    def test_explanation_uses_model_feature_evidence(self):
        message = "Congratulations you have won a free prize call now"
        for experiment_models in self.models.values():
            for model in experiment_models.values():
                explanation = explain_message(model, message)
                self.assertTrue(explanation["method"])
                self.assertTrue(explanation["terms"])
                self.assertTrue(
                    all(item["contribution"] > 0 for item in explanation["spam_terms"])
                )
                self.assertTrue(
                    all(item["contribution"] < 0 for item in explanation["ham_terms"])
                )
        with self.assertRaises(ValueError):
            explain_message(self.models["Baseline"]["SVM"], "")

    def test_suspicious_rules_are_separate_and_validated(self):
        findings = analyze_suspicious_patterns(
            "URGENT! Claim your cash prize at https://bit.ly/offer and verify your account"
        )
        self.assertGreaterEqual(len(findings), 4)
        indicators = {finding["indicator"] for finding in findings}
        self.assertIn("Urgency or pressure", indicators)
        self.assertIn("Link or shortened URL", indicators)
        with self.assertRaises(ValueError):
            analyze_suspicious_patterns("")

    def test_history_crud_and_filters(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "history.sqlite3"
            first = save_prediction(
                "claim your prize", "Spam", "SVM", "Balanced", path
            )
            save_prediction("hello friend", "Ham", "Naive Bayes", "Baseline", path)
            self.assertEqual(len(get_history(path)), 2)
            self.assertEqual(len(get_history(path, search="prize", label="Spam")), 1)
            self.assertEqual(len(get_history(path, model="Naive Bayes")), 1)
            self.assertTrue(delete_prediction(first, path))
            self.assertEqual(len(get_history(path)), 1)
            self.assertEqual(clear_history(path), 1)
            self.assertTrue(get_history(path).empty)
            with self.assertRaises(ValueError):
                save_prediction("", "Ham", "SVM", "Baseline", path)

    def test_batch_csv_classification_and_export(self):
        frame = read_message_csv(
            b"message,source\nCongratulations win a prize,one\n,empty\n"
        )
        results = classify_csv(
            frame,
            "message",
            self.models["Balanced"],
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results.loc[1, "Prediction"], "Not classified")
        self.assertIn("Model disagreement", results.columns)
        csv_export = results.to_csv(index=False).encode("utf-8")
        self.assertEqual(len(pd.read_csv(pd.io.common.BytesIO(csv_export))), 2)
        with self.assertRaises(ValueError):
            read_message_csv(b"")
        with self.assertRaises(ValueError):
            read_message_csv(b"\xff")
        with self.assertRaises(ValueError):
            read_message_csv(b"message,label\nhello,ham,extra\n")
        with self.assertRaises(ValueError):
            classify_csv(frame, "missing", self.models["Baseline"])

    def test_evaluation_uses_reported_metrics_and_test_data(self):
        comparison = load_comparison(ROOT / "outputs" / "final_model_comparison.csv")
        balanced = comparison[
            (comparison["Experiment"] == "Balanced")
            & (comparison["Dataset"] == "Test")
        ]
        self.assertEqual(balanced.loc[balanced["F1 Score"].idxmax(), "Model"], "SVM")
        evaluation = evaluate_test_models(
            self.models["Balanced"], ROOT / "dataset" / "test.csv"
        )
        self.assertEqual(set(evaluation), set(MODEL_FILES["Balanced"]))
        for result in evaluation.values():
            self.assertEqual(result["confusion_matrix"].shape, (2, 2))
            self.assertIn("roc", result)

    def test_robustness_transformations(self):
        message = "Congratulations! You won a free prize."
        variations = message_variations(message)
        self.assertEqual(len(variations), 7)
        self.assertNotEqual(variations["Uppercase"], message)
        result = test_robustness(self.models["Balanced"]["SVM"], message)
        self.assertEqual(len(result), 7)
        self.assertTrue(result.loc[0, "Matches original"])


if __name__ == "__main__":
    unittest.main()
