import pandas as pd

baseline = pd.read_csv("outputs/model_comparison.csv")
balanced = pd.read_csv("outputs/balanced_model_comparison.csv")

# Keep only baseline test results for a fair comparison
baseline = baseline[baseline["Dataset"] == "Test"].copy()

baseline["Experiment"] = "Baseline"
balanced["Experiment"] = "Balanced"
balanced["Dataset"] = "Test"

comparison = pd.concat(
    [baseline, balanced],
    ignore_index=True
)

comparison = comparison[
    ["Model", "Dataset", "Experiment",
     "Accuracy", "Precision", "Recall", "F1 Score"]
]

comparison.to_csv(
    "outputs/final_model_comparison.csv",
    index=False
)

print("\nFinal Test Model Comparison")
print(comparison.to_string(index=False))