import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# Create output directories
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load datasets
train_df = pd.read_csv(
    "dataset/train.csv",
    keep_default_na=False
)

test_df = pd.read_csv(
    "dataset/test.csv",
    keep_default_na=False
)

X_train = train_df["message"].astype(str)
y_train = train_df["label"].astype(int)

X_test = test_df["message"].astype(str)
y_test = test_df["label"].astype(int)

# Define three ML models
models = {
    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "SVM": LinearSVC(
        random_state=42
    )
}

results = []
trained_models = {}
roc_results = {}

# Train and evaluate models
for name, classifier in models.items():

    print(f"\n{'='*50}")
    print(f"TRAINING: {name}")
    print('='*50)

    # Create pipeline
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                min_df=2
            )
        ),
        ("classifier", classifier)
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Predictions
    train_predictions = pipeline.predict(X_train)
    test_predictions = pipeline.predict(X_test)

    # Evaluate training and testing performance
    for split_name, actual, predicted in [
        ("Train", y_train, train_predictions),
        ("Test", y_test, test_predictions)
    ]:

        accuracy = accuracy_score(actual, predicted)

        precision = precision_score(
            actual, predicted, zero_division=0
        )

        recall = recall_score(
            actual, predicted, zero_division=0
        )

        f1 = f1_score(
            actual, predicted, zero_division=0
        )

        results.append({
            "Model": name,
            "Dataset": split_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        print(f"\n{split_name} Performance:")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

    # Detailed test classification report
    print("\nTest Classification Report:")
    print(classification_report(
        y_test,
        test_predictions,
        target_names=["Ham", "Spam"],
        zero_division=0
    ))

    # Save confusion matrix
    cm = confusion_matrix(y_test, test_predictions)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"]
    )

    plt.title(f"{name} - Test Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(
        f"outputs/{filename}_confusion_matrix.png",
        dpi=300
    )
    plt.close()

    # ROC curve
    if hasattr(pipeline, "predict_proba"):
        scores = pipeline.predict_proba(X_test)[:, 1]
    else:
        scores = pipeline.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, scores)
    auc = roc_auc_score(y_test, scores)

    roc_results[name] = (fpr, tpr, auc)

    print(f"ROC-AUC: {auc:.4f}")

    # Save trained model
    joblib.dump(
        pipeline,
        f"models/{filename}.joblib"
    )

    trained_models[name] = pipeline


# Save comparison results
results_df = pd.DataFrame(results)

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

print("\n--- MODEL COMPARISON ---")
print(results_df.to_string(index=False))


# Plot model comparison (test results)
test_results = results_df[
    results_df["Dataset"] == "Test"
]

test_results.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
].plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("ML Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    "outputs/model_comparison.png",
    dpi=300
)
plt.close()


# Plot ROC curves
plt.figure(figsize=(8, 6))

for name, (fpr, tpr, auc) in roc_results.items():
    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/roc_comparison.png",
    dpi=300
)
plt.close()

print("\nAll models trained successfully!")
print("Models saved in models/")
print("Evaluation graphs saved in outputs/")