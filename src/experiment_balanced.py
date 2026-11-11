import pandas as pd
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load training and testing data
train_df = pd.read_csv("dataset/train.csv", keep_default_na=False)
print("Training dataset columns:")
print(train_df.columns.tolist())

print("\nTraining label values:")
print(train_df["label"].value_counts())

print("\nSample training data:")
print(train_df.head())

test_df = pd.read_csv("dataset/test.csv", keep_default_na=False)

# Separate majority and minority classes
ham = train_df[train_df["label"] == 0]
spam = train_df[train_df["label"] == 1]

# Oversample spam messages in training data
spam_oversampled = resample(
    spam,
    replace=True,
    n_samples=len(ham),
    random_state=42
)

# Combine and shuffle balanced training data
balanced_train = pd.concat([ham, spam_oversampled])
balanced_train = balanced_train.sample(frac=1, random_state=42)

print("Original training distribution:")
print(train_df["label"].value_counts())

print("\nBalanced training distribution:")
print(balanced_train["label"].value_counts())

X_train = balanced_train["message"]
y_train = balanced_train["label"]

X_test = test_df["message"]
y_test = test_df["label"]

# Define models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42
    ),
    "SVM": LinearSVC(random_state=42)
}

results = []

# Train and evaluate each model
for name, classifier in models.items():

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            min_df=2
        )),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, pos_label=1
    )
    recall = recall_score(
        y_test, predictions, pos_label=1
    )
    f1 = f1_score(
        y_test, predictions, pos_label=1
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(f"\n{name}")
    print(classification_report(y_test, predictions))

    import joblib
    filename = name.lower().replace(" ", "_")
    joblib.dump(pipeline, f"models/{filename}_balanced.joblib")

# Save comparison results
results_df = pd.DataFrame(results)
results_df.to_csv("outputs/balanced_model_comparison.csv", index=False)

print("\nBalanced Model Comparison:")
print(results_df.to_string(index=False))