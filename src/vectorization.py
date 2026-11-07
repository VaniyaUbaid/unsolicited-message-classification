import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load training and testing datasets
train_df = pd.read_csv(
    "dataset/train.csv",
    keep_default_na=False
)

test_df = pd.read_csv(
    "dataset/test.csv",
    keep_default_na=False
)

X_train = train_df["message"]
X_test = test_df["message"]

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    min_df=2
)

# Fit only on training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform testing data using the same vocabulary
X_test_tfidf = tfidf.transform(X_test)

# Display results
print("--- TF-IDF VECTORIZATION ---")

print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))

print("\nTraining TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)

print("\nVocabulary size:", len(tfidf.vocabulary_))

print("\nSample vocabulary:")
print(list(tfidf.vocabulary_.keys())[:20])

print("\nVectorization completed successfully!")