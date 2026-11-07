import pandas as pd
from sklearn.model_selection import train_test_split

# Load preprocessed dataset
df = pd.read_csv("dataset/processed_sms.csv")

# Separate features and target
X = df["cleaned_message"]
y = df["label_encoded"]

# Stratified 80:20 split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create training and testing DataFrames
train_df = pd.DataFrame({
    "message": X_train,
    "label": y_train
})

test_df = pd.DataFrame({
    "message": X_test,
    "label": y_test
})

# Save the datasets
train_df.to_csv("dataset/train.csv", index=False)
test_df.to_csv("dataset/test.csv", index=False)

# Display results
print("--- TRAIN TEST SPLIT ---")

print("Total records:", len(df))
print("Training records:", len(train_df))
print("Testing records:", len(test_df))

print("\nTraining class distribution:")
print(train_df["label"].value_counts())

print("\nTesting class distribution:")
print(test_df["label"].value_counts())

print("\nTraining percentage:", len(train_df) / len(df) * 100)
print("Testing percentage:", len(test_df) / len(df) * 100)

print("\nTrain-test split completed successfully!")