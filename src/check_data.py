
import pandas as pd

train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

print("--- TRAIN DATA ---")
print(train_df.isnull().sum())

print("\n--- TEST DATA ---")
print(test_df.isnull().sum())

print("\nTraining message data types:")
print(train_df["message"].apply(type).value_counts())

print("\nRows with missing training messages:")
print(train_df[train_df["message"].isna()])

print("\nRows with missing testing messages:")
print(test_df[test_df["message"].isna()])
