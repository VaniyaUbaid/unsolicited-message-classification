
import pandas as pd

# Load the SMS Spam Collection dataset
df = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

# Display basic information
print("\n--- DATASET OVERVIEW ---")
print("Shape:", df.shape)

print("\nFirst 5 records:")
print(df.head())

print("\nDataset information:")
df.info()

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check duplicate records
print("\nDuplicate records:", df.duplicated().sum())

# Check class distribution
print("\nClass distribution:")
print(df["label"].value_counts())

print("\nClass distribution (%):")
print(df["label"].value_counts(normalize=True) * 100)
