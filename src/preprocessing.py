import pandas as pd
import re
import os

# Load original dataset
df = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("Original dataset size:", len(df))

# Remove exact duplicate records
df = df.drop_duplicates().reset_index(drop=True)

print("After removing duplicates:", len(df))

# Text cleaning function
def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation but preserve numbers
    text = re.sub(r"[^\w\s]", " ", text)

    # Normalize extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Apply text cleaning
df["cleaned_message"] = df["message"].apply(clean_text)

# Encode labels: ham = 0, spam = 1
df["label_encoded"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Check for invalid labels
assert df["label_encoded"].notna().all()

# Save processed dataset
os.makedirs("dataset", exist_ok=True)

df.to_csv(
    "dataset/processed_sms.csv",
    index=False
)

print("\n--- PREPROCESSING SUMMARY ---")
print("Final dataset shape:", df.shape)

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nSample processed messages:")
print(df[["label", "message", "cleaned_message"]].head())

print("\nProcessed dataset saved successfully!")