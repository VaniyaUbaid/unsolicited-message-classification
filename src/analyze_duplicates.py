import pandas as pd

df = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Find exact duplicate rows
print("Exact duplicate rows:", df.duplicated().sum())

# Find messages appearing more than once
duplicate_messages = df[
    df.duplicated(subset=["message"], keep=False)
]

print(
    "\nTotal rows belonging to repeated messages:",
    len(duplicate_messages)
)

# Check whether identical messages have conflicting labels
label_counts = df.groupby("message")["label"].nunique()

conflicting_messages = label_counts[label_counts > 1]

print(
    "\nMessages having conflicting labels:",
    len(conflicting_messages)
)

if len(conflicting_messages) > 0:
    print("\nExamples of conflicting messages:")
    for message in conflicting_messages.index[:5]:
        print("\nMessage:", message)
        print(df[df["message"] == message])