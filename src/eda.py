import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Load dataset
df = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Create output directory if needed
import os
os.makedirs("outputs", exist_ok=True)

sns.set_theme(style="whitegrid")

# 1. Class Distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="label", hue="label",
              palette={"ham": "seagreen", "spam": "tomato"},
              legend=False)
plt.title("SMS Class Distribution")
plt.xlabel("Message Category")
plt.ylabel("Number of Messages")
plt.savefig("outputs/class_distribution.png", dpi=300)
plt.show()

# 2. Message Length Distribution
df["message_length"] = df["message"].apply(len)

plt.figure(figsize=(9, 5))
sns.histplot(
    data=df,
    x="message_length",
    hue="label",
    bins=40,
    element="step",
    stat="density",
    common_norm=False
)
plt.title("Message Length Distribution")
plt.xlabel("Message Length (Characters)")
plt.ylabel("Density")
plt.savefig("outputs/message_length.png", dpi=300)
plt.show()

# 3 & 4. Most Frequent Words
def get_word_frequency(messages):
    words = []

    for message in messages:
        tokens = re.findall(r"\b[a-zA-Z]+\b", message.lower())
        words.extend(tokens)

    return Counter(words).most_common(15)


for category in ["spam", "ham"]:
    word_counts = get_word_frequency(
        df[df["label"] == category]["message"]
    )

    words, counts = zip(*word_counts)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(counts), y=list(words), color="steelblue")
    plt.title(f"Most Frequent Words in {category.upper()} Messages")
    plt.xlabel("Frequency")
    plt.ylabel("Words")
    plt.tight_layout()

    plt.savefig(f"outputs/{category}_word_frequency.png", dpi=300)
    plt.show()

print("EDA completed successfully!")