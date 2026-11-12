from typing import Any

import pandas as pd

from src.prediction import label_name, predict_message


def message_variations(message: str) -> dict[str, str]:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Enter a non-empty message for robustness testing.")
    compact = " ".join(message.split())
    substitutions = (
        ("congratulations", "congratulashuns"),
        ("winner", "winer"),
        ("claim", "cla im"),
        ("free", "fre"),
    )
    spelling = message
    for original, variation in substitutions:
        spelling = spelling.replace(original, variation)
        spelling = spelling.replace(original.title(), variation.title())

    return {
        "Original": message,
        "Uppercase": message.upper(),
        "Lowercase": message.lower(),
        "Extra spaces": f"  {compact}   ".replace(" ", "  ", 1),
        "Repeated punctuation": message.rstrip("!?.") + "!!!",
        "Spelling variations": spelling,
        "Character substitution (e to 3)": message.replace("e", "3").replace(
            "E", "3"
        ),
    }


def test_robustness(model: Any, message: str) -> pd.DataFrame:
    variations = message_variations(message)
    baseline = label_name(predict_message(model, variations["Original"]))
    rows = []
    for name, text in variations.items():
        prediction = label_name(predict_message(model, text))
        rows.append(
            {
                "Variation": name,
                "Prediction": prediction,
                "Matches original": prediction == baseline,
            }
        )
    return pd.DataFrame(rows)
