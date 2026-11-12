import csv
from io import BytesIO, StringIO
from typing import Any

import pandas as pd

from src.prediction import label_name, majority_prediction, predict_message


def read_message_csv(data: bytes) -> pd.DataFrame:
    if not data:
        raise ValueError("The uploaded CSV is empty.")
    try:
        decoded = data.decode("utf-8-sig")
        reader = csv.reader(StringIO(decoded))
        headers = next(reader, None)
        if not headers or not any(header.strip() for header in headers):
            raise ValueError("The uploaded CSV has no header row.")
        expected_fields = len(headers)
        for row in reader:
            if row and len(row) != expected_fields:
                raise ValueError(
                    f"Malformed CSV row {reader.line_num}: expected "
                    f"{expected_fields} fields but found {len(row)}."
                )
        frame = pd.read_csv(BytesIO(data), keep_default_na=False)
    except (csv.Error, pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(
            ("The uploaded CSV", "Malformed CSV")
        ):
            raise
        raise ValueError(f"Could not read the uploaded CSV: {error}") from error
    if frame.empty or len(frame.columns) == 0:
        raise ValueError("The uploaded CSV has no rows or columns.")
    return frame


def classify_messages(
    messages: list[Any],
    models: dict[str, Any],
) -> pd.DataFrame:
    if not models:
        raise ValueError("Select at least one available model.")

    rows = []
    for message in messages:
        text = message if isinstance(message, str) else ""
        predictions = {}
        if text.strip():
            predictions = {
                name: label_name(predict_message(model, text))
                for name, model in models.items()
            }

        if len(models) == 1:
            label = next(iter(predictions.values()), "Not classified")
            rows.append(
                {
                    "Prediction": label,
                    "Model disagreement": "",
                    "Majority suggestion": "",
                }
            )
            continue

        labels = list(predictions.values())
        majority = majority_prediction(predictions) if predictions else None
        rows.append(
            {
                **{f"{name} prediction": label for name, label in predictions.items()},
                "Prediction": (
                    majority
                    if majority is not None
                    else ("Disagreement" if predictions else "Not classified")
                ),
                "Model disagreement": (
                    len(set(labels)) > 1 if predictions else ""
                ),
                "Majority suggestion": majority if majority is not None else "",
            }
        )
    return pd.DataFrame(rows)


def classify_csv(
    frame: pd.DataFrame,
    message_column: str,
    models: dict[str, Any],
) -> pd.DataFrame:
    if message_column not in frame.columns:
        raise ValueError("Select a message column present in the CSV.")
    predictions = classify_messages(frame[message_column].tolist(), models)
    return pd.concat([frame.reset_index(drop=True), predictions], axis=1)
