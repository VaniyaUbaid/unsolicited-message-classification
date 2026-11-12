from pathlib import Path
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator

import pandas as pd


HISTORY_COLUMNS = [
    "id",
    "message",
    "predicted_label",
    "selected_model",
    "experiment_type",
    "timestamp",
]


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            predicted_label TEXT NOT NULL CHECK (predicted_label IN ('Spam', 'Ham')),
            selected_model TEXT NOT NULL,
            experiment_type TEXT NOT NULL,
            timestamp TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
        )
        """
    )
    return connection


@contextmanager
def _connection(db_path: Path) -> Iterator[sqlite3.Connection]:
    connection = _connect(db_path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def save_prediction(
    message: str,
    predicted_label: str,
    selected_model: str,
    experiment_type: str,
    db_path: Path,
) -> int:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("A non-empty message is required for history.")
    if predicted_label not in {"Spam", "Ham"}:
        raise ValueError("Predicted label must be Spam or Ham.")
    with _connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions
                (message, predicted_label, selected_model, experiment_type)
            VALUES (?, ?, ?, ?)
            """,
            (message, predicted_label, selected_model, experiment_type),
        )
        return int(cursor.lastrowid)


def get_history(
    db_path: Path,
    search: str = "",
    label: str = "All",
    model: str = "All",
) -> pd.DataFrame:
    conditions = []
    parameters: list[Any] = []
    if search:
        conditions.append("message LIKE ?")
        parameters.append(f"%{search}%")
    if label in {"Spam", "Ham"}:
        conditions.append("predicted_label = ?")
        parameters.append(label)
    if model != "All":
        conditions.append("selected_model = ?")
        parameters.append(model)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with _connection(db_path) as connection:
        rows = connection.execute(
            f"SELECT {', '.join(HISTORY_COLUMNS)} FROM predictions "
            f"{where} ORDER BY id DESC",
            parameters,
        ).fetchall()
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def delete_prediction(record_id: int, db_path: Path) -> bool:
    with _connection(db_path) as connection:
        cursor = connection.execute(
            "DELETE FROM predictions WHERE id = ?", (int(record_id),)
        )
        return cursor.rowcount > 0


def clear_history(db_path: Path) -> int:
    with _connection(db_path) as connection:
        cursor = connection.execute("DELETE FROM predictions")
        return int(cursor.rowcount)
