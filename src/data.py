"""Loading, splitting and decoding the prepared UPI transaction dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATASET, LABEL_COLUMN, SEED, TIME_COLUMN


def load_data(path: Path = DATASET) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (features, labels, timestamps) with the label and time columns removed
    from the feature frame."""
    if not path.exists():
        sys.exit(f"Dataset not found: {path}\n"
                 f"Run the preprocessing notebook first (notebooks/01_preprocessing.ipynb).")
    df = pd.read_csv(path, parse_dates=[TIME_COLUMN])
    ts = df.pop(TIME_COLUMN)
    y = df.pop(LABEL_COLUMN)
    return df, y, ts


def stratified_split(X, y, ts):
    """70 / 15 / 15 stratified random split - keeps the fraud rate identical in every
    partition, which matters when the positive class has only 480 rows."""
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y.iloc[tmp], random_state=SEED)
    return tr, va, te


def chronological_split(X, y, ts):
    """70 / 15 / 15 split by transaction time - the deployment-realistic setting: the
    model only ever sees transactions older than the ones it scores."""
    order = np.argsort(ts.values, kind="stable")
    n = len(order)
    return order[: int(0.70 * n)], order[int(0.70 * n) : int(0.85 * n)], order[int(0.85 * n) :]


def get_splitter(split: str):
    return stratified_split if split == "stratified" else chronological_split


# ------------------------------------------------------------------------- decoding

# Preprocessing used pd.get_dummies(drop_first=True), so each group is missing its
# first category - an all-zero group means that dropped reference value.
ONEHOT_GROUPS = [
    ("Type", "transaction type_", "P2M"),
    ("Merchant", "merchant_category_", "Bills"),
    ("Device", "device_type_", "Android"),
    ("Network", "network_type_", "3G"),
    ("Sender bank", "sender_bank_", "Axis"),
    ("Sender state", "sender_state_", "Andhra Pradesh"),
]
BINARY_FLAGS = ("is_weekend", "is_night_transaction", "is_high_amount")


def decode(row, columns) -> dict:
    """Turn one one-hot encoded row back into readable attributes for display."""
    out = {"Amount": f"Rs {row['amount (INR)']:,.0f}",
           "Hour": f"{int(row['hour_of_day']):02d}:00"}
    for label, prefix, reference in ONEHOT_GROUPS:
        hot = [c for c in columns if c.startswith(prefix) and row[c] == 1]
        out[label] = hot[0][len(prefix):] if hot else f"{reference} (ref)"
    flags = [f for f in BINARY_FLAGS if row[f] == 1]
    out["Flags"] = ", ".join(f.replace("is_", "").replace("_", " ") for f in flags) or "none"
    return out


def load_score_file(path: Path, columns) -> tuple[pd.DataFrame, pd.Series | None]:
    """Read an external CSV of transactions to score. Must carry the same feature
    columns as the training data; timestamp and fraud_flag are optional."""
    if not path.exists():
        sys.exit(f"No such file: {path}")
    df = pd.read_csv(path)
    truth = df.pop(LABEL_COLUMN) if LABEL_COLUMN in df.columns else None
    df = df.drop(columns=[TIME_COLUMN], errors="ignore")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        sys.exit(f"{path.name} is missing {len(missing)} feature column(s), "
                 f"e.g. {missing[:3]}\n"
                 f"Generate a valid file with: python src/make_test_samples.py")
    return df[list(columns)], truth
