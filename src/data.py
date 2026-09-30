"""Loading, splitting and decoding the prepared UPI transaction dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATASET, ID_COLUMN, LABEL_COLUMN, LEAKY_PREFIXES, SEED, TIME_COLUMN


def load_data(path: Path = DATASET) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (features, labels, timestamps) with the label and time columns removed
    from the feature frame."""
    if not path.exists():
        sys.exit(f"Dataset not found: {path}\n"
                 f"Run the preprocessing notebook first (notebooks/01_preprocessing.ipynb).")
    df = pd.read_csv(path, parse_dates=[TIME_COLUMN])
    ts = df.pop(TIME_COLUMN)
    y = df.pop(LABEL_COLUMN)
    leaky = [c for c in df.columns if c.startswith(LEAKY_PREFIXES)]
    df = df.drop(columns=[ID_COLUMN, *leaky], errors="ignore")   # identifier + label leakage
    return df, y, ts


def stratified_split(X, y, ts):
    """70 / 15 / 15 stratified random split - keeps the fraud rate identical in every
    partition, which matters when the positive class has few rows."""
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

# Preprocessing one-hot encoded every category (no reference level dropped), so each
# group always has exactly one active column.
ONEHOT_GROUPS = [
    ("Type", "transaction type_"),
    ("Merchant", "merchant_category_"),
    ("Device", "device_type_"),
    ("Network", "network_type_"),
    ("Sender bank", "sender_bank_"),
    ("Sender state", "sender_state_"),
]
BINARY_FLAGS = ("is_weekend", "is_night_transaction", "is_high_amount")


def decode(row, columns) -> dict:
    """Turn one one-hot encoded row back into readable attributes for display."""
    out = {"Amount": f"Rs {row['amount (INR)']:,.0f}",
           "Hour": f"{int(row['hour_of_day']):02d}:00"}
    for label, prefix in ONEHOT_GROUPS:
        hot = [c for c in columns if c.startswith(prefix) and row[c] == 1]
        out[label] = hot[0][len(prefix):] if hot else "unknown"
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
