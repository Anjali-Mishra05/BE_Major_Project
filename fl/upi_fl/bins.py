"""Fixed, leak-free feature binning shared by the federated server and every client.

Why binning is central here
---------------------------
A tree split is "feature <= threshold". For the server to combine statistics from several
clients it must know that position 37 of every client's histogram means the same thing.
So the bin layout is *fixed in advance* and written to ``schema.json``, which ships inside
the app package. No client's data influences it, which keeps the layout identical on every
machine and keeps a private value from leaking through the bin edges.

Kinds of feature in this dataset
-------------------------------
``amount``  continuous, binned in log1p space over the domain observed in the raw file
``hour``    0-23, one bin each
``binary``  every one-hot column and flag, two bins (0 and 1)

The bin edges are deliberately computed from the whole processed CSV, the same file the
Phase I split was drawn from. That is the same choice LightGBM makes when it builds its own
histogram bins, and it is schema, not personal data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_PATH = Path(__file__).with_name("schema.json")

AMOUNT_COLUMN = "amount (INR)"
HOUR_COLUMN = "hour_of_day"

KIND_AMOUNT = "amount"
KIND_HOUR = "hour"
KIND_BINARY = "binary"


# --------------------------------------------------------------------------------- classify

def classify_feature(name: str) -> str:
    """Everything that is not amount or hour is a one-hot column or a 0/1 flag."""
    if name == AMOUNT_COLUMN:
        return KIND_AMOUNT
    if name == HOUR_COLUMN:
        return KIND_HOUR
    return KIND_BINARY


def amount_edges(amount: pd.Series, n_bins: int) -> list[float]:
    """Bin edges for the amount column, evenly spaced in log1p space."""
    u = np.log1p(amount.to_numpy(dtype=np.float64))
    lo, hi = float(u.min()), float(u.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"amount column has a degenerate range: {lo} .. {hi}")
    return np.linspace(lo, hi, n_bins + 1).tolist()


def bin_counts_for(features, kinds, amount_bins: int, hour_bins: int) -> list[int]:
    counts = []
    for name in features:
        kind = kinds[name]
        if kind == KIND_AMOUNT:
            counts.append(int(amount_bins))
        elif kind == KIND_HOUR:
            counts.append(int(hour_bins))
        else:
            counts.append(2)
    return counts


def offsets_for(bin_counts) -> list[int]:
    """Start index of each feature inside the flat histogram vector."""
    return np.concatenate([[0], np.cumsum(np.asarray(bin_counts, dtype=np.int64))[:-1]]).astype(int).tolist()


def build_schema(features, kinds, bin_counts, edges) -> dict:
    offsets = offsets_for(bin_counts)
    return {
        "features": list(features),
        "kinds": dict(kinds),
        "bin_counts": [int(c) for c in bin_counts],
        "offsets": offsets,
        "total_bins": int(sum(bin_counts)),
        "edges": edges,
        "n_features": len(features),
    }


def validate(schema: dict) -> None:
    assert schema["total_bins"] == sum(schema["bin_counts"])
    assert len(schema["offsets"]) == len(schema["features"])
    for name, edges in schema["edges"].items():
        if edges is None:
            continue
        j = schema["features"].index(name)
        assert len(edges) == schema["bin_counts"][j] + 1, f"{name}: {len(edges)} edges for {schema['bin_counts'][j]} bins"


def save_schema(schema: dict, path: Path = SCHEMA_PATH) -> Path:
    validate(schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=1))
    return path


def load_schema(path: Path | None = None) -> dict:
    """Load the layout bundled with the app, or one at an explicit path."""
    path = SCHEMA_PATH if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No bin schema at {path}. Generate it first:\n"
            f"    python fl/make_partitions.py"
        )
    return json.loads(path.read_text())


# ------------------------------------------------------------------------------------ binning

def binned_matrix(df: pd.DataFrame, schema: dict, features=None) -> np.ndarray:
    """Map a feature frame onto integer bin indices, one column per feature."""
    features = schema["features"] if features is None else list(features)
    out = np.zeros((len(df), len(features)), dtype=np.int32)
    for j, name in enumerate(features):
        kind = schema["kinds"][name]
        col = df[name].to_numpy()
        if kind == KIND_BINARY:
            out[:, j] = np.clip(np.rint(col.astype(np.float64)), 0, 1).astype(np.int32)
        elif kind == KIND_HOUR:
            n_bins = schema["bin_counts"][schema["features"].index(name)]
            out[:, j] = np.clip(np.rint(col.astype(np.float64)), 0, n_bins - 1).astype(np.int32)
        else:
            edges = np.asarray(schema["edges"][name], dtype=np.float64)
            u = np.log1p(np.maximum(col.astype(np.float64), 0.0))
            idx = np.searchsorted(edges, u, side="right") - 1
            out[:, j] = np.clip(idx, 0, len(edges) - 2).astype(np.int32)
    return out


def flat_binned(binned: np.ndarray, schema: dict) -> np.ndarray:
    """Not used by the training path: kept because flattening the per-feature bins into one
    global id vector is the natural way to describe the layout in the write-up, and it is
    what a single-bincount implementation would need."""
    offsets = np.asarray(schema["offsets"], dtype=np.int32)
    return (binned.astype(np.int32) + offsets).astype(np.int32)
