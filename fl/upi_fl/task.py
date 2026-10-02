"""Client-side work: hold one bank's shard, turn it into histograms, keep the running score.

This module is deliberately free of Flower imports so ``fl/simulate.py`` and the real
ClientApp share exactly the same code. Whatever the tests prove about the simulation is
therefore true of the networked run.

The two constants below duplicate ``src/config.py``. They are duplicated rather than
imported because a Flower app is packaged on its own and cannot reach outside its own
directory; if they ever drift, ``fl/compare.py`` will not find the label column and fails
loudly.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import bins, gbdt

LABEL_COLUMN = "fraud_flag"
TIME_COLUMN = "timestamp"


@dataclass
class Shard:
    """One client's private view: its rows, binned, plus the score it is carrying."""

    path: str
    binned: np.ndarray          # (n, n_features) int32 bin indices
    y: np.ndarray               # (n,) float64 labels
    raw: np.ndarray             # (n,) float64 current model score, updated tree by tree
    trees_seen: int = 0
    hostname: str = "unknown"

    @property
    def n(self) -> int:
        return int(len(self.y))

    @property
    def n_pos(self) -> int:
        return int((self.y > 0).sum())

    @property
    def n_neg(self) -> int:
        return int((self.y <= 0).sum())


_CACHE: dict[tuple[str, int], Shard] = {}


def load_shard(path, schema: dict, use_cache: bool = True) -> Shard:
    """Read a client's CSV and bin it. Cached, because a run touches this once per client
    but the boosting loop would otherwise re-read it on every message."""
    key = (str(path), id(schema))
    if use_cache and key in _CACHE:
        return _CACHE[key]

    df = pd.read_csv(path)
    if LABEL_COLUMN not in df.columns:
        raise KeyError(f"{path} has no '{LABEL_COLUMN}' column - is it a shard, not raw data?")
    y = df.pop(LABEL_COLUMN).to_numpy(dtype=np.float64)
    df = df.drop(columns=[TIME_COLUMN], errors="ignore")

    features = schema["features"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"{path} is missing {len(missing)} feature column(s), e.g. {missing[:3]}")

    binned = bins.binned_matrix(df[features], schema)
    shard = Shard(path=str(path), binned=binned,
                  y=y, raw=np.zeros(len(y), dtype=np.float64), hostname=socket.gethostname())
    if use_cache:
        _CACHE[key] = shard
    return shard


def reset(shard: Shard, base_score: float) -> None:
    """Start (or restart) a run from an empty model."""
    shard.raw = np.full(shard.n, float(base_score), dtype=np.float64)
    shard.trees_seen = 0


def apply_tree(shard: Shard, tree) -> None:
    """Fold one finished tree into the client's running score. This is what keeps the
    client in step with the server without the server ever resending the whole model."""
    if isinstance(tree, dict):
        tree = gbdt.Tree.from_arrays(tree)
    shard.raw += tree.leaf_for_rows(shard.binned)
    shard.trees_seen += 1


def histograms(shard: Shard, tree, node_ids, pos_weight: float, schema: dict) -> np.ndarray:
    """Per-bin gradient/hessian/count totals for each requested node, packed flat.

    This is the entire outbound payload of a client for one round: nothing here can be
    reversed into a transaction, and every value is a sum that the other clients' values
    simply add to.
    """
    g, h = gbdt.gradients(shard.raw, shard.y, pos_weight)
    node = tree.node_index(shard.binned)
    stats = [gbdt.node_histogram(shard.binned, g, h, np.flatnonzero(node == nid), schema)
             for nid in node_ids]
    return gbdt.pack_histograms(stats)


def payload_floats(node_count: int, total_bins: int) -> int:
    """Size of one client's reply, so communication overhead can be reported rather than
    estimated."""
    return int(node_count * 3 * total_bins)


def local_loss(shard: Shard) -> float:
    """Mean log-loss on this client's own rows, at whatever score it currently carries.

    Reported as a metric so the server can plot progress without holding any client data.
    """
    p = gbdt.sigmoid(shard.raw)
    eps = 1e-12
    ll = -(shard.y * np.log(p + eps) + (1.0 - shard.y) * np.log(1.0 - p + eps))
    return float(ll.mean())
