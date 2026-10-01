"""
Splits the prepared dataset into federated clients and a server-held test set.

Objective 4 asks for several simulated bank/POS clients, each training locally on its own
transactions. This script produces exactly that layout:

    data/fl/client_0.csv ... client_{n-1}.csv   one shard per simulated bank
    data/fl/server_test.csv                     the ONLY labelled rows the server keeps
    data/fl/partition_summary.json              what went where, for the write-up
    fl/upi_fl/schema.json                       the fixed bin layout both sides share

The split is built from ``src/data.py`` unchanged, so the rows the federated model is
scored on are the same rows the Phase I baseline was scored on. Without that, the
Objective 5 comparison would not be like-for-like.

Two partitioning modes:

    --mode bank   non-IID and realistic: a client only sees the banks assigned to it, so
                  its transactions genuinely look different from the others' (default)
    --mode iid    control: the same rows shuffled evenly, fraud rate preserved. If the
                  federated model works here but not with --mode bank, the gap is the
                  non-IID data, not the protocol.

    python fl/make_partitions.py                     # 3 clients, grouped by bank
    python fl/make_partitions.py --clients 5
    python fl/make_partitions.py --mode iid          # IID control split
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "fl"))

import config  # noqa: E402  - sets OpenMP env vars before the numeric stack loads
from config import DATASET, LABEL_COLUMN, SEED, TIME_COLUMN  # noqa: E402
from data import load_data, stratified_split  # noqa: E402

from upi_fl import bins  # noqa: E402

FL_DIR = ROOT / "data" / "fl"
BANK_PREFIX = "sender_bank_"
W = 78


def rule(char: str = "-") -> None:
    print(char * W)


# ----------------------------------------------------------------------------- bank labels

def bank_labels(X: pd.DataFrame) -> np.ndarray:
    """Recover each row's sender bank from the one-hot block (`sender_bank_*`)."""
    cols = [c for c in X.columns if c.startswith(BANK_PREFIX)]
    if not cols:
        return np.full(len(X), "unknown", dtype=object)
    block = X[cols].to_numpy()
    picked = block.argmax(axis=1)
    present = block.max(axis=1) == 1
    names = np.array([c[len(BANK_PREFIX):] for c in cols], dtype=object)
    return np.where(present, names[picked], "unknown")


def assign_banks_to_clients(counts: dict[str, int], n_clients: int) -> dict[str, int]:
    """Largest bank first, each one going to the least loaded client.

    Bank sizes here are uneven, so a naive round-robin would hand one simulated bank most
    of the traffic and turn the comparison into a story about shard size rather than about
    federation.
    """
    assignment: dict[str, int] = {}
    load = [0] * n_clients
    for bank, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        target = int(np.argmin(load))
        assignment[bank] = target
        load[target] += count
    return assignment


def iid_assignment(y: pd.Series, n_clients: int, seed: int) -> np.ndarray:
    """Round-robin inside each class, so every client sees the same fraud rate."""
    rng = np.random.default_rng(seed)
    out = np.empty(len(y), dtype=np.int32)
    values = y.to_numpy()
    for cls in np.unique(values):
        idx = rng.permutation(np.flatnonzero(values == cls))
        out[idx] = np.arange(len(idx)) % n_clients
    return out


# --------------------------------------------------------------------------------- writing

def write_rows(path: Path, X: pd.DataFrame, y: pd.Series, ts: pd.Series, idx: np.ndarray) -> None:
    out = pd.concat([ts.iloc[idx].rename(TIME_COLUMN), X.iloc[idx], y.iloc[idx]], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Partition the dataset for federated training.")
    ap.add_argument("--mode", choices=("bank", "iid"), default="bank",
                    help="bank = non-IID grouped by sender bank (default); iid = control split")
    ap.add_argument("--clients", type=int, default=3,
                    help="number of simulated bank/POS clients (default 3)")
    ap.add_argument("--seed", type=int, default=SEED, help=f"seed (default {SEED})")
    ap.add_argument("--amount-bins", type=int, default=64, help="amount bins (default 64)")
    ap.add_argument("--hour-bins", type=int, default=24, help="hour bins (default 24)")
    ap.add_argument("--out", type=Path, default=FL_DIR, help=f"output directory (default {FL_DIR})")
    args = ap.parse_args()

    if args.clients < 2:
        sys.exit("--clients must be at least 2: a federation of one is not federated.")

    print()
    rule("=")
    print(" PHASE II - PARTITIONING FOR FEDERATED LEARNING".center(W))
    print(" Objective 4  |  simulated bank/POS clients, no raw data pooling".center(W))
    rule("=")

    X, y, ts = load_data()
    tr, va, te = stratified_split(X, y, ts)
    pool = np.concatenate([tr, va])                    # what the clients will hold
    print(f"\n  {len(X):,} rows x {X.shape[1]} features   fraud {y.sum():,} ({y.mean()*100:.2f}%)")
    print(f"  client pool {len(pool):,} rows   server test {len(te):,} rows")
    print("  (the server test rows are the Phase I test rows, so Objective 5 compares the")
    print("   federated model and the centralized baseline on identical data)")

    # --- assignment
    if args.mode == "bank":
        labels = bank_labels(X)
        counts = {str(k): int(v) for k, v in pd.Series(labels).value_counts().items()}
        if args.clients > len(counts):
            sys.exit(f"--clients {args.clients} exceeds the {len(counts)} sender banks available")
        mapping = assign_banks_to_clients(counts, args.clients)
        client_of_row = np.array([mapping[str(b)] for b in labels], dtype=np.int32)
        banks_of_client = {c: sorted(b for b, k in mapping.items() if k == c)
                           for c in range(args.clients)}
    else:
        client_of_row = iid_assignment(y, args.clients, args.seed)
        banks_of_client = {c: ["(iid: all banks)"] for c in range(args.clients)}

    pool_client = client_of_row[pool]
    print(f"\n  mode: {args.mode}   clients: {args.clients}")

    # --- write one shard per client
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": args.mode,
        "seed": args.seed,
        "n_clients": args.clients,
        "dataset": str(DATASET.relative_to(ROOT)),
        "split": "stratified (Phase I partition boundaries)",
        "rows_total": int(len(X)),
        "rows_pooled_at_clients": int(len(pool)),
        "server_test_file": str((args.out / "server_test.csv").relative_to(ROOT)),
        "clients": [],
    }

    rule()
    print(f"  {'client':<8}{'rows':>10}{'fraud':>9}{'fraud rate':>12}   banks")
    rule()
    for c in range(args.clients):
        idx = pool[pool_client == c]
        path = args.out / f"client_{c}.csv"
        write_rows(path, X, y, ts, idx)
        yc = y.iloc[idx]
        banks = banks_of_client[c]
        summary["clients"].append({
            "client": c,
            "file": str(path.relative_to(ROOT)),
            "rows": int(len(idx)),
            "fraud": int(yc.sum()),
            "fraud_rate": round(float(yc.mean()) if len(idx) else 0.0, 6),
            "banks": banks,
        })
        shown = ", ".join(banks)
        shown = shown if len(shown) <= 34 else shown[:31] + "..."
        print(f"  {c:<8}{len(idx):>10,}{int(yc.sum()):>9,}{yc.mean()*100:>11.2f}%   {shown}")

    write_rows(args.out / "server_test.csv", X, y, ts, te)
    summary["server_test"] = {
        "rows": int(len(te)),
        "fraud": int(y.iloc[te].sum()),
        "fraud_rate": round(float(y.iloc[te].mean()), 6),
    }
    rule()
    print(f"  {'server':<8}{len(te):>10,}{int(y.iloc[te].sum()):>9,}"
          f"{y.iloc[te].mean()*100:>11.2f}%   labelled rows, used for scoring only")

    # --- the fixed bin layout, which every client and the server must agree on
    features = list(X.columns)
    kinds = {name: bins.classify_feature(name) for name in features}
    bin_counts = bins.bin_counts_for(features, kinds, args.amount_bins, args.hour_bins)
    edges = {bins.AMOUNT_COLUMN: bins.amount_edges(X[bins.AMOUNT_COLUMN], args.amount_bins)}
    schema = bins.build_schema(features, kinds, bin_counts, edges)
    schema_path = bins.save_schema(schema)
    summary["bins"] = {
        "amount_bins": args.amount_bins,
        "hour_bins": args.hour_bins,
        "total_bins": schema["total_bins"],
        "schema_file": str(schema_path.relative_to(ROOT)),
    }

    summary_path = args.out / "partition_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print()
    print(f"  bin layout: {schema['n_features']} features -> {schema['total_bins']} bins "
          f"({args.amount_bins} amount, {args.hour_bins} hour, 2 per flag)")
    print(f"  wrote {summary_path.relative_to(ROOT)}")
    print(f"  wrote {schema_path.relative_to(ROOT)}   (ships inside the Flower app bundle)")
    rule("=")


if __name__ == "__main__":
    main()

