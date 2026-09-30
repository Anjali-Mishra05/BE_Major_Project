"""
Builds a small file of unseen transactions to score during a demo.

Rows are drawn from the held-out test partition, so the model has genuinely never
seen them - the same guarantee the reported metrics rest on. The output keeps the
fraud_flag column so evaluation.py can report whether each call was right.

    python src/make_test_samples.py                    # 20 transactions
    python src/make_test_samples.py --n 20 --fraud 5   # control the fraud/legit mix
    python src/make_test_samples.py --out data/samples/panel_demo.csv

NOTE ON THE MIX. Fraud is a minority class, so 20 rows drawn at random may contain few or
none. --fraud oversamples the fraud class so a demo can show both
outcomes. That makes the file useful for demonstration but NOT a valid sample for
measuring performance - quote metrics from the full test set instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402  - sets OpenMP env vars before lightgbm loads

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import DEFAULT_SAMPLES, ROOT, SEED, SPLITS, TIME_COLUMN, ensure_dirs  # noqa: E402
from data import get_splitter, load_data  # noqa: E402

MAX_SAMPLES = 20


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a file of unseen transactions to score.")
    ap.add_argument("--n", type=int, default=MAX_SAMPLES,
                    help=f"total transactions (default {MAX_SAMPLES}, max {MAX_SAMPLES})")
    ap.add_argument("--fraud", type=int, default=5,
                    help="how many should be actual fraud (default 5)")
    ap.add_argument("--split", choices=SPLITS, default="stratified",
                    help="which held-out partition to draw from (default: stratified)")
    ap.add_argument("--out", type=Path, default=DEFAULT_SAMPLES,
                    help=f"output CSV (default {DEFAULT_SAMPLES.name})")
    ap.add_argument("--seed", type=int, default=SEED, help="sampling seed (default 42)")
    args = ap.parse_args()

    if not 1 <= args.n <= MAX_SAMPLES:
        sys.exit(f"--n must be between 1 and {MAX_SAMPLES}")
    if not 0 <= args.fraud <= args.n:
        sys.exit(f"--fraud must be between 0 and --n ({args.n})")

    ensure_dirs()
    X, y, ts = load_data()
    _, _, te = get_splitter(args.split)(X, y, ts)

    yte = y.iloc[te]
    fraud_pos, legit_pos = te[yte.values == 1], te[yte.values == 0]
    n_fraud = min(args.fraud, len(fraud_pos))
    if n_fraud < args.fraud:
        print(f"  note: only {n_fraud} fraud rows available in this partition")
    n_legit = args.n - n_fraud

    rng = np.random.default_rng(args.seed)
    pick = np.concatenate([rng.choice(fraud_pos, n_fraud, replace=False),
                           rng.choice(legit_pos, n_legit, replace=False)])
    rng.shuffle(pick)

    out = pd.concat([ts.iloc[pick].rename(TIME_COLUMN), X.iloc[pick], y.iloc[pick]], axis=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    rel = args.out.relative_to(ROOT) if args.out.is_absolute() else args.out
    print(f"Wrote {rel}")
    print(f"  {len(out)} transactions  |  {n_fraud} fraud, {n_legit} legitimate")
    print(f"  drawn from the held-out {args.split} test partition - unseen during training")
    print(f"\nScore them with:")
    print(f"  python src/evaluation.py --score-file {rel}")


if __name__ == "__main__":
    main()
