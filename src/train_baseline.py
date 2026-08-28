"""
Trains the baseline centralized fraud-detection model.

Objective 2 of the Phase I report: develop and evaluate a baseline centralized AI
fraud detection model using accuracy, precision, recall, F1-score and AUC-ROC.

A LightGBM classifier is trained under two splits - stratified random and
chronological - and measured against two reference controls that establish the noise
floor for this dataset: a majority-class dummy, and the same model trained on
permuted labels.

    python src/train_baseline.py

Writes models/lightgbm_baseline_<split>.txt, reports/baseline_metrics.json and
reports/figures/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402  - sets OpenMP env vars before lightgbm loads

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

import modeling  # noqa: E402
import plots  # noqa: E402
from config import (  # noqa: E402
    DATASET,
    N_PERMUTATIONS,
    REPORTS_DIR,
    ROOT,
    SEED,
    SPLITS,
    ensure_dirs,
    model_path,
)
from data import get_splitter, load_data  # noqa: E402


def run_split(name: str, X, y, ts, out: dict) -> None:
    print(f"\n{'=' * 72}\n{name.upper()} SPLIT\n{'=' * 72}")
    tr, va, te = get_splitter(name)(X, y, ts)
    Xtr, Xva, Xte = X.iloc[tr], X.iloc[va], X.iloc[te]
    ytr, yva, yte = y.iloc[tr], y.iloc[va], y.iloc[te]
    print(f"  train {len(tr):>7,} ({ytr.sum():>3} fraud)   "
          f"val {len(va):>6,} ({yva.sum():>3})   test {len(te):>6,} ({yte.sum():>3})")

    section = {"sizes": {"train": len(tr), "val": len(va), "test": len(te)},
               "fraud": {"train": int(ytr.sum()), "val": int(yva.sum()), "test": int(yte.sum())}}

    # --- model
    print("\n  [1/2] LightGBM")
    t0 = time.perf_counter()
    model = modeling.train(Xtr, ytr, Xva, yva)
    train_seconds = time.perf_counter() - t0

    scores_va = model.predict(Xva, num_iteration=model.best_iteration)
    t0 = time.perf_counter()
    scores_te = model.predict(Xte, num_iteration=model.best_iteration)
    infer_us = (time.perf_counter() - t0) / len(Xte) * 1e6

    threshold = modeling.best_f1_threshold(yva, scores_va)
    section["lightgbm"] = {
        "test_default_threshold": modeling.evaluate(yte, scores_te, 0.5),
        "test_tuned_threshold": modeling.evaluate(yte, scores_te, threshold),
        "val_tuned_threshold": modeling.evaluate(yva, scores_va, threshold),
        "n_trees": int(model.best_iteration or model.num_trees()),
        "train_seconds": round(train_seconds, 2),
        "inference_us_per_txn": round(infer_us, 2),
    }
    print(f"    trees={section['lightgbm']['n_trees']}  "
          f"test ROC-AUC={section['lightgbm']['test_tuned_threshold']['roc_auc']:.4f}  "
          f"PR-AUC={section['lightgbm']['test_tuned_threshold']['pr_auc']:.4f}  "
          f"{infer_us:.1f} us/txn")

    # --- controls
    print("\n  [2/2] Reference controls")
    section["majority_class_dummy"] = (
        modeling.evaluate(yte, np.zeros(len(yte)), 0.5) | {"roc_auc": 0.5})
    print(f"    majority-class dummy: accuracy="
          f"{section['majority_class_dummy']['accuracy']:.4f}, recall=0.0000")

    aucs, aps = [], []
    for k in range(N_PERMUTATIONS):
        rng = np.random.default_rng(SEED + k)
        y_perm = pd.Series(rng.permutation(ytr.values), index=ytr.index)
        m = modeling.train(Xtr, y_perm, Xva, yva, verbose=False)
        p = m.predict(Xte, num_iteration=m.best_iteration)
        aucs.append(roc_auc_score(yte, p))
        aps.append(average_precision_score(yte, p))
    section["permuted_label_control"] = {
        "n_runs": N_PERMUTATIONS,
        "roc_auc_mean": round(float(np.mean(aucs)), 6),
        "roc_auc_std": round(float(np.std(aucs)), 6),
        "roc_auc_max": round(float(np.max(aucs)), 6),
        "pr_auc_mean": round(float(np.mean(aps)), 6),
    }
    print(f"    permuted labels ({N_PERMUTATIONS} runs): ROC-AUC = "
          f"{np.mean(aucs):.4f} +/- {np.std(aucs):.4f} (max {np.max(aucs):.4f})")

    # --- artefacts
    section["figures"] = {"curves": str(plots.curves(yte, scores_te, name).relative_to(ROOT))}
    model.save_model(str(model_path(name)))
    print(f"\n  saved {model_path(name).relative_to(ROOT)}")
    if name == "stratified":
        section["figures"]["importance"] = str(
            plots.importance(model, list(X.columns)).relative_to(ROOT))

    out[name] = section


def main() -> dict:
    ensure_dirs()
    print("Loading", DATASET.relative_to(ROOT))
    X, y, ts = load_data()
    print(f"  {X.shape[0]:,} transactions x {X.shape[1]} features | "
          f"fraud = {y.sum()} ({y.mean() * 100:.3f}%) | "
          f"{ts.min().date()} -> {ts.max().date()}")

    out = {"dataset": {"rows": int(X.shape[0]), "features": int(X.shape[1]),
                       "fraud_count": int(y.sum()), "fraud_rate": round(float(y.mean()), 6),
                       "date_range": [str(ts.min()), str(ts.max())]},
           "seed": SEED}
    for split in SPLITS:
        run_split(split, X, y, ts, out)

    path = REPORTS_DIR / "baseline_metrics.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nMetrics written to {path.relative_to(ROOT)}")
    return out


if __name__ == "__main__":
    main()
