"""Training, loading and scoring the LightGBM baseline, plus the metric helpers."""

from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import SEED, model_path

# Lightweight GBDT: shallow trees, few leaves - small enough for edge inference and
# for the tree-count / latency budget in the Phase I report (5.8).
PARAMS = {
    "objective": "binary",
    "metric": ["auc", "average_precision"],
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 5,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": SEED,
    "verbosity": -1,
    "num_threads": 1,
}
NUM_BOOST_ROUND = 600
EARLY_STOPPING = 50


def train(Xtr, ytr, Xva, yva, verbose: bool = True) -> lgb.Booster:
    """Fit the baseline. scale_pos_weight compensates for the 0.192% fraud rate."""
    params = dict(PARAMS)
    params["scale_pos_weight"] = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    dtr = lgb.Dataset(Xtr, label=ytr)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)
    callbacks = [lgb.early_stopping(EARLY_STOPPING, verbose=False)]
    if verbose:
        callbacks.append(lgb.log_evaluation(100))
    return lgb.train(params, dtr, num_boost_round=NUM_BOOST_ROUND, valid_sets=[dva],
                     valid_names=["val"], callbacks=callbacks)


def load(split: str) -> tuple[lgb.Booster, "Path"]:
    """Load the model trained on `split`. Scoring one split with the other split's
    model would leak - those test rows were in its training partition."""
    path = model_path(split)
    if not path.exists():
        sys.exit(f"No model trained on the {split} split: {path}\n"
                 f"Run:  python src/train_baseline.py")
    return lgb.Booster(model_file=str(path)), path


# -------------------------------------------------------------------------- metrics

def best_f1_threshold(y_true, scores) -> float:
    """Threshold maximising fraud-class F1 on the validation set. Tuned on validation
    only, then applied unchanged to test."""
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    if len(thr) == 0:
        return 0.5
    return float(thr[min(int(np.argmax(f1)), len(thr) - 1)])


def evaluate(y_true, scores, threshold: float) -> dict:
    """Every metric the Phase I report asks for, at one decision threshold."""
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 6),
        "accuracy": round(accuracy_score(y_true, y_pred), 6),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 6),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 6),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 6),
        "roc_auc": round(roc_auc_score(y_true, scores), 6),
        "pr_auc": round(average_precision_score(y_true, scores), 6),
        "fpr": round(fp / (fp + tn) if (fp + tn) else 0.0, 6),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_samples": int(len(y_true)),
        "n_fraud": int(y_true.sum()),
    }
