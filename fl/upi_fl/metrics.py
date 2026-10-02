"""Metric helpers for the federated run.

These mirror ``src/modeling.py`` key for key, using the same sklearn functions, so the
federated numbers and the Phase I baseline numbers are produced the same way. It is a copy
rather than an import because a Flower app ships on its own and cannot reach ``src/``.

``fl/compare.py`` - which does not run inside the app - imports the original functions from
``src`` instead, and checks that both agree.
"""

from __future__ import annotations

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


def roc_auc(y_true, scores) -> float:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def best_f1_threshold(y_true, scores) -> float:
    """Threshold maximising fraud-class F1 - the same rule the baseline uses."""
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    if len(thr) == 0:
        return 0.5
    return float(thr[min(int(np.argmax(f1)), len(thr) - 1)])


def evaluate(y_true, scores, threshold: float) -> dict:
    """Every metric the Phase I report asks for, at one decision threshold."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(scores) >= threshold).astype(int)
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
