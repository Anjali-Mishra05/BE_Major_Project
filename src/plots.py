"""Figures written to reports/figures/."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from config import FIGURES_DIR  # noqa: E402

ACCENT = "#2F5DA8"


def curves(y_true, scores, split_name: str):
    """ROC and precision-recall, each against its no-skill reference."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    fpr, tpr, _ = roc_curve(y_true, scores)
    axes[0].plot(fpr, tpr, color=ACCENT,
                 label=f"LightGBM (AUC={roc_auc_score(y_true, scores):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="random (AUC=0.500)")

    prec, rec, _ = precision_recall_curve(y_true, scores)
    axes[1].plot(rec, prec, color=ACCENT,
                 label=f"LightGBM (AP={average_precision_score(y_true, scores):.4f})")
    prevalence = float(np.mean(y_true))
    axes[1].axhline(prevalence, color="k", ls="--", lw=1,
                    label=f"prevalence ({prevalence:.4f})")

    axes[0].set(xlabel="False positive rate", ylabel="True positive rate",
                title=f"ROC - {split_name} split")
    axes[1].set(xlabel="Recall (fraud)", ylabel="Precision (fraud)",
                title=f"Precision-Recall - {split_name} split")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out = FIGURES_DIR / f"curves_{split_name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def importance(model, feature_names, top: int = 20):
    gain = model.feature_importance(importance_type="gain")
    order = np.argsort(gain)[-top:]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([feature_names[i] for i in order], gain[order], color="#4C78A8")
    ax.set(xlabel="Total split gain", title=f"LightGBM feature importance (top {top})")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    out = FIGURES_DIR / "lightgbm_feature_importance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
