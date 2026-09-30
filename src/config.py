"""Paths, constants and environment setup shared by every entrypoint.

Import this FIRST in any script that touches LightGBM: on macOS LightGBM ships its
own OpenMP runtime (libomp) which conflicts with the one in Anaconda's numeric stack
(libiomp), and parallel sections deadlock unless the thread count is pinned before
the library loads.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATASET = PROCESSED_DIR / "upi_transactions_ml_ready_final.csv"
DEFAULT_SAMPLES = SAMPLES_DIR / "demo_test_samples.csv"

LABEL_COLUMN = "fraud_flag"
ID_COLUMN = "transaction id"
# `behaviour_*` is generated together with fraud_flag (e.g. every Device/Network Anomaly
# row is fraud) and would not exist at scoring time, so it is excluded as label leakage.
LEAKY_PREFIXES = ("behaviour_",)
TIME_COLUMN = "timestamp"

SEED = 42
N_PERMUTATIONS = 5          # permuted-label control runs per split
SPLITS = ("stratified", "chronological")


def model_path(split: str) -> Path:
    return MODELS_DIR / f"lightgbm_baseline_{split}.txt"


def ensure_dirs() -> None:
    for d in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR, SAMPLES_DIR):
        d.mkdir(parents=True, exist_ok=True)
