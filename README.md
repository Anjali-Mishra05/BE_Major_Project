# UPI Fraud Detection — Phase I Baseline

Major Project (Phase I) · *Federated Learning, Secure Aggregation, Differential Privacy
& Edge AI for Real-Time UPI Fraud Detection via a Smart POS Prototype*

This repository covers **Objective 2**: develop and evaluate a baseline centralized AI
fraud detection model using accuracy, precision, recall, F1-score and AUC-ROC.

## Result

Trained on `data/processed/upi_transactions_ml_ready_final.csv` (332,693 transactions,
25.0 % fraud, synthetic fraud added), 71 features:

| Split | ROC-AUC | PR-AUC | Recall (tuned) | Precision (tuned) | ROC-AUC, permuted labels |
|---|---|---|---|---|---|
| Stratified | 0.9115 | 0.8577 | 0.698 | 0.846 | 0.555 ± 0.065 |
| Chronological | 0.9105 | 0.8557 | 0.711 | 0.822 | 0.540 ± 0.065 |

> **Leakage excluded.** The dataset's `behaviour` column is generated with the label
> (every "Device Anomaly" and "Network Anomaly" row is fraud), so it is dropped before
> training. Including it gives ROC-AUC 0.99. The fraud is still synthetic, so these
> numbers do not predict real-world performance. Details:
> [`reports/baseline_results.md`](reports/baseline_results.md).

> **Reporting accuracy.** A majority-class dummy reaches 75 % accuracy and catches zero
> fraud. Always quote accuracy alongside it.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Train the baseline (~1 min). Writes models/, reports/baseline_metrics.json, figures.
python src/train_baseline.py

# 2. Evaluate. Loads the saved model, retrains nothing, runs in seconds.
python src/evaluation.py
python src/evaluation.py --split chronological    # time-ordered split
python src/evaluation.py --samples 20             # score 20 individual transactions
python src/evaluation.py --html --save            # projectable HTML page + JSON

# 3. Build a file of new, unseen transactions and score it.
python src/make_test_samples.py --n 20 --fraud 5
python src/evaluation.py --score-file data/samples/demo_test_samples.csv
```

For a live demo, run `evaluation.py` in the terminal — a real run is more convincing than
a slide. `--html` writes `reports/evaluation_report.html` as the projectable leave-behind
(print to PDF from the browser).

## Layout

```
data/
  raw/          upi_transactions_2024.csv          source dataset
  processed/    upi_transactions_ml_ready_final.csv  one-hot encoded, model-ready
  samples/      demo_test_samples.csv              unseen transactions for demos
notebooks/
  01_preprocessing.ipynb                           raw -> processed
src/
  config.py             paths, seed, constants, OpenMP setup
  data.py               loading, splits, one-hot decoding
  modeling.py           LightGBM training/loading, metric helpers
  plots.py              ROC/PR curves, feature importance
  report_html.py        HTML report renderer
  train_baseline.py     entrypoint: train + controls
  evaluation.py         entrypoint: evaluate + demo
  make_test_samples.py  entrypoint: build a new-test-data file
models/
  lightgbm_baseline_stratified.txt                 the demo model
  lightgbm_baseline_chronological.txt
reports/
  baseline_results.md      full write-up
  baseline_metrics.json    training metrics, both splits
  evaluation_report.json   evaluation metrics
  evaluation_report.html   projectable report
  figures/                 ROC/PR curves, feature importance
```

**One model per split, deliberately.** The chronological test rows sit inside the
stratified training partition, so scoring one split with the other's model would leak.
`evaluation.py --split X` loads the model trained on X.

## Notes

- **Reproducible.** Seed 42 throughout; reruns give identical metrics.
- **macOS/OpenMP.** LightGBM ships `libomp`, Anaconda ships `libiomp`; with both loaded,
  parallel sections deadlock. `src/config.py` pins the thread count before LightGBM
  imports, which is why it must be imported first in every entrypoint.
- **Phase II.** FedAvg/FedProx and per-example gradient clipping need a differentiable
  model, which LightGBM is not. Federating a GBDT calls for a different algorithm family
  (SecureBoost, FedTree) than report §5.3 currently specifies.
