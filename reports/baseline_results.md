# Baseline Centralized Fraud Detection — Results

**Phase I, Objective 2** — *Develop and evaluate a baseline centralized AI fraud detection
model using accuracy, precision, recall, F1-score and AUC-ROC.*

Run: `python src/train_baseline.py` · Artifacts: `reports/baseline_metrics.json`,
`reports/figures/`, `models/`

---

## Headline finding

Both baselines score at chance level, and **the model trained on the real labels performs
identically to the same model trained on randomly shuffled labels**. The prepared dataset
contains no learnable relationship between the 60 features and `fraud_flag`.

| Split | ROC-AUC (real labels) | ROC-AUC (permuted labels, 5 runs) |
|---|---|---|
| Stratified | 0.5068 | 0.5068 ± 0.0192 |
| Chronological | 0.4929 | 0.4907 ± 0.0240 |

The permuted-label control is the decisive test: it destroys any real signal by shuffling
the training labels. A model with genuine predictive power scores well above its permuted
control. Here the two are indistinguishable.

## Dataset

| Property | Value |
|---|---|
| Transactions | 250,000 |
| Features | 60 (after one-hot encoding, `timestamp` excluded) |
| Fraud cases | 480 (**0.192 %**) |
| Date range | 2024-01-01 → 2024-12-30 |
| Missing values | 0 |

## Model

| | LightGBM |
|---|---|
| Architecture | 15 leaves, depth 5, ≤600 rounds w/ early stopping |
| Size | 21 trees (42 KB) |
| Imbalance handling | `scale_pos_weight` = 520 |
| Inference | 1.2–3.0 µs/txn |

Well inside the Raspberry Pi / Jetson Nano budget of report §5.8.

The compact MLP evaluated in the first pass was removed — LightGBM is the model going
forward. Note for Phase II: FedAvg/FedProx and per-example gradient clipping need a
differentiable model, so federating a GBDT requires a different algorithm family
(SecureBoost, FedTree) than report §5.3 currently specifies.

## Full metrics

**Stratified split** — test n = 37,500 (72 fraud)

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR |
|---|---|---|---|---|---|---|---|---|
| LightGBM | 0.50 | 0.8016 | 0.0023 | 0.2361 | 0.0045 | 0.5068 | 0.0019 | 0.1973 |
| LightGBM | tuned | 0.9196 | 0.0010 | 0.0417 | 0.0020 | 0.5068 | 0.0019 | 0.0787 |
| Majority-class dummy | — | **0.9981** | 0.0000 | 0.0000 | 0.0000 | 0.5000 | — | 0.0000 |

**Chronological split** — test n = 37,500 (67 fraud)

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR |
|---|---|---|---|---|---|---|---|---|
| LightGBM | 0.50 | 0.0564 | 0.0018 | 0.9403 | 0.0035 | 0.4929 | 0.0018 | 0.9452 |
| LightGBM | tuned | 0.9465 | 0.0026 | 0.0746 | 0.0050 | 0.4929 | 0.0018 | 0.0519 |
| Majority-class dummy | — | **0.9982** | 0.0000 | 0.0000 | 0.0000 | 0.5000 | — | 0.0000 |

Thresholds marked *tuned* maximise fraud-class F1 on the validation split and are then
applied unchanged to test.

**Read accuracy with care.** The majority-class dummy — a model that labels every
transaction legitimate — reaches 99.81 % accuracy and catches zero fraud. At 0.19 %
prevalence, accuracy is not a meaningful metric; PR-AUC, recall and FPR are.

Both PR-AUC values sit at 0.0019, exactly the fraud prevalence, which is the value a random
scorer achieves.

## Statistical pre-check

Run before training, and consistent with the results:

- **χ² per binary feature:** 2 of 58 features reach p < 0.05, against 2.9 expected by
  chance alone. None survives Bonferroni correction (α = 0.00086); the strongest,
  `is_high_amount`, gives p = 0.0012.
- **Transaction amount:** Mann-Whitney U, p = 0.99 — fraud and legitimate amount
  distributions are indistinguishable (medians ₹618.5 vs ₹629).
- **Hour of day:** two-sample KS, p = 0.61 — no temporal concentration of fraud.

## Interpretation

`fraud_flag` in this Kaggle-sourced synthetic dataset appears to have been assigned
independently of the transaction attributes. This is a property of the data, not of the
models or the preprocessing: no classifier can recover a relationship that was never
encoded. Adding capacity, resampling (SMOTE), or further tuning cannot change this — they
would only fit noise, and the permuted-label control would rise to meet them.

## Reproducing and demonstrating

```bash
python src/train_baseline.py                     # trains, ~80 s, writes models/ and reports/
python src/evaluation.py                         # demo report on the held-out test set
python src/evaluation.py --split chronological   # same, on the time-ordered split
python src/evaluation.py --samples 20            # score 20 individual transactions
python src/evaluation.py --html --save           # projectable HTML page + JSON output

python src/make_test_samples.py --n 20 --fraud 5 # build a new-test-data file
python src/evaluation.py --score-file data/samples/demo_test_samples.csv
```

`make_test_samples.py` writes `data/samples/demo_test_samples.csv` — transactions drawn
from the held-out test partition, so the model has genuinely never seen them. `--fraud`
oversamples the fraud class (20 random rows would contain none at 0.192% prevalence), which
makes the file good for demonstration but **not** a valid sample for measuring performance;
quote the full-test-set metrics above instead.

For a panel demo, run it in the terminal &mdash; a live run is more convincing than a slide.
`--html` writes the same report as a styled page you can project or hand over as a PDF
(print from the browser); it embeds the ROC/PR figure by relative path, so keep it inside
`reports/`.

`evaluation.py` loads the saved models and retrains nothing, so it runs in seconds and is
safe to drive live. It prints the dataset and split, model sizes, the full metric table,
confusion matrices, reference baselines, feature importance, inference latency, and
individual transactions with their fraud scores.

## Reporting this result

The dataset stays as-is for Phase I. That makes the negative result itself the finding to
report, and it is a defensible one — a chance-level baseline is only meaningful because it
was verified against a permutation control and a statistical pre-check, both of which are
included above.

Two things to state explicitly in the Phase I write-up:

1. **The limitation is the data, not the method.** The pipeline, splits, imbalance
   handling, threshold tuning and metrics are all sound and are what Phase II builds on.
   No classifier can recover a relationship that was never encoded in the labels.
2. **Accuracy must be reported alongside the majority-class dummy.** Quoting 99.81 %
   accuracy without it would misrepresent the model, since the dummy reaches the same
   number while catching zero fraud.

If a dataset with verified fraud signal is approved later, `src/train_baseline.py` is
dataset-agnostic: point `DATA` at the new CSV and set the label column. Splits, imbalance
handling, threshold tuning, metrics, controls and figures carry over unchanged.
