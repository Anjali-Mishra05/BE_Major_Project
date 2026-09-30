# Baseline Centralized Fraud Detection — Results

**Phase I, Objective 2** — *Develop and evaluate a baseline centralized AI fraud detection
model using accuracy, precision, recall, F1-score and AUC-ROC.*

Run: `python src/train_baseline.py` · Artifacts: `reports/baseline_metrics.json`,
`reports/figures/`, `models/`

---

## Headline finding

On the updated dataset (synthetic fraud added, 25 % prevalence) the LightGBM baseline,
trained **without** the label-correlated `behaviour` column, ranks fraud well above chance
and clearly beats the shuffled-label control.

| Split | ROC-AUC | PR-AUC | ROC-AUC, permuted labels (5 runs) |
|---|---|---|---|
| Stratified | 0.9115 | 0.8577 | 0.555 ± 0.065 |
| Chronological | 0.9105 | 0.8557 | 0.540 ± 0.065 |

**Why `behaviour` is excluded.** The dataset contains a `behaviour` column (one-hot
encoded as `behaviour_*`) generated together with `fraud_flag`:

| behaviour | legit | fraud |
|---|---|---|
| Device Anomaly | 0 | 12,537 |
| Network Anomaly | 0 | 12,352 |
| Normal | 220,540 | 411 |
| Unusual Time + High Amount | 882 | 21,767 |
| High Amount | 11,575 | 22,117 |
| Unusual Time | 16,523 | 13,989 |

It was the top feature by gain (about 14x the next one) and would not exist at scoring
time, so it is dropped in `load_data` (`LEAKY_PREFIXES` in `src/config.py`). Including it
gave ROC-AUC 0.9913 / PR-AUC 0.9802 (stratified), which reflects the generator's rule, not
detection ability. The remaining fraud is still synthetic, so 0.91 shows the model
recovers the generator's feature-level patterns (amount, hour, night/high-amount flags),
not real-world performance.

## Dataset

| Property | Value |
|---|---|
| Transactions | 332,693 |
| Features | 71 (after one-hot encoding; `timestamp`, `transaction id` and `behaviour_*` excluded) |
| Fraud cases | 83,173 (**25.0 %**) |
| Date range | 2024-01-01 -> 2024-12-30 |
| Source file | `data/processed/upi_transactions_ml_ready_final.csv` |

## Model

| | LightGBM |
|---|---|
| Architecture | 15 leaves, depth 5, <=600 rounds w/ early stopping |
| Size | 186 trees (stratified), 206 trees (chronological) |
| Imbalance handling | `scale_pos_weight` = train legit / train fraud (about 3) |
| Inference | about 8.5 us/txn |

Note for Phase II: FedAvg/FedProx and per-example gradient clipping need a differentiable
model, so federating a GBDT requires a different algorithm family (SecureBoost, FedTree)
than report section 5.3 currently specifies.

## Full metrics

Test n = 49,904 for both splits.

**Stratified split** - 12,476 fraud in test

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR |
|---|---|---|---|---|---|---|---|---|
| LightGBM | 0.50 | 0.8625 | 0.6978 | 0.7937 | 0.7426 | 0.9115 | 0.8577 | 0.1146 |
| LightGBM | tuned (0.691) | 0.8926 | 0.8457 | 0.6975 | 0.7645 | 0.9115 | 0.8577 | 0.0424 |
| Majority-class dummy | - | 0.7500 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2500 | 0.0000 |

**Chronological split** - 12,459 fraud in test

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR |
|---|---|---|---|---|---|---|---|---|
| LightGBM | 0.50 | 0.8610 | 0.6956 | 0.7882 | 0.7390 | 0.9105 | 0.8557 | 0.1148 |
| LightGBM | tuned (0.659) | 0.8894 | 0.8220 | 0.7112 | 0.7626 | 0.9105 | 0.8557 | 0.0512 |
| Majority-class dummy | - | 0.7503 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2500 | 0.0000 |

Thresholds marked *tuned* maximise fraud-class F1 on the validation split and are then
applied unchanged to test. The dummy reaches 75 % accuracy by labelling everything
legitimate, so quote accuracy alongside it.

The permuted-label control is not exactly 0.5 (0.555 and 0.540, std about 0.065): at
this size the shuffled-label runs stop early on very few trees and are noisy, but none
approaches the real-label score (max 0.659).

## History

The previous dataset (250,000 rows, 480 fraud, 0.192 %) gave chance-level scores
(ROC-AUC 0.507 / 0.493), identical to permuted-label controls, and its statistical
pre-check found no relationship between features and `fraud_flag`. The updated dataset
adds synthetic fraud with feature-level patterns plus a leaky `behaviour` column (excluded).

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
oversamples the fraud class (a random 20 rows would rarely show both outcomes), which
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

State explicitly in the Phase I write-up:

1. **The fraud is synthetic.** The 25 % prevalence and its patterns come from the
   generator, so the scores show the model recovers those rules, not real fraud.
2. **`behaviour` was excluded as label leakage.** Including it gives 0.99 ROC-AUC; state
   that the reported 0.91 is the leakage-free figure.
3. **Accuracy must be quoted with the majority-class dummy** (75 %).
