# Software validation — 14 November 2026

The implementation was tested locally on Windows with Python 3.12.14. Dependency
versions are recorded in requirements-tested.txt and dashboard/package-lock.json.

## Completed checks

- 20 automated regression tests passed, including the existing Flower protocol and
  tree-math tests, secure mask cancellation, DP accounting, privacy budget refusal,
  idempotent retries after client restart, auth/schema validation, missing-client
  deadline, final-model installation, local inference, and Windows model loading.
- React production build passed; package installation audit reported zero vulnerabilities.
- Real HTTP deployment: one coordinator and two independent client processes trained
  20 depth-3 trees in plain, secure and secure + DP modes; all 80 exchanges per run
  completed and both clients acknowledged the final model checksum.
- A separate synthetic three-process acceptance test stopped and restarted a client
  mid-run: training recovered and completed. A dropout test failed within its configured
  deadline and did not aggregate a partial cohort.
- Browser checks confirmed live server metrics, client status, privacy information,
  measured comparisons, and a successful local transaction prediction.
- Synthetic scaling across 2, 3, and 5 clients preserved predictions to within 6e-11.
  This measures compute and aggregation on one host, not network scalability.
- After the Windows baseline-loading correction, a fresh isolated launcher run completed
  training, delivery, inference timing and baseline comparison end-to-end, then stopped
  its child processes normally.

## Measured results

Same 49,904 held-out transactions; threshold 0.5; fraud labels are synthetic.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Saved centralized baseline, re-evaluated | .8625 | .6978 | .7937 | .7426 | .9115 | .8577 |
| Plain federated | .8903 | .8646 | .6653 | .7519 | .9060 | .8393 |
| Secure federated | .8903 | .8646 | .6653 | .7519 | .9060 | .8393 |
| Secure + DP | .8724 | .7843 | .6753 | .7257 | .8930 | .8059 |

DP: clipping norm 10, noise multiplier 8, 80 releases per client, run epsilon 6.50197
at delta 1e-6. The DP runtime uses unweighted labels and the histogram model; the baseline
uses class weighting and a different tree-growth algorithm, so this is not a controlled
comparison of federation alone. Plain versus secure is the masking control.

| Mode | Training seconds | Inference p50 ms | Inference p95 ms | Model KB |
|---|---:|---:|---:|---:|
| Plain | 107.0 | 7.18 | 12.42 | 39.4 |
| Secure | 104.0 | 8.66 | 13.69 | 39.4 |
| Secure + DP | 194.8 | 7.62 | 10.30 | 39.5 |

Inference timing includes feature binning and model scoring for one record, excludes HTTP,
and uses 25 requests per mode. These are single-run measurements without confidence intervals.
Machine load and polling influence timing. Full metrics and application payload sizes
are in software_comparison.json; network acceptance and scaling have separate JSON reports.

## Defects found and corrected

- Test rows were influencing bin definitions. Bins now use a fixed public INR domain.
- Federated fitting included validation rows while the baseline did not. Both now use
  the same training partition; validation remains separate.
- SQLite context managers did not close database connections on Windows. All ledger
  connections now close explicitly, including failure paths.
- Windows CRLF checkout corrupted LightGBM's byte offsets when loading model files.
  Loading normalizes line endings and .gitattributes preserves LF for future checkouts.
- Narrow layouts showed sidebar text outside the icon rail. Small-screen text is now hidden.

## Remaining acceptance boundaries

This does not certify zero future bugs, production cryptographic security, real-bank
accuracy, Raspberry Pi/Jetson latency, or physical three-device Wi-Fi behavior. Run the
documented LAN acceptance sequence when the other devices are available. Hardware remains
deferred. The implementation is histogram GBDT, not GNN/FedAvg. Privacy assumptions and
floating-point sampling limitations are explicitly documented in docs/privacy.md.
