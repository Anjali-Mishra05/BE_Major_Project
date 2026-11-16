# Semester software validation — 16 November 2026

This record covers local, uncommitted software changes. Nothing was pushed to GitHub.
Hardware/POS peripherals and physical edge-device acceptance remain deferred.

## Implemented

- Flower gRPC coordination of the dashboard's existing histogram-tree protocol,
  including secure aggregation, DP accounting and final-model acknowledgement.
- Two/three-POS Docker configuration generation; the tested configuration has one
  coordinator and three POS clients. Each client mounts only its own shard and identity.
- Federated GBDT export to ONNX, parity checking before acknowledgement and local
  ONNX Runtime CPU inference. Public binning remains a local preprocessing step.
- Logistic regression, decision tree and random forest comparisons on identical
  training/test splits, with ONNX conversion parity checks.
- Scheduled transaction-load benchmarks reporting inference and queue latency,
  achieved throughput, process CPU and sampled process RSS.
- React comparison tables and ONNX engine labeling; installed-model version is
  separate from a new training run's progress.

## Observed checks

| Check | Result |
|---|---|
| Python regression suite | 28 tests passed |
| React production build | Passed on Windows and in the Linux Docker build |
| Python dependency consistency | Passed on Windows and in the server container |
| Native-process Flower recovery | Restart recovered; missing participant caused bounded failure without partial aggregation |
| Three-POS Docker plain/secure/DP | All completed; nine histogram exchanges per run for three depth-two trees |
| Docker client restart during secure training | Recovered and delivered the final model to all three clients |
| Docker dropout injection | Failed within the configured round deadline; no partial aggregate published |
| Docker final-image normal runs | Plain, secure and DP completed; ONNX predictions matched across all three POS clients |
| Browser workflow | Comparison page rendered; local sample loaded and ONNX risk prediction displayed |
| Git exclusions | Runtime credentials, generated model binaries and dashboard build output confirmed ignored |

Evidence: [Docker fault acceptance](docker_acceptance.json),
[final-image normal runs](docker_smoke.json),
[Flower process acceptance](flower_dashboard_acceptance.json), and
[algorithm/load measurements](algorithm_benchmarks.json).

Testing exposed and fixed Flower's requirement to register signal handlers on the
main thread, transient Windows checkpoint replacement locks, temporary-folder cleanup
before training threads exited, and localhost port publication on this Docker version.
Checkpoint retries are bounded and retain the previous committed file on persistent
failure. Tests cover successful retries and persistent failures.

## Interpretation

The federated model remains histogram GBDT, not GNN or FedAvg. Docker devices share
one laptop and do not establish physical-device or LAN performance. The separate
algorithm benchmarks are centralized comparisons; their CPU/RAM measurements concern
Windows ONNX serving processes, not Docker training workloads. Classification uses
49,904 held-out rows and threshold 0.5; no tuning uses the test split. Labels are synthetic.

Flower 1.39's compatibility Client/Server API emits deprecation warnings; the version
is pinned. The demo uses plaintext gRPC on a private Docker bridge, with no gRPC host
port published and dashboard ports bound to localhost. Physical LAN deployment needs
its own transport configuration and acceptance run. This research DP implementation
retains the threat-model and numerical limitations documented in `docs/privacy.md`.

Passing these checks demonstrates the tested scenarios, not an absence of all possible
future failures. Follow `docs/semester_demo.md` to reproduce the presentation.
