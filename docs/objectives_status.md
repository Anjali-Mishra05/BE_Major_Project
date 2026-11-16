# Software objectives and acceptance evidence

The supplied objective document is the requirements source. Hardware procurement,
physical POS assembly, Jetson/Raspberry Pi deployment and device-specific latency remain
outside this software pass, as requested.

| Objective | Software implementation | Evidence / remaining boundary |
|---|---|---|
| 1. Dataset and preprocessing | Existing anonymized research dataset and preprocessing notebook; non-IID bank partitioner; 70/15/15 disjoint train/validation/test; label-derived behaviour columns excluded | Partition summary records 232,885 training rows and 49,904 held-out test rows. Public fixed amount bins remove test/private-extrema dependence. Source provenance and real-world collection are not independently verified. |
| 2. Centralized baseline | Existing trained LightGBM baseline and evaluation tools retained | Saved baseline re-evaluated on the same test rows in the comparison report; historical training results remain labeled historical. |
| 3. Lightweight edge model | Compact tree model exported to ONNX; local ONNX Runtime inference; parity and checksum acknowledgement before deployment completes | Runs on laptop CPU. Physical edge-device validation is deferred. |
| 4. Federated clients | Flower gRPC server with three Docker POS clients, each mounting its own shard; HTTP runtime also retained | Docker normal/restart/dropout acceptance passed. Three physical laptops still need a LAN acceptance run. |
| 5. Central aggregation | Server sums per-bin statistics and grows global trees; baseline comparison | Plain and securely masked runs compared on identical held-out data; tests compare predictions and reject incomplete cohorts. |
| 6. Secure aggregation and DP | Pairwise cryptographic masks; record clipping and Gaussian noise; durable cumulative accountant | Mask cancellation, replay, budget and recovery tests. Research threat model and numerical limitations documented in privacy.md. |
| 7. Integrated prototype | React server/client dashboards, model delivery, local transaction inference | Functional software prototype. Payment rails, scanners and physical POS integration deferred. |
| 8. Evaluation | Classification metrics, inference p50/p95, model bytes, application payload bytes, privacy spending, synthetic 2/3/5-client scaling | Reports under reports/. Network payload figures exclude HTTP/TLS/TCP framing. Scaling test is single-process and does not establish physical-network capacity. |

No finite test suite can promise zero future bugs or prevent a device/network from failing.
Acceptance here means observed normal runs complete, injected failures terminate within a
deadline, protected retries preserve the release, and incomplete aggregates are rejected.

The original project uses histogram GBDT. The integrated runtime retains that model family.
If the academic specification mandates **GNN** specifically (rather than the broader
AI-model wording in the objective document), that is a separate unfulfilled modeling
requirement; this implementation must not be presented as GNN or FedAvg.

See dashboard_runbook.md for repeatable commands and privacy.md for assumptions.

## Semester simulation additions

The one-laptop simulation now has a Flower gRPC adapter for the integrated dashboard,
Docker configuration generation for two or three isolated POS clients, ONNX export and
local ONNX Runtime inference. Separate algorithm benchmarks train logistic regression,
decision tree and random forest models and report held-out classification metrics plus
latency, throughput, CPU and RAM under scheduled transaction loads. Generated models,
credentials and shards remain ignored; small reports provide evidence.

See [semester_demo.md](semester_demo.md) for exact commands and methodology. These
additions address the semester's software demonstration scope; they do not establish
physical-device performance or fulfill a separately mandated GNN requirement.
