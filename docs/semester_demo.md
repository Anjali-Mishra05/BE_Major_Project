# Semester presentation: simulated POS devices

The software demonstration uses one laptop, one Flower coordinator and two or three
Docker POS containers. Each POS sees only its own training CSV, credential file and
persistent state volume. The coordinator sees a held-out evaluation CSV and client
authentication tokens, not client pairwise masking keys or training CSVs. All containers
share the laptop kernel/CPU: these are simulated devices, not physical edge hardware.

## Prepare and launch

### On this already-prepared laptop

The demo instances were stopped after testing. Keep the existing credentials and Docker
volumes. Start Docker Desktop, open PowerShell, and run:

```powershell
cd C:\Users\ABCOM\Desktop\BE_Major_Porject
docker compose -f runtime/docker-demo/compose.json up -d
docker compose -f runtime/docker-demo/compose.json ps
```

Wait until the server and all three POS services show healthy. Open the four URLs below.
The saved models and privacy ledgers survive shutdown. **Do not rerun the preparation
command on this existing directory.** If you changed source files, use `up --build -d`.

### On a fresh checkout

Start Docker Desktop in Linux-container mode. From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dashboard.txt
.\.venv\Scripts\python.exe -m service.docker_demo --clients 3
docker compose -f runtime/docker-demo/compose.json up --build -d
docker compose -f runtime/docker-demo/compose.json ps
```

The preparation command runs once per new experiment directory. It refuses to overwrite
existing credentials or ledgers. Use `--clients 2` for two POS clients; use `--out` to
choose a separate empty directory. Docker state volumes are preserved across restarts.
Do not reset volumes to pretend prior privacy spending never happened.

Open the server at http://localhost:8080 and POS dashboards at ports 8081, 8082 and 8083.
Stop any earlier dashboard launcher using those ports first. Read the administrator token
from `runtime/docker-demo/keys/server.json` and enter it in the server's Run settings.
Keep that file out of screenshots and Git.

Start with three trees, depth two and a 90-second round deadline for a short demonstration;
use twenty trees/depth three for a longer evaluation. Show each client's update being
accepted, server aggregation, final model delivery and local transaction inference.
The final JSON model is converted and parity-checked on every POS before acknowledgement;
the prediction API then uses ONNX Runtime CPU. Public fixed feature binning runs locally
before the ONNX graph. The ONNX input is binned features, not raw payment records.

The Docker adapter uses the Flower 1.39 compatibility Client/Server gRPC API with the
existing histogram aggregation algorithm. It is **not FedAvg**. Flower emits deprecation
warnings for this API; the version is pinned for this demo. The original SuperLink/
SuperNode app remains available separately. The original HTTP dashboard launcher also
continues to work. Docker's gRPC port is only on the private bridge network; dashboards
are bound to host loopback. This configuration is for a trusted local demonstration.

## Compare algorithms and transaction loads

```powershell
.\.venv\Scripts\python.exe -m service.benchmarks --seconds 5 --rates 50 250 1000
```

This trains logistic regression, a decision tree and a random forest on the same
232,885 training rows and evaluates on the same 49,904 held-out rows. The validation
split is reserved and unused; hyperparameters are fixed, not tuned on the test set.
Every model is exported to ONNX and compared with native probabilities before benchmarking.
The dashboard's Comparison page shows accuracy, precision, recall, F1, ROC-AUC, PR-AUC,
model size, ONNX parity error, latency, achieved throughput, CPU and RAM for each load.
These centralized algorithm experiments are separate from plain/secure/DP federated runs.

Each model/load starts a fresh subprocess, uses one ONNX thread and scheduled single-row
requests. Inference time excludes scheduling and feature preprocessing; latency with queue
includes scheduling delay and backlog. CPU is process CPU time divided by wall time;
100% means one logical core. RAM is process RSS, including interpreter, model and loaded
test inputs, sampled after each inference. The short run is a demonstration benchmark,
not a sustained capacity certification. Close other heavy workloads before measuring.
Dataset fraud labels are synthetic; results do not establish real-bank performance.

Models and benchmark input arrays stay under ignored `runtime/`; the small measured
`reports/algorithm_benchmarks.json` is the presentation evidence. Refresh Comparison
after rerunning. Reports are mounted read-only into the server container.

## Acceptance checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m service.demo_check --faults
```

The Docker check requires the containers to be running. It trains plain, secure and DP
runs, verifies ONNX predictions on every POS, restarts one POS during a secure run and
injects a dropout to verify bounded failure without partial aggregation. It restores
the stopped container and writes `reports/docker_acceptance.json` only after passing.
It deliberately leaves the server's last run failed after the dropout test; start a fresh
run for the presentation. Tests consume genuine DP ledger budget.

Stop the demonstration without deleting ledgers:

```powershell
docker compose -f runtime/docker-demo/compose.json down
```

## Presentation order

1. Problem, dataset limitations and separate private POS partitions.
2. Diagram: three POS containers exchange protected updates with Flower; raw training rows stay local.
3. Live server and individual POS dashboards during training and model delivery.
4. Score a local transaction with the deployed ONNX model.
5. Compare different algorithms, inference time and CPU/RAM under transaction loads.
6. Show secure aggregation, DP epsilon/delta and the accuracy/privacy tradeoff.
7. Show restart/dropout acceptance evidence and explain deferred physical edge deployment.

Hardware integration and device-specific performance remain next-semester work. Do not
claim all eight original objectives are fully accepted or describe histogram GBDT as GNN.

## What to show the examiner: 10–12 minute walkthrough

| Time | Screen / action | Explanation |
|---|---|---|
| 0–1 min | Introduce the problem and architecture | “We simulate three POS terminals on one laptop. Each trains on its own transaction partition. Flower coordinates updates; the server builds the shared fraud model.” |
| 1–2 min | Show `docker compose ... ps`, then the server's Clients tab | Identify one server and three separate POS containers. Wait for **3 / 3 connected**. Explain that containers simulate devices; physical POS/Edge AI deployment is deferred. |
| 2–4 min | Server → Run settings → start a secure run | Enter the local admin token privately. Choose Secure aggregation, 3 trees, depth 2, timeout 90 seconds. Start training, then return to Overview. Show client updates arriving, aggregation events, completed exchanges and the changing global metrics. |
| During training | Switch to one POS → Overview | Show that device's local transaction count, training state, cumulative privacy spending and installed model. The installed model can remain available while a new model trains. Individual clients send protected histograms, not whole transaction records. |
| 4–5 min | Server shows completed / global model deployed | Explain that every POS must acknowledge the same final model. The server cannot declare successful deployment with a missing participant. |
| 5–6 min | One POS → Local inference | Click **Load one local transaction**, then **Assess risk**. Point to fraud risk, review threshold, latency, tree count and **ONNX Runtime CPU**. A score is a model prediction, not proof of actual fraud. |
| 6–8 min | Server → Comparison, first two sections | Show the three different ML algorithms and their accuracy/precision/recall/F1/AUC. Scroll through load measurements at 50, 250 and 1,000 transactions/sec; compare inference p50/p95, achieved TPS, queue delay, CPU and peak RAM. These are saved measured results, not measurements being recomputed by opening the page. |
| 8–10 min | Server → Privacy; then Comparison's lower sections | Explain plain vs secure vs secure+DP. Secure aggregation hides individual contributions under the stated threat model; DP clips contributions and adds noise. Show epsilon/delta and cumulative client budget. Use the saved privacy comparison to discuss accuracy/privacy tradeoffs. |
| 10–12 min | Open validation reports; conclude with next semester | Show passing Docker/ONNX/restart/dropout evidence. Explain that dropout produces a controlled failure rather than accepting an incomplete aggregate. Physical edge-device latency, peripherals and deployment remain future work. |

### Browser tabs to prepare

| Tab | URL | Purpose |
|---|---|---|
| Server | http://localhost:8080 | Overview, Clients, Run settings, Privacy and Comparison |
| POS 1 | http://localhost:8081 | Client 0's private data, training and ONNX inference |
| POS 2 | http://localhost:8082 | Client 1's independent workspace |
| POS 3 | http://localhost:8083 | Client 2's independent workspace |

Use the server Overview and one POS tab side by side during training. All three POS
pages are equivalent interfaces backed by different shards and ledgers. The generated
bank split contains 77,005 / 81,318 / 74,562 training rows respectively. They are simulated
POS datasets, not three physical payment terminals or a live payment network.

For an optional live DP run, choose **Secure aggregation + DP**, 3 trees, depth 2,
sigma 8, clipping norm 10, timeout 90 seconds. Check the cumulative budget first.
Do not delete the ledger to bypass a budget limit. Avoid running load benchmarks or
fault injection during the main presentation: prepare their evidence beforehand.

## Where to find the benchmarks already completed

All paths below are relative to the repository root. The saved JSON reports are small
and intended for Git. You do **not** need to rerun training to show these tables.

| Evidence | File / dashboard location | What it proves |
|---|---|---|
| Different algorithms + load benchmarks | `reports/algorithm_benchmarks.json`; server → Comparison → first two sections | Same 49,904-row held-out set; three algorithms; ONNX parity; inference, throughput, CPU and RAM at three offered rates |
| Plain / secure / DP comparison | `reports/software_comparison.json`; server → Comparison → lower sections | Earlier 20-tree privacy experiments and saved centralized baseline; different experiment from the short live 3-tree demonstration |
| Docker restart/dropout test | `reports/docker_acceptance.json` | Three POS clients; all modes completed; model predictions agree; restart recovered; missing client rejected |
| Final-image normal runs | `reports/docker_smoke.json` | Plain, secure and DP runs completed with matching ONNX predictions on all three POS clients |
| Native Flower process recovery | `reports/flower_dashboard_acceptance.json` | Flower transport restart and dropout checks outside Docker |
| Overall test record | `reports/semester_validation.md` | 28 passing automated tests, builds, dependency checks and interpretation limits |
| Existing scaling experiment | `reports/scalability.json` | Synthetic in-process 2/3/5-client scaling; not a physical-network benchmark |

The latest saved algorithm accuracy is **86.82% logistic regression**, **89.26% decision
tree**, and **87.35% random forest**. Read the report for exact metrics and per-load timings;
do not present these as federated accuracies or hardware-edge measurements. The algorithm
benchmarks run on Windows ONNX CPU processes; Docker tests demonstrate federation.

To inspect a report in PowerShell:

```powershell
Get-Content reports/algorithm_benchmarks.json
Get-Content reports/docker_acceptance.json
```

Benchmark ONNX files and inputs are under ignored `runtime/benchmarks/`. Deployed Docker
models live at `/state/model.onnx` inside each POS's persistent volume. Model binaries are
regenerated locally; the committed reports retain the results. To refresh the benchmark
tables after a new measurement, reload the dashboard or leave and re-enter Comparison.
**Export evidence** downloads the current dashboard state; it does not bundle all saved
benchmark reports. Use the files above for the presentation's complete evidence.

## Rehearsal and shutdown checklist

1. Start Docker Desktop and the existing compose setup; verify all four services healthy.
2. Open all dashboard tabs; confirm three clients connected and saved comparisons populated.
3. Keep the admin token private. Rehearse one short secure run and one local ONNX prediction.
4. Keep the benchmark JSON and validation Markdown available as backup presentation evidence.
5. If a run fails, show the actual error and inspect logs; do not describe it as completed.
   Use `docker compose -f runtime/docker-demo/compose.json logs --tail 50` to diagnose.
6. Stop with `docker compose -f runtime/docker-demo/compose.json down`; close dashboard tabs.
   Do not add `--volumes` / `-v`, which would delete the persistent models and privacy ledgers.
