# Federated Learning Runbook — Multi-Device Deployment

This runbook covers three deployment modes for the UPI fraud detection federated GBDT,
from simplest to production-like.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | ≥ 3.10 | Same on all machines |
| Flower | 1.39.0 | `pip install -r requirements-fl.txt` |
| Dataset | `data/processed/upi_transactions_engineered.csv` | Only needed on the partitioning machine |

```bash
# On every machine
pip install -r requirements-fl.txt

# On the machine that creates the partitions
python fl/make_partitions.py --clients 3
```

---

## Mode 1: In-Process Simulation (1 machine)

No network, no Flower server process. Tests the algorithm in isolation.

```bash
# Quick sanity check (aggregation invariant only)
python fl/simulate.py --check-only

# Full training run
python fl/simulate.py --trees 60 --learning-rate 0.2 --save

# Compare with baseline
python fl/compare.py
```

**What to check:**
- `max |pooled − sum of clients|` should be ≤ 1e-6
- `federated vs pooled max probability difference` should be ≤ 1e-12
- `reports/fl/simulation.json` has the run details

---

## Mode 2: Localhost (1 machine, separate processes)

Real Flower runtime with server and clients as separate processes on the same machine.

### Step 1: Start the Flower SuperLink (server runtime)

```bash
# Terminal 1
flwr server --insecure
```

### Step 2: Start SuperNodes (client runtimes)

Open 3 more terminals, one per client:

```bash
# Terminal 2 — client 0
flwr client --insecure --superlink 127.0.0.1:9092 \
  --node-config "partition-id=0 data-path=data/fl/client_0.csv num-partitions=3"

# Terminal 3 — client 1
flwr client --insecure --superlink 127.0.0.1:9092 \
  --node-config "partition-id=1 data-path=data/fl/client_1.csv num-partitions=3"

# Terminal 4 — client 2
flwr client --insecure --superlink 127.0.0.1:9092 \
  --node-config "partition-id=2 data-path=data/fl/client_2.csv num-partitions=3"
```

### Step 3: Run the federated app

```bash
# Terminal 5, from the fl/ directory
cd fl
flwr run . local-simulation   # OR: the federation target matching localhost
```

### Step 4: Verify

```bash
# Check the round log
python -c "import json; d=json.load(open('reports/fl/fl_rounds.json')); print('hosts:', {h for t in d['trees'] for h in t['hostnames']})"

# Compare
python fl/compare.py --model models/fl/global_final.json
```

---

## Mode 3: Multi-Device (3 machines on the same network)

This is the deployment mode that demonstrates Objective 4 end-to-end.

### Network Layout

| Role | Machine | IP (example) | What it runs |
|------|---------|-------------|-------------|
| Server | A | 192.168.1.100 | SuperLink + ServerApp |
| Client 0 | B | 192.168.1.101 | SuperNode + ClientApp |
| Client 1 | C | 192.168.1.102 | SuperNode + ClientApp |
| Client 2 | A or B | (same) | Can double up on any machine |

### Step 1: Prepare data on each client machine

Copy the client's own shard to each machine. **Do not** copy other clients' data.

```bash
# On machine B
scp user@A:~/BE_Major_Project/data/fl/client_0.csv data/fl/client_0.csv

# On machine C
scp user@A:~/BE_Major_Project/data/fl/client_1.csv data/fl/client_1.csv
```

### Step 2: Start the SuperLink on machine A

```bash
# Machine A
flwr server --insecure
```

The SuperLink listens on port `9092` by default.

### Step 3: Start SuperNodes on machines B and C

```bash
# Machine B
flwr client --insecure --superlink 192.168.1.100:9092 \
  --node-config "partition-id=0 data-path=/path/to/client_0.csv num-partitions=3"

# Machine C
flwr client --insecure --superlink 192.168.1.100:9092 \
  --node-config "partition-id=1 data-path=/path/to/client_1.csv num-partitions=3"
```

Replace `192.168.1.100` with machine A's actual IP.

### Step 4: Run the federated app from machine A

```bash
cd fl
flwr run . --run-config "server-test-path=/abs/path/to/data/fl/server_test.csv"
```

### Step 5: Verify multi-device participation

```bash
python -c "
import json
d = json.load(open('reports/fl/fl_rounds.json'))
hosts = {h for t in d['trees'] for h in t['hostnames']}
print(f'Machines that contributed: {sorted(hosts)}')
print(f'Trees: {len(d[\"trees\"])}, Rounds: {d[\"federation_rounds\"]}')
for node_id, info in d['nodes'].items():
    print(f'  Node {node_id}: {info[\"hostname\"]} partition {info[\"partition_id\"]} ({info[\"rows\"]:,} rows)')
"
```

**The round log must show B's and C's hostnames** — this is the evidence for Objective 4.

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `no client answered the hello round` | SuperNodes not connected | Check firewall, IP, port 9092 |
| `could not find the schema` | Working directory wrong | Pass absolute `--run-config "schema-path=/abs/path/to/schema.json"` |
| `node out of sync - resending forest` | Client restarted mid-run | Normal — the resync protocol handles this automatically |
| ROC-AUC much lower than expected | Too few trees | Increase `num-trees` in pyproject.toml or `--run-config "num-trees=60"` |

---

## Configuration Reference

All parameters are in `fl/pyproject.toml` under `[tool.flwr.app.config]` and can be
overridden per run:

```bash
flwr run . --run-config "num-trees=60 learning-rate=0.2 max-depth=4"
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num-trees` | 60 | Number of boosting rounds |
| `max-depth` | 4 | Maximum tree depth |
| `learning-rate` | 0.2 | Shrinkage per tree |
| `lambda-l2` | 1.0 | L2 regularisation |
| `min-child-samples` | 50 | Minimum samples per leaf |
| `eval-every-trees` | 10 | Server-side scoring interval (0 = off) |
| `server-test-path` | `data/fl/server_test.csv` | Labelled test set for server scoring |
| `aggregation` | `sum` | Aggregation mode (Obj 6 adds `secure-agg`) |
