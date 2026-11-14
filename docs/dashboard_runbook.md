# Three-device software demonstration

Topology: device A coordinates training; devices B and C each train on one private shard.
The React dashboard is served by Python, so Node is only needed to build it. No GitHub
push, cloud account, or external hosting service is required.

## Quick local demonstration (one laptop, three processes)

Use Python 3.12 and Node 20 or newer. In PowerShell, from the project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-tested.txt
npm.cmd ci --prefix dashboard
npm.cmd run build --prefix dashboard
.\.venv\Scripts\python.exe fl/make_partitions.py --clients 2
.\.venv\Scripts\python.exe -m service.lab --trees 20 --depth 3 --serve
```

If the environment and dashboard are already built, only the last command is needed.
To reopen the most complete saved lab model without retraining, run `./Start-Dashboard.ps1`.
Do not start a second launcher while ports 8080–8082 are already in use.
The launcher runs plain, secure, and DP experiments, then leaves all three dashboards
running. Stop the launcher with Ctrl+C to stop its child services.

- Server: http://127.0.0.1:8080
- Client 0: http://127.0.0.1:8081
- Client 1: http://127.0.0.1:8082

The launcher prints the location of `server.json`. Read its `admin_token` locally and
paste it into **Run settings** to start or cancel additional experiments. Never commit
credential files. Tokens remain in the page's memory, not browser storage.

Generated credentials, client privacy ledgers, logs and models are under ignored
`runtime/lab-.../`. Measured comparison evidence is saved to
`reports/software_comparison.json`. Lab mode creates fresh simulated client identities;
do not use it to claim a cumulative privacy guarantee across repeated lab invocations.

## Prepare three physical devices

1. Build the dashboard and install the same Python environment on each device. On a Mac,
   use `python3.12 -m venv .venv` and `.venv/bin/python`. The tested lock file records the
   Windows environment; `requirements-dashboard.txt` provides portable dependency ranges.
2. Generate the two non-IID bank shards once with `fl/make_partitions.py --clients 2`.
   This also writes the schema, validation set, test set, and partition summary.
3. Provision credentials on a trusted setup machine:

   ```text
   python -m service.provision --out runtime/lan-keys --clients 2 --server https://SERVER_LAN_IP:8080
   ```

4. Device A receives only `server.json`, `fl/upi_fl/schema.json`, and the held-out
   `server_test.csv`. Device B receives only `client-0.json` and `client_0.csv`.
   Device C receives only `client-1.json` and `client_1.csv`. Both clients also receive
   the identical schema. Do not copy the complete data or credentials folder onto the server.
5. Edit each client configuration locally: set `data` and `state_dir` to paths on that
   device; set `ca_cert` to the trusted server certificate/CA file. Keep its `pair_keys`
   and token private. Do not reset or delete the ledger to bypass a privacy limit.
6. Use a certificate valid for the server hostname/IP and trusted by both clients. The
   runtime supports TLS directly; no certificate-verification bypass is needed.

Server (device A):

```text
python -m service.server --credentials runtime/lan-keys/server.json --host 0.0.0.0 --port 8080 --cert runtime/tls/server.crt --key runtime/tls/server.key
```

Client 0 (device B):

```text
python -m service.client --config runtime/client-0.json --port 8081
```

Client 1 (device C):

```text
python -m service.client --config runtime/client-1.json --port 8081
```

Open the server dashboard at `https://SERVER_LAN_IP:8080`; each client opens its own
`http://127.0.0.1:8081`. Client inference UI intentionally binds to loopback: local
transaction examples are not made available to the other devices. Permit inbound TCP
8080 to the server on the private network. Clients only require outbound access.
For an isolated lab using synthetic data, HTTP is supported, but it is unencrypted and
must not be described as secure transport. No firewall settings are changed automatically.

## Demonstration sequence

1. Show both client dashboards with their own local record count and coordinator connection.
2. On the server, choose secure aggregation, 20 trees, depth 3, and start.
3. Show the event timeline: dispatch → client update receipt → aggregate → global tree.
4. Show **completed** only after both clients acknowledge the final model checksum.
5. On each client, open **Local inference**, load a local example, and assess risk.
   This reports model risk and measured inference time, not a payment authorization.
6. Run secure + DP and compare accuracy, precision, recall, F1, ROC-AUC and PR-AUC.
   Client ε is cumulative; the server's ε is for that run only.
7. Export evidence. The JSON snapshot contains no credentials or transaction records.

## Failure behavior

- Training runs in a background worker, independent of HTTP request timeouts.
- Polling, uploads, and connection attempts use bounded network timeouts.
- Requests identify the complete model and query. A restarted client rebuilds scores.
- A durable SQLite transaction commits each protected release and privacy cost before upload.
- Duplicate replies are idempotent; conflicting duplicates and stale rounds are rejected.
- Missing clients cause a bounded failed run. Partial secure aggregates are never used.
- A coordinator restart marks an active run interrupted. Start a new run; checkpoint
  files are evidence, not a promise of automatic server-side resume.
- Cancelling does not refund already released privacy budget.
- Previously installed valid client models remain usable if a later run fails.

## Repeatable verification

```text
python -m unittest discover -s tests -v
python -m service.network_check
python -m service.scalability
npm run build --prefix dashboard
```

The network check uses ports 8180–8182 and synthetic fixtures. It actually terminates and
restarts a child client, then verifies a separate missing-client deadline. The scalability
check compares 2, 3, and 5 simulated clients in one process; it is not a LAN benchmark.

The original Flower runtime is retained for compatibility and its tests still run.
Its original commands do **not** enable the new protections or dashboard. Use `service.server`
and `service.client` for the integrated software demonstration.
