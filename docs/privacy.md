# Privacy design and limits

## Secure sum

Each pair of clients shares a 256-bit provisioned secret, absent from the server's
credential file. HMAC-SHA256 derives a stream key from that secret and a unique query ID.
ChaCha20 generates equal masks; the lexically lower client adds its mask and the other
subtracts it. Addition uses unsigned 64-bit modular arithmetic. Protected floating-point
histograms are rounded to fixed-point integers at scale 1,000,000 before masking.

All members of the fixed cohort must contribute. No dropout recovery secrets are given
to the server; a missing client aborts. Quantization error is at most 0.5 / scale per
client and component. Inputs outside the allowed numeric range are rejected.

Threat model: authenticated, honest-but-curious coordinator; honest, non-colluding clients;
trusted initial provisioning. This is not Byzantine-robust aggregation or a complete
malicious-server protocol. In a two-client federation, server/client collusion reveals
the other client's protected contribution by subtraction. Provisioning is performed on a
trusted machine; the local test intentionally co-locates everything and demonstrates
functional masking, not secrecy against the owner of that machine.

TLS protects credentials and data in transit. Pairwise masking does not provide transport
authentication. Use the server TLS flags and the client `ca_cert` setting for the LAN.

## Differential privacy mechanism

Privacy unit: one transaction record, add/remove adjacency, fixed public schema and cohort.
At a frontier, each record contributes to one node and one bin per feature. For gradient
g, Hessian h and count 1, the concatenated L2 norm is
`sqrt(number_of_features * (g² + h² + 1))`. All three contributions are jointly scaled by
`min(1, C / norm)`. This bounds the full frontier query's record sensitivity by C.

Each client adds independent Gaussian noise with standard deviation `sigma * C` to every
coordinate before quantization and masking. No raw counts, fraud rates or losses are
sent as training telemetry. Class weighting is fixed at 1, avoiding a non-private label
count query. Negative noisy Hessians/counts are clamped after aggregation as postprocessing.

The Gaussian accountant uses `rho = 1/(2*sigma²)` per released query and composes rho
across queries. At configured delta, `epsilon = rho + 2*sqrt(rho*log(1/delta))`.
There is no subsampling amplification claim. A query includes all frontier nodes jointly;
there is one charge per frontier release, including leaves. The client ledger composes
across runs with different sigma values and enforces its cumulative budget before release.
The server reports only current-run spending. Plain and secure-only modes have **no DP
guarantee**; epsilon zero in their internal accountant means no DP mechanism was run,
not perfect privacy. The dashboard labels them not applicable.

Duplicate requests reuse a persisted noisy response. Reusing the query ID with different
content is refused. Keys, cached responses, model files, and the SQLite ledger are local
state; protect and back them up together. A new identity or deleted ledger cannot erase
the mathematical privacy loss from prior releases of the same underlying data.

The guarantee is for the modeled histogram mechanism and its postprocessing, conditional
on public schema/cohort and trusted setup. The implementation uses OS-backed random
Gaussian sampling but floating-point sampling has not received an adversarial numerical
audit. Do not claim certified production DP. Data-dependent runtime, logs on the local
device, malicious clients, and private evaluation datasets are outside the guarantee.
Evaluation uses a designated held-out research dataset, not client training records.

## References

- Bun and Steinke, [Concentrated Differential Privacy](https://arxiv.org/abs/1605.02065):
  Gaussian zCDP and composition/conversion to epsilon-delta privacy.
- Cryptography, [symmetric encryption primitives](https://cryptography.io/en/latest/hazmat/primitives/symmetric-encryption/):
  ChaCha20 interface and key/nonce requirements.

## Utility interpretation

The dataset contains synthetic fraud labels. Report both ROC-AUC and PR-AUC, precision,
recall, F1, confusion matrix, and the decision threshold. A lower epsilon can reduce model
utility. Results are not evidence of real-bank fraud performance. The tree model is a
lightweight edge-compatible software model, not a GNN or neural FedAvg implementation.
