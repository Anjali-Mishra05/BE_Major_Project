"""
Runs the whole federated boosting loop inside one process: no Flower, no network.

Two jobs.

1. Test the algorithm before it is wired into Flower, so a failure here is a maths bug and
   not a transport bug.
2. Assert the invariant the whole approach rests on: summing the clients' per-bin
   histograms gives exactly the histogram you would get by pooling the rows first. That is
   what makes the federated model not an approximation of a centralized one, and it is why
   the aggregation can be a plain - and later, a securely masked - sum.

    python fl/simulate.py                     # 3 clients, 30 trees, depth 4
    python fl/simulate.py --trees 120 --depth 4 --save
    python fl/simulate.py --check-only        # just the aggregation invariant

Writes reports/fl/simulation.json and, with --save, models/fl/simulated_model.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fl"))

from upi_fl import bins, gbdt, metrics, server_core, task  # noqa: E402

FL_DIR = ROOT / "data" / "fl"
REPORTS = ROOT / "reports" / "fl"
W = 78


def rule(char: str = "-") -> None:
    print(char * W)


def pooled_shard(shards: list[task.Shard]) -> task.Shard:
    """One client that happens to hold every row: the centralized stand-in, used only to
    check that federation changed nothing."""
    return task.Shard(
        path="(pooled)",
        binned=np.vstack([s.binned for s in shards]),
        y=np.concatenate([s.y for s in shards]),
        raw=np.zeros(sum(s.n for s in shards), dtype=np.float64),
        hostname="pooled-control",
    )


def load_server_test(schema: dict, path: Path):
    df = pd.read_csv(path)
    y = df.pop(task.LABEL_COLUMN).to_numpy(dtype=np.float64)
    df = df.drop(columns=[task.TIME_COLUMN], errors="ignore")
    return bins.binned_matrix(df[schema["features"]], schema), y


def test_aggregation_exact(schema, shards, pooled, params, pos_weight, base_score):
    """Grow one tree from the pooled histogram, and at every node compare it with the sum
    of the clients' histograms. Any difference would mean the federated model is not the
    centralized model, which is the claim the report makes."""
    total_bins = schema["total_bins"]
    for s in shards:
        task.reset(s, base_score)
    task.reset(pooled, base_score)

    tree = gbdt.Tree(params["max_depth"])
    worst = {"delta": 0.0, "nodes": 0}

    def fetch(node_ids, depth):
        node = tree.node_index(pooled.binned)
        g, h = gbdt.gradients(pooled.raw, pooled.y, pos_weight)
        pooled_stats = {nid: gbdt.node_histogram(pooled.binned, g, h,
                                                np.flatnonzero(node == nid), schema)
                        for nid in node_ids}
        per_client = [gbdt.unpack_histograms(
            task.histograms(s, tree, node_ids, pos_weight, schema), node_ids, total_bins)
            for s in shards]
        summed = server_core.sum_histograms(per_client, node_ids, total_bins)
        for nid in node_ids:
            worst["nodes"] += 1
            for A, B in zip(pooled_stats[nid], summed[nid]):
                worst["delta"] = max(worst["delta"], float(np.max(np.abs(A - B))))
        return pooled_stats

    gbdt.grow_tree(schema, params, tree, fetch)
    return worst


def make_fetch(shards, pos_weight, base_score, schema, ledger, comm):
    """The simulation's stand-in for the network: one call = one server<->client round."""
    total_bins = schema["total_bins"]

    def fetch(k, tree, frontier, prev_tree, depth):
        if ledger["last_k"] is None:                      # first ever round: start clean
            for s in shards:
                task.reset(s, base_score)
        if ledger["last_k"] != k:
            if prev_tree is not None:                     # fold in the tree that just closed
                for s in shards:
                    task.apply_tree(s, prev_tree)
            ledger["last_k"] = k

        per_client = []
        for s in shards:
            buf = task.histograms(s, tree, frontier, pos_weight, schema)
            comm["client_floats"] += int(buf.size)
            per_client.append(gbdt.unpack_histograms(buf, frontier, total_bins))
        return server_core.sum_histograms(per_client, frontier, total_bins)

    return fetch


def train(schema, params, shards, num_trees, pos_weight, base_score,
          ledger=None, comm=None, on_tree=None):
    ledger = {} if ledger is None else ledger
    comm = {"client_floats": 0} if comm is None else comm
    ledger.setdefault("last_k", None)
    forest = server_core.train_forest(
        schema, params, num_trees,
        make_fetch(shards, pos_weight, base_score, schema, ledger, comm),
        pos_weight=pos_weight, base_score=base_score, on_tree=on_tree)
    return forest, comm


# ------------------------------------------------------------------------------------ main

def baseline_reference() -> dict | None:
    """The Phase I numbers, read only. Nothing here writes to the baseline reports."""
    path = ROOT / "reports" / "baseline_metrics.json"
    if not path.exists():
        return None
    blob = json.loads(path.read_text())
    return blob.get("stratified", {}).get("lightgbm", {})


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulate the federated GBDT locally.")
    ap.add_argument("--clients", type=int, default=3, help="clients to simulate (default 3)")
    ap.add_argument("--trees", type=int, default=30, help="boosting rounds (default 30)")
    ap.add_argument("--depth", type=int, default=4, help="max tree depth (default 4)")
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--lambda-l2", type=float, default=1.0)
    ap.add_argument("--min-child-samples", type=int, default=50)
    ap.add_argument("--check-only", action="store_true",
                    help="run only the aggregation invariant and exit")
    ap.add_argument("--skip-control", action="store_true",
                    help="skip the pooled single-client control run (it doubles the runtime)")
    ap.add_argument("--save", action="store_true", help="write models/fl/simulated_model.json")
    args = ap.parse_args()

    print()
    rule("=")
    print(" FEDERATED GBDT - LOCAL SIMULATION".center(W))
    print(" Objectives 4 and 5  |  no Flower, no network: algorithm only".center(W))
    rule("=")

    schema = bins.load_schema()
    total_bins = schema["total_bins"]
    params = {"max_depth": args.depth, "learning_rate": args.learning_rate,
              "lambda_l2": args.lambda_l2, "min_child_samples": args.min_child_samples,
              "min_split_gain": 1e-6}

    shards = []
    for i in range(args.clients):
        path = FL_DIR / f"client_{i}.csv"
        if not path.exists():
            sys.exit(f"missing {path}\n"
                     f"Run first:  python fl/make_partitions.py --clients {args.clients}")
        shards.append(task.load_shard(path, schema))
    pooled = pooled_shard(shards)

    n_pos = sum(s.n_pos for s in shards)
    n_neg = sum(s.n_neg for s in shards)
    pos_weight = n_neg / max(n_pos, 1)
    base_score = 0.0

    print(f"\n  {len(shards)} clients, {sum(s.n for s in shards):,} rows total "
          f"({n_pos:,} fraud, {n_neg:,} legitimate)")
    print(f"  pos_weight {pos_weight:.4f}   total bins {total_bins}   depth {args.depth}   "
          f"learning rate {args.learning_rate}")
    for i, s in enumerate(shards):
        print(f"    client {i}: {s.n:>7,} rows   {s.n_pos:>6,} fraud   "
              f"({s.n_pos / s.n * 100:.2f}%)   {s.hostname}")

    # --- 1. the invariant
    rule()
    print("  [1/3] aggregation invariant: summed client histograms vs pooled histogram")
    worst = test_aggregation_exact(schema, shards, pooled, params, pos_weight, base_score)
    ok = worst["delta"] <= 1e-6
    print(f"    {worst['nodes']} nodes compared, max |pooled - sum of clients| = "
          f"{worst['delta']:.3e}   {'OK' if ok else 'FAILED'}")
    print("    (this is what lets a client send statistics instead of rows)")
    if not ok:
        sys.exit("\nAggregation is not exact - stopping before the training run.")
    if args.check_only:
        return

    # --- 2. the federated run
    rule()
    print(f"  [2/3] federated run: {args.trees} trees x depth {args.depth} x "
          f"{len(shards)} clients")
    ledger: dict = {}
    comm = {"client_floats": 0}
    t0 = time.perf_counter()

    def progress(k, tree, forest):
        if (k + 1) % 10 == 0 or k == 0:
            print(f"    tree {k + 1:>3}/{args.trees}   nodes {tree.n_nodes}   "
                  f"{time.perf_counter() - t0:6.1f}s")

    fed, comm = train(schema, params, shards, args.trees, pos_weight, base_score,
                      ledger, comm, on_tree=progress)
    fed_seconds = time.perf_counter() - t0

    # --- 3. same algorithm, one client that happens to hold every row
    if args.skip_control:
        ctrl, ctrl_metrics, pooled_note = None, None, "pooled control skipped"
        max_gap = float("nan")
    else:
        print(f"\n  [3/3] control: the same {args.trees} trees on a single pooled client")
        ctrl, _ = train(schema, params, [pooled], args.trees, pos_weight, base_score)
        pooled_note = "pooled control (1 client)"

    test_binned, test_y = load_server_test(schema, FL_DIR / "server_test.csv")
    fed_p = fed.predict_proba(test_binned)
    if ctrl is not None:
        ctrl_p = ctrl.predict_proba(test_binned)
        max_gap = float(np.max(np.abs(fed_p - ctrl_p)))
        ctrl_metrics = metrics.evaluate(test_y, ctrl_p, 0.5)
    fed_metrics = metrics.evaluate(test_y, fed_p, 0.5)

    rule("=")
    print(" RESULT".center(W))
    rule("=")
    print(f"  server test set: {len(test_y):,} rows, {int(test_y.sum()):,} fraud "
          f"- the Phase I test partition")
    if ctrl is not None:
        print(f"  federated vs pooled control: max probability difference {max_gap:.3e}")
    else:
        print("  pooled control skipped (--skip-control)")
    print(f"\n  {'model':<28}{'ROC-AUC':>10}{'PR-AUC':>10}{'acc@0.5':>10}{'F1@0.5':>10}")
    rule()
    print(f"  {'federated (' + str(len(shards)) + ' clients)':<28}"
          f"{fed_metrics['roc_auc']:>10.4f}{fed_metrics['pr_auc']:>10.4f}"
          f"{fed_metrics['accuracy']:>10.4f}{fed_metrics['f1']:>10.4f}")
    if ctrl_metrics is not None:
        print(f"  {pooled_note:<28}"
              f"{ctrl_metrics['roc_auc']:>10.4f}{ctrl_metrics['pr_auc']:>10.4f}"
              f"{ctrl_metrics['accuracy']:>10.4f}{ctrl_metrics['f1']:>10.4f}")

    base = baseline_reference()
    if base:
        tuned = base["test_tuned_threshold"]
        print(f"  {'Phase I LightGBM baseline':<28}{tuned['roc_auc']:>10.4f}"
              f"{tuned['pr_auc']:>10.4f}{tuned['accuracy']:>10.4f}{tuned['f1']:>10.4f}")
        print(f"  (the baseline is quoted at its validation-tuned threshold "
              f"{tuned['threshold']:.3f}; the federated split has no server-side")
        print("   validation partition to tune on, so ROC-AUC and PR-AUC are the fair")
        print("   comparison and the accuracy/F1 columns are at a fixed 0.5)")

    payload = {
        "clients": len(shards),
        "num_trees": args.trees,
        "max_depth": args.depth,
        "learning_rate": args.learning_rate,
        "lambda_l2": args.lambda_l2,
        "pos_weight": round(pos_weight, 6),
        "seconds": round(fed_seconds, 2),
        "rounds": args.trees * (args.depth + 1) + 1,
        "aggregation_max_delta": worst["delta"],
        "aggregation_nodes": worst["nodes"],
        "federated_vs_pooled_max_probability_gap": max_gap,
        "federated": fed_metrics,
        "pooled_control": ctrl_metrics,
        "client_floats_total": comm["client_floats"],
        "client_megabytes_total": round(comm["client_floats"] * 8 / 1e6, 4),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "simulation.json"
    out.write_text(json.dumps(payload, indent=2))
    if args.save:
        fed.save(ROOT / "models" / "fl" / "simulated_model.json")
        print("\n  model written to models/fl/simulated_model.json")
    print(f"  report written to {out.relative_to(ROOT)}")
    rule("=")


if __name__ == "__main__":
    main()
