"""Flower ClientApp: one simulated bank/POS client.

The client holds its own transactions and nothing else. Every value it sends out is a sum,
so the server learns aggregated statistics and never a row, a model, or a single
transaction.

Protocol (the server drives; see ``server_app`` for the other half)

    op = hello   report row count, fraud count and hostname. The server needs the global
                 positive rate to weight the two classes the way ScalePosWeight does.
    op = hist    the server names the nodes it needs statistics for; reply with the per-bin
                 gradient/hessian/count totals for each of them.

State across messages
---------------------
A client keeps its shard and its running score, so the server only ever sends the tree that
just closed rather than the whole model. If the two ever disagree about how many trees the
client has folded in, the client answers ``need-resync`` and the server resends the forest -
so a client that restarts mid-run rejoins instead of silently training on a stale model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from flwr.common import (
    Array,
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.client import ClientApp

from upi_fl import bins, gbdt, paths, task

app = ClientApp()

# One instance serves every message of a run, so the shard is read and binned once. The
# cache is module level so it also survives the instance being recreated between messages.
_STATE: dict = {}


def _state(context: Context) -> dict:
    if _STATE:
        return _STATE
    schema = bins.load_schema(
        paths.resolve_path(context.run_config.get("schema-path", ""),
                           default=Path(__file__).with_name("schema.json"),
                           what="schema"))
    partition_id = int(context.node_config.get("partition-id", 0))
    data_path = context.node_config.get("data-path", "") or f"data/fl/client_{partition_id}.csv"
    shard = task.load_shard(paths.resolve_path(data_path, what="data"), schema)
    _STATE.update(schema=schema, shard=shard, partition_id=partition_id,
                  num_partitions=int(context.node_config.get("num-partitions", 1)))
    return _STATE


def _unpack_arrays(ar: ArrayRecord) -> dict:
    """Read numpy arrays back from an ArrayRecord."""
    return {k: ar[k].numpy() for k in ar}


def _reply(msg: Message, arrays: dict | None = None,
           configs: dict | None = None, metrics: dict | None = None) -> Message:
    """Build a reply with optional arrays (numpy), config (str) and metrics (numeric)."""
    content = RecordDict()
    if arrays:
        ar = ArrayRecord()
        for key, val in arrays.items():
            ar[key] = Array(ndarray=np.asarray(val))
        content["arrays"] = ar
    if configs:
        content["config"] = ConfigRecord(configs)
    if metrics:
        content["metrics"] = MetricRecord(metrics)
    return msg.create_reply(content)


def _base_reply_data(shard: task.Shard, state: dict) -> tuple[dict, dict]:
    """Return (config_dict, metric_dict) with base client info.

    String fields (hostname, partition-id) go in ConfigRecord;
    numeric fields go in MetricRecord.
    """
    configs = {
        "hostname": shard.hostname,
        "partition-id": state["partition_id"],
    }
    mets = {
        "num-examples": shard.n,
        "n": shard.n,
        "n-pos": shard.n_pos,
        "n-neg": shard.n_neg,
        "tree-count": shard.trees_seen,
    }
    return configs, mets


@app.train()
def train(msg: Message, context: Context) -> Message:
    state = _state(context)
    shard, schema = state["shard"], state["schema"]
    cfg = msg.content["config"]

    if cfg["op"] == "hello":
        configs, mets = _base_reply_data(shard, state)
        return _reply(msg, configs=configs, metrics=mets)

    # --- one round of growing the current tree -----------------------------------------
    tree_index = int(cfg["tree-index"])
    base_score = float(cfg["base-score"])

    # Unpack the arrays from the incoming message
    incoming = _unpack_arrays(msg.content["arrays"])

    if int(cfg["resync"]):
        task.reset(shard, base_score)
        for t in gbdt.unpack_trees(incoming, "forest"):
            task.apply_tree(shard, t)
    else:
        # Only the first level of a tree carries it, and on a resync the forest already
        # contains it, so the two branches are exclusive.
        for t in gbdt.unpack_trees(incoming, "prev"):
            task.apply_tree(shard, t)

    configs, mets = _base_reply_data(shard, state)
    if shard.trees_seen != tree_index:
        mets["need-resync"] = 1
        return _reply(msg, configs=configs, metrics=mets)

    frontier = [int(x) for x in incoming["frontier"]]
    current = gbdt.unpack_trees(incoming, "tree")[0]
    buf = task.histograms(shard, current, frontier, float(cfg["pos-weight"]), schema)

    mets["payload-floats"] = int(buf.size)
    # Local loss for the model as it stands (tree_index trees in): the server averages
    # these to show progress without needing a single row of client data.
    mets["loss"] = round(task.local_loss(shard), 8)
    return _reply(msg, arrays={"hist": buf}, configs=configs, metrics=mets)
