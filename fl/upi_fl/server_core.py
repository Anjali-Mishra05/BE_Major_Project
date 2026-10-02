"""The server-side training loop, shared by ``fl/simulate.py`` and the Flower ServerApp.

Keeping it here means the local simulation and the three-device run execute the same code
path; the only difference is what ``fetch`` does when the server needs statistics.
"""

from __future__ import annotations

import numpy as np

from .gbdt import Forest, Tree, grow_tree


def train_forest(schema: dict, params: dict, num_trees: int, fetch, *,
                 pos_weight: float = 1.0, base_score: float = 0.0,
                 on_tree=None) -> Forest:
    """Build ``num_trees`` trees, one server<->client exchange per depth level per tree.

    ``fetch(tree_index, tree, frontier, prev_tree, depth) -> {node_id: (G, H, C)}`` is one
    round: the server names the nodes it needs statistics for and gets back the summed
    per-bin totals. ``prev_tree`` is the tree the clients must fold into their running score
    before they can compute gradients for the tree being grown, and it is only sent on the
    first level of the tree.
    """
    forest = Forest(schema, params, base_score)
    for k in range(num_trees):
        tree = Tree(forest.params["max_depth"])
        prev = forest.trees[-1] if forest.trees else None

        def fetch_level(frontier, depth, k=k, tree=tree, prev=prev):
            return fetch(k, tree, frontier, prev, depth)

        gains = grow_tree(schema, forest.params, tree, fetch_level)
        for j, g in gains:
            forest.gain[j] += g
        forest.add_tree(tree)
        if on_tree is not None:
            on_tree(k, tree, forest)
    return forest


def sum_histograms(per_client: list[dict], node_ids, total_bins: int) -> dict:
    """Add the clients' per-bin totals. This is the aggregation Objective 5 asks for: a
    plain sum, which is also the exact shape Secure Aggregation needs if Objective 6 is
    picked up later."""
    out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for nid in node_ids:
        G = np.zeros(total_bins, dtype=np.float64)
        H = np.zeros(total_bins, dtype=np.float64)
        C = np.zeros(total_bins, dtype=np.float64)
        for stats in per_client:
            g, h, c = stats[nid]
            G += g
            H += h
            C += c
        out[nid] = (G, H, C)
    return out
