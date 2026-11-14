"""Record-level Gaussian mechanism and all-participant pairwise secure sum.

Threat model: honest-but-curious server, non-colluding clients, provisioned pair keys.
No dropout unmasking: losing a participant aborts the round. See docs/privacy.md.
"""
import base64
import hashlib
import math
import secrets

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from .common import gbdt

SCALE = 1_000_000


def epsilon(queries, sigma, delta):
    if queries == 0:
        return 0.0
    if sigma <= 0 or not 0 < delta < 1:
        raise ValueError('invalid privacy parameters')
    rho = queries / (2 * sigma * sigma)
    return rho + 2 * math.sqrt(rho * math.log(1 / delta))


def private_histogram(shard, tree, frontier, schema, clip, sigma):
    # Each record appears at exactly one frontier node and one bin per feature.
    # Clip its JOINT gradient, Hessian and count contribution across all features.
    g, h = gbdt.gradients(shard.raw, shard.y, 1.0)
    w = np.minimum(1.0, clip / np.sqrt(schema['n_features'] * (g*g + h*h + 1)))
    nodes = tree.node_index(shard.binned)
    stats = []
    for nid in frontier:
        rows = np.flatnonzero(nodes == nid)
        parts = [np.zeros(schema['total_bins']) for _ in range(3)]
        for j, (offset, size) in enumerate(zip(schema['offsets'], schema['bin_counts'])):
            for dst, weights in zip(parts, (g*w, h*w, w)):
                dst[offset:offset+size] = np.bincount(shard.binned[rows, j], weights=weights[rows], minlength=size)
        stats.append(parts)
    buf = gbdt.pack_histograms(stats)
    # SystemRandom supplies OS-backed entropy; never reuse a public experiment seed.
    rng = secrets.SystemRandom()
    noise = np.fromiter((rng.gauss(0, sigma * clip) for _ in range(buf.size)), float)
    return buf + noise


def mask(values, client_id, pair_keys, query_id):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or np.max(np.abs(values), initial=0) > 1e10:
        raise ValueError('histogram outside fixed-point range')
    encoded = np.rint(values * SCALE).astype('<i8').view('<u8').copy()
    for peer, key in sorted(pair_keys.items()):
        # Derive a query-specific stream key, with domain separation and 256-bit context.
        import hmac
        stream_key = hmac.new(bytes.fromhex(key), query_id.encode(), hashlib.sha256).digest()
        stream = Cipher(algorithms.ChaCha20(stream_key, bytes(16)), mode=None).encryptor()
        pad = np.frombuffer(stream.update(bytes(encoded.nbytes)), dtype='<u8')
        if client_id < peer:
            encoded = encoded + pad
        else:
            encoded = encoded - pad
    return base64.b64encode(encoded.tobytes()).decode()


def decode_payload(payload, expected_size):
    raw = base64.b64decode(payload, validate=True)
    if len(raw) != expected_size * 8:
        raise ValueError('incorrect histogram length')
    return np.frombuffer(raw, dtype='<u8')


def secure_sum(payloads, size):
    total = np.zeros(size, dtype='<u8')
    for payload in payloads:
        total += decode_payload(payload, size)
    return total.view('<i8').astype(float) / SCALE
