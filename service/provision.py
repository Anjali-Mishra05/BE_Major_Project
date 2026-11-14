"""Generate separate server and client credentials for a trusted lab setup."""
import argparse
import secrets
from pathlib import Path
from .common import atomic_json


def provision(out, count=2, server='http://127.0.0.1:8080'):
    if count < 2 or count > 20:
        raise ValueError('choose 2 to 20 clients')
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('credential directory must be empty; existing keys are never overwritten')
    ids = [f'client-{i}' for i in range(count)]
    tokens = {cid: secrets.token_urlsafe(32) for cid in ids}
    pairs = {cid: {} for cid in ids}
    for i, a in enumerate(ids):
        for b in ids[i+1:]:
            key = secrets.token_hex(32)
            pairs[a][b] = pairs[b][a] = key
    atomic_json(out / 'server.json', {'clients': tokens, 'admin_token': secrets.token_urlsafe(32)})
    for i, cid in enumerate(ids):
        atomic_json(out / f'{cid}.json', {'id': cid, 'token': tokens[cid], 'server': server,
            'pair_keys': pairs[cid], 'data': f'data/fl/client_{i}.csv',
            'state_dir': f'runtime/{cid}', 'max_epsilon': 20.0, 'delta': 1e-6})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='runtime/credentials')
    ap.add_argument('--clients', type=int, default=2)
    ap.add_argument('--server', default='http://127.0.0.1:8080')
    args = ap.parse_args()
    provision(args.out, args.clients, args.server)
    print('Created separate credentials. Keep client pair keys off the server in the LAN deployment.')
