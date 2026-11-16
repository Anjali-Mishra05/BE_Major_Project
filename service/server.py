"""One coordinator process; HTTP polling keeps slow training outside request lifetimes."""
import argparse
import copy
import hashlib
import json
import secrets
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .common import ROOT, atomic_json, bins, clean, gbdt, metrics
from .privacy import decode_payload, secure_sum, epsilon


class RunConfig(BaseModel):
    trees: int = Field(default=20, ge=1, le=200)
    depth: int = Field(default=3, ge=1, le=6)
    learning_rate: float = Field(default=0.2, gt=0, le=1)
    min_child_samples: int = Field(default=50, ge=1)
    mode: str = Field(default='secure', pattern='^(plain|secure|dp)$')
    sigma: float = Field(default=8.0, ge=0.1, le=100)
    clip: float = Field(default=10.0, gt=0, le=100)
    timeout: float = Field(default=120, ge=1, le=3600)


class Coordinator:
    def __init__(self, credentials, out, schema_path, test_path=None):
        self.credentials = credentials
        self.out = Path(out)
        self.schema = bins.load_schema(Path(schema_path))
        self.schema_hash = hashlib.sha256(json.dumps(self.schema, sort_keys=True).encode()).hexdigest()
        self.test_path = test_path
        self.cv = threading.Condition(threading.RLock())
        self.clients = {}
        self.job = None
        self.replies = {}
        self.cancelled = False
        self.training_thread = None
        self.state = {'status': 'idle', 'events': [], 'history': [], 'clients': {}, 'run_id': None}
        self.model = None
        # A crashed coordinator never silently resumes with reused privacy queries.
        if (self.out / 'state.json').exists():
            self.state = json.loads((self.out / 'state.json').read_text())
            if self.state['status'] in ('training', 'distributing'):
                self.state.update(status='interrupted', error='Coordinator restarted; start a new run.')
        if (self.out / 'model.json').exists():
            self.model = json.loads((self.out / 'model.json').read_text())

    def event(self, stage, message):
        self.state['events'] = (self.state['events'] + [{'at': time.time(), 'stage': stage, 'message': message}])[-120:]
        atomic_json(self.out / 'state.json', self.state)

    def snapshot(self):
        with self.cv:
            result = copy.deepcopy(self.state)
            result['clients'] = {cid: {**self.clients.get(cid, {}),
                'online': time.time() - self.clients.get(cid, {}).get('last_seen', 0) < 15}
                for cid in self.credentials['clients']}
            result['schema_features'] = self.schema['n_features']
            return clean(result)

    def start(self, config):
        with self.cv:
            if self.state['status'] in ('training', 'distributing'):
                raise ValueError('A run is already active')
            if any(time.time() - self.clients.get(cid, {}).get('last_seen', 0) > 15 for cid in self.credentials['clients']):
                raise ValueError('All provisioned clients must be online before starting')
            if len(self.credentials['clients']) < 2:
                raise ValueError('At least two clients are required')
            self.cancelled = False
            self.model = None
            self.state = {'status': 'training', 'run_id': uuid.uuid4().hex, 'config': config,
                'events': [], 'history': [], 'tree': 0, 'depth': 0, 'rounds': 0,
                'upload_bytes': 0, 'download_bytes': 0, 'started': time.time(), 'received': [],
                'privacy': {'epsilon': 0, 'delta': 1e-6, 'scope': 'record add/remove; per run'}}
            self.event('start', 'All clients connected. Training started.')
            self.training_thread = threading.Thread(target=self.train, daemon=True)
            self.training_thread.start()
            return self.state['run_id']

    def exchange(self, job, timeout):
        with self.cv:
            self.job = clean(job)
            self.replies = {}
            self.state['received'] = []
            self.event(job['op'], f"Dispatched {job['query_id']} to all clients")
            deadline = time.monotonic() + timeout
            while len(self.replies) < len(self.credentials['clients']):
                if self.cancelled:
                    raise RuntimeError('Run cancelled; incomplete aggregate discarded')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    missing = sorted(set(self.credentials['clients']) - set(self.replies))
                    raise TimeoutError(f'Round deadline exceeded; missing {missing}. No partial aggregate used.')
                self.cv.wait(min(remaining, 1))
            result = dict(self.replies)
            self.job = None
            return result

    def submit(self, cid, body):
        with self.cv:
            if self.job is None or body.get('query_id') != self.job['query_id']:
                raise ValueError('stale or unknown query')
            if cid in self.replies:
                if self.replies[cid] != body:
                    raise ValueError('conflicting duplicate response')
                return
            if self.job['op'] == 'hist':
                size = len(self.job['frontier']) * 3 * self.schema['total_bins']
                if self.job['config']['mode'] == 'plain':
                    a = np.asarray(body.get('hist'), dtype=float)
                    if a.shape != (size,) or not np.isfinite(a).all():
                        raise ValueError('invalid histogram')
                else:
                    decode_payload(body.get('payload', ''), size)
            elif body.get('model_hash') != self.job['model_hash']:
                raise ValueError('model acknowledgement checksum mismatch')
            self.replies[cid] = body
            self.state['received'] = sorted(self.replies)
            self.state['upload_bytes'] += len(json.dumps(body).encode())
            self.event('receive', f'{cid}: update accepted ({len(self.replies)}/{len(self.credentials["clients"])})')
            self.cv.notify_all()

    def train(self):
        try:
            cfg = self.state['config']
            run = self.state['run_id']
            params = {'max_depth': cfg['depth'], 'learning_rate': cfg['learning_rate'],
                      'min_child_samples': cfg['min_child_samples']}
            forest = gbdt.Forest(self.schema, params)
            evaluation = None
            if self.test_path and Path(self.test_path).exists():
                import pandas as pd
                df = pd.read_csv(self.test_path)
                evaluation = (bins.binned_matrix(df[self.schema['features']], self.schema), df['fraud_flag'].to_numpy())
            for k in range(cfg['trees']):
                tree = gbdt.Tree(cfg['depth'])
                def fetch(frontier, depth):
                    with self.cv:
                        self.state.update(tree=k, depth=depth)
                    qid = f'{run}:{k}:{depth}'
                    responses = self.exchange({'op': 'hist', 'query_id': qid,
                        'run_id': run, 'config': cfg, 'model': forest.to_json(),
                        'tree': tree.to_json(), 'frontier': list(frontier),
                        'participants': sorted(self.credentials['clients'])}, cfg['timeout'])
                    size = len(frontier) * 3 * self.schema['total_bins']
                    if cfg['mode'] == 'plain':
                        total = np.sum([r['hist'] for r in responses.values()], axis=0)
                    else:
                        total = secure_sum([r['payload'] for r in responses.values()], size)
                    stats = gbdt.unpack_histograms(total, frontier, self.schema['total_bins'])
                    if cfg['mode'] == 'dp':
                        # Nonnegative Hessians/counts are postprocessing of the noisy release.
                        stats = {nid: (g, np.maximum(h, 0), np.maximum(c, 0)) for nid, (g,h,c) in stats.items()}
                    with self.cv:
                        self.state['rounds'] += 1
                        if cfg['mode'] == 'dp':
                            self.state['privacy']['epsilon'] = epsilon(self.state['rounds'], cfg['sigma'], 1e-6)
                        self.event('aggregate', 'All updates combined; server selecting global tree splits.')
                    return stats
                gains = gbdt.grow_tree(self.schema, forest.params, tree, fetch)
                for j, gain in gains:
                    forest.gain[j] += gain
                forest.add_tree(tree)
                score = metrics.evaluate(evaluation[1], forest.predict_proba(evaluation[0]), 0.5) if evaluation else None
                with self.cv:
                    self.state['tree'] = k+1
                    self.state['history'].append({'tree': k+1, 'metrics': score, 'seconds': time.time()-self.state['started']})
                    self.event('model', f'Global tree {k+1} committed')
                    atomic_json(self.out / 'checkpoint.json', forest.to_json())
            model = clean(forest.to_json())
            model_hash = hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()
            with self.cv:
                self.state['status'] = 'distributing'
            self.exchange({'op': 'deploy', 'query_id': f'{run}:deploy', 'run_id': run,
                           'model': model, 'model_hash': model_hash}, cfg['timeout'])
            with self.cv:
                self.model = model
                atomic_json(self.out / 'model.json', model)
                self.state.update(status='completed', seconds=time.time()-self.state['started'], model_hash=model_hash)
                self.event('complete', 'Final model verified and installed on every client.')
                atomic_json(self.out / 'runs' / f'{run}.json', self.state)
        except Exception as exc:
            with self.cv:
                self.job = None
                self.state.update(status='cancelled' if self.cancelled else 'failed', error=str(exc))
                self.event('error', str(exc))


def create_app(coordinator):
    app = FastAPI(title='UPI Federation')
    c = coordinator
    def auth(token, expected):
        if not token or not secrets.compare_digest(token, expected):
            raise HTTPException(401, 'Invalid access token')
    def client_auth(cid, token):
        if cid not in c.credentials['clients']:
            raise HTTPException(401, 'Unknown client')
        auth(token, c.credentials['clients'][cid])

    @app.get('/api/state')
    def state():
        return c.snapshot()

    @app.post('/api/runs')
    def start(config: RunConfig, x_token: str = Header(default='')):
        auth(x_token, c.credentials['admin_token'])
        try:
            return {'run_id': c.start(config.model_dump())}
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @app.post('/api/cancel')
    def cancel(x_token: str = Header(default='')):
        auth(x_token, c.credentials['admin_token'])
        with c.cv:
            c.cancelled = True
            c.cv.notify_all()
        return {'ok': True}

    @app.post('/api/clients/{cid}/poll')
    def poll(cid: str, body: dict, x_token: str = Header(default='')):
        client_auth(cid, x_token)
        with c.cv:
            if body.get('schema_hash') != c.schema_hash:
                raise HTTPException(409, 'Schema mismatch; copy the same schema to every device')
            # Explicit allowlist: no raw rows, labels, or client loss in server telemetry.
            c.clients[cid] = {k: body.get(k) for k in ('hostname', 'stage', 'trees', 'epsilon', 'error')}
            c.clients[cid]['last_seen'] = time.time()
            job = c.job if cid not in c.replies else None
            c.state['download_bytes'] = c.state.get('download_bytes', 0) + len(json.dumps(job).encode())
            return {'job': job, 'status': c.state['status']}

    @app.post('/api/clients/{cid}/result')
    def result(cid: str, body: dict, x_token: str = Header(default='')):
        client_auth(cid, x_token)
        try:
            c.submit(cid, body)
            return {'accepted': True}
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(409, str(exc))

    @app.get('/api/baseline')
    def baseline():
        path = ROOT / 'reports/baseline_metrics.json'
        return json.loads(path.read_text()) if path.exists() else {}

    @app.get('/api/comparison')
    def comparison():
        path = ROOT / 'reports/software_comparison.json'
        return json.loads(path.read_text()) if path.exists() else {}

    @app.get('/api/benchmarks')
    def benchmarks():
        path = ROOT / 'reports/algorithm_benchmarks.json'
        return json.loads(path.read_text()) if path.exists() else {}

    dist = ROOT / 'dashboard/dist'
    if dist.exists():
        app.mount('/', StaticFiles(directory=dist, html=True), name='dashboard')
    return app


if __name__ == '__main__':
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument('--credentials', default='runtime/credentials/server.json')
    ap.add_argument('--out', default='runtime/server')
    ap.add_argument('--schema', default='fl/upi_fl/schema.json')
    ap.add_argument('--test', default='data/fl/server_test.csv')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--cert')
    ap.add_argument('--key')
    args = ap.parse_args()
    coordinator = Coordinator(json.loads(Path(args.credentials).read_text()), args.out, args.schema, args.test)
    uvicorn.run(create_app(coordinator), host=args.host, port=args.port, ssl_certfile=args.cert, ssl_keyfile=args.key)
