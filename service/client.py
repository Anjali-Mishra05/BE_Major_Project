"""A single device's private data, training worker, durable privacy ledger and inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
import math
import socket
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from .common import ROOT, atomic_json, bins, clean, forest_from, task, tree_from
from .privacy import mask, private_histogram


class Worker:
    def __init__(self, config, schema_path='fl/upi_fl/schema.json'):
        self.config = config
        self.schema = bins.load_schema(Path(schema_path))
        self.schema_hash = hashlib.sha256(json.dumps(self.schema, sort_keys=True).encode()).hexdigest()
        self.shard = task.load_shard(config['data'], self.schema, use_cache=False)
        if self.shard.n == 0 or not np.isin(self.shard.y, [0, 1]).all():
            raise ValueError('Client data must be nonempty with binary fraud labels')
        self.out = Path(config['state_dir'])
        self.out.mkdir(parents=True, exist_ok=True)
        self.db = self.out / 'ledger.sqlite'
        with closing(sqlite3.connect(self.db)) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS releases (id TEXT PRIMARY KEY, digest TEXT, response TEXT, rho REAL)')
        self.state = {'id': config['id'], 'hostname': socket.gethostname(), 'stage': 'ready',
                      'trees': 0, 'rows': self.shard.n, 'epsilon': self.spent(), 'error': None}
        self.stop = threading.Event()
        self.model = None
        self.onnx = None
        if (self.out / 'model.json').exists():
            self.model = forest_from(json.loads((self.out / 'model.json').read_text()))
            self.state['trees'] = len(self.model.trees)
            self.install_onnx(self.model)

    def install_onnx(self, model):
        from .onnx_model import export_forest, session, predict
        runtime = session(export_forest(model, self.out / 'model.onnx'))
        sample = self.shard.binned[:128]
        if not np.allclose(predict(runtime, sample), model.predict_proba(sample), atol=2e-6):
            raise ValueError('ONNX/native prediction parity failed')
        self.onnx = runtime
        self.state['inference_engine'] = 'ONNX Runtime CPU'

    def spent(self):
        with closing(sqlite3.connect(self.db)) as db:
            rho = db.execute('SELECT COALESCE(SUM(rho),0) FROM releases').fetchone()[0]
        return rho + 2 * math.sqrt(rho * math.log(1 / self.config.get('delta', 1e-6)))

    def compute(self, job):
        qid = job['query_id']
        digest = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
        # Cache and budget are committed together BEFORE the response leaves this machine.
        # Retries after a lost response or process restart resend the identical release.
        with closing(sqlite3.connect(self.db, timeout=120)) as db, db:
            db.execute('BEGIN IMMEDIATE')
            cached = db.execute('SELECT digest,response FROM releases WHERE id=?', (qid,)).fetchone()
            if cached:
                if cached[0] != digest:
                    raise ValueError('Query identifier reused with different content')
                return json.loads(cached[1])
            self.state.update(stage='synchronizing', error=None)
            model = forest_from(job['model'])
            if model.schema != self.schema:
                raise ValueError('Model schema mismatch')
            rho = 0
            if job['op'] == 'deploy':
                model_hash = hashlib.sha256(json.dumps(job['model'], sort_keys=True).encode()).hexdigest()
                if model_hash != job['model_hash']:
                    raise ValueError('Final model checksum mismatch')
                self.install_onnx(model)
                atomic_json(self.out / 'model.json', job['model'])
                self.model = model
                self.state.update(stage='model ready', trees=len(model.trees))
                response = {'query_id': qid, 'model_hash': model_hash}
            elif job['op'] == 'hist':
                cfg = job['config']
                peers = sorted([self.config['id'], *self.config['pair_keys']])
                if sorted(job['participants']) != peers:
                    raise ValueError('Participant roster differs from provisioned secure cohort')
                if cfg['mode'] == 'dp':
                    rho = 1 / (2 * cfg['sigma']**2)
                    previous = db.execute('SELECT COALESCE(SUM(rho),0) FROM releases').fetchone()[0]
                    total = previous + rho
                    eps = total + 2 * math.sqrt(total * math.log(1 / self.config.get('delta', 1e-6)))
                    if eps > self.config.get('max_epsilon', 20):
                        raise ValueError('Cumulative privacy budget exhausted; release refused')
                self.shard.raw = model.raw_score(self.shard.binned)
                self.shard.trees_seen = len(model.trees)
                self.state.update(stage='training', trees=len(model.trees))
                tree = tree_from(job['tree'])
                if cfg['mode'] == 'dp':
                    buf = private_histogram(self.shard, tree, job['frontier'], self.schema, cfg['clip'], cfg['sigma'])
                else:
                    buf = task.histograms(self.shard, tree, job['frontier'], 1.0, self.schema)
                self.state['stage'] = 'protecting update'
                response = {'query_id': qid}
                if cfg['mode'] == 'plain':
                    response['hist'] = buf.tolist()
                else:
                    response['payload'] = mask(buf, self.config['id'], self.config['pair_keys'], qid)
            else:
                raise ValueError('Unknown operation')
            db.execute('INSERT INTO releases VALUES (?,?,?,?)', (qid, digest, json.dumps(response), rho))
        self.state['epsilon'] = self.spent()
        return response

    def run(self):
        config = self.config
        base = config['server'].rstrip('/') + '/api/clients/' + config['id']
        session = requests.Session()
        session.headers['X-Token'] = config['token']
        session.verify = config.get('ca_cert', True)
        future = None
        pending = None
        failed_query = None
        with ThreadPoolExecutor(max_workers=1) as pool:
            while not self.stop.is_set():
                try:
                    if future and future.done():
                        try:
                            pending = future.result()
                        except Exception as exc:
                            self.state.update(stage='error', error=str(exc))
                        future = None
                    if pending:
                        response = session.post(base + '/result', json=pending, timeout=(5, 10))
                        if response.status_code == 409:
                            pending = None  # expired round; cached release and budget stay intact
                        else:
                            response.raise_for_status()
                            pending = None
                            self.state['stage'] = 'waiting'
                    result = session.post(base + '/poll', json={**self.state, 'schema_hash': self.schema_hash}, timeout=(5, 10))
                    result.raise_for_status()
                    server_state = result.json()
                    job = server_state['job']
                    if job and future is None and pending is None and job['query_id'] != failed_query:
                        failed_query = job['query_id']
                        future = pool.submit(self.compute, job)
                    if not job and future is None and not pending and not self.state.get('error'):
                        self.state['stage'] = 'model ready' if self.model and server_state['status'] == 'completed' else 'waiting'
                    self.state['connected'] = True
                except requests.RequestException as exc:
                    self.state.update(connected=False, connection_error=str(exc))
                self.stop.wait(0.5)


def create_app(worker):
    app = FastAPI(title='UPI local client')
    @app.get('/api/local')
    def local():
        return clean({**worker.state, 'features': worker.schema['features'],
                     'model_trees': len(worker.model.trees) if worker.model else 0,
                     'model_ready': worker.model is not None, 'privacy_limit': worker.config.get('max_epsilon', 20)})

    @app.post('/api/predict')
    def predict(body: dict):
        if worker.model is None:
            raise HTTPException(409, 'No global model installed yet')
        try:
            import pandas as pd
            rows = body['rows']
            if not isinstance(rows, list) or not 1 <= len(rows) <= 100:
                raise ValueError('Provide 1 to 100 feature objects')
            frame = pd.DataFrame(rows)[worker.schema['features']]
            if not np.isfinite(frame.to_numpy(dtype=float)).all():
                raise ValueError('All features must be finite numbers')
            start = time.perf_counter()
            from .onnx_model import predict as onnx_predict
            proba = onnx_predict(worker.onnx, bins.binned_matrix(frame, worker.schema))
            return {'risk': proba.tolist(), 'flagged': (proba >= 0.5).tolist(),
                    'latency_ms': (time.perf_counter()-start)*1000, 'trees': len(worker.model.trees),
                    'engine': 'ONNX Runtime CPU'}
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, str(exc))

    @app.get('/api/sample')
    def sample():
        import pandas as pd
        # Served only by the client's loopback UI, never uploaded to the coordinator.
        df = pd.read_csv(worker.config['data'], nrows=1)
        return clean(df[worker.schema['features']].iloc[0].to_dict())

    rows_cache = {}

    @app.get('/api/row')
    def row(n: int | None = None, kind: str = 'any'):
        """Return one local transaction by 1-based row number, or a random one (any/fraud/legit).

        Served only by the client's loopback UI, never uploaded to the coordinator.
        """
        import random
        import pandas as pd
        if 'df' not in rows_cache:
            rows_cache['df'] = pd.read_csv(worker.config['data'])
        df = rows_cache['df']
        total = len(df)
        label = df['fraud_flag'] if 'fraud_flag' in df.columns else None
        if n is None:
            pool = np.arange(total)
            if kind in ('fraud', 'legit'):
                if label is None:
                    raise HTTPException(422, 'This device has no labels to filter by')
                pool = np.flatnonzero(label.to_numpy() == (1 if kind == 'fraud' else 0))
            elif kind != 'any':
                raise HTTPException(422, "kind must be 'any', 'fraud' or 'legit'")
            if len(pool) == 0:
                raise HTTPException(404, 'No matching rows on this device')
            n = int(random.choice(pool.tolist())) + 1
        if not 1 <= n <= total:
            raise HTTPException(422, f'Row number must be between 1 and {total}')
        record = df.iloc[n - 1]
        return {'row': n, 'total': total,
                'actual_fraud': int(record['fraud_flag']) if label is not None else None,
                'features': clean(record[worker.schema['features']].to_dict())}

    dist = ROOT / 'dashboard/dist'
    if dist.exists():
        app.mount('/', StaticFiles(directory=dist, html=True), name='dashboard')
    return app


if __name__ == '__main__':
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--schema', default='fl/upi_fl/schema.json')
    ap.add_argument('--port', type=int, default=8081)
    args = ap.parse_args()
    worker = Worker(json.loads(Path(args.config).read_text()), args.schema)
    threading.Thread(target=worker.run, daemon=True).start()
    try:
        uvicorn.run(create_app(worker), host='127.0.0.1', port=args.port)
    finally:
        worker.stop.set()
