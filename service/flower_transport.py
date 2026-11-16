"""Flower 1.39 gRPC transport for the dashboard's histogram protocol.

Uses Flower's compatibility Client/Server API, not FedAvg: histogram sums build trees.
Intended for the private Docker demo network; gRPC is not exposed on the host.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import secrets
import threading
import time
import flwr as fl
from flwr.common import Code, Status, Parameters, FitIns, FitRes, GetPropertiesIns, GetPropertiesRes
from .common import clean
from .client import Worker, create_app as client_app
from .server import Coordinator, create_app as server_app

OK = Status(code=Code.OK, message='OK')


class POSClient(fl.client.Client):
    def __init__(self, worker):
        self.worker = worker

    def get_properties(self, ins):
        w = self.worker
        w.state['connected'] = True
        return GetPropertiesRes(status=OK, properties={
            'id': w.config['id'], 'token': w.config['token'], 'schema_hash': w.schema_hash,
            'telemetry': json.dumps(clean({k: w.state.get(k) for k in
                ('hostname', 'stage', 'trees', 'epsilon', 'error')}))})

    def fit(self, ins):
        w = self.worker
        try:
            job = json.loads(ins.config['job'])
            response = w.compute(job)
            w.state['stage'] = 'model ready' if job['op'] == 'deploy' else 'waiting'
            return FitRes(status=OK, parameters=Parameters(tensors=[], tensor_type=''), num_examples=0,
                metrics={'response': json.dumps(response)})
        except Exception as exc:
            w.state.update(stage='error', error=str(exc))
            # No partial histograms or private data in error responses.
            return FitRes(status=OK, parameters=Parameters(tensors=[], tensor_type=''), num_examples=0,
                metrics={'error': str(exc)})


class DashboardServer(fl.server.Server):
    def __init__(self, coordinator):
        super().__init__(client_manager=fl.server.SimpleClientManager())
        self.coordinator = coordinator
        self.stop = threading.Event()

    def fit(self, num_rounds, timeout):
        c = self.coordinator
        pending, identities, last_poll, attempted = {}, {}, {}, {}
        pool = ThreadPoolExecutor(max_workers=max(4, len(c.credentials['clients']) * 2))
        try:
            while not self.stop.wait(0.1):
                proxies = self.client_manager().all()
                for key, (future, kind, cid, qid) in list(pending.items()):
                    if not future.done():
                        continue
                    del pending[key]
                    try:
                        result = future.result()
                        if result.status.code != Code.OK:
                            raise ValueError('Flower request refused')
                        if kind == 'poll':
                            props = result.properties
                            cid = props.get('id')
                            expected = c.credentials['clients'].get(cid)
                            if not expected or not secrets.compare_digest(str(props.get('token', '')), expected):
                                raise ValueError('Unknown POS credentials')
                            if props.get('schema_hash') != c.schema_hash:
                                raise ValueError('POS schema mismatch')
                            if any(other != key and ident == cid and other in proxies for other, ident in identities.items()):
                                raise ValueError('Duplicate POS identity')
                            identities[key] = cid
                            telemetry = json.loads(props['telemetry'])
                            with c.cv:
                                c.clients[cid] = {k: telemetry.get(k) for k in ('hostname', 'stage', 'trees', 'epsilon', 'error')}
                                c.clients[cid]['last_seen'] = time.time()
                        else:
                            if result.metrics.get('error'):
                                raise ValueError(result.metrics['error'])
                            c.submit(cid, json.loads(result.metrics['response']))
                    except Exception as exc:
                        if cid in c.credentials['clients']:
                            with c.cv:
                                c.clients.setdefault(cid, {}).update(error=str(exc))
                        # Failed jobs are not retried endlessly; the coordinator deadline
                        # fails the round without accepting a partial aggregate.
                    last_poll[key] = time.monotonic()
                for key in list(identities):
                    if key not in proxies:
                        identities.pop(key, None)
                        attempted.pop(key, None)
                        last_poll.pop(key, None)
                for key, proxy in proxies.items():
                    if key in pending:
                        continue
                    cid = identities.get(key)
                    with c.cv:
                        job = c.job
                        send = cid and job and cid not in c.replies and attempted.get(key) != job['query_id']
                        if send:
                            payload = json.dumps(job)
                            c.state['download_bytes'] += len(payload.encode())
                            c.clients[cid].update(stage='training' if job['op']=='hist' else 'synchronizing')
                            deadline = c.state['config']['timeout']
                    if send:
                        attempted[key] = job['query_id']
                        pending[key] = (pool.submit(proxy.fit, FitIns(Parameters([], ''), {'job': payload}),
                            deadline, None), 'fit', cid, job['query_id'])
                    elif time.monotonic() - last_poll.get(key, 0) > 1:
                        pending[key] = (pool.submit(proxy.get_properties, GetPropertiesIns({}), 5, None), 'poll', cid, None)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        return fl.server.history.History(), 0.0


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument('role', choices=['server', 'client'])
    ap.add_argument('--config', required=True)
    ap.add_argument('--schema', default='fl/upi_fl/schema.json')
    ap.add_argument('--address', default='server:8085')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--out', default='runtime/server')
    ap.add_argument('--test', default='data/fl/server_test.csv')
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.role == 'server':
        coordinator = Coordinator(config, args.out, args.schema, args.test)
        server = DashboardServer(coordinator)
        app = server_app(coordinator)
        # Flower registers OS signal handlers and must own the main thread.
        threading.Thread(target=uvicorn.run, args=(app,),
            kwargs={'host':args.host, 'port':args.port}, daemon=True).start()
        fl.server.start_server(server_address=args.address, server=server,
            config=fl.server.ServerConfig(num_rounds=1))
        return
    else:
        worker = Worker(config, args.schema)
        def connect():
            while not worker.stop.is_set():
                try:
                    fl.client.start_client(server_address=args.address, client=POSClient(worker), insecure=True)
                except Exception as exc:
                    worker.state.update(connected=False, connection_error=str(exc))
                worker.stop.wait(2)
        threading.Thread(target=connect, daemon=True).start()
        app = client_app(worker)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
