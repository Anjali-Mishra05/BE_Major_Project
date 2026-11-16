"""Prepare an isolated 2/3-POS Docker Compose configuration without copying secrets into images."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from .common import ROOT, atomic_json
from .provision import provision


def prepare(count, directory):
    out = Path(directory).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose an empty demo directory; existing ledgers and credentials are never overwritten')
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT/'fl/make_partitions.py'), '--clients', str(count),
        '--out', str(out/'data')], cwd=ROOT, check=True)
    provision(out/'keys', count, 'http://server:8080')
    def mount(source, target):
        return {'type': 'bind', 'source': str(source), 'target': target, 'read_only': True}
    services = {}
    volumes = {}
    for i in range(-1, count):
        name = 'server' if i < 0 else f'pos-{i+1}'
        config = out/'keys'/('server.json' if i < 0 else f'client-{i}.json')
        if i >= 0:
            cfg = json.loads(config.read_text())
            cfg.update(data='/data/local.csv', state_dir='/state')
            atomic_json(config, cfg)
        data = out/'data'/('server_test.csv' if i < 0 else f'client_{i}.csv')
        volumes[name] = {}
        services[name] = dict(image='upi-pos-demo:local', build={'context': str(ROOT)},
            command=['server' if i < 0 else 'client', '--config', '/config/identity.json',
                '--host', '0.0.0.0', '--port', '8080', '--address', '0.0.0.0:8085' if i < 0 else 'server:8085',
                '--out', '/state', '--test', '/data/local.csv'],
            ports=[f'127.0.0.1:{8080+i+1}:8080'],
            volumes=[mount(config, '/config/identity.json'), mount(data, '/data/local.csv'),
                {'type':'volume', 'source':name, 'target':'/state'}],
            networks=['pos'], init=True, stop_grace_period='10s',
            healthcheck={'test':['CMD','python','-c',
                "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/"+
                ('state' if i < 0 else 'local')+"', timeout=3)"], 'interval':'10s', 'timeout':'5s', 'retries':6})
        if i < 0:
            services[name]['volumes'].append(mount(ROOT/'reports', '/app/reports'))
        else:
            services[name]['depends_on'] = {'server': {'condition': 'service_healthy'}}
    atomic_json(out/'compose.json', {'name': 'upi-semester-demo', 'services': services,
        'networks': {'pos': {'driver': 'bridge'}}, 'volumes': volumes})
    print(f'Prepared {out / "compose.json"}')
    print('Run: docker compose -f "'+str(out/'compose.json')+'" up --build -d')
    print('Server http://localhost:8080; POS dashboards at ports '+', '.join(str(8081+i) for i in range(count)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--clients', type=int, choices=[2,3], default=3)
    ap.add_argument('--out', default='runtime/docker-demo')
    args = ap.parse_args()
    prepare(args.clients, args.out)
