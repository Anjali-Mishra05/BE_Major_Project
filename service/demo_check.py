"""Acceptance check against the running Flower/Docker demo; never deletes ledgers."""
import argparse
import json
import subprocess
import time
from pathlib import Path
import requests
from .common import ROOT, atomic_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--directory', default='runtime/docker-demo')
    ap.add_argument('--faults', action='store_true')
    ap.add_argument('--report', help='Optional output JSON path')
    args = ap.parse_args()
    directory = Path(args.directory)
    creds = json.loads((directory/'keys/server.json').read_text())
    count = len(creds['clients'])
    base = 'http://127.0.0.1:8080'
    headers = {'X-Token': creds['admin_token']}
    def get(path, port=8080):
        response = requests.get(f'http://127.0.0.1:{port}'+path, timeout=10)
        response.raise_for_status()
        return response.json()
    def docker(action, service):
        subprocess.run(['docker', 'compose', '-f', str(directory/'compose.json'), action, service], check=True, timeout=60)
    def ready():
        deadline = time.monotonic()+120
        while time.monotonic()<deadline:
            try:
                state = get('/api/state')
                if all(c['online'] and not c.get('error') for c in state['clients'].values()):
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError('Clients not ready')
    def run(mode, fault=None):
        ready()
        response = requests.post(base+'/api/runs', headers=headers,
            json={'mode':mode, 'trees':3, 'depth':2, 'timeout':5 if fault=='dropout' else 90}, timeout=10)
        response.raise_for_status()
        if fault:
            docker('stop', 'pos-1')
            if fault=='restart':
                docker('start', 'pos-1')
        deadline = time.monotonic()+360
        while time.monotonic()<deadline:
            state = get('/api/state')
            if state['status'] in ('failed', 'completed', 'cancelled'):
                return state
            time.sleep(.5)
        raise TimeoutError('Run did not terminate')
    report = {'transport':'Flower gRPC, Docker server + private POS containers',
        'clients':count, 'runs':[], 'checked_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    try:
        for mode in ('plain','secure','dp'):
            print('Checking '+mode, flush=True)
            state = run(mode, 'restart' if args.faults and mode=='secure' else None)
            if state['status']!='completed':
                raise AssertionError(state.get('error',state['status']))
            sample = get('/api/sample',8081)
            scores = []
            for port in range(8081,8081+count):
                response = requests.post(f'http://127.0.0.1:{port}/api/predict',json={'rows':[sample]},timeout=10)
                response.raise_for_status()
                prediction = response.json()
                assert prediction['engine']=='ONNX Runtime CPU' and prediction['trees']==3
                scores.append(prediction['risk'][0])
            assert max(scores)-min(scores)<1e-7, scores
            report['runs'].append({'mode':mode, 'status':state['status'], 'rounds':state['rounds'],
                'model_hash':state['model_hash'], 'all_pos_onnx_predictions_match':True,
                'restart_mid_run':bool(args.faults and mode=='secure'), 'seconds':state['seconds']})
        if args.faults:
            state = run('secure','dropout')
            assert state['status']=='failed' and 'No partial aggregate' in state.get('error',''), state
            report['dropout'] = state['error']
    finally:
        if args.faults:
            docker('start','pos-1')
    output = Path(args.report) if args.report else ROOT/'reports'/('docker_acceptance.json' if args.faults else 'docker_smoke.json')
    atomic_json(output,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
