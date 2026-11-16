"""Real-process restart/dropout acceptance check on synthetic fixtures."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import requests
import pandas as pd
from .common import ROOT, atomic_json, bins
from .provision import provision
from .lab import wait_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--flower',action='store_true',help='Use Flower gRPC instead of HTTP polling')
    args=ap.parse_args()
    root=ROOT/'runtime'/f'network-check-{time.time_ns()}'
    base='http://127.0.0.1:8180'
    provision(root/'keys',2,base)
    schema=bins.build_schema(['x'],{'x':'binary'},[2],{})
    bins.save_schema(schema,root/'schema.json')
    for i in range(2):
        path=root/'keys'/f'client-{i}.json'
        config=json.loads(path.read_text())
        data=root/f'{i}.csv'
        pd.DataFrame({'x':[0,1]*100,'fraud_flag':[0,1]*100}).to_csv(data,index=False)
        config.update(data=str(data),state_dir=str(root/f'client-{i}'))
        atomic_json(path,config)
    creds=json.loads((root/'keys/server.json').read_text())
    commands=[[sys.executable,'-m','service.server','--credentials',str(root/'keys/server.json'),
        '--out',str(root/'server'),'--schema',str(root/'schema.json'),'--test',str(root/'0.csv'),'--port','8180']]
    commands += [[sys.executable,'-m','service.client','--config',str(root/'keys'/f'client-{i}.json'),
        '--schema',str(root/'schema.json'),'--port',str(8181+i)] for i in range(2)]
    if args.flower:
        commands=[[sys.executable,'-m','service.flower_transport','server','--config',str(root/'keys/server.json'),
            '--out',str(root/'server'),'--schema',str(root/'schema.json'),'--test',str(root/'0.csv'),
            '--port','8180','--address','127.0.0.1:8185']]
        commands += [[sys.executable,'-m','service.flower_transport','client','--config',str(root/'keys'/f'client-{i}.json'),
            '--schema',str(root/'schema.json'),'--port',str(8181+i),'--address','127.0.0.1:8185'] for i in range(2)]
    processes=[];logs=[]
    def launch(i):
        log=(root/f'process-{i}-{time.time_ns()}.log').open('w');logs.append(log)
        return subprocess.Popen(commands[i],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    def stop(p):
        if p.poll() is None:
            p.terminate();p.wait(timeout=10)
    def wait_clients():
        end=time.monotonic()+60
        while time.monotonic()<end:
            s=wait_json(base+'/api/state',processes)
            if all(c['online'] for c in s['clients'].values()):return
            time.sleep(.3)
        raise TimeoutError('Clients not online')
    def start(timeout):
        r=requests.post(base+'/api/runs',json={'trees':4,'depth':2,'min_child_samples':1,'mode':'dp','timeout':timeout},headers={'X-Token':creds['admin_token']},timeout=5)
        r.raise_for_status()
    def terminal():
        end=time.monotonic()+90
        while time.monotonic()<end:
            s=requests.get(base+'/api/state',timeout=5).json()
            if s['status'] in ('completed','failed'):return s
            time.sleep(.2)
        raise TimeoutError('Run did not terminate')
    try:
        processes=[launch(i) for i in range(3)]
        wait_clients();start(30)
        # Interrupt the first client while the run is active, then restore its durable state.
        time.sleep(1)
        stop(processes[1]);time.sleep(1);processes[1]=launch(1)
        recovered=terminal()
        assert recovered['status']=='completed',recovered
        wait_clients();start(2)
        stop(processes[2])
        failed=terminal()
        assert failed['status']=='failed' and 'No partial aggregate' in failed['error'],failed
        result={'transport':'Flower gRPC, three child processes' if args.flower else 'real loopback HTTP, three child processes',
            'restart_mid_run':'passed; final model acknowledged by both clients',
            'dropout':'passed; bounded failure without partial aggregation',
            'recovered_rounds':recovered['rounds'],'dropout_error':failed['error']}
        atomic_json(ROOT/'reports'/('flower_dashboard_acceptance.json' if args.flower else 'network_acceptance.json'),result)
        print(json.dumps(result,indent=2))
    finally:
        for p in processes:stop(p)
        for log in logs:log.close()


if __name__=='__main__':main()
