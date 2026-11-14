"""Launch three local processes, exercise real HTTP, and retain comparison evidence.

This is a single-host deployment test, not evidence of three physical devices.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from .common import ROOT, atomic_json
from .provision import provision


def wait_json(url, processes, timeout=60):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if any(p.poll() is not None for p in processes):
            raise RuntimeError('A service exited; inspect runtime/lab logs')
        try:
            response=requests.get(url,timeout=2)
            if response.ok:
                return response.json()
        except requests.RequestException:
            pass
        time.sleep(.5)
    raise TimeoutError(f'Service did not become ready: {url}')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--trees',type=int,default=20)
    ap.add_argument('--depth',type=int,default=3)
    ap.add_argument('--modes',nargs='+',default=['plain','secure','dp'],choices=['plain','secure','dp'])
    ap.add_argument('--serve',action='store_true',help='Keep dashboards running after evaluation')
    ap.add_argument('--port',type=int,default=8080)
    ap.add_argument('--resume-dir',type=Path,help='Reuse a prior lab directory and its client privacy ledgers')
    ap.add_argument('--preview-only',action='store_true',help='Serve dashboards without running new experiments')
    ap.add_argument('--report',type=Path,default=ROOT/'reports/software_comparison.json')
    args=ap.parse_args()
    run_dir=args.resume_dir.resolve() if args.resume_dir else ROOT/'runtime'/f'lab-{time.time_ns()}'
    keys=run_dir/'credentials'
    base=f'http://127.0.0.1:{args.port}'
    if not args.resume_dir:
        provision(keys,2,base)
    server_config=json.loads((keys/'server.json').read_text())
    for i in range(2):
        path=keys/f'client-{i}.json'
        cfg=json.loads(path.read_text())
        cfg['server']=base
        cfg['state_dir']=str(run_dir/f'client-{i}')
        atomic_json(path,cfg)
    commands=[[sys.executable,'-m','service.server','--credentials',str(keys/'server.json'),'--out',str(run_dir/'server'),'--port',str(args.port)]]
    commands += [[sys.executable,'-m','service.client','--config',str(keys/f'client-{i}.json'),'--port',str(args.port+i+1)] for i in range(2)]
    processes=[];logs=[]
    try:
        for i,cmd in enumerate(commands):
            log=(run_dir/f'process-{i}.log').open('w')
            logs.append(log)
            processes.append(subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0))
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            state=wait_json(base+'/api/state',processes)
            if sum(c.get('online',False) for c in state['clients'].values())==2:
                break
            time.sleep(1)
        else:
            raise TimeoutError('Both clients did not join')
        print(f'Dashboards: {base}, http://127.0.0.1:{args.port+1}, http://127.0.0.1:{args.port+2}',flush=True)
        print(f'Local credentials: {keys / "server.json"}',flush=True)
        if args.preview_only:
            print('Preview ready; no experiment started. Press Ctrl+C to stop.',flush=True)
            while True:
                time.sleep(1)
        results={'environment':'one host, three independent processes, real HTTP; not a physical LAN test',
                 'dataset':'repository dataset with synthetic fraud labels', 'runs':[]}
        for mode in args.modes:
            response=requests.post(base+'/api/runs',json={'mode':mode,'trees':args.trees,'depth':args.depth,'sigma':8,'timeout':120},headers={'X-Token':server_config['admin_token']},timeout=10)
            response.raise_for_status()
            deadline=time.monotonic()+1800
            last=-1
            while time.monotonic()<deadline:
                state=wait_json(base+'/api/state',processes)
                if state.get('tree')!=last:
                    last=state.get('tree')
                    print(f'{mode}: {last}/{args.trees} trees, {state["status"]}',flush=True)
                if state['status'] in ('completed','failed','cancelled'):
                    break
                time.sleep(1)
            if state['status']!='completed':
                raise RuntimeError(state.get('error','experiment exceeded its deadline'))
            atomic_json(run_dir/f'{mode}-state.json',state)
            atomic_json(run_dir/f'{mode}-model.json',json.loads((run_dir/'server/model.json').read_text()))
            local_base=f'http://127.0.0.1:{args.port+1}'
            sample=requests.get(local_base+'/api/sample',timeout=5).json()
            latencies=[]
            for _ in range(25):
                r=requests.post(local_base+'/api/predict',json={'rows':[sample]},timeout=5)
                r.raise_for_status()
                latencies.append(r.json()['latency_ms'])
            import numpy as np
            results['runs'].append({'mode':mode,'metrics':state['history'][-1]['metrics'],
                'seconds':state['seconds'],'rounds':state['rounds'],'privacy':state['privacy'],
                'upload_bytes':state['upload_bytes'],'download_bytes':state['download_bytes'],
                'inference_ms_p50':float(np.percentile(latencies,50)),
                'inference_ms_p95':float(np.percentile(latencies,95)),
                'model_bytes':(run_dir/f'{mode}-model.json').stat().st_size,
                'model_hash':state['model_hash']})
            atomic_json(args.report,results)
        # Re-evaluate the existing centralized model on the identical held-out rows.
        import lightgbm as lgb
        import pandas as pd
        from .common import metrics
        df=pd.read_csv(ROOT/'data/fl/server_test.csv')
        model=lgb.Booster(model_str=(ROOT/'models/lightgbm_baseline_stratified.txt').read_text(encoding='utf-8'))
        schema=json.loads((ROOT/'fl/upi_fl/schema.json').read_text())
        p=model.predict(df[schema['features']])
        results['centralized_saved_baseline']=metrics.evaluate(df['fraud_flag'],p,.5)
        results['baseline_note']='Saved baseline re-evaluated, not retrained; identical train/test indices, different tree algorithm and class weighting.'
        atomic_json(args.report,results)
        print(f'Evidence saved to {args.report}',flush=True)
        if args.serve:
            print('Keeping dashboards available. Stop this command to stop all three services.',flush=True)
            while True:
                time.sleep(1)
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for log in logs:log.close()


if __name__=='__main__':
    main()
