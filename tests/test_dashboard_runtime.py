import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from service.common import bins, clean, gbdt
from service.client import Worker, create_app as client_app
from service.server import Coordinator, RunConfig, create_app
from service.provision import provision
from service.privacy import mask, secure_sum, epsilon


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        provision(self.root / 'keys', 2)
        self.schema = bins.build_schema(['x'], {'x': 'binary'}, [2], {})
        self.schema_path = self.root / 'schema.json'
        bins.save_schema(self.schema, self.schema_path)
        self.workers = []
        for i in range(2):
            data = self.root / f'{i}.csv'
            pd.DataFrame({'x': [0,1]*40, 'fraud_flag': [0,1]*40}).to_csv(data,index=False)
            cfg = json.loads((self.root/'keys'/f'client-{i}.json').read_text())
            cfg.update(data=str(data), state_dir=str(self.root/f'worker-{i}'))
            self.workers.append(Worker(cfg, self.schema_path))
        creds = json.loads((self.root/'keys/server.json').read_text())
        self.c = Coordinator(creds, self.root/'server', self.schema_path, self.root/'0.csv')
        self.api = TestClient(create_app(self.c))
        for w in self.workers:
            r = self.api.post(f'/api/clients/{w.config["id"]}/poll', headers={'X-Token':w.config['token']}, json={'schema_hash':w.schema_hash, **w.state})
            self.assertEqual(r.status_code,200)

    def drive(self, mode='secure', duplicate=False):
        self.c.start(RunConfig(trees=2, depth=2, min_child_samples=1, mode=mode, sigma=8, timeout=3).model_dump())
        end = time.monotonic()+15
        while self.c.state['status'] in ('training','distributing') and time.monotonic()<end:
            with self.c.cv:
                job = copy.deepcopy(self.c.job)
                replies = dict(self.c.replies)
            if job:
                for w in self.workers:
                    if w.config['id'] not in replies:
                        response = w.compute(job)
                        self.c.submit(w.config['id'],response)
                        if duplicate:
                            # Lock protects this assertion from the next round starting.
                            with self.c.cv:
                                if self.c.job and self.c.job['query_id']==job['query_id']:
                                    self.c.submit(w.config['id'],response)
            time.sleep(.01)
        self.assertEqual(self.c.state['status'],'completed',self.c.state)
        return self.c.model

    def test_secure_matches_plain_and_deploys_everywhere(self):
        plain=self.drive('plain')
        secure=self.drive('secure',True)
        from service.common import forest_from
        x=np.array([[0],[1]])
        np.testing.assert_allclose(forest_from(plain).predict_proba(x), forest_from(secure).predict_proba(x),atol=1e-5)
        for w in self.workers:
            self.assertEqual(len(w.model.trees),2)
            api=TestClient(client_app(w))
            r=api.post('/api/predict',json={'rows':[{'x':1}]})
            self.assertEqual(r.status_code,200)
            self.assertGreater(r.json()['risk'][0],.5)

    def job(self):
        return clean({'op':'hist','query_id':'run:0:0','run_id':'run',
            'config':RunConfig(mode='dp').model_dump(), 'model':gbdt.Forest(self.schema).to_json(),
            'tree':gbdt.Tree(1).to_json(), 'frontier':[0], 'participants':['client-0','client-1']})

    def test_restart_resends_identical_noise_and_budget(self):
        w=self.workers[0]
        job=self.job()
        first=w.compute(job)
        spent=w.spent()
        restarted=Worker(w.config,self.schema_path)
        self.assertEqual(first,restarted.compute(job))
        self.assertEqual(spent,restarted.spent())
        changed=copy.deepcopy(job)
        changed['config']['sigma']=3
        with self.assertRaisesRegex(ValueError,'reused'):
            restarted.compute(changed)

    def test_budget_refuses_release(self):
        self.workers[0].config['max_epsilon']=.001
        with self.assertRaisesRegex(ValueError,'budget'):
            self.workers[0].compute(self.job())
        self.assertEqual(self.workers[0].spent(),0)

    def test_dp_training_completes(self):
        self.drive('dp')
        self.assertGreater(self.c.state['privacy']['epsilon'],0)
        self.assertGreater(self.workers[0].spent(),0)

    def test_dropout_times_out_without_publishing(self):
        self.c.start(RunConfig(trees=1,timeout=1).model_dump())
        deadline=time.monotonic()+4
        while self.c.state['status']=='training' and time.monotonic()<deadline:
            time.sleep(.05)
        self.assertEqual(self.c.state['status'],'failed')
        self.assertIn('No partial',self.c.state['error'])
        self.assertFalse((self.root/'server/model.json').exists())

    def test_auth_stale_schema_and_duplicate_start(self):
        self.assertEqual(self.api.post('/api/runs',json={}).status_code,401)
        w=self.workers[0]
        self.assertEqual(self.api.post('/api/clients/client-0/poll',json={'schema_hash':'wrong'},headers={'X-Token':w.config['token']}).status_code,409)
        self.c.start(RunConfig(timeout=1).model_dump())
        with self.assertRaisesRegex(ValueError,'already'):
            self.c.start(RunConfig().model_dump())
        with self.assertRaisesRegex(ValueError,'stale'):
            self.c.submit('client-0',{'query_id':'old'})
        with self.c.cv:
            self.c.cancelled=True
            self.c.cv.notify_all()
        deadline=time.monotonic()+3
        while self.c.state['status']=='training' and time.monotonic()<deadline:
            time.sleep(.01)

    def test_masks_cancel_and_missing_mask_does_not(self):
        a=np.array([1.3,-2.1,10.5]);b=np.array([-9.4,4.5,2.0])
        payloads=[mask(v,w.config['id'],w.config['pair_keys'],'unique') for v,w in zip([a,b],self.workers)]
        np.testing.assert_allclose(secure_sum(payloads,3),a+b,atol=1e-6)
        self.assertFalse(np.allclose(secure_sum(payloads[:1],3),a))
        self.assertGreater(epsilon(20,8,1e-6),epsilon(10,8,1e-6))


if __name__=='__main__':
    unittest.main()
