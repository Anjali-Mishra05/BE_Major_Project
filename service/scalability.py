"""Synthetic cohort scaling: same records and algorithm, changing client count."""
import time
import secrets
import numpy as np
from .common import ROOT, atomic_json, bins, gbdt, task
from .privacy import mask, secure_sum


def main():
    schema=bins.build_schema(['x'],{'x':'binary'},[2],{})
    x=np.tile([[0],[1]],(3000,1));y=x[:,0].astype(float)
    records=[]
    predictions=[]
    for count in (2,3,5):
        ids=[f'client-{i}' for i in range(count)]
        pairs={cid:{} for cid in ids}
        for i,a in enumerate(ids):
            for b in ids[i+1:]:pairs[a][b]=pairs[b][a]=secrets.token_hex(32)
        chunks=np.array_split(np.arange(len(y)),count)
        shards=[task.Shard(cid,x[ix],y[ix],np.zeros(len(ix))) for cid,ix in zip(ids,chunks)]
        forest=gbdt.Forest(schema,{'max_depth':2,'min_child_samples':1})
        start=time.perf_counter();byte_count=0;rounds=0
        for k in range(5):
            tree=gbdt.Tree(2)
            for shard in shards:shard.raw=forest.raw_score(shard.binned)
            def fetch(frontier,depth):
                nonlocal byte_count,rounds
                payloads=[mask(task.histograms(s,tree,frontier,1,schema),cid,pairs[cid],f'{count}:{k}:{depth}') for cid,s in zip(ids,shards)]
                byte_count+=sum(len(p) for p in payloads);rounds+=1
                return gbdt.unpack_histograms(secure_sum(payloads,len(frontier)*6),frontier,2)
            gbdt.grow_tree(schema,forest.params,tree,fetch);forest.add_tree(tree)
        predictions.append(forest.predict_proba(x))
        records.append({'clients':count,'records':len(y),'trees':5,'seconds':time.perf_counter()-start,
                        'masked_base64_bytes':byte_count,'rounds':rounds})
    gap=max(float(np.max(np.abs(p-predictions[0]))) for p in predictions)
    assert gap<1e-5,gap
    report={'scope':'synthetic, single-process compute/aggregation benchmark; not network scaling',
            'cohorts':records,'max_prediction_gap':gap}
    atomic_json(ROOT/'reports/scalability.json',report)
    print(report)


if __name__=='__main__':main()
