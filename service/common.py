import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'fl'))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from upi_fl import bins, gbdt, task, metrics


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def tree_from(blob):
    blob = dict(blob)
    blob['value'] = [float('nan') if x is None else x for x in blob['value']]
    return gbdt.Tree.from_arrays(blob)


def forest_from(blob):
    model = gbdt.Forest(blob['schema'], blob['params'], blob['base_score'])
    model.trees = [tree_from(t) for t in blob['trees']]
    return model


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(clean(value), allow_nan=False, indent=2), encoding='utf-8')
    tmp.replace(path)
