import json
import math
import os
import sys
import tempfile
import time
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
    payload = json.dumps(clean(value), allow_nan=False, indent=2)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Windows antivirus/indexers can briefly open a file without delete sharing.
        # Retry only permission failures, with a bounded deadline; never truncate the
        # previous committed checkpoint or silently swallow a persistent failure.
        for attempt in range(8):
            try:
                tmp.replace(path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(min(.02 * 2**attempt, .25))
    finally:
        tmp.unlink(missing_ok=True)
