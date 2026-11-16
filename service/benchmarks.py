"""Train distinct algorithms on the same split; measure isolated ONNX serving loads."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from .common import ROOT, atomic_json, metrics
from .onnx_model import session


def load_test(model_path, data_path, rate, seconds):
    """Single request worker with a scheduled arrival stream; latency includes backlog."""
    import psutil
    runtime = session(model_path)
    X = np.load(data_path)
    process = psutil.Process()
    name = runtime.get_inputs()[0].name
    for i in range(30):
        runtime.run(None, {name: X[i % len(X):i % len(X)+1]})
    baseline = peak = process.memory_info().rss
    cpu0 = process.cpu_times()
    start = time.perf_counter()
    latency, service = [], []
    count = max(1, int(rate * seconds))
    for i in range(count):
        due = start + i / rate
        remaining = due - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        before = time.perf_counter()
        runtime.run(None, {name: X[i % len(X):i % len(X)+1]})
        end = time.perf_counter()
        latency.append((end - due) * 1000)
        service.append((end - before) * 1000)
        peak = max(peak, process.memory_info().rss)
    elapsed = time.perf_counter() - start
    cpu1 = process.cpu_times()
    return dict(offered_tps=rate, completed=count, errors=0, duration_s=elapsed,
        achieved_tps=count / max(elapsed, seconds),
        latency_p50_ms=float(np.percentile(latency, 50)), latency_p95_ms=float(np.percentile(latency, 95)),
        inference_p50_ms=float(np.percentile(service, 50)), inference_p95_ms=float(np.percentile(service, 95)),
        cpu_one_core_percent=100 * (cpu1.user+cpu1.system-cpu0.user-cpu0.system) / elapsed,
        rss_baseline_mb=baseline / 1e6, rss_peak_mb=peak / 1e6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rates', type=int, nargs='+', default=[50, 250, 1000])
    ap.add_argument('--seconds', type=float, default=5)
    ap.add_argument('--worker', nargs=2, metavar=('MODEL', 'DATA'))
    args = ap.parse_args()
    if args.seconds <= 0 or any(x <= 0 for x in args.rates):
        ap.error('Rates and duration must be positive')
    if args.worker:
        print(json.dumps(load_test(*args.worker, args.rates[0], args.seconds)))
        return
    sys.path.insert(0, str(ROOT / 'src'))
    from data import load_data, stratified_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from skl2onnx import to_onnx
    X, y, ts = load_data()
    tr, va, te = stratified_split(X, y, ts)
    train, test = X.iloc[tr].to_numpy(dtype=np.float32), X.iloc[te].to_numpy(dtype=np.float32)
    out = ROOT / 'runtime/benchmarks'
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'test.npy', test)
    algorithms = {
        'Logistic regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)),
        'Decision tree': DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, random_state=42),
        'Random forest': RandomForestClassifier(n_estimators=80, max_depth=10, min_samples_leaf=30, n_jobs=1, random_state=42),
    }
    report = dict(created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        environment=f'{platform.system()} {platform.machine()}, Python {platform.python_version()}',
        train_rows=len(tr), validation_rows=len(va), test_rows=len(te), features=list(X.columns),
        test_indices_sha256=hashlib.sha256(np.asarray(te, dtype=np.int64).tobytes()).hexdigest(),
        methodology='Same stratified 70/15/15 split, seed 42; fixed threshold 0.5; validation unused; no test tuning. '
        'Centralized algorithm comparison, separate from federated privacy experiments. ONNX CPU, one thread. '
        'Each model/load runs in a fresh process. Scheduled single-transaction arrivals; end-to-end latency includes backlog; '
        'inference latency excludes scheduler and preprocessing. CPU is process time / wall time (100% = one core); '
        'RSS includes interpreter, runtime and loaded test inputs, sampled after each request. Synthetic labels.', models=[])
    for index, (name, model) in enumerate(algorithms.items()):
        print(f'Training {name}', flush=True)
        start = time.perf_counter()
        model.fit(train, y.iloc[tr])
        training_seconds = time.perf_counter() - start
        native = model.predict_proba(test)[:, 1]
        classifier = model.steps[-1][1] if hasattr(model, 'steps') else model
        converted = to_onnx(model, train[:1], options={id(classifier): {'zipmap': False}}, target_opset=17)
        path = out / f'model-{index}.onnx'
        path.write_bytes(converted.SerializeToString())
        runtime = session(path)
        predicted = runtime.run(None, {runtime.get_inputs()[0].name: test})[1][:, 1]
        delta = float(np.max(np.abs(predicted-native)))
        if delta > 2e-5:
            raise RuntimeError(f'{name}: ONNX conversion mismatch {delta}')
        row = dict(name=name, metrics=metrics.evaluate(y.iloc[te], predicted, 0.5),
            training_seconds=training_seconds, model_bytes=path.stat().st_size,
            onnx_max_abs_error=delta, loads=[])
        for rate in args.rates:
            result = subprocess.run([sys.executable, '-m', 'service.benchmarks', '--worker', str(path),
                str(out / 'test.npy'), '--rates', str(rate), '--seconds', str(args.seconds)],
                cwd=ROOT, capture_output=True, text=True, timeout=max(120, args.seconds * 10), check=True)
            row['loads'].append(json.loads(result.stdout))
        report['models'].append(row)
    # Publish only a complete comparison; failed reruns retain prior evidence.
    atomic_json(ROOT / 'reports/algorithm_benchmarks.json', report)
    print('Saved reports/algorithm_benchmarks.json', flush=True)


if __name__ == '__main__':
    main()
