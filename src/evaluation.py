"""
Demo / evaluation harness for the trained LightGBM baseline.

Loads the saved model and prints a full evaluation on a held-out test set. Does NOT
retrain, so it runs in a few seconds and is safe to drive live in front of an audience.

    python src/evaluation.py                          # full report, stratified split
    python src/evaluation.py --split chronological
    python src/evaluation.py --samples 20             # score 20 individual transactions
    python src/evaluation.py --score-file data/samples/demo_test_samples.csv
    python src/evaluation.py --html --save            # projectable page + JSON

Sections printed:
    1  Dataset and split
    2  Trained model
    3  Classification metrics - accuracy, precision, recall, F1, ROC-AUC, PR-AUC, FPR
    4  Confusion matrix
    5  Reference baselines - what a trivial model scores on the same data
    6  Feature importance (LightGBM total split gain)
    7  Inference latency - single-transaction and batch, for the edge budget
    8  Transaction-level scoring - individual transactions with their fraud scores
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402  - sets OpenMP env vars before lightgbm loads

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import modeling  # noqa: E402
from config import (  # noqa: E402
    DATASET,
    FIGURES_DIR,
    REPORTS_DIR,
    ROOT,
    SPLITS,
    ensure_dirs,
)
from data import decode, get_splitter, load_data, load_score_file  # noqa: E402

W = 78  # console width
MODEL_NAME = "LightGBM"
MAX_SAMPLES = 20


def rule(char: str = "-") -> None:
    print(char * W)


def header(n: int, title: str) -> None:
    print()
    rule("=")
    print(f" {n}. {title.upper()}")
    rule("=")


def bar(value: float, vmax: float, width: int = 26) -> str:
    filled = int(round(width * value / vmax)) if vmax > 0 else 0
    return "#" * filled + "." * (width - filled)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the trained UPI fraud baseline.")
    ap.add_argument("--split", choices=SPLITS, default="stratified",
                    help="which held-out test set to score (default: stratified)")
    ap.add_argument("--samples", type=int, default=MAX_SAMPLES,
                    help=f"individual transactions to score, max {MAX_SAMPLES}")
    ap.add_argument("--score-file", type=Path, default=None,
                    help="CSV of new transactions to score instead of sampling the test set")
    ap.add_argument("--top-features", type=int, default=12,
                    help="feature importances to list (default: 12)")
    ap.add_argument("--save", action="store_true", help="write reports/evaluation_report.json")
    ap.add_argument("--html", action="store_true",
                    help="write reports/evaluation_report.html for projecting to an audience")
    args = ap.parse_args()
    args.samples = max(1, min(args.samples, MAX_SAMPLES))

    ensure_dirs()
    print()
    rule("=")
    print(" UPI FRAUD DETECTION - BASELINE MODEL EVALUATION".center(W))
    print(" Phase I, Objective 2  |  centralized LightGBM baseline".center(W))
    ctx = {"split": args.split}  # collected for the optional HTML render

    booster, model_file = modeling.load(args.split)
    print(f" Split: {args.split}".center(W))
    rule("=")

    # --- 1. dataset -----------------------------------------------------------
    header(1, "Dataset and split")
    X, y, ts = load_data()
    tr, va, te = get_splitter(args.split)(X, y, ts)
    Xva, yva = X.iloc[va], y.iloc[va]
    Xte, yte = X.iloc[te], y.iloc[te]

    print(f"  Source            {DATASET.relative_to(ROOT)}")
    print(f"  Transactions      {len(X):,}")
    print(f"  Features          {X.shape[1]}")
    print(f"  Fraud cases       {int(y.sum()):,}  ({y.mean() * 100:.3f}% of all transactions)")
    print(f"  Period            {ts.min().date()} to {ts.max().date()}")
    print()
    print(f"  Split strategy    {args.split}")
    for name, idx in (("Train", tr), ("Validation", va), ("Test", te)):
        n, f = len(idx), int(y.iloc[idx].sum())
        print(f"    {name:<12}{n:>8,} transactions{f:>6} fraud  ({f / n * 100:.3f}%)")
    print()
    print("  The model below never saw the test rows during training.")

    ctx["dataset"] = {"rows": int(len(X)), "features": int(X.shape[1]),
                      "fraud": int(y.sum()), "rate": float(y.mean()),
                      "start": str(ts.min().date()), "end": str(ts.max().date()),
                      "test_size": int(len(te)), "test_fraud": int(yte.sum())}
    ctx["partitions"] = [(n, len(i), int(y.iloc[i].sum()))
                         for n, i in (("Train", tr), ("Validation", va), ("Test", te))]

    # --- 2. model -------------------------------------------------------------
    header(2, "Trained model")
    n_trees = booster.num_trees()
    size_kb = model_file.stat().st_size / 1024
    print(f"  {MODEL_NAME:<14}{model_file.relative_to(ROOT)}")
    print(f"                {n_trees} trees, {size_kb:.1f} KB on disk")
    print()
    print("  Fits comfortably on Raspberry Pi / Jetson Nano class hardware.")
    ctx["models"] = [{"name": MODEL_NAME, "detail": f"{n_trees} trees",
                      "size_kb": size_kb, "file": model_file.name}]

    # --- 3. metrics -----------------------------------------------------------
    header(3, "Classification metrics on the test set")
    scores_va, scores_te = booster.predict(Xva), booster.predict(Xte)
    tuned = modeling.best_f1_threshold(yva, scores_va)
    results = {MODEL_NAME: {"default": modeling.evaluate(yte, scores_te, 0.5),
                            "tuned": modeling.evaluate(yte, scores_te, tuned)}}

    cols = ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "FPR")
    keys = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fpr")
    print(f"  {'Model':<14}{'Threshold':>10}" + "".join(f"{c:>10}" for c in cols))
    rule()
    for mode in ("default", "tuned"):
        r = results[MODEL_NAME][mode]
        label = "0.50" if mode == "default" else f"{r['threshold']:.4f}*"
        print(f"  {MODEL_NAME:<14}{label:>10}" + "".join(f"{r[k]:>10.4f}" for k in keys))
    rule()
    print("  Two rows, one model: the same scores judged at two decision thresholds.")
    print("  ROC-AUC and PR-AUC are identical across them because both are")
    print("  threshold-independent - they measure ranking quality at every cut-off.")
    print("  * tuned to maximise fraud-class F1 on validation, then applied")
    print("    unchanged to test - never tuned on the test set.")
    ctx["metrics"] = results

    # --- 4. confusion matrix --------------------------------------------------
    header(4, "Confusion matrix (tuned threshold)")
    cm = results[MODEL_NAME]["tuned"]["confusion_matrix"]
    print("                        predicted legit   predicted fraud")
    print(f"    actual legit  {cm['tn']:>16,}  {cm['fp']:>16,}")
    print(f"    actual fraud  {cm['fn']:>16,}  {cm['tp']:>16,}")
    print(f"    -> caught {cm['tp']} of {cm['tp'] + cm['fn']} fraud cases, "
          f"raised {cm['fp']:,} false alarms")

    # --- 5. reference baselines ----------------------------------------------
    header(5, "Reference baselines on the same test set")
    dummy = modeling.evaluate(yte, np.zeros(len(yte)), 0.5)
    print(f"  Majority-class dummy   accuracy {dummy['accuracy']:.4f}   "
          f"recall {dummy['recall']:.4f}   fraud caught 0")
    print("    A model that labels every transaction legitimate. It needs no features")
    print("    and no training, and it catches no fraud at all.")
    print()
    print(f"  Random scorer          ROC-AUC 0.5000   PR-AUC "
          f"{yte.mean():.4f} (= fraud prevalence)")
    print()
    print(f"  HOW TO READ THIS. At {yte.mean() * 100:.1f}% fraud the dummy already reaches "
          f"{dummy['accuracy'] * 100:.1f}% accuracy,")
    print("  so accuracy alone says little. Compare ROC-AUC, PR-AUC and recall against")
    print("  the dummy and random scorer above. The label-correlated `behaviour` column")
    print("  is excluded from training (see reports/baseline_results.md).")
    ctx["dummy"] = dummy
    ctx["prevalence"] = float(yte.mean())

    # --- 6. feature importance ------------------------------------------------
    header(6, f"Feature importance (top {args.top_features})")
    gain = booster.feature_importance(importance_type="gain")
    names = list(X.columns)  # LightGBM sanitises spaces in its own feature names
    order = np.argsort(gain)[::-1][: args.top_features]
    vmax = float(gain[order[0]]) if len(order) and gain[order[0]] > 0 else 1.0
    print(f"  {'Feature':<30}{'Gain':>14}  Relative")
    rule()
    for i in order:
        print(f"  {names[i][:29]:<30}{gain[i]:>14,.1f}  {bar(float(gain[i]), vmax)}")
    rule()
    print("  Total split gain - how much each feature reduced training loss.")
    ctx["importance"] = [(names[i], float(gain[i])) for i in order]

    # --- 7. latency -----------------------------------------------------------
    header(7, "Inference latency")
    one = Xte.iloc[[0]]
    for _ in range(20):
        booster.predict(one)
    timings = []
    for _ in range(200):
        t0 = time.perf_counter()
        booster.predict(one)
        timings.append((time.perf_counter() - t0) * 1e3)
    timings = np.array(timings)

    t0 = time.perf_counter()
    booster.predict(Xte)
    batch_s = time.perf_counter() - t0

    print(f"  {MODEL_NAME:<14} single transaction   p50 {np.percentile(timings, 50):6.3f} ms   "
          f"p95 {np.percentile(timings, 95):6.3f} ms")
    print(f"  {MODEL_NAME:<14} batch of {len(Xte):,}    {batch_s * 1e3:6.1f} ms total   "
          f"{len(Xte) / batch_s:,.0f} txn/s")
    print()
    print("  Measured on this machine, single-threaded. Edge hardware will be slower,")
    print("  but the headroom against a real-time UPI decision budget is large.")
    ctx["latency"] = [{"model": MODEL_NAME, "p50": float(np.percentile(timings, 50)),
                       "p95": float(np.percentile(timings, 95)),
                       "batch_ms": batch_s * 1e3, "throughput": len(Xte) / batch_s}]

    # --- 8. transaction-level scoring ----------------------------------------
    if args.score_file:
        Xs, truth = load_score_file(args.score_file, X.columns)
        source = str(args.score_file)
        header(8, f"Scoring {len(Xs)} new transactions")
        print(f"  Source: {source}")
        if truth is None:
            print("  No fraud_flag column - showing scores only, no correctness check.")
    else:
        rng = np.random.default_rng(7)
        fraud_pos = np.flatnonzero(yte.values == 1)
        legit_pos = np.flatnonzero(yte.values == 0)
        n_fraud = min(max(args.samples // 4, 1), len(fraud_pos))
        pick = np.concatenate([rng.choice(fraud_pos, n_fraud, replace=False),
                               rng.choice(legit_pos, args.samples - n_fraud, replace=False)])
        rng.shuffle(pick)
        Xs, truth = Xte.iloc[pick], yte.iloc[pick]
        source = f"held-out {args.split} test set"
        header(8, f"Scoring {len(Xs)} individual transactions")
        print(f"  Source: {source} (unseen during training)")
    print()

    scores_new = booster.predict(Xs)
    ctx["samples"] = []
    correct = 0
    for j in range(len(Xs)):
        attrs = decode(Xs.iloc[j], list(X.columns))
        score = float(scores_new[j])
        decision = "FLAG" if score >= tuned else "pass"
        line = f"  Transaction {j + 1:>2}"
        if truth is not None:
            actual = int(truth.iloc[j])
            label = "FRAUD" if actual else "legitimate"
            ok = (actual == 1) == (decision == "FLAG")
            correct += ok
            line += f"   actual: {label:<11}  model: {'correct' if ok else 'WRONG'}"
            sample_truth = label
        else:
            sample_truth = "unlabelled"
        print(line)
        print("    " + "  ".join(f"{k}: {v}" for k, v in list(attrs.items())[:4]))
        print("    " + "  ".join(f"{k}: {v}" for k, v in list(attrs.items())[4:]))
        print(f"      {MODEL_NAME:<14} fraud score {score:.4f}   decision: {decision}")
        print()
        ctx["samples"].append({"attrs": attrs, "truth": sample_truth,
                               "scores": {MODEL_NAME: (score, decision)}})

    if truth is not None:
        print(f"  {correct} of {len(Xs)} decisions matched the true label "
              f"({correct / len(Xs) * 100:.0f}%).")
        print("  On a sample this small, and with an oversampled fraud mix, treat this")
        print("  as illustrative only - the section 3 metrics are the measured result.")
    ctx["score_source"] = source

    # --- outputs --------------------------------------------------------------
    if args.save:
        payload = {"split": args.split, "test_size": int(len(yte)),
                   "test_fraud": int(yte.sum()), "models": results,
                   "majority_class_dummy": dummy,
                   "random_scorer": {"roc_auc": 0.5, "pr_auc": round(float(yte.mean()), 6)}}
        out = REPORTS_DIR / "evaluation_report.json"
        out.write_text(json.dumps(payload, indent=2))
        print(f"\n  Written: {out.relative_to(ROOT)}")

    if args.html:
        from report_html import render

        # Relative path: the page lives in reports/, so this resolves when opened from disk.
        fig = FIGURES_DIR / f"curves_{args.split}.png"
        if fig.exists():
            ctx["figure_block"] = (
                '<section><p class="kicker">8 &middot; Curves</p>'
                '<h2>ROC and precision-recall</h2><figure>'
                f'<img alt="ROC and precision-recall curves, {args.split} split" '
                f'src="figures/{fig.name}">'
                '<figcaption>LightGBM against the random diagonal and the prevalence '
                'floor.</figcaption></figure></section>')
        out = REPORTS_DIR / "evaluation_report.html"
        out.write_text(render(ctx))
        print(f"  Written: {out.relative_to(ROOT)}    <- open this in a browser")

    rule("=")
    print(" Evaluation complete.".center(W))
    rule("=")
    print()


if __name__ == "__main__":
    main()
