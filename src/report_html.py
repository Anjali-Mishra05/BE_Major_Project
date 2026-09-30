"""
Renders an evaluation run as a self-contained HTML page for projection.

`evaluation.py --html` collects a context dict while it prints its console report,
then hands it here. The page is written to reports/ so the relative figure paths
resolve when it is opened from disk; it needs no server and no build step.
"""

from __future__ import annotations

import html
from datetime import datetime

METRIC_COLS = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
               ("f1", "F1"), ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("fpr", "FPR")]

CSS = """
:root{
  --ground:#F5F6F8;--surface:#FFFFFF;--sunk:#EDEFF3;--ink:#14181F;--mid:#414B59;
  --mute:#6B7683;--rule:#DCE1E7;--strong:#C3CBD5;--accent:#2F5DA8;--null:#B0722A;
  --null-soft:#F4E8D8;
  --sans:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  --serif:"IBM Plex Serif",Georgia,serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0F1319;--surface:#161B23;--sunk:#1B212B;--ink:#E6EAF0;--mid:#B4BECB;
  --mute:#8A95A4;--rule:#262E39;--strong:#37414F;--accent:#7BA3E8;--null:#D9A15C;
  --null-soft:#2E2519;}}
:root[data-theme="dark"]{
  --ground:#0F1319;--surface:#161B23;--sunk:#1B212B;--ink:#E6EAF0;--mid:#B4BECB;
  --mute:#8A95A4;--rule:#262E39;--strong:#37414F;--accent:#7BA3E8;--null:#D9A15C;
  --null-soft:#2E2519;}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--serif);
  font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:960px;margin:0 auto;padding:0 28px 80px}
header{background:var(--surface);border-bottom:1px solid var(--rule);padding:44px 0 28px;margin-bottom:40px}
header .wrap{padding-bottom:0}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--accent);margin:0 0 14px}
h1{font-family:var(--sans);font-weight:600;font-size:clamp(26px,4vw,34px);letter-spacing:-.02em;
  margin:0 0 12px;text-wrap:balance}
.meta{display:flex;flex-wrap:wrap;gap:6px 22px;font-family:var(--mono);font-size:11.5px;
  color:var(--mute);border-top:1px solid var(--rule);padding-top:14px;margin-top:18px}
.meta b{font-weight:500;color:var(--mid)}
section{margin:0 0 44px}
h2{font-family:var(--sans);font-weight:600;font-size:19px;margin:0 0 4px;letter-spacing:-.01em}
.kicker{font-family:var(--mono);font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;
  color:var(--mute);margin:0 0 8px}
p{margin:0 0 14px;max-width:70ch}
code{font-family:var(--mono);font-size:.87em;background:var(--sunk);padding:.1em .36em;border-radius:3px}
.scroll{overflow-x:auto;border:1px solid var(--rule);background:var(--surface);margin:0 0 12px}
table{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:12.5px}
caption{caption-side:top;text-align:left;font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--mute);padding:13px 15px 9px}
th,td{padding:8px 14px;text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}
thead th{font-weight:500;font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--mute);border-bottom:1px solid var(--strong)}
tbody tr{border-bottom:1px solid var(--rule)}
tbody tr:last-child{border-bottom:0}
tr.control{background:var(--sunk);color:var(--mid)}
tr.control td:first-child{font-weight:500}
td.flag{color:var(--null);font-weight:500}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:1px;
  background:var(--rule);border:1px solid var(--rule)}
.fact{background:var(--surface);padding:15px 17px}
.fact dt{font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--mute);margin:0 0 5px}
.fact dd{margin:0;font-family:var(--sans);font-size:21px;font-weight:600;font-variant-numeric:tabular-nums}
.fact dd small{display:block;font-family:var(--mono);font-size:10.5px;font-weight:400;
  color:var(--mute);margin-top:3px}
.fact.hi dd{color:var(--null)}
.callout{background:var(--null-soft);border-left:2px solid var(--null);padding:18px 20px;margin:0 0 16px}
.callout p:last-child{margin-bottom:0}
.callout .kicker{color:var(--null)}
.cms{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.cm{border:1px solid var(--rule);background:var(--surface);padding:16px 18px}
.cm h3{font-family:var(--sans);font-size:14px;font-weight:600;margin:0 0 12px}
.cm table{font-size:12px}
.cm td,.cm th{padding:6px 10px}
.cm .note{font-family:var(--mono);font-size:11px;color:var(--mid);margin:10px 0 0}
.bar{display:inline-block;height:8px;background:var(--accent);border-radius:1px;vertical-align:middle}
.txn{border:1px solid var(--rule);background:var(--surface);padding:14px 16px;margin-bottom:10px}
.txn .hd{display:flex;justify-content:space-between;align-items:baseline;gap:12px;margin-bottom:8px}
.txn .id{font-family:var(--sans);font-size:14px;font-weight:600}
.tag{font-family:var(--mono);font-size:10px;letter-spacing:.08em;text-transform:uppercase;
  padding:2px 8px;border-radius:2px;border:1px solid var(--strong);color:var(--mute)}
.tag.fraud{color:var(--null);border-color:var(--null);background:var(--null-soft)}
.attrs{display:flex;flex-wrap:wrap;gap:4px 18px;font-family:var(--mono);font-size:11.5px;
  color:var(--mid);margin-bottom:10px}
.attrs b{font-weight:400;color:var(--mute)}
.scores{display:flex;flex-wrap:wrap;gap:8px 24px;font-family:var(--mono);font-size:12px;
  border-top:1px solid var(--rule);padding-top:9px;font-variant-numeric:tabular-nums}
.dec{font-weight:500}
.dec.FLAG{color:var(--null)}
figure{margin:0;border:1px solid var(--rule);background:#FFF;padding:12px}
figure img{display:block;width:100%;height:auto}
figure figcaption{font-family:var(--mono);font-size:11px;color:#5B6572;padding:10px 3px 2px}
footer{border-top:1px solid var(--rule);padding-top:18px;font-family:var(--mono);font-size:11px;
  color:var(--mute);line-height:1.7}
@media print{body{background:#FFF}header{padding-top:0}section{break-inside:avoid}}
"""


def _esc(v) -> str:
    return html.escape(str(v))


def _metric_rows(metrics: dict) -> str:
    out = []
    for name, modes in metrics.items():
        for mode in ("default", "tuned"):
            r = modes[mode]
            thr = "0.50" if mode == "default" else f"{r['threshold']:.4f}*"
            cells = "".join(
                f'<td class="flag">{r[k]:.4f}</td>' if k in ("roc_auc", "pr_auc")
                else f"<td>{r[k]:.4f}</td>" for k, _ in METRIC_COLS)
            out.append(f"<tr><td>{_esc(name)}</td><td>{thr}</td>{cells}</tr>")
    return "\n".join(out)


def _confusion(metrics: dict) -> str:
    cards = []
    for name, modes in metrics.items():
        cm = modes["tuned"]["confusion_matrix"]
        total = cm["tp"] + cm["fn"]
        cards.append(f"""<div class="cm"><h3>{_esc(name)}</h3>
<table><thead><tr><th></th><th>pred. legit</th><th>pred. fraud</th></tr></thead>
<tbody><tr><td>actual legit</td><td>{cm['tn']:,}</td><td>{cm['fp']:,}</td></tr>
<tr><td>actual fraud</td><td>{cm['fn']:,}</td><td>{cm['tp']:,}</td></tr></tbody></table>
<p class="note">Caught {cm['tp']} of {total} fraud cases &middot; {cm['fp']:,} false alarms</p></div>""")
    return "\n".join(cards)


def _importance(rows) -> str:
    vmax = max((g for _, g in rows), default=1.0) or 1.0
    return "\n".join(
        f'<tr><td>{_esc(n)}</td><td>{g:,.1f}</td>'
        f'<td style="text-align:left"><span class="bar" style="width:{g / vmax * 160:.1f}px"></span></td></tr>'
        for n, g in rows)


def _samples(samples) -> str:
    out = []
    for i, s in enumerate(samples, 1):
        tag = "fraud" if s["truth"] == "FRAUD" else ""
        attrs = "".join(f"<span><b>{_esc(k)}</b> {_esc(v)}</span>" for k, v in s["attrs"].items())
        scores = "".join(
            f'<span>{_esc(m)} &middot; score {sc:.4f} &middot; '
            f'<span class="dec {d}">{d}</span></span>'
            for m, (sc, d) in s["scores"].items())
        out.append(f"""<div class="txn"><div class="hd"><span class="id">Transaction {i}</span>
<span class="tag {tag}">actual: {_esc(s['truth'])}</span></div>
<div class="attrs">{attrs}</div><div class="scores">{scores}</div></div>""")
    return "\n".join(out)


def render(ctx: dict) -> str:
    d, m = ctx["dataset"], ctx["metrics"]
    parts = "".join(
        f"<tr><td>{_esc(n)}</td><td>{c:,}</td><td>{f}</td><td>{f / c * 100:.3f}%</td></tr>"
        for n, c, f in ctx["partitions"])
    models = "".join(
        f"<tr><td>{_esc(x['name'])}</td><td style='text-align:left'>{_esc(x['detail'])}</td>"
        f"<td>{x['size_kb']:.1f} KB</td><td style='text-align:left'>{_esc(x['file'])}</td></tr>"
        for x in ctx["models"])
    lat = "".join(
        f"<tr><td>{_esc(x['model'])}</td><td>{x['p50']:.3f} ms</td><td>{x['p95']:.3f} ms</td>"
        f"<td>{x['batch_ms']:.1f} ms</td><td>{x['throughput']:,.0f}/s</td></tr>"
        for x in ctx["latency"])
    dummy = ctx["dummy"]
    source = _esc(ctx.get("score_source", "held-out test set"))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>UPI Fraud Baseline &mdash; Evaluation Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@400;500&display=swap">
<style>{CSS}</style></head><body>

<header><div class="wrap">
<p class="eyebrow">Phase I &middot; Objective 2 &middot; Baseline evaluation</p>
<h1>UPI Fraud Detection &mdash; Baseline Model Evaluation</h1>
<div class="meta">
<span><b>Split</b> {_esc(ctx['split'])}</span>
<span><b>Test set</b> {d['test_size']:,} transactions, {d['test_fraud']} fraud</span>
<span><b>Generated</b> {datetime.now():%Y-%m-%d %H:%M}</span>
<span><b>Source</b> src/evaluation.py</span>
</div></div></header>

<div class="wrap">

<section><p class="kicker">1 &middot; Data</p><h2>Dataset and split</h2>
<dl class="facts">
<div class="fact"><dt>Transactions</dt><dd>{d['rows']:,}</dd></div>
<div class="fact"><dt>Features</dt><dd>{d['features']}</dd></div>
<div class="fact hi"><dt>Fraud rate</dt><dd>{d['rate'] * 100:.3f}%<small>{d['fraud']} cases</small></dd></div>
<div class="fact"><dt>Period</dt><dd>2024<small>{d['start']} to {d['end']}</small></dd></div>
</dl>
<div class="scroll"><table><caption>Partitions &mdash; {_esc(ctx['split'])} split</caption>
<thead><tr><th>Partition</th><th>Transactions</th><th>Fraud</th><th>Rate</th></tr></thead>
<tbody>{parts}</tbody></table></div>
<p>The models never saw the test rows during training.</p></section>

<section><p class="kicker">2 &middot; Model</p><h2>Trained baseline</h2>
<div class="scroll"><table>
<thead><tr><th>Model</th><th>Architecture</th><th>Size</th><th>Artefact</th></tr></thead>
<tbody>{models}</tbody></table></div>
<p>Fits comfortably on Raspberry Pi / Jetson Nano class hardware.</p></section>

<section><p class="kicker">3 &middot; Results</p><h2>Classification metrics on the test set</h2>
<div class="scroll"><table>
<caption>Two rows per model &mdash; one decision threshold each</caption>
<thead><tr><th>Model</th><th>Threshold</th>{''.join(f'<th>{l}</th>' for _, l in METRIC_COLS)}</tr></thead>
<tbody>{_metric_rows(m)}
<tr class="control"><td>Majority-class dummy</td><td>&mdash;</td><td>{dummy['accuracy']:.4f}</td>
<td>0.0000</td><td>0.0000</td><td>0.0000</td><td>0.5000</td><td>&mdash;</td><td>0.0000</td></tr>
<tr class="control"><td>Random scorer</td><td>&mdash;</td><td>&mdash;</td><td>&mdash;</td><td>&mdash;</td>
<td>&mdash;</td><td>0.5000</td><td>{ctx['prevalence']:.4f}</td><td>&mdash;</td></tr>
</tbody></table></div>
<p>* Threshold that maximised fraud-class F1 on validation, then applied unchanged to test.
ROC-AUC and PR-AUC are identical across each model's two rows because both are
threshold-independent &mdash; they measure ranking quality across all thresholds at once.</p>
<div class="callout"><p class="kicker">How to read this</p>
<p>At {d['rate'] * 100:.2f}% fraud, accuracy is not a useful metric. The majority-class dummy
&mdash; which labels every transaction legitimate and needs no training &mdash; reaches
<strong>{dummy['accuracy'] * 100:.2f}% accuracy</strong> while catching zero fraud. The metrics
that matter are ROC-AUC, PR-AUC and recall, compared against the dummy and random scorer above.
The label-correlated <code>behaviour</code> column is excluded from training; see
<code>reports/baseline_results.md</code>.</p></div></section>

<section><p class="kicker">4 &middot; Errors</p><h2>Confusion matrix</h2>
<div class="cms">{_confusion(m)}</div></section>

<section><p class="kicker">5 &middot; Attribution</p><h2>Feature importance</h2>
<div class="scroll"><table><caption>Top {len(ctx['importance'])} by total split gain</caption>
<thead><tr><th>Feature</th><th>Gain</th><th style="text-align:left">Relative</th></tr></thead>
<tbody>{_importance(ctx['importance'])}</tbody></table></div></section>

<section><p class="kicker">6 &middot; Systems</p><h2>Inference latency</h2>
<div class="scroll"><table>
<thead><tr><th>Model</th><th>Single p50</th><th>Single p95</th><th>Batch total</th><th>Throughput</th></tr></thead>
<tbody>{lat}</tbody></table></div>
<p>Measured single-threaded on the development machine. Edge hardware will be slower, but the
headroom against a real-time UPI decision budget is large.</p></section>

<section><p class="kicker">7 &middot; Demonstration</p><h2>Transaction-level scoring</h2>
<p>Source: <code>{source}</code>. Decoded back to readable attributes and scored at the
tuned threshold.</p>
{_samples(ctx['samples'])}</section>

{ctx.get('figure_block', '')}

<footer>Generated by <code>src/evaluation.py --html</code> &middot; models from
<code>src/train_baseline.py</code> &middot; seed 42<br>
Full write-up: <code>reports/baseline_results.md</code></footer>

</div></body></html>
"""
