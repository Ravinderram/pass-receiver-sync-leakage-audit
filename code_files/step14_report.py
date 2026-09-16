"""Step 14 - One report a human can actually read.

The pipeline leaves its output as npz arrays, CSV tables and loose PNGs. A
supervisor cannot judge any of that. This assembles everything into a single
self-contained HTML file - figures embedded, nothing to unpack, opens offline.

The report leads with the validation ledger rather than the accuracy, because on
this project the accuracy was wrong twice before the checks caught it. What
someone reviewing this needs first is whether the pipeline can be believed.

Run:  python step14_report.py
Out:  report.html
"""

import base64
import glob
import os

import numpy as np
import pandas as pd

import config as C
from common import banner, load_npz

OUT = "report.html"


# ----------------------------------------------------------------- gathering
def read_csv(name):
    p = f"{C.WORK}/{name}"
    return pd.read_csv(p) if os.path.isfile(p) else None


def embed(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def dataset_summary():
    rows = []
    for mid in C.MATCHES:
        try:
            d = load_npz(f"dataset_{mid}")
        except FileNotFoundError:
            continue
        home, away, div = C.MATCHES[mid]
        rows.append({
            "Match": mid, "Fixture": f"{home} v {away}", "Div": div,
            "Samples": len(d["X"]),
            "Role": "test" if mid in C.TEST_MATCHES else "train",
        })
    return pd.DataFrame(rows)


def checks(synced, leak_rows):
    """The validation ledger. Each row is a claim we can check against a number
    from the dataset paper or from physics, not against our own hope."""
    out = []

    if synced is not None:
        b, a = synced["dist_before"].mean(), synced["dist_after"].median()
        out.append(("Event/tracking offset before sync", f"{b:.2f} m",
                    "9.37 m (paper)", abs(b - 9.37) < 4))
        out.append(("Event/tracking offset after sync (median)", f"{a:.2f} m",
                    "2.61 m (paper)", a < 5))
        if "orientation" in synced:
            o = synced["orientation"].value_counts().index[0]
            out.append(("Coordinate orientation chosen", o, "shift only",
                        o == "shift only"))
        if "ball_to_passer" in synced:
            bp = synced["ball_to_passer"].median()
            out.append(("Ball to passer at the chosen frame", f"{bp:.2f} m",
                        "small, annotation-free", bp < 4))

    total = sum(r["Samples"] for _, r in dataset_summary().iterrows()) \
        if len(dataset_summary()) else 0
    out.append(("Samples built", f"{total}", "~4,300 successful open-play",
                total > 3000))

    for name, val, ref, ok in leak_rows:
        out.append((name, val, ref, ok))
    return out


def leak_checks():
    rows = []
    for mid in C.TEST_MATCHES:
        try:
            d = load_npz(f"dataset_{mid}")
        except FileNotFoundError:
            continue
        X, y = d["X"], d["y"]
        if not len(X):
            continue
        import probes as PR
        snap = PR.snapshot_from_samples(X, y, t=-1)
        leak = PR.run_probe("ball_direction", snap)["acc"]
        yard = PR.run_probe("nearest_teammate", snap)["acc"]
        rows.append((f"Ball-direction shortcut on {mid}", f"{leak:.3f}",
                     f"nearest-teammate yardstick {yard:.3f}",
                     leak - yard < PR.MARGIN + 0.08))
    return rows


# -------------------------------------------------------------------- render
CSS = """
:root{
  --paper:#FCFCFA; --ink:#14171A; --muted:#6E7378; --rule:#E2E4DF;
  --pass:#1F7A4D; --fail:#B3402F; --data:#2E6F9E; --wash:#F2F3EF;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
.wrap{max-width:1000px;margin:0 auto;padding:56px 28px 96px}
h1{font-size:31px;line-height:1.15;margin:0 0 6px;letter-spacing:-.02em;font-weight:650}
h2{font-size:12px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted);
  font-weight:650;margin:56px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--rule)}
h3{font-size:16px;margin:26px 0 8px;font-weight:600}
p{margin:0 0 12px;max-width:74ch}
.lede{font-size:17px;color:#3A3F44;max-width:70ch}
.meta{font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--muted);
  margin-top:14px}
table{border-collapse:collapse;width:100%;margin:6px 0 18px;font-size:14px}
th{text-align:left;font-weight:600;font-size:11px;letter-spacing:.09em;
  text-transform:uppercase;color:var(--muted);padding:7px 10px 7px 0;
  border-bottom:1px solid var(--rule)}
td{padding:9px 10px 9px 0;border-bottom:1px solid var(--rule);vertical-align:top}
.num{font:13px ui-monospace,SFMono-Regular,Menlo,monospace;font-variant-numeric:tabular-nums}
.chip{display:inline-block;font:11px ui-monospace,Menlo,monospace;padding:2px 8px;
  border-radius:2px;letter-spacing:.04em}
.ok{background:rgba(31,122,77,.11);color:var(--pass)}
.no{background:rgba(179,64,47,.11);color:var(--fail)}
.ledger td:first-child{width:44%}
figure{margin:20px 0 26px}
figure img{width:100%;border:1px solid var(--rule);background:#fff;display:block}
figcaption{font-size:12.5px;color:var(--muted);margin-top:7px}
.note{border-left:2px solid var(--data);padding:2px 0 2px 16px;margin:18px 0;
  color:#3A3F44;font-size:14.5px}
.warn{border-left-color:var(--fail)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:1px;
  background:var(--rule);border:1px solid var(--rule);margin:18px 0 26px}
.cell{background:var(--paper);padding:14px 16px}
.cell b{display:block;font:24px ui-monospace,Menlo,monospace;font-weight:600;
  letter-spacing:-.02em}
.cell span{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
code{font:12.5px ui-monospace,Menlo,monospace;background:var(--wash);padding:1px 5px}
@media(max-width:640px){.wrap{padding:32px 18px 64px}h1{font-size:25px}}
"""


def table_html(df, num_cols=()):
    if df is None or not len(df):
        return "<p class='meta'>not available — run the matching step</p>"
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = ""
    for _, r in df.iterrows():
        cells = ""
        for c in df.columns:
            v = r[c]
            cls = " class='num'" if c in num_cols or isinstance(v, (int, float,
                                                                   np.floating)) else ""
            v = f"{v:.3f}" if isinstance(v, (float, np.floating)) else v
            cells += f"<td{cls}>{v}</td>"
        body += f"<tr>{cells}</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build():
    synced = read_csv("passes_synced.csv")
    base = read_csv("baseline_results.csv")
    model = read_csv("model_results.csv")
    abl = read_csv("ablation_results.csv")
    ds = dataset_summary()

    ledger = checks(synced, leak_checks())
    n_ok = sum(1 for *_, ok in ledger if ok)

    rows = "".join(
        f"<tr><td>{n}</td><td class='num'>{v}</td><td class='num'>{ref}</td>"
        f"<td><span class='chip {'ok' if ok else 'no'}'>"
        f"{'pass' if ok else 'check'}</span></td></tr>"
        for n, v, ref, ok in ledger)

    headline = ""
    if model is not None:
        a = model[model.setting.str.startswith("A")].groupby("test_match")["top1"].mean()
        b = model[model.setting.str.startswith("B")].groupby("test_match")["top1"].mean()
        g = base[base.model == "gradient boosting"].set_index("test_match")["top1"] \
            if base is not None else None
        cells = ""
        for mid in a.index:
            cells += (f"<div class='cell'><b>{a[mid]:.3f}</b>"
                      f"<span>model · {mid}</span></div>")
            if g is not None and mid in g.index:
                cells += (f"<div class='cell'><b>{g[mid]:.3f}</b>"
                          f"<span>best baseline · {mid}</span></div>")
        gap = (a.mean() - b.mean()) * 100
        cells += (f"<div class='cell'><b>{gap:+.1f}</b>"
                  f"<span>points from 1.5 s of history</span></div>")
        headline = f"<div class='grid'>{cells}</div>"

    figs = ""
    for pat, cap in [
        ("figures/inspect_*.png",
         "One pass across the prediction window. Percentages are the model's belief; "
         "the green ring is who actually received. Confident-and-wrong panels are "
         "where the model's assumptions show."),
        ("figures/07_breakdown_*.png",
         "Accuracy by pass length, pitch zone and opponents in the passing lane."),
        ("figures/08_calibration_*.png",
         "When the model says 70%, is it right 70% of the time?"),
        ("figures/06_sync.png",
         "Event-to-ball distance before and after synchronisation, and the time "
         "correction applied to each event."),
    ]:
        for p in sorted(glob.glob(pat))[:2]:
            figs += (f"<figure><img src='data:image/png;base64,{embed(p)}'>"
                     f"<figcaption>{os.path.basename(p)} — {cap}</figcaption></figure>")

    abl_tbl = ""
    if abl is not None:
        piv = abl.pivot_table(index="setting", columns="test_match",
                              values="top1", aggfunc="mean").reset_index()
        abl_tbl = table_html(piv.round(3))

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pass receiver prediction — report</title><style>{CSS}</style></head><body>
<div class="wrap">

<h1>Predicting the pass receiver<br>from Bundesliga tracking data</h1>
<p class="lede">Given 1.5 seconds of play ending {C.PREDICT_LEAD_S}&nbsp;s before the
ball is struck, which of the passer's ten teammates receives it?</p>
<p class="meta">Bassek, Rein, Weber &amp; Memmert (2025), Sci Data 12:195 · CC-BY 4.0 ·
7 matches · 1,002,644 frames · {len(ds)} matches processed</p>

<h2>Validation ledger</h2>
<p>Accuracy on this project was wrong twice before these checks caught it, so the
report leads with them. Each row compares a measured value against a number from
the dataset paper or from physics — not against what we hoped to see.
<b>{n_ok} of {len(ledger)} pass.</b></p>
<table class="ledger"><thead><tr><th>Check</th><th>Measured</th>
<th>Reference</th><th></th></tr></thead><tbody>{rows}</tbody></table>

<div class="note warn"><b>Why the ball-direction check exists.</b> The synchroniser
emits the frame that best matches the pass event, and at that frame the ball is
already travelling, so its direction alone identifies the receiver far more often
than the nearest-teammate yardstick. The window therefore ends
{C.PREDICT_LEAD_S}&nbsp;s before that emitted frame (not before ball contact,
which this dataset does not record). Current measurements:
<code>work/leakage_probes.csv</code> and <code>work/leadsweep_summary.csv</code>.
Whether a trained model <i>uses</i> that shortcut is a separate question, answered
by <code>work/cross_offset_matrix.csv</code> and
<code>work/ball_mask_summary.csv</code>.</div>

<h2>Results</h2>
{headline}
{table_html(base) if base is not None else ""}

<h2>Which improvements helped</h2>
<p>Six factors taken from the literature, each measured on its own rather than
switched on together: train-time pitch reflection, test-time reflection averaging,
space features, a geometric edge bias in the attention, self-supervised
pretraining, and per-type normalisation. Rows marked <i>composite</i> combine
several factors and are never counted as separate interventions; the arithmetic is
in <code>work/ablation_summary.csv</code> and
<code>work/ablation_additivity.csv</code>.</p>
{abl_tbl}

<h2>Data</h2>
{table_html(ds)}
<div class="note">Five of the seven matches are the same home team, and those five
are exactly the second-division matches. Any split other than holding out a
first-division match leaks both team identity and division.</div>

<h2>Figures</h2>
{figs}

<h2>Limitations</h2>
<p>Seven matches. One held-out match is roughly 600 passes, so a single accuracy
carries a 95% interval of about four points — differences smaller than that are
not differences. Published work on this task trains on hundreds of thousands of
passes, so no claim here is about football; the claims are about method.</p>
<p class="meta">Generated by step14_report.py</p>
</div></body></html>"""

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    return n_ok, len(ledger)


if __name__ == "__main__":
    banner("Step 14 - building the report")
    ok, total = build()
    size = os.path.getsize(OUT) / 1e6
    print(f"validation ledger: {ok}/{total} checks pass")
    print(f"\nSaved -> {OUT}  ({size:.1f} MB, self-contained)")
    print("Open it in a browser. Figures are embedded, so it can be emailed as-is.")
