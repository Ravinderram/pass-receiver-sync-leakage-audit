"""Step 17 - Accuracy as a function of how early the window ends.

The offset is measured BACKWARDS FROM THE FRAME EMITTED BY THE SYNCHRONISER
(step3), not from true ball contact, which no public dataset provides. Because
the emitted frame itself sits after contact by a match-varying amount, a given
offset does not correspond to the same moment relative to the kick in every
match. The curve is a trade-off against a tool-dependent origin.

At every offset in config.LEAD_SWEEP_LEADS this reports, on the test matches:
  model             top-1 of the model trained at that offset, canonical protocol
                    (fit matches, early stopping on the validation match), one
                    model per seed; the 0.4 s point IS the canonical model,
                    loaded from the shared checkpoint cache
  nearest_teammate  the yardstick
  ball_direction    probe: the teammate the ball's velocity points at
  passer_direction  probe: the teammate the passer's velocity points at

The probes show whether a shortcut is AVAILABLE at that offset. Whether a model
USES it is step24 (cross-offset) and step25 (ball masking). A passer signal that
rises towards the anchor is reported as unresolved: legitimate decision
execution also becomes more informative as the pass approaches.

Datasets for non-canonical offsets are built into work/lead_cache/; the canonical
work/dataset_*.npz files are never overwritten (the old version did overwrite them).

Run:  python step17_leadsweep.py
Out:  work/leadsweep_results.csv (per seed), work/leadsweep_summary.csv,
      figures/paper/figD_leadsweep.png
"""

import numpy as np
import pandas as pd

import config as C
import experiment as E
import probes as PR
import provenance as P
from device import banner as device_banner

LEADS = C.LEAD_SWEEP_LEADS


def probe_row(split):
    X, y = split["X"], split["y"]
    s = PR.snapshot_from_samples(X, y, t=-1)
    return {name: PR.run_probe(name, s)["acc"]
            for name in ("nearest_teammate", "ball_direction", "passer_direction")}


if __name__ == "__main__":
    device_banner()
    run_id = P.canonical_run_id()
    P.print_header("Step 17 - accuracy versus window-end offset", run_id,
                   offsets_s=LEADS, seeds=C.SEEDS, fit=C.FIT_MATCHES,
                   validation=C.VAL_MATCH, test=C.TEST_MATCHES,
                   training=f"cap {C.EPOCHS}, patience {C.PATIENCE}, best epoch restored")

    rows = []
    for lead in LEADS:
        print(f"\n--- offset {lead:.2f} s before the emitted frame ---")
        tests = {m: E.load_split([m], lead) for m in C.TEST_MATCHES}
        probes = {m: probe_row(t) for m, t in tests.items()}
        for seed in C.SEEDS:
            model, stats, info = E.get_or_train(lead, seed)
            for m, te in tests.items():
                acc = float((E.evaluate(model, stats, te).argmax(1) == te["y"]).mean())
                rows.append({"lead_s": lead, "test_match": m, "seed": seed,
                             "n": len(te["y"]), "model_top1": acc,
                             "selected_epoch": info["selected_epoch"],
                             **probes[m]})
        for m in C.TEST_MATCHES:
            accs = [r["model_top1"] for r in rows if r["lead_s"] == lead and r["test_match"] == m]
            p = probes[m]
            print(f"  {m}  model {np.mean(accs):.3f} (sd {np.std(accs, ddof=1) if len(accs) > 1 else float('nan'):.3f})"
                  f"   nearest {p['nearest_teammate']:.3f}   ball {p['ball_direction']:.3f}"
                  f"   passer {p['passer_direction']:.3f}")

    df = P.stamp(pd.DataFrame(rows), run_id, model_variant="leadsweep")
    df.to_csv(f"{C.WORK}/leadsweep_results.csv", index=False)
    summ = (df.groupby(["lead_s", "test_match"])
            .agg(n=("n", "first"), model_mean=("model_top1", "mean"),
                 model_sd=("model_top1", lambda v: v.std(ddof=1)),
                 n_seeds=("seed", "nunique"),
                 nearest_teammate=("nearest_teammate", "first"),
                 ball_direction=("ball_direction", "first"),
                 passer_direction=("passer_direction", "first"))
            .reset_index())
    for p in ("ball_direction", "passer_direction"):
        summ[f"{p}_excess"] = summ[p] - summ["nearest_teammate"]
    summ = P.stamp(summ, run_id, model_variant="leadsweep_summary")
    summ.to_csv(f"{C.WORK}/leadsweep_summary.csv", index=False)
    print(f"\nSaved -> {C.WORK}/leadsweep_results.csv")
    print(f"Saved -> {C.WORK}/leadsweep_summary.csv")

    for m in C.TEST_MATCHES:
        s = summ[summ.test_match == m].sort_values("lead_s", ascending=False)
        for p in ("ball_direction", "passer_direction"):
            d = PR.flatness_diagnostic(s[p].to_numpy(), s["nearest_teammate"].to_numpy())
            print(f"  {m} {p:17s} across offsets: {PR.LABEL_TEXT[d['label']]}")

    try:
        from step31_paper_figures import figure_d_leadsweep
        figure_d_leadsweep()
    except Exception as exc:
        print(f"(figure D not drawn: {exc})")
