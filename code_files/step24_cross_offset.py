"""Step 24 - Does a trained model DEPEND on the near-anchor shortcut?

A probe that scores highly at the emitted frame shows a shortcut is AVAILABLE.
It does not show that a model trained there USES it: the model might have
learned the tactical picture and merely ignored the moving ball.

The cross-offset matrix separates the two. A model is trained at every offset in
config.CROSS_OFFSET_LEADS (canonical protocol, shared checkpoint cache) and each
trained model is evaluated on test sets rebuilt at EVERY offset.

  If the model trained at 0.0 s scores far above the 0.4 s model on 0.0 s inputs
  but collapses to or below the 0.4 s model's level when tested at 0.4 s, its
  extra accuracy came from information that exists only near the anchor: it
  depends on the shortcut.
  If it stays about as good as the 0.4 s model everywhere, it did not.

To keep the matrix comparable, each test match is restricted to the passes that
exist at every offset (a few early passes drop out at larger offsets); n is
reported. Offsets are relative to the emitted synchronised frame, not to contact.

Run:  python step24_cross_offset.py
Out:  work/cross_offset_results.csv (per seed and test match),
      work/cross_offset_matrix.csv (mean over seeds, pooled over test matches),
      work/cross_offset_metadata.json, figures/paper/figB_cross_offset.png
"""

import numpy as np
import pandas as pd

import config as C
import experiment as E
import provenance as P
from device import banner as device_banner

LEADS = C.CROSS_OFFSET_LEADS


def common_rows(match_id):
    rows = None
    for lead in LEADS:
        pr = set(E.load_split([match_id], lead)["pass_row"].tolist())
        rows = pr if rows is None else rows & pr
    return np.array(sorted(rows))


def restricted(match_id, lead, keep):
    s = E.load_split([match_id], lead)
    order = np.argsort(s["pass_row"])
    s = E.subset(s, order)
    return E.subset(s, np.isin(s["pass_row"], keep))


if __name__ == "__main__":
    device_banner()
    run_id = P.canonical_run_id()
    P.print_header("Step 24 - cross-offset dependence", run_id,
                   offsets_s=LEADS, seeds=C.SEEDS, fit=C.FIT_MATCHES,
                   validation=C.VAL_MATCH, test=C.TEST_MATCHES,
                   training_runs=len(LEADS) * len(C.SEEDS))

    keep = {m: common_rows(m) for m in C.TEST_MATCHES}
    tests = {(m, lead): restricted(m, lead, keep[m])
             for m in C.TEST_MATCHES for lead in LEADS}
    for m in C.TEST_MATCHES:
        print(f"  {m}: {len(keep[m])} passes present at every offset")

    rows = []
    for train_lead in LEADS:
        print(f"\n--- models trained at {train_lead:.2f} s ---")
        for seed in C.SEEDS:
            model, stats, info = E.get_or_train(train_lead, seed)
            for test_lead in LEADS:
                for m in C.TEST_MATCHES:
                    te = tests[(m, test_lead)]
                    acc = float((E.evaluate(model, stats, te).argmax(1) == te["y"]).mean())
                    rows.append({"train_lead_s": train_lead, "test_lead_s": test_lead,
                                 "seed": seed, "test_match": m, "n_common": len(te["y"]),
                                 "top1": acc, "selected_epoch": info["selected_epoch"]})
        line = []
        for test_lead in LEADS:
            v = [r["top1"] for r in rows if r["train_lead_s"] == train_lead
                 and r["test_lead_s"] == test_lead]
            line.append(f"{test_lead:.1f}: {np.mean(v):.3f}")
        print("  tested at  " + "   ".join(line))

    df = pd.DataFrame(rows)
    df.insert(0, "lead_s", df["train_lead_s"])
    df = P.stamp(df, run_id, model_variant="cross_offset")
    df.to_csv(f"{C.WORK}/cross_offset_results.csv", index=False)

    w = df.assign(hits=df.top1 * df.n_common)
    per_seed = (w.groupby(["train_lead_s", "test_lead_s", "seed"])
                .agg(hits=("hits", "sum"), n=("n_common", "sum")).reset_index())
    per_seed["top1_pooled"] = per_seed.hits / per_seed.n
    mat = (per_seed.groupby(["train_lead_s", "test_lead_s"])
           .agg(top1_mean=("top1_pooled", "mean"),
                top1_sd=("top1_pooled", lambda v: v.std(ddof=1)),
                n_pooled=("n", "first"), n_seeds=("seed", "nunique"))
           .reset_index())
    P.stamp(mat, run_id, model_variant="cross_offset_matrix").to_csv(
        f"{C.WORK}/cross_offset_matrix.csv", index=False)
    P.write_json(f"{C.WORK}/cross_offset_metadata.json", P.base_metadata(
        "cross_offset", run_id, offsets_s=LEADS, seeds=C.SEEDS,
        test_matches=C.TEST_MATCHES, n_common={m: int(len(v)) for m, v in keep.items()}))

    piv = mat.pivot(index="train_lead_s", columns="test_lead_s", values="top1_mean")
    print("\nmean top-1, pooled test matches (rows = trained at, columns = tested at):")
    print(piv.round(3).to_string())
    canon = C.PREDICT_LEAD_S
    if 0.0 in piv.index and canon in piv.columns:
        own = piv.loc[0.0, 0.0]
        moved = piv.loc[0.0, canon]
        ref = piv.loc[canon, canon]
        print(f"\nmodel trained at 0.00 s: {own:.3f} on 0.00 s inputs, {moved:.3f} on "
              f"{canon:.2f} s inputs")
        print(f"model trained at {canon:.2f} s on {canon:.2f} s inputs: {ref:.3f}")
        print(f"-> drop of the 0.00 s model when the near-anchor information is removed: "
              f"{own - moved:+.3f}; gap to the {canon:.2f} s model on the same inputs: "
              f"{moved - ref:+.3f}")
        print("A large drop to (or below) the reference level is evidence that the")
        print("0.00 s model relies on information that exists only near the anchor.")
    print(f"\nSaved -> {C.WORK}/cross_offset_results.csv")
    print(f"Saved -> {C.WORK}/cross_offset_matrix.csv")
    try:
        from step31_paper_figures import figure_b_cross_offset
        figure_b_cross_offset()
    except Exception as exc:
        print(f"(figure B not drawn: {exc})")
