"""Step 30 - Do the technique-transfer effects stabilise with more data?

The paper's own gap is about the small-sample regime: at ~10^3 training passes
several ablation rows change sign between runs. This asks a narrower, answerable
question: as the amount of training data grows from one fit match to all of them,
do the individual intervention gains settle down, or do they keep moving?

Design
  sizes        config.SIZE_CURVE_N_FIT matches, taken as NESTED prefixes of
               config.FIT_MATCHES, so a larger size always contains the smaller
               one and the curve is not confounded by which matches were picked
  settings     config.SIZE_CURVE_SETTINGS, using the step12 definitions; each is
               compared with "plain" trained on the SAME subset
  seeds        config.SEEDS, paired: the gain is computed per seed and then
               averaged, so seed-to-seed variation is inside the error bar
  validation   unchanged (config.VAL_MATCH) at every size, so early stopping is
               not itself a function of training size
  test         unchanged

This is NOT a scaling law. Four sizes, one dataset and a handful of seeds cannot
support one. The claim it can support is narrower: whether the sign and rough
magnitude of an effect are stable across this range.

Run:  python step30_size_curve.py
Out:  work/size_curve_results.csv, work/size_curve_summary.csv,
      figures/paper/figF_size_curve.png
"""

import numpy as np
import pandas as pd

import config as C
import experiment as E
import metrics_lib as M
import provenance as P
from device import banner as device_banner
from step12_ablation import COMPONENTS, KIND, run_setting

SIZES = C.SIZE_CURVE_N_FIT
def _sign_consistent(paired):
    """True unless the paired differences include both a positive and a negative.

    Exact zeros are not a sign change: with few seeds a factor can leave the
    predictions untouched, and calling that "the sign changes" would overstate
    the instability.
    """
    v = np.asarray(paired, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return True
    return not ((v > 0).any() and (v < 0).any())


SETTINGS = C.SIZE_CURVE_SETTINGS


if __name__ == "__main__":
    device_banner()
    run_id = P.canonical_run_id()
    subsets = {n: C.FIT_MATCHES[:n] for n in SIZES}
    P.print_header("Step 30 - training size versus intervention gain", run_id,
                   sizes=subsets, settings=SETTINGS, seeds=C.SEEDS,
                   validation=C.VAL_MATCH, test=C.TEST_MATCHES,
                   runs=len(SIZES) * len(set(SETTINGS + ["plain"])) * len(C.SEEDS))

    rows = []
    for n, matches in subsets.items():
        n_samples = len(E.load_split(matches)["y"])
        print(f"\n--- {n} fit match(es) {matches}: {n_samples} passes ---")
        for name in dict.fromkeys(["plain"] + SETTINGS):
            r = run_setting(name, C.SEEDS, fit_matches=matches,
                            subset_tag=f"sizecurve_{n}", verbose=False)
            for row in r:
                row.update({"n_fit_matches": n, "fit_matches": "+".join(matches),
                            "n_fit_samples": n_samples})
            rows += r
            df_n = pd.DataFrame([x for x in r])
            print(f"  {name:16s} " + "   ".join(
                f"{m} {g.top1.mean():.3f}" for m, g in df_n.groupby("test_match")))

    df = P.stamp(pd.DataFrame(rows), run_id, lead_s=C.PREDICT_LEAD_S)
    df["model_variant"] = df["setting"]
    df.to_csv(f"{C.WORK}/size_curve_results.csv", index=False)

    out = []
    for n in SIZES:
        sub = df[df.n_fit_matches == n]
        base = sub[sub.setting == "plain"]
        for name in SETTINGS:
            if name == "plain":
                continue
            g = sub[sub.setting == name]
            paired = [((g[g.seed == s].groupby("test_match").top1.mean()
                        - base[base.seed == s].groupby("test_match").top1.mean()).mean() * 100)
                      for s in sorted(set(g.seed) & set(base.seed))]
            mean, sd, lo, hi = M.mean_sd_ci(paired)
            out.append({"n_fit_matches": n,
                        "n_fit_samples": int(sub.n_fit_samples.iloc[0]),
                        "setting": name, "kind": KIND[name],
                        "components": "+".join(COMPONENTS[name]),
                        "n_seeds": len(paired), "gain_pp": mean, "paired_sd_pp": sd,
                        "ci95_lo_pp": lo, "ci95_hi_pp": hi,
                        "clears_2sd": bool(np.isfinite(sd) and abs(mean) > 2 * sd),
                        "sign_consistent": _sign_consistent(paired)})
    summ = pd.DataFrame(out)
    P.stamp(summ, run_id, model_variant="size_curve_summary").to_csv(
        f"{C.WORK}/size_curve_summary.csv", index=False)
    print("\n" + summ[["n_fit_matches", "n_fit_samples", "setting", "gain_pp",
                       "paired_sd_pp", "ci95_lo_pp", "ci95_hi_pp", "clears_2sd",
                       "sign_consistent"]].round(2).to_string(index=False))

    print("\nstability across sizes (sign of the gain):")
    for name, g in summ.groupby("setting"):
        signs = np.sign(g.sort_values("n_fit_matches").gain_pp.to_numpy())
        stable = len(set(signs[np.isfinite(signs)])) == 1
        print(f"  {name:16s} {'same sign at every size' if stable else 'SIGN CHANGES across sizes'}"
              f"   gains " + ", ".join(f"{v:+.1f}" for v in
                                       g.sort_values('n_fit_matches').gain_pp))
    print("\nFour sizes on one dataset cannot establish a scaling law, and none is")
    print("claimed. This shows only whether an effect's sign is stable in range.")
    print(f"\nSaved -> {C.WORK}/size_curve_results.csv, size_curve_summary.csv")
    try:
        from step31_paper_figures import figure_f_size_curve
        figure_f_size_curve()
    except Exception as exc:
        print(f"(figure F not drawn: {exc})")
