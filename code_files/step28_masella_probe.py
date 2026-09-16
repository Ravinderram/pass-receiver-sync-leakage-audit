"""Step 28 - A one-line probe built from the Masella passer-facing angle.

Masella et al. (arXiv:2605.25696) predict receiver selection with an
edge-conditioned MPNN on a star graph. One of their three edge features is a
signed angle (their eq. 4)

    theta_pj = atan2(u_x w_y - u_y w_x, u . w)

where u is the normalised passer facing direction, defined there as the vector
from the ball to the passer, and w the vector from passer to teammate.

This script implements that angle on the IDSSE data and uses it as a SINGLE-LINE
PROBE: pick the teammate with the smallest |theta| (or the largest), and see how
often that is the receiver. Both readings are reported because the sign
convention of "from ball to passer" admits both, and the one applied to the test
matches is selected on the non-test matches only.

This is NOT a reproduction of their model. Their dataset (369 international
matches) and code are not public, no MPNN is trained here, and a one-line angle
rule is not their architecture. Nothing here bears on their reported accuracy.

Two conditions of their setup are worth stating because they matter for
interpretation, not as criticism:
  * they split passes randomly 70/15/15, so passes from one match appear in both
    training and test. Whether that changes their numbers is not established here
    and cannot be, without their data.
  * their pipeline uses the same Needleman-Wunsch family of synchronisers whose
    emitted frames this project probes.

Run:  python step28_masella_probe.py
Out:  work/masella_probe.csv, work/masella_probe_curve.csv,
      figures/paper/fig_masella_probe.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
import experiment as E
import probes as PR
import provenance as P
from common import banner

OUT = f"{C.FIGS}/paper"
INK, MUTED, DATA, WARN, GOOD = "#14171A", "#6E7378", "#2E6F9E", "#B3402F", "#1F7A4D"
RULES = ["masella_min_abs", "masella_max_abs"]


def probe_at(match_ids, lead, t=-1):
    rows = []
    for m in match_ids:
        sp = E.load_split([m], lead)
        s = PR.snapshot_from_samples(sp["X"], sp["y"], t=t)
        base = PR.run_probe("nearest_teammate", s)
        for rule in RULES + ["passer_direction"]:
            r = PR.run_probe(rule, s)
            rows.append({"match": m, "lead_s": lead, "probe": rule, "n": r["n"],
                         "n_defined": r["n_defined"], "accuracy": r["acc"],
                         "accuracy_where_defined": r["acc_defined"],
                         "nearest_teammate": base["acc"],
                         "excess": r["acc"] - base["acc"]})
    return rows


if __name__ == "__main__":
    banner("Step 28 - Masella-style passer-facing angle probe")
    run_id = P.canonical_run_id()
    P.print_header("passer-facing angle as a one-line probe", run_id,
                   angle="atan2(u x w, u . w), u = ball->passer, w = passer->teammate",
                   measured_at=f"the last window frame, {C.PREDICT_LEAD_S} s before "
                               "the emitted synchronised frame",
                   min_ball_passer_m=C.MASELLA_MIN_BALL_PASSER_M,
                   rules=RULES)

    sel = pd.DataFrame(probe_at(C.FIT_MATCHES + [C.VAL_MATCH], C.PREDICT_LEAD_S))
    means = sel[sel.probe.isin(RULES)].groupby("probe").accuracy.mean()
    chosen = str(means.idxmax())
    print("\nrule selection on NON-TEST matches only:")
    print(means.round(3).to_string())
    print(f"-> applying {chosen} to the test matches\n")

    rows = probe_at(list(C.MATCHES), C.PREDICT_LEAD_S)
    df = pd.DataFrame(rows)
    df["role"] = np.where(df.match.isin(C.TEST_MATCHES), "test",
                          np.where(df.match == C.VAL_MATCH, "validation", "fit"))
    df["selected_rule"] = chosen
    print(df[df.probe.isin(RULES + ["passer_direction"])]
          [["match", "role", "probe", "n", "n_defined", "accuracy",
            "nearest_teammate", "excess"]].round(3).to_string(index=False))

    curve = []
    for lead in C.LEAD_SWEEP_LEADS:
        curve += probe_at(C.TEST_MATCHES, lead)
    cdf = pd.DataFrame(curve)
    P.stamp(df, run_id, lead_s=C.PREDICT_LEAD_S, model_variant="masella_probe").to_csv(
        f"{C.WORK}/masella_probe.csv", index=False)
    P.stamp(cdf, run_id, model_variant="masella_probe_curve").to_csv(
        f"{C.WORK}/masella_probe_curve.csv", index=False)

    print("\nacross window-end offsets (test matches, mean):")
    piv = cdf.pivot_table(index="lead_s", columns="probe",
                          values=["accuracy", "excess"]).round(3)
    print(piv.to_string())
    g = cdf[cdf.probe == chosen].groupby("lead_s")
    acc = g.accuracy.mean().sort_index(ascending=False).to_numpy()
    yard = g.nearest_teammate.mean().sort_index(ascending=False).to_numpy()
    d = PR.flatness_diagnostic(acc, yard)
    print(f"\n{chosen} across offsets: {PR.LABEL_TEXT[d['label']]}")

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for rule, col, mk in ((chosen, DATA, "-o"), ("passer_direction", GOOD, "-s")):
        gg = cdf[cdf.probe == rule].groupby("lead_s").accuracy.mean()
        ax.plot(gg.index, gg.values, mk, color=col, lw=2.2, ms=6, label=rule)
    gg = cdf.groupby("lead_s").nearest_teammate.mean()
    ax.plot(gg.index, gg.values, "--", color=MUTED, lw=1.4, label="nearest teammate")
    ax.set_xlabel("window-end offset before the emitted synchronised frame (s)")
    ax.set_ylabel("probe accuracy")
    ax.set_title("Passer-facing angle as a one-line probe (not the Masella model)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig_masella_probe.{ext}", bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"\nSaved -> {C.WORK}/masella_probe.csv, masella_probe_curve.csv, "
          f"{OUT}/fig_masella_probe.png")
    print("\nThis is a probe built from one of their edge features, not their model,")
    print("and it says nothing about their published accuracy.")
