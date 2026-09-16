"""Step 31 - The paper's figures, every one drawn from a results file.

No number is typed into this file. Each function reads a CSV written by the
experiment that produced it, and says so in the figure's own caption line.

  figA_topk           top-1 / top-2 / top-3 / top-5 by test match
                      <- work/canonical_metrics_summary.csv
  figB_cross_offset   train-offset x test-offset accuracy heatmap
                      <- work/cross_offset_matrix.csv
  figC_ball_mask      normal versus ball-masked accuracy
                      <- work/ball_mask_summary.csv
  figD_leadsweep      model, nearest, ball probe and passer probe versus offset
                      <- work/leadsweep_summary.csv
  figE_ablation       gain per intervention with seed variation
                      <- work/ablation_summary.csv
  figF_size_curve     training-data size versus intervention gain
                      <- work/size_curve_results.csv / size_curve_summary.csv

Offsets in every figure are measured from the EMITTED SYNCHRONISED FRAME, which
is an estimate of the pass moment, not a ground-truth contact time; the axis
labels say so.

Run:  python step31_paper_figures.py [A B C D E F]
Out:  figures/paper/fig<letter>_*.png and .pdf
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
import provenance as P

OUT = f"{C.FIGS}/paper"
INK, MUTED, RULE = "#14171A", "#6E7378", "#D8DAD5"
DATA, WARN, GOOD, ALT = "#2E6F9E", "#B3402F", "#1F7A4D", "#C98A2B"
OFFSET_LABEL = ("window-end offset before the emitted synchronised frame (s)\n"
                "(an estimate of the pass moment, not a ground-truth contact time)")

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 300, "font.size": 11,
                     "axes.titleweight": "600", "legend.frameon": False})


def _read(name, run_id=None, required=True):
    path = f"{C.WORK}/{name}"
    if not os.path.isfile(path):
        if required:
            raise FileNotFoundError(f"{path} missing; run the experiment that writes it")
        return None
    df = pd.read_csv(path)
    if run_id and "run_id" in df.columns:
        P.check_run_id(df, run_id, name, allow_stale="--allow-stale" in sys.argv)
    return df


def _save(fig, name, source):
    os.makedirs(OUT, exist_ok=True)
    fig.text(0.005, -0.02, f"source: {source}", fontsize=7.5, color=MUTED)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {OUT}/{name}.png  +  .pdf   (from {source})")


# --------------------------------------------------------------------- A
def figure_a_topk():
    run_id = P.canonical_run_id()
    s = _read("canonical_metrics_summary.csv", run_id)
    s = s[(s.model == "canonical") & (s.aggregation == "mean_over_seeds")
          & (s.match != "pooled")]
    ks = C.TOPK_REPORTED
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    width = 0.8 / len(s)
    for i, (_, r) in enumerate(s.iterrows()):
        vals = [r[f"top{k}"] for k in ks]
        errs = [r.get(f"top{k}_sd", np.nan) for k in ks]
        pos = np.arange(len(ks)) + (i - (len(s) - 1) / 2) * width
        ax.bar(pos, vals, width * 0.9, yerr=errs, capsize=3, color=DATA,
               alpha=0.55 + 0.45 * i, label=f"{r['match']} (n={int(r['n'])})",
               error_kw=dict(lw=1, ecolor=INK, alpha=0.7))
        for x, v in zip(pos, vals):
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=8.5, color=INK)
    for j, k in enumerate(ks):
        ax.plot([j - 0.42, j + 0.42], [k / C.N_TEAMMATES] * 2, color=WARN,
                ls=(0, (4, 3)), lw=1.2)
    ax.plot([], [], color=WARN, ls=(0, (4, 3)), lw=1.2, label="chance at that k")
    ax.set_xticks(np.arange(len(ks)))
    ax.set_xticklabels([f"top-{k}" for k in ks])
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Ranking accuracy by test match (mean over seeds, sd bars)")
    ax.legend(fontsize=9.5)
    ax.grid(axis="y", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "figA_topk_by_match", "work/canonical_metrics_summary.csv")


# --------------------------------------------------------------------- B
def figure_b_cross_offset():
    run_id = P.canonical_run_id()
    m = _read("cross_offset_matrix.csv", run_id)
    piv = m.pivot(index="train_lead_s", columns="test_lead_s", values="top1_mean")
    sd = m.pivot(index="train_lead_s", columns="test_lead_s", values="top1_sd")
    fig, ax = plt.subplots(figsize=(1.25 * len(piv.columns) + 3.4,
                                    1.05 * len(piv.index) + 2.6))
    im = ax.imshow(piv.values, cmap="Blues", aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            e = sd.values[i, j]
            ax.text(j, i, f"{v:.3f}\n±{e:.3f}" if np.isfinite(e) else f"{v:.3f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if v > np.nanmean(piv.values) else INK)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{c:.1f}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{r:.1f}" for r in piv.index])
    ax.set_xlabel("tested at offset (s before the emitted frame)")
    ax.set_ylabel("trained at offset (s)")
    ax.set_title("Top-1 accuracy across train and test offsets\n"
                 "a row that collapses when moved right depended on near-anchor information",
                 fontsize=11.5)
    fig.colorbar(im, ax=ax, shrink=0.85, label="top-1 accuracy")
    _save(fig, "figB_cross_offset", "work/cross_offset_matrix.csv")


# --------------------------------------------------------------------- C
def figure_c_ball_mask():
    run_id = P.canonical_run_id()
    s = _read("ball_mask_summary.csv", run_id)
    masks = sorted(s["mask"].unique())
    leads = sorted(s.train_lead_s.unique())
    fig, axes = plt.subplots(1, len(masks), figsize=(5.6 * len(masks), 4.3),
                             sharey=True)
    for ax, mask in zip(np.atleast_1d(axes), masks):
        sm = s[s["mask"] == mask]
        x = np.arange(len(leads))
        for k, (lab, col) in enumerate((("normal_mean", DATA),
                                        ("masked_mean", WARN))):
            vals = [sm[sm.train_lead_s == l][lab].mean() for l in leads]
            ax.bar(x + (k - 0.5) * 0.38, vals, 0.36, color=col, alpha=0.85,
                   label="normal" if k == 0 else "ball masked")
            for xi, v in zip(x + (k - 0.5) * 0.38, vals):
                ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=8.5)
        for xi, l in zip(x, leads):
            d = sm[sm.train_lead_s == l]
            ax.text(xi, 0.02, f"drop {d.abs_drop_mean.mean():+.3f}", ha="center",
                    fontsize=9, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([f"trained at\n{l:.2f} s" for l in leads])
        ax.set_title(f"mask: {mask}")
        ax.grid(axis="y", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("top-1 accuracy")
    np.atleast_1d(axes)[0].legend(fontsize=9.5)
    fig.suptitle("Same trained model, ball information removed at test time\n"
                 "the contrast between the two training offsets is the evidence, "
                 "not either drop alone", fontsize=12, weight="600", y=1.04)
    fig.tight_layout()
    _save(fig, "figC_ball_mask", "work/ball_mask_summary.csv")


# --------------------------------------------------------------------- D
def figure_d_leadsweep():
    run_id = P.canonical_run_id()
    s = _read("leadsweep_summary.csv", run_id)
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.6))
    for i, m in enumerate(sorted(s.test_match.unique())):
        g = s[s.test_match == m].sort_values("lead_s")
        ax[0].errorbar(g.lead_s, g.model_mean, yerr=g.model_sd, fmt="-o",
                       color=DATA, alpha=1 - 0.4 * i, lw=2.2, ms=6, capsize=3,
                       label=f"model, {m}")
        ax[0].plot(g.lead_s, g.nearest_teammate, "--", color=MUTED, lw=1.3,
                   alpha=1 - 0.4 * i,
                   label="nearest teammate" if i == 0 else None)
    ax[0].set_xlabel(OFFSET_LABEL, fontsize=9.5)
    ax[0].set_ylabel("top-1 accuracy")
    ax[0].set_title("(a) What earliness costs")
    ax[0].legend(fontsize=9.5)

    g = s.groupby("lead_s")[["ball_direction", "passer_direction",
                             "nearest_teammate"]].mean()
    ax[1].plot(g.index, g.ball_direction, "-o", color=WARN, lw=2.2, ms=6,
               label="ball direction probe")
    ax[1].plot(g.index, g.passer_direction, "-s", color=GOOD, lw=2.2, ms=6,
               label="passer direction probe")
    ax[1].plot(g.index, g.nearest_teammate, "--", color=MUTED, lw=1.4,
               label="nearest teammate")
    ax[1].set_xlabel(OFFSET_LABEL, fontsize=9.5)
    ax[1].set_ylabel("accuracy of a one-line rule")
    ax[1].set_title("(b) Which shortcuts are available, and when")
    ax[1].legend(fontsize=9.5)
    for a in ax:
        a.grid(alpha=0.45)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Receiver prediction against window-end offset "
                 "(offsets measured from the emitted frame)",
                 fontsize=12.5, weight="600", y=1.05)
    fig.tight_layout()
    _save(fig, "figD_leadsweep", "work/leadsweep_summary.csv")


# --------------------------------------------------------------------- E
def figure_e_ablation(robust=False):
    run_id = P.canonical_run_id()
    suffix = "_robust" if robust else ""
    g = _read(f"ablation_summary{suffix}.csv", run_id)
    g = g.sort_values("gain_pp")
    fig, ax = plt.subplots(figsize=(8.6, 0.52 * len(g) + 2.4))
    ypos = np.arange(len(g))
    colors = [GOOD if v > 0 else WARN for v in g.gain_pp]
    alphas = [0.9 if c else 0.45 for c in g.clears_2sd]
    for y, (_, r), col, al in zip(ypos, g.iterrows(), colors, alphas):
        ax.barh(y, r.gain_pp, height=0.6, color=col, alpha=al)
        ax.plot([r.ci95_lo_pp, r.ci95_hi_pp], [y, y], color=INK, lw=1.4)
        ax.plot([r.gain_pp - 2 * r.paired_sd_pp, r.gain_pp + 2 * r.paired_sd_pp],
                [y + 0.28] * 2, color=MUTED, lw=1.0)
        ax.text(r.gain_pp + (0.25 if r.gain_pp >= 0 else -0.25), y,
                f"{r.gain_pp:+.1f}", va="center",
                ha="left" if r.gain_pp >= 0 else "right", fontsize=9)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{r.setting}  ({r.kind}, {int(r.n_seeds)} seeds)"
                        for _, r in g.iterrows()], fontsize=10)
    ax.set_xlabel("change in top-1 accuracy versus the plain model (percentage points)")
    ax.set_title("Ablation: gain, 95% interval over seeds (thick), ±2 paired sd (thin)\n"
                 "solid bars clear twice their paired seed spread", fontsize=11.5)
    ax.grid(axis="x", alpha=0.45)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, f"figE_ablation{suffix}", f"work/ablation_summary{suffix}.csv")


# --------------------------------------------------------------------- F
def figure_f_size_curve():
    run_id = P.canonical_run_id()
    s = _read("size_curve_summary.csv", run_id)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for i, (name, g) in enumerate(s.groupby("setting")):
        g = g.sort_values("n_fit_samples")
        col = [DATA, GOOD, WARN, ALT, MUTED][i % 5]
        ax.errorbar(g.n_fit_samples, g.gain_pp,
                    yerr=[g.gain_pp - g.ci95_lo_pp, g.ci95_hi_pp - g.gain_pp],
                    fmt="-o", color=col, lw=2, ms=6, capsize=3, label=name)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xlabel("training passes (nested subsets of the fit matches)")
    ax.set_ylabel("gain over the plain model (percentage points)")
    ax.set_title("Does an intervention's effect settle as data grow?\n"
                 "four sizes on one dataset: a stability check, not a scaling law",
                 fontsize=11.5)
    ax.legend(fontsize=9.5)
    ax.grid(alpha=0.45)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "figF_size_curve", "work/size_curve_summary.csv")


FIGURES = {"A": figure_a_topk, "B": figure_b_cross_offset,
           "C": figure_c_ball_mask, "D": figure_d_leadsweep,
           "E": figure_e_ablation, "F": figure_f_size_curve}


if __name__ == "__main__":
    want = [a.upper() for a in sys.argv[1:] if a.upper() in FIGURES] or list(FIGURES)
    print(f"drawing figures {', '.join(want)}\n")
    for k in want:
        try:
            FIGURES[k]()
        except FileNotFoundError as exc:
            print(f"  fig{k}: skipped, {exc}")
    print("\nEvery figure above is drawn from a results file; none contains a "
          "hand-entered number.")
