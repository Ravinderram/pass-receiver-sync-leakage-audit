"""Step 18 - Ranking, calibration and per-position metrics.

This file no longer trains anything. Every metric is computed from the per-pass
predictions step23 exported, through metrics_lib.py, which is also what
verify/section8_results.py uses. That is the point: the figure and the table can
no longer come from different models.

Reported here
  ranking      top-1/2/3/5, MRR, mean rank of the true receiver
  calibration  Brier score, expected calibration error, reliability curve
  positions    precision, recall and F1 over RECEIVING POSITION CATEGORIES, with
               support. These evaluate whether the model finds, say, strikers;
               they are NOT ten-way teammate precision and recall, because a
               teammate slot index is not a category. Getting the wrong centre
               back counts as a correct "IV" here and as a miss in top-1.

Run:  python step18_metrics.py [--allow-stale]
Out:  work/metrics.csv, work/metrics_by_position.csv, work/reliability.csv,
      work/confusion_by_position.csv, figures/paper/fig15_metrics.png,
      figures/paper/fig17_confusion.png
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
import metrics_lib as M
import provenance as P
from common import banner

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify"))
import vutil  # noqa: E402

OUT = f"{C.FIGS}/paper"
K = C.N_TEAMMATES
INK, MUTED, DATA, WARN, GOOD = "#14171A", "#6E7378", "#2E6F9E", "#B3402F", "#1F7A4D"
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 300, "font.size": 11,
                     "axes.titleweight": "600"})

# kept for scripts that import them
ranking_metrics = M.ranking_metrics


def calibration(prob, y, bins=None):
    return {"ece": M.ece(prob, y, bins), "brier": M.brier(prob, y)}


def probs_of(pred, variant, match, seed=None):
    g = pred[(pred.model_variant == variant) & (pred.test_match == match)]
    if seed is not None:
        g = g[g.seed == seed]
    g = g.sort_values("sample_idx")
    return (g[[f"prob_{j}" for j in range(K)]].to_numpy(float),
            g["true_receiver"].to_numpy(int), g)


def figure(per_match, rank_all, prob_all, y_all):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ks = C.TOPK_REPORTED
    for i, (mid, m) in enumerate(per_match.items()):
        ax[0].plot(ks, [m[f"top{k}"] for k in ks], "-o", lw=2.2, ms=6,
                   color=DATA, alpha=1 - 0.4 * i, label=mid)
    ax[0].plot(ks, [k / K for k in ks], "--", color=WARN, lw=1.2, label="chance")
    ax[0].set_xlabel("k")
    ax[0].set_ylabel("accuracy within top k")
    ax[0].set_title("The answer is usually near the top")
    ax[0].set_xticks(ks)
    ax[0].legend(fontsize=10)

    ax[1].hist(rank_all, bins=np.arange(0.5, K + 1.5), color=DATA, alpha=0.85)
    ax[1].set_xlabel("rank of the true receiver")
    ax[1].set_ylabel("passes")
    ax[1].set_title(f"Mean rank {rank_all.mean():.2f} of {K}")
    ax[1].set_xticks(range(1, K + 1))

    rel = M.reliability_table(prob_all, y_all)
    keep = rel[rel.n >= 10]
    ax[2].plot([0, 1], [0, 1], "k--", lw=1)
    ax[2].plot(keep.mean_confidence, keep.accuracy, "-o", color=GOOD, lw=2.2, ms=6)
    ax[2].set_xlabel("predicted probability")
    ax[2].set_ylabel("actual hit rate")
    ax[2].set_title(f"Calibration (ECE {M.ece(prob_all, y_all):.3f})")
    for a in ax:
        a.grid(alpha=0.4)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Beyond top-1: ranking quality and calibration",
                 fontsize=13, weight="600", y=1.03)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig15_metrics.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {OUT}/fig15_metrics.png  +  .pdf")
    return rel


def confusion_figure(conf_tables, per_class_rows):
    n = len(conf_tables)
    fig, axes = plt.subplots(1, n + 1, figsize=(5.2 * (n + 1), 4.4))
    axes = np.atleast_1d(axes)
    for ax, (mid, tbl) in zip(axes, conf_tables.items()):
        norm = tbl.div(tbl.sum(axis=1).replace(0, 1), axis=0)
        ax.imshow(norm.values, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(norm.columns)))
        ax.set_xticklabels(norm.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(norm.index)))
        ax.set_yticklabels(norm.index, fontsize=9)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                v = norm.values[i, j]
                if v > 0.01:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                            color="white" if v > 0.55 else INK)
        ax.set_xlabel("predicted position")
        ax.set_ylabel("actual position")
        ax.set_title(f"{mid} - row-normalised")

    ax = axes[-1]
    pc = pd.concat(per_class_rows)
    pc = pc[~pc.position.str.startswith("_")]
    pc = pc.groupby("position")[["precision", "recall", "f1"]].mean().sort_values("f1")
    y = np.arange(len(pc))
    ax.barh(y - 0.25, pc["precision"], height=0.25, color=DATA, label="precision")
    ax.barh(y, pc["recall"], height=0.25, color=GOOD, label="recall")
    ax.barh(y + 0.25, pc["f1"], height=0.25, color=WARN, label="F1")
    ax.set_yticks(y)
    ax.set_yticklabels(pc.index, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("score")
    ax.set_title("Per receiving position")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Confusion and per-class metrics over RECEIVING POSITION "
                 "CATEGORIES, not teammate slots", fontsize=12.5, weight="600", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig17_confusion.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {OUT}/fig17_confusion.png  +  .pdf")


if __name__ == "__main__":
    banner("Step 18 - metrics from the canonical predictions (no training here)")
    os.makedirs(OUT, exist_ok=True)
    run_id, pred = vutil.canonical_predictions()
    print(f"run id {run_id}\n")

    rows, per_match, per_class_rows, conf_tables = [], {}, [], {}
    all_prob, all_y, all_rank = [], [], []

    for mid in C.TEST_MATCHES:
        seeds = sorted(pred[(pred.model_variant == "canonical")
                            & (pred.test_match == mid)].seed.unique())
        per_seed = []
        for seed in seeds:
            prob, y, _ = probs_of(pred, "canonical", mid, seed)
            per_seed.append({**M.ranking_metrics(prob, y), **calibration(prob, y)})
        agg = {k: float(np.mean([r[k] for r in per_seed])) for k in per_seed[0]}
        sd = {f"{k}_sd": float(np.std([r[k] for r in per_seed], ddof=1))
              for k in per_seed[0]} if len(per_seed) > 1 else {}
        prob_e, y_e, _ = probs_of(pred, "canonical_seed_ensemble", mid)
        rows.append({"test_match": mid, "n": len(y_e), "n_seeds": len(seeds),
                     **agg, **sd,
                     **{f"ensemble_{k}": v for k, v in
                        {**M.ranking_metrics(prob_e, y_e),
                         **calibration(prob_e, y_e)}.items()}})
        per_match[mid] = agg
        all_prob.append(prob_e)
        all_y.append(y_e)
        all_rank.append(M.ranks_from_probs(prob_e, y_e))

        print(f"--- {mid}  (n={len(y_e)}, {len(seeds)} seeds) ---")
        print("  " + "   ".join(f"top-{k} {agg[f'top{k}']:.3f}" for k in C.TOPK_REPORTED))
        print(f"  MRR {agg['mrr']:.3f}   mean rank {agg['mean_rank']:.2f} of {K}")
        print(f"  Brier {agg['brier']:.3f}   ECE {agg['ece']:.3f}   "
              "(mean over seeds; the ensemble is reported separately)")

        g = pred[(pred.model_variant == "canonical_seed_ensemble")
                 & (pred.test_match == mid)]
        tab, dropped = M.per_position(g.true_position.astype(str),
                                      g.pred_position.astype(str))
        if len(tab):
            tab.insert(0, "test_match", mid)
            tab.insert(1, "dropped_unknown_position", dropped)
            per_class_rows.append(tab)
            conf_tables[mid] = pd.crosstab(
                pd.Series(g.true_position.astype(str).to_numpy(), name="actual"),
                pd.Series(g.pred_position.astype(str).to_numpy(), name="predicted"))
            print("\n  per receiving POSITION CATEGORY (not teammate identity):")
            print("   " + tab.head(8).round(3).to_string(index=False)
                  .replace("\n", "\n   "))
            if dropped:
                print(f"   ({dropped} passes dropped: position code unknown)")
        print()

    df = P.stamp(pd.DataFrame(rows), run_id, lead_s=C.PREDICT_LEAD_S,
                 model_variant="canonical")
    df.to_csv(f"{C.WORK}/metrics.csv", index=False)

    print("=" * 62)
    print("HEADLINE (mean over seeds)")
    print("=" * 62)
    for _, r in df.iterrows():
        print(f"  {r['test_match']}:  top-1 {r['top1']:.3f}   "
              f"(n={int(r['n'])}, chance {1 / K:.2f})")
    print(f"\n  mean across test matches: {df['top1'].mean():.3f}")

    prob_all = np.concatenate(all_prob)
    y_all = np.concatenate(all_y)
    rel = figure(per_match, np.concatenate(all_rank), prob_all, y_all)
    P.stamp(rel, run_id, lead_s=C.PREDICT_LEAD_S,
            model_variant="canonical_seed_ensemble").to_csv(
        f"{C.WORK}/reliability.csv", index=False)
    if per_class_rows:
        P.stamp(pd.concat(per_class_rows), run_id, lead_s=C.PREDICT_LEAD_S,
                model_variant="canonical_seed_ensemble").to_csv(
            f"{C.WORK}/metrics_by_position.csv", index=False)
        pd.concat(conf_tables, names=["match"]).to_csv(
            f"{C.WORK}/confusion_by_position.csv")
        confusion_figure(conf_tables, per_class_rows)
    print(f"\nSaved -> {C.WORK}/metrics.csv, metrics_by_position.csv, reliability.csv")
    print("The same predictions feed verify/section8_results.py, so these numbers")
    print("and the paper's tables cannot disagree.")
