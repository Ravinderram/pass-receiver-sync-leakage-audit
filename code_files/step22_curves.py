"""Step 22 - Learning curves, ROC and precision-recall.

Nothing is trained here. Both inputs come from the canonical run:

  learning curves   work/canonical_history.csv, written by step23. Train,
                    validation and monitored-test accuracy per epoch, with the
                    SELECTED epoch marked. The test curve is monitoring only: no
                    selection decision ever used it.
  ROC and PR        work/metrics_predictions_<run_id>.csv. Every (pass, candidate)
                    pair is one binary decision, "is this player the receiver".
                    DESCRIPTIVE ONLY: one positive per pass and candidates within
                    a pass are not independent, so no test is run on these.
                    Precision-recall is the more informative of the two at a
                    1-in-10 positive rate; ROC's false-positive rate is diluted
                    by a large true-negative pool.

Run:  python step22_curves.py [--allow-stale]
Out:  figures/paper/fig18_learning_curves.png, fig19_roc_pr.png,
      work/curves_summary.csv
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

import config as C
import metrics_lib as M
import provenance as P
from common import banner

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify"))
import vutil  # noqa: E402

OUT = f"{C.FIGS}/paper"
K = C.N_TEAMMATES
INK, MUTED = "#14171A", "#6E7378"
DATA, WARN, GOOD, ALT = "#2E6F9E", "#B3402F", "#1F7A4D", "#C98A2B"
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 300, "font.size": 11,
                     "axes.titleweight": "600"})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {OUT}/{name}.png  +  .pdf")


def learning_curves(hist, meta):
    seeds = sorted(hist.seed.unique())
    sel = {s["seed"]: s["selected_epoch"] for s in (meta or {}).get("per_seed", [])}
    test_cols = [c for c in hist.columns if c.startswith("test_") and c.endswith("_top1")]
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.5))
    for i, seed in enumerate(seeds):
        h = hist[hist.seed == seed].sort_values("epoch")
        a = 1 - 0.25 * i
        ls = ["-", "--", ":", "-."][i % 4]
        ax[0].plot(h.epoch, h.train_top1, ls, color=WARN, lw=1.8, alpha=a,
                   label="train" if i == 0 else None)
        if "val_top1" in h:
            ax[0].plot(h.epoch, h.val_top1, ls, color=GOOD, lw=2.0, alpha=a,
                       label=f"validation ({C.VAL_MATCH})" if i == 0 else None)
        for j, c in enumerate(test_cols):
            ax[0].plot(h.epoch, h[c], ls, color=[DATA, ALT][j % 2], lw=2.0, alpha=a,
                       label=c.replace("test_", "test ").replace("_top1", "")
                       + " (monitored only)" if i == 0 else None)
        if seed in sel:
            ax[0].axvline(sel[seed], color=GOOD, ls=(0, (2, 3)), lw=1.1, alpha=a)
        ax[1].plot(h.epoch, h.train_loss, ls, color=WARN, lw=1.8, alpha=a,
                   label="train loss" if i == 0 else None)
        if "val_loss" in h:
            ax[1].plot(h.epoch, h.val_loss, ls, color=GOOD, lw=1.8, alpha=a,
                       label="validation loss" if i == 0 else None)
    if sel:
        ax[0].annotate(f"selected epoch(s): {sorted(set(sel.values()))}",
                       (0.02, 0.02), xycoords="axes fraction", color=GOOD, fontsize=9.5)
    ax[0].axhline(1 / K, color=MUTED, ls=(0, (4, 3)), lw=1)
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("top-1 accuracy")
    ax[0].set_title("(a) Accuracy per epoch, all seeds")
    ax[0].legend(fontsize=8.5, loc="lower right")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("cross-entropy loss")
    ax[1].set_title("(b) Loss per epoch")
    ax[1].legend(fontsize=9)
    for a_ in ax:
        a_.grid(alpha=0.4)
        a_.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Training and validation per epoch; the test curve is monitored, "
                 "never selected on", fontsize=12, weight="600", y=1.03)
    fig.tight_layout()
    save(fig, "fig18_learning_curves")


def roc_pr(pred):
    fig, ax = plt.subplots(1, 2, figsize=(11.8, 4.6))
    rows = []
    for i, mid in enumerate(sorted(pred.test_match.unique())):
        g = pred[(pred.model_variant == "canonical_seed_ensemble")
                 & (pred.test_match == mid)].sort_values("sample_idx")
        prob = g[[f"prob_{j}" for j in range(K)]].to_numpy(float)
        y = g.true_receiver.to_numpy(int)
        present = g[[f"present_{j}" for j in range(K)]].to_numpy(int).astype(bool)
        s, lab = M.pair_arrays(prob, y, present)
        auc, apc = M.pair_auc_ap(prob, y, present)
        a = 1 - 0.4 * i
        fpr, tpr, _ = roc_curve(lab, s)
        ax[0].plot(fpr, tpr, color=DATA, lw=2.2, alpha=a, label=f"{mid}   AUC {auc:.3f}")
        pr, rc, _ = precision_recall_curve(lab, s)
        ax[1].plot(rc, pr, color=GOOD, lw=2.2, alpha=a, label=f"{mid}   AP {apc:.3f}")
        rows.append({"test_match": mid, "n": len(y), "n_pairs": int(len(lab)),
                     "positive_rate": float(lab.mean()), "roc_auc_pairs": auc,
                     "ap_pairs": apc})
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="random  AUC 0.500")
    ax[0].set_xlabel("false positive rate")
    ax[0].set_ylabel("true positive rate")
    ax[0].set_title("(a) ROC over all (pass, candidate) pairs")
    ax[1].axhline(float(np.mean([r["positive_rate"] for r in rows])), color=MUTED,
                  ls="--", lw=1.2, label="random (positive rate)")
    ax[1].set_xlabel("recall")
    ax[1].set_ylabel("precision")
    ax[1].set_ylim(0, 1.02)
    ax[1].set_title("(b) Precision-recall over all pairs")
    for a_ in ax:
        a_.grid(alpha=0.4)
        a_.legend(fontsize=9.5)
        a_.spines[["top", "right"]].set_visible(False)
    fig.suptitle("One binary decision per candidate: descriptive only, "
                 "candidates within a pass are not independent",
                 fontsize=12, weight="600", y=1.03)
    fig.tight_layout()
    save(fig, "fig19_roc_pr")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    banner("Step 22 - curves from the canonical run (no training here)")
    run_id, pred = vutil.canonical_predictions()
    meta = P.read_json(f"{C.WORK}/run_metadata_{run_id}.json")
    hist_path = f"{C.WORK}/canonical_history.csv"
    if not os.path.isfile(hist_path):
        sys.exit(f"{hist_path} missing: run step23_canonical.py")
    hist = pd.read_csv(hist_path)
    P.check_run_id(hist, run_id, "canonical_history.csv", vutil.allow_stale())
    print(f"run id {run_id}   {hist.seed.nunique()} seeds, "
          f"{int(hist.groupby('seed').size().max())} epochs at most\n")

    learning_curves(hist, meta)
    summ = roc_pr(pred)
    gen = (hist.sort_values("epoch").groupby("seed").last()
           [["epoch", "train_top1"] + [c for c in hist.columns
                                       if c.startswith("test_") and c.endswith("_top1")]])
    print("\nlast-epoch training accuracy vs monitored test accuracy per seed:")
    print(gen.round(3).to_string())
    print("\npair-level, seed ensemble:")
    print(summ.round(3).to_string(index=False))
    P.stamp(summ, run_id, lead_s=C.PREDICT_LEAD_S,
            model_variant="canonical_seed_ensemble").to_csv(
        f"{C.WORK}/curves_summary.csv", index=False)
    print(f"\nSaved -> {C.WORK}/curves_summary.csv")
