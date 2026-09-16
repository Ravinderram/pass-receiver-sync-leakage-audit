"""Step 10 - Error analysis.

Four things:
  1. accuracy broken down by pass distance, pitch zone and crowding
  2. calibration - when the model says 70%, is it right 70% of the time?
  3. the confident mistakes, which are the informative ones
  4. one situation with the predicted probability next to each teammate

Run:  python step10_analysis.py
Out:  figures/07..10_*.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config as C
from common import banner, load_match
from step7_baselines import candidate_features
from device import DEVICE, banner as device_banner, to_numpy

os.makedirs(C.FIGS, exist_ok=True)
plt.rcParams.update({"figure.dpi": 140, "font.size": 10, "axes.titleweight": "bold"})
BLUE, RED = "#4C72B0", "#C44E52"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{C.FIGS}/{name}.png")
    plt.close(fig)
    print(f"  {C.FIGS}/{name}.png")


def train_final(Xtr, ytr, seed=0, Xval=None, yval=None):
    """Train through the shared trainer so this file cannot drift from step23.

    Prefer analysing the canonical checkpoints: __main__ below loads them with
    experiment.get_or_train, and this function is only the fallback for callers
    that hand in their own arrays.
    """
    import experiment as E
    model, _ = E.fit_model(Xtr, ytr, Xval, yval, seed)
    return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        p = to_numpy(torch.softmax(model(torch.tensor(X).to(DEVICE)), dim=1))
    return p


def fig_breakdown(prob, y, Xte_raw, test_id):
    pred = prob.argmax(1)
    hit = pred == y
    feats = candidate_features(Xte_raw)
    dist = feats[np.arange(len(y)), y, 0]     # distance to the true receiver
    lane = feats[np.arange(len(y)), y, 3]     # opponents in the lane

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))

    bins = [0, 10, 20, 30, 100]
    labels = ["<10 m", "10-20", "20-30", ">30"]
    idx = np.digitize(dist, bins) - 1
    ax[0].bar(labels, [hit[idx == i].mean() if (idx == i).any() else 0
                       for i in range(4)], color=BLUE)
    ax[0].set_title("Accuracy by pass length")
    ax[0].set_ylabel("top-1 accuracy")

    zx = Xte_raw[:, -1, 0, 0]                 # passer x, attacking +x
    zbins = [-60, -20, 20, 60]
    zlab = ["own third", "middle", "final third"]
    zidx = np.digitize(zx, zbins) - 1
    ax[1].bar(zlab, [hit[zidx == i].mean() if (zidx == i).any() else 0
                     for i in range(3)], color=BLUE)
    ax[1].set_title("Accuracy by pitch zone")

    lbins = [0, 1, 2, 20]
    llab = ["0 in lane", "1", "2+"]
    lidx = np.digitize(lane, lbins) - 1
    ax[2].bar(llab, [hit[lidx == i].mean() if (lidx == i).any() else 0
                     for i in range(3)], color=BLUE)
    ax[2].set_title("Accuracy by opponents in the passing lane")

    for a in ax:
        a.set_ylim(0, 1)
        a.axhline(0.1, color=RED, ls="--", lw=1)
    save(fig, f"07_breakdown_{test_id}")


def fig_calibration(prob, y, test_id):
    conf = prob.max(1)
    correct = prob.argmax(1) == y
    edges = np.linspace(0, 1, 11)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() >= 10:
            xs.append(conf[m].mean())
            ys.append(correct[m].mean())
            ns.append(int(m.sum()))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(xs, ys, "o-", color=BLUE, label="model")
    for x, yv, n in zip(xs, ys, ns):
        ax.annotate(str(n), (x, yv), fontsize=7, xytext=(3, -9),
                    textcoords="offset points")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("actual hit rate")
    ax.set_title(f"Calibration - {test_id}")
    ax.legend()
    save(fig, f"08_calibration_{test_id}")


def fig_confident_mistakes(prob, y, X, test_id, k=3):
    """Confident and wrong is where the model's assumptions show."""
    wrong = prob.argmax(1) != y
    conf = prob.max(1)
    order = np.argsort(-np.where(wrong, conf, -1))[:k]

    fig, axes = plt.subplots(1, k, figsize=(5 * k, 4.5))
    for ax, i in zip(np.atleast_1d(axes), order):
        now = X[i, -1]
        mates = np.where(now[:, 4] == 1)[0]
        opps = np.where(now[:, 5] == 1)[0]
        ax.scatter(now[opps, 0], now[opps, 1], c=RED, s=45)
        ax.scatter(now[mates, 0], now[mates, 1], c=BLUE, s=45)
        ax.scatter(now[0, 0], now[0, 1], c="gold", s=110, edgecolor="k")
        true_j = mates[min(y[i], len(mates) - 1)]
        pred_j = mates[min(int(prob[i, :len(mates)].argmax()), len(mates) - 1)]
        ax.scatter(*now[true_j, :2], facecolors="none", edgecolors="lime",
                   s=220, lw=2)
        ax.scatter(*now[pred_j, :2], facecolors="none",
                   edgecolors="magenta", s=300, lw=2)
        ax.set_title(f"model {conf[i]:.0%} sure, wrong", fontsize=9)
        ax.set_xlim(-55, 55)
        ax.set_ylim(-36, 36)
        ax.set_aspect("equal")
    fig.suptitle("Confident mistakes  (green = actual, magenta = predicted)")
    save(fig, f"09_confident_mistakes_{test_id}")


def fig_explain(prob, y, X, test_id, match_id):
    """The one figure that explains the project in ten seconds."""
    i = int(np.argmax(prob.max(1)))
    now = X[i, -1]
    m = load_match(match_id)

    fig, ax = plt.subplots(figsize=(9, 6))
    m["pitch"].plot(ax=ax)
    mates = np.where(now[:, 4] == 1)[0]
    opps = np.where(now[:, 5] == 1)[0]

    ax.scatter(now[opps, 0], now[opps, 1], c=RED, s=70, label="opponents")
    # prob always has N_TEAMMATES entries; after a red card fewer of those slots
    # hold a real player. Objects 1..len(mates) are the present ones in order,
    # so the probabilities line up once truncated.
    sc = ax.scatter(now[mates, 0], now[mates, 1], c=prob[i, :len(mates)],
                    cmap="Blues", s=220, vmin=0, vmax=1, edgecolor="k", zorder=3,
                    label="teammates (shaded by probability)")
    for k, s in enumerate(mates):
        ax.annotate(f"{prob[i, k]:.0%}", (now[s, 0], now[s, 1]),
                    fontsize=8, ha="center", va="center", zorder=4)
    ax.scatter(now[0, 0], now[0, 1], c="gold", s=160, edgecolor="k",
               label="passer", zorder=5)
    ax.scatter(*now[mates[y[i]], :2], facecolors="none", edgecolors="lime",
               s=300, lw=2.5, label="actual receiver", zorder=6)

    fig.colorbar(sc, ax=ax, label="predicted probability")
    ax.set_title(f"{test_id}: who gets the ball? (team attacks right)")
    ax.legend(loc="upper right", fontsize=8)
    save(fig, f"10_explained_{test_id}")


if __name__ == "__main__":
    banner("Step 10 - error analysis")
    device_banner()
    print()
    import experiment as E
    print(f"using the canonical checkpoints (seed {C.SEEDS[0]}), so this analysis")
    print("describes the same model the results tables report\n")
    model, stats, info = E.get_or_train(C.PREDICT_LEAD_S, C.SEEDS[0])

    for test_id in C.TEST_MATCHES:
        te = E.load_split([test_id])
        Xte_raw, yte = te["X"], te["y"]
        prob = E.softmax(E.evaluate(model, stats, te))

        print(f"--- {test_id} ---")
        print(f"  top-1 accuracy: {(prob.argmax(1) == yte).mean():.3f}"
              f"   (selected epoch {info['selected_epoch']})")

        fig_breakdown(prob, yte, Xte_raw, test_id)
        fig_calibration(prob, yte, test_id)
        fig_confident_mistakes(prob, yte, Xte_raw, test_id, k=3)
        fig_explain(prob, yte, Xte_raw, test_id, test_id)
        print()
