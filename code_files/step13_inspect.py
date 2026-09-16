"""Step 13 - Look at individual passes.

The dataset lives in npz arrays of shape (n, 13, 23, 11). Nobody can check an
array. This renders one pass as a strip of pitch frames across the 1.5 s window,
with the model's probability written next to each teammate on the final frame.

Two things this is for:

  Checking the data. Are the players where they should be? Does the passing team
  attack right? Is the ball still at the passer's foot at the end of the window,
  or has it already gone - the leakage we spent a whole round chasing?

  Checking the model. When it is confident and wrong, was the situation genuinely
  ambiguous, or is the model missing something a coach would see instantly?

Run:  python step13_inspect.py [MATCH_ID] [N]
Out:  figures/inspect_<match>_<i>.png
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C
from common import banner, load_npz
from device import banner as device_banner

os.makedirs(C.FIGS, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

MATE, OPP, PASSER, BALL = "#2E6F9E", "#C0504D", "#E8A33D", "#222222"
PITCH = "#F4F6F4"


def draw_pitch(ax):
    """A quiet pitch. The data is the subject, not the grass."""
    ax.add_patch(plt.Rectangle((-52.5, -34), 105, 68, fc=PITCH, ec="#C8CFC8", lw=1))
    ax.plot([0, 0], [-34, 34], color="#C8CFC8", lw=1)
    ax.add_patch(plt.Circle((0, 0), 9.15, fc="none", ec="#C8CFC8", lw=1))
    for side in (-1, 1):
        ax.add_patch(plt.Rectangle((side * 52.5 - side * 16.5, -20.16), side * 16.5,
                                   40.32, fc="none", ec="#C8CFC8", lw=1))
    ax.set_xlim(-56, 56)
    ax.set_ylim(-37, 37)
    ax.set_aspect("equal")
    ax.axis("off")


def frame_panel(ax, fr, title, probs=None, truth=None):
    draw_pitch(ax)
    present = fr[:, C.IDX_PRESENT] > 0.5
    mates = np.where((fr[:, 4] == 1) & present)[0]
    opps = np.where((fr[:, 5] == 1) & present)[0]

    ax.scatter(fr[opps, 0], fr[opps, 1], c=OPP, s=34, zorder=3)
    ax.scatter(fr[mates, 0], fr[mates, 1], c=MATE, s=34, zorder=3)
    ax.scatter(fr[0, 0], fr[0, 1], c=PASSER, s=90, ec="k", lw=0.8, zorder=5)
    ax.scatter(fr[-1, 0], fr[-1, 1], c=BALL, s=26, zorder=6)

    # velocity arrows say more about a football situation than dots do
    for grp, col in ((mates, MATE), (opps, OPP), ([0], PASSER)):
        for j in grp:
            ax.arrow(fr[j, 0], fr[j, 1], fr[j, 2] * 0.6, fr[j, 3] * 0.6,
                     color=col, width=0.25, head_width=1.1, alpha=0.55, zorder=2)

    if probs is not None:
        for k, j in enumerate(mates):
            if k < len(probs):
                ax.annotate(f"{probs[k]:.0%}", (fr[j, 0], fr[j, 1]),
                            xytext=(0, 7), textcoords="offset points",
                            ha="center", fontsize=7.5, weight="bold",
                            color="#14304A", zorder=7)
    if truth is not None and truth < len(mates):
        ax.scatter(fr[mates[truth], 0], fr[mates[truth], 1], fc="none",
                   ec="#1F9D55", s=200, lw=2.2, zorder=8)

    ax.set_title(title, fontsize=8.5, color="#444")


def inspect(X, y, i, probs=None, tag=""):
    s = X[i]
    steps = [0, len(s) // 3, 2 * len(s) // 3, len(s) - 1]

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    for ax, t in zip(axes, steps):
        lead = (len(s) - 1 - t) * C.STRIDE / C.FRAMERATE + C.PREDICT_LEAD_S
        last = t == len(s) - 1
        frame_panel(ax, s[t], f"{lead:.2f} s before the ball is struck",
                    probs=probs[i] if (probs is not None and last) else None,
                    truth=y[i] if last else None)

    ball_speed = np.linalg.norm(s[-1, -1, 2:4])
    ball_gap = np.linalg.norm(s[-1, -1, :2] - s[-1, 0, :2])
    verdict = ("ball still with the passer - good"
               if ball_gap < 3 and ball_speed < 6 else
               "ball already moving - check PREDICT_LEAD_S")

    sub = f"ball {ball_speed:.1f} m/s, {ball_gap:.1f} m from the passer  |  {verdict}"
    if probs is not None:
        p = probs[i]
        sub += f"  |  model says {p.argmax()} at {p.max():.0%}, actual {y[i]}"
    fig.suptitle(f"{tag}pass #{i}   —   passing team attacks right\n{sub}",
                 fontsize=9.5, y=1.06)
    fig.tight_layout()
    out = f"{C.FIGS}/inspect_{tag.strip().replace(' ', '')}{i}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    banner("Step 13 - inspecting individual passes")
    device_banner()
    print()
    match_id = sys.argv[1] if len(sys.argv) > 1 else C.TEST_MATCHES[0]
    n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    d = load_npz(f"dataset_{match_id}")
    X_raw, y = d["X"], d["y"]
    if len(X_raw) == 0:
        sys.exit(f"no samples for {match_id}")

    import experiment as E
    print("loading the canonical model so the panels show the reported model's")
    print("probabilities, not a separately trained one...")
    model, stats, _ = E.get_or_train(C.PREDICT_LEAD_S, C.SEEDS[0])
    te = E.load_split([match_id])
    X_raw, y = te["X"], te["y"]
    probs = E.softmax(E.evaluate(model, stats, te))

    acc = (probs.argmax(1) == y).mean()
    print(f"top-1 on {match_id}: {acc:.3f}\n")

    conf = probs.max(1)
    hit = probs.argmax(1) == y
    picks = {
        "right ": np.argsort(-np.where(hit, conf, -1))[:n_show // 2 or 1],
        "wrong ": np.argsort(-np.where(~hit, conf, -1))[:n_show // 2 or 1],
    }
    for tag, idxs in picks.items():
        for i in idxs:
            print(" ", inspect(X_raw, y, int(i), probs, tag))

    print("\nGreen ring = who actually received. Percentages = model belief.")
    print("Arrows are velocity. Confident-and-wrong panels are the useful ones.")
