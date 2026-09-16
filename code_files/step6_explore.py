"""Step 6 - Explore and visualise.

Six figures. Two of them are checks, four are for the presentation.

Note: pass distance and angle are plotted here only to understand the data.
They are outputs of the pass, so they never enter the model as features.

Run:  python step6_explore.py
Out:  figures/*.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from common import banner, load_match, load_npz

os.makedirs(C.FIGS, exist_ok=True)
plt.rcParams.update({"figure.dpi": 140, "font.size": 10, "axes.titleweight": "bold"})
BLUE, RED, GREY = "#4C72B0", "#C44E52", "#808080"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{C.FIGS}/{name}.png")
    plt.close(fig)
    print(f"  {C.FIGS}/{name}.png")


def fig_sync(p):
    """CHECK: did synchronisation actually help?"""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(p["dist_before"].dropna(), bins=60, color=GREY, label="before")
    ax[0].hist(p["dist_after"].dropna(), bins=60, color=BLUE, alpha=0.8, label="after")
    ax[0].set_xlabel("event location to ball position (m)")
    ax[0].set_ylabel("passes")
    ax[0].set_title("Synchronisation effect")
    ax[0].legend()

    shift = (p["sync_frame"] - p["naive_frame"]) / C.FRAMERATE
    ax[1].hist(shift, bins=60, color=BLUE)
    ax[1].set_xlabel("time correction applied (s)")
    ax[1].set_title("How far each event moved")
    save(fig, "06_sync")


def fig_counts(p):
    fig, ax = plt.subplots(figsize=(7, 4))
    n = p.groupby("match_id").size().reindex(C.MATCHES.keys())
    colors = [RED if C.MATCHES[m][2] == 1 else BLUE for m in n.index]
    ax.bar(n.index, n.values, color=colors)
    ax.set_ylabel("usable passes")
    ax.set_title("Passes per match (red = 1. Bundesliga, held out for testing)")
    ax.tick_params(axis="x", rotation=30)
    save(fig, "01_pass_counts")


def fig_pass_geometry(p):
    """Understanding only. These are outputs of the pass, never model inputs."""
    # ev_x / ev_y come from step 3 and are already in TRACKING coordinates.
    # at_x / at_y are the raw corner-based event coordinates - do not plot those
    # against anything from the position data.
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(p["ev_x"].dropna(), bins=50, color=BLUE)
    ax[0].set_xlabel("pass origin x (m, centre-based)")
    ax[0].set_title("Where passes start (length of pitch)")
    ax[1].hist(p["ev_y"].dropna(), bins=50, color=BLUE)
    ax[1].set_xlabel("pass origin y (m, centre-based)")
    ax[1].set_title("Where passes start (width of pitch)")
    save(fig, "02_pass_origins")


def fig_label_balance():
    """How often is each teammate index the receiver? Should be roughly flat."""
    ys = []
    for mid in C.MATCHES:
        ys.append(load_npz(f"dataset_{mid}")["y"])
    y = np.concatenate(ys)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(C.N_TEAMMATES), np.bincount(y, minlength=C.N_TEAMMATES), color=BLUE)
    ax.axhline(len(y) / C.N_TEAMMATES, color=RED, ls="--", label="uniform")
    ax.set_xlabel("teammate index")
    ax.set_ylabel("times the receiver")
    ax.set_title("Class balance (index order is arbitrary, so this should be flat)")
    ax.legend()
    save(fig, "03_label_balance")


def fig_situation(match_id="J03WMX"):
    """One pass, drawn on a real pitch. The figure that explains the project."""
    d = load_npz(f"dataset_{match_id}")
    X, y = d["X"], d["y"]
    if len(X) == 0:
        print("  (no samples, skipping situation plot)")
        return

    i = 0
    s = X[i]                       # (T, 23, 8)
    now = s[-1]                    # the moment of the pass
    m = load_match(match_id)

    fig, ax = plt.subplots(figsize=(9, 6))
    m["pitch"].plot(ax=ax)

    # trails
    for j in range(C.N_OBJECTS - 1):
        col = BLUE if now[j, 4] or now[j, 7] else RED
        ax.plot(s[:, j, 0], s[:, j, 1], color=col, alpha=0.35, lw=1)
    ax.plot(s[:, -1, 0], s[:, -1, 1], color="k", alpha=0.5, lw=1)

    mates = np.where(now[:, 4] == 1)[0]
    opps = np.where(now[:, 5] == 1)[0]
    ax.scatter(now[mates, 0], now[mates, 1], c=BLUE, s=70, label="teammates", zorder=3)
    ax.scatter(now[opps, 0], now[opps, 1], c=RED, s=70, label="opponents", zorder=3)
    ax.scatter(now[0, 0], now[0, 1], c="gold", s=140, edgecolor="k",
               label="passer", zorder=4)
    ax.scatter(now[-1, 0], now[-1, 1], c="k", s=45, label="ball", zorder=5)

    r = mates[min(y[i], len(mates) - 1)]
    ax.scatter(now[r, 0], now[r, 1], facecolors="none", edgecolors="lime",
               s=260, lw=2.5, label="actual receiver", zorder=6)

    ax.set_title(f"{match_id}: one pass, {C.WINDOW_SECONDS}s of history "
                 f"(passing team attacks to the right)")
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "04_situation")


def fig_speed_check(match_id="J03WMX"):
    """CHECK: are the velocities we derived physically sensible?"""
    d = load_npz(f"clean_{match_id}")
    v = d["firstHalf_Home_vel"]
    speed = np.linalg.norm(v, axis=2)
    speed = speed[~np.isnan(speed)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(speed, bins=80, color=BLUE)
    ax.axvline(10, color=RED, ls="--", label="10 m/s = 36 km/h (elite sprint)")
    ax.set_xlabel("player speed (m/s)")
    ax.set_yscale("log")
    ax.set_title("Speed distribution after filtering")
    ax.legend()
    save(fig, "05_speed_check")
    over = (speed > 12).mean() * 100
    print(f"  speeds above 12 m/s: {over:.3f} %  (should be near zero)")


if __name__ == "__main__":
    banner("Step 6 - exploration figures")
    p = pd.read_csv(f"{C.WORK}/passes_synced.csv")
    fig_counts(p)
    fig_pass_geometry(p)
    fig_label_balance()
    fig_situation()
    fig_speed_check()
    fig_sync(p)
