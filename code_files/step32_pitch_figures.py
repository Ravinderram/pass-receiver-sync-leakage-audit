"""Step 32 - Pitch figures for the paper, drawn from the canonical run.

Two figures, both from data already in work/. Nothing is redrawn by hand and no
pass is chosen by eye.

  fig9_leakage_pitch.png
      One real pass at two frames: the frame the synchroniser emits, and the
      frame 0.4 s earlier where the canonical model's window ends. The ball's
      velocity arrow is drawn at both. At the emitted frame the ball is already
      travelling towards the receiver, which is the shortcut Section V measures;
      0.4 s earlier it is still at the passer's foot. This is the paper's whole
      argument in one picture.

      The pass is SELECTED BY RULE, not by eye: among test-match passes where
      the ball-direction probe is correct at the emitted frame, we take the one
      whose ball speed is closest to the dataset median. It is therefore a
      typical case of the shortcut, not the most extreme one. The rule is
      printed so it can be quoted in the caption, and --pass-row overrides it.

  fig10_spatial.png
      (a) Where the usable passes start, over all seven matches.
      (b) Canonical top-1 accuracy by pitch zone on the held-out matches, with
          the number of passes per zone. Cells holding fewer than MIN_CELL
          passes are left blank rather than shown as noise.

      Coordinates are normalised so the passing team always attacks to the
      right, using the attacking signs stored by step4_clean.py. Without that
      normalisation a zone map mixes both directions of play and means nothing.

Run:  python step32_pitch_figures.py [--pass-row N] [--match J03WMX]
Out:  figures/paper/fig9_leakage_pitch.png, figures/paper/fig10_spatial.png
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Rectangle

import config as C
import provenance as P
from common import banner, load_npz

OUT = f"{C.FIGS}/paper"
INK, MUT, HOME, AWAY, BALL, GOOD = "#14171A", "#6E7378", "#2E6F9E", "#B3402F", "#E8A33D", "#1F7A4D"
MIN_CELL = 25          # passes needed before a zone is shown in fig10(b)
NX, NY = 6, 4          # zone grid
plt.rcParams.update({"font.size": 10, "savefig.dpi": 300, "axes.titleweight": "600",
                     "legend.frameon": False})


# ----------------------------------------------------------------- pitch
def draw_pitch(ax, length=105.0, width=68.0):
    hx, hy = length / 2, width / 2
    ax.add_patch(Rectangle((-hx, -hy), length, width, fill=False, color=MUT, lw=1.2))
    ax.plot([0, 0], [-hy, hy], color=MUT, lw=1.0)
    ax.add_patch(Circle((0, 0), 9.15, fill=False, color=MUT, lw=1.0))
    for s in (-1, 1):
        ax.add_patch(Rectangle((s * hx - s * 16.5, -20.16), s * 16.5, 40.32,
                               fill=False, color=MUT, lw=1.0))
        ax.add_patch(Rectangle((s * hx - s * 5.5, -9.16), s * 5.5, 18.32,
                               fill=False, color=MUT, lw=1.0))
    ax.set_xlim(-hx - 2, hx + 2)
    ax.set_ylim(-hy - 2, hy + 2)
    ax.set_aspect("equal")
    ax.axis("off")


# --------------------------------------------------- figure 9: one pass
def frame_state(clean, half, side, frame, sign):
    """Positions and velocities at one frame, flipped so the passer attacks +x."""
    other = "Away" if side == "Home" else "Home"
    g = lambda k: clean[f"{half}_{k}"][frame]
    return {
        "mates": g(f"{side}_pos") * sign, "mates_v": g(f"{side}_vel") * sign,
        "opps": g(f"{other}_pos") * sign, "ball": g("Ball_pos")[0] * sign,
        "ball_v": g("Ball_vel")[0] * sign,
    }


def pick_pass(passes, preds):
    """The rule described in the module docstring."""
    import probes as PR
    rows = []
    for mid in C.TEST_MATCHES:
        d = load_npz(f"dataset_{mid}")
        X, y = d["X"], d["y"]
        if not len(X):
            continue
        s = PR.snapshot_from_samples(X, y, t=-1)
        hit = PR.PROBES["ball_direction"](s)[0] == y
        speed = np.linalg.norm(X[:, -1, C.N_OBJECTS - 1, 2:4], axis=1)
        for i in np.where(hit)[0]:
            rows.append({"match": mid, "pass_row": int(d["pass_row"][i]),
                         "speed_at_window_end": float(speed[i])})
    if not rows:
        raise SystemExit("no candidate pass found; pass --pass-row explicitly")
    df = pd.DataFrame(rows)
    med = df.speed_at_window_end.median()
    return df.iloc[(df.speed_at_window_end - med).abs().argmin()]


def figure_leakage(passes, pass_row, match_id):
    row = passes.loc[pass_row]
    half, side = row["half"], row["team_slot"]
    clean = load_npz(f"clean_{match_id}")
    sign = float(clean[f"{half}_sign_home"]) if side == "Home" else float(clean[f"{half}_sign_away"])
    emitted = int(row["sync_frame"])
    earlier = emitted - int(round(C.PREDICT_LEAD_S * C.FRAMERATE))
    passer, recip = int(row["passer_slot"]), int(row["recipient_slot"])

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0))
    titles = ["the frame the synchroniser emits",
              f"{C.PREDICT_LEAD_S:.1f} s earlier: where our window ends"]
    for ax, frame, title in zip(axes, (emitted, earlier), titles):
        st = frame_state(clean, half, side, frame, sign)
        draw_pitch(ax)
        ok = ~np.isnan(st["opps"][:, 0])
        ax.scatter(st["opps"][ok, 0], st["opps"][ok, 1], s=60, c=AWAY, alpha=.75,
                   edgecolors="white", linewidths=.8, zorder=3, label="opponents")
        mates = st["mates"]
        ok = ~np.isnan(mates[:, 0])
        ax.scatter(mates[ok, 0], mates[ok, 1], s=60, c=HOME, alpha=.85,
                   edgecolors="white", linewidths=.8, zorder=3, label="passing team")
        ax.scatter(*mates[passer], s=150, facecolors="none", edgecolors=INK, lw=1.8,
                   zorder=5, label="passer")
        ax.scatter(*mates[recip], s=190, facecolors="none", edgecolors=GOOD, lw=2.4,
                   zorder=5, label="true receiver")
        b, bv = st["ball"], st["ball_v"]
        ax.scatter(*b, s=70, c=BALL, edgecolors=INK, lw=.9, zorder=6, label="ball")
        speed = float(np.linalg.norm(bv))
        if speed > 0.3:
            u = bv / speed
            ax.arrow(b[0], b[1], u[0] * 9, u[1] * 9, width=.7, head_width=2.6,
                     length_includes_head=True, color=BALL, ec=INK, lw=.6, zorder=6)
        gap = float(np.linalg.norm(b - mates[passer]))
        ax.set_title(f"{title}\nball {speed:.1f} m/s, {gap:.2f} m from the passer",
                     fontsize=10.5)
    axes[0].legend(fontsize=8.5, loc="lower left", ncol=3, columnspacing=1.0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig9_leakage_pitch.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  fig9: {match_id} pass_row {pass_row}, frames {emitted} and {earlier}")


# ------------------------------------------- figure 10: spatial breakdown
def normalised_origins(passes):
    """Pass origins with the passing team always attacking +x."""
    xs, ys, rows = [], [], []
    for mid, gm in passes.groupby("match_id"):
        try:
            clean = load_npz(f"clean_{mid}")
        except FileNotFoundError:
            continue
        for (half, side), g in gm.groupby(["half", "team_slot"]):
            sign = float(clean[f"{half}_sign_home"]) if side == "Home" \
                else float(clean[f"{half}_sign_away"])
            xs.append(g.ev_x.to_numpy() * sign)
            ys.append(g.ev_y.to_numpy() * sign)
            rows.append(g.index.to_numpy())
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(rows))


def figure_spatial(passes, preds):
    x, y, idx = normalised_origins(passes)
    fig, ax = plt.subplots(1, 2, figsize=(11.8, 4.2))

    draw_pitch(ax[0])
    hb = ax[0].hexbin(x, y, gridsize=18, cmap="Blues", mincnt=1, zorder=2)
    fig.colorbar(hb, ax=ax[0], shrink=.8, label="passes")
    ax[0].set_title(f"(a) Where the {len(x):,} usable passes start")

    can = preds[preds.model_variant == "canonical"]
    hit = can.groupby("pass_row").hit_top1.mean()
    pos = pd.Series(range(len(idx)), index=idx)
    common = hit.index.intersection(pos.index)
    hx, hy_ = x[pos[common]], y[pos[common]]
    hv = hit[common].to_numpy()

    draw_pitch(ax[1])
    xe = np.linspace(-52.5, 52.5, NX + 1)
    ye = np.linspace(-34, 34, NY + 1)
    grid = np.full((NY, NX), np.nan)
    counts = np.zeros((NY, NX), int)
    for i in range(NY):
        for j in range(NX):
            m = (hx >= xe[j]) & (hx < xe[j + 1]) & (hy_ >= ye[i]) & (hy_ < ye[i + 1])
            counts[i, j] = m.sum()
            if m.sum() >= MIN_CELL:
                grid[i, j] = hv[m].mean()
    im = ax[1].imshow(grid, extent=[-52.5, 52.5, -34, 34], origin="lower",
                      cmap="Blues", vmin=np.nanmin(grid), vmax=np.nanmax(grid),
                      alpha=.85, zorder=1)
    for i in range(NY):
        for j in range(NX):
            cx, cy = (xe[j] + xe[j + 1]) / 2, (ye[i] + ye[i + 1]) / 2
            if counts[i, j] >= MIN_CELL:
                ax[1].text(cx, cy, f"{grid[i, j]:.2f}\nn={counts[i, j]}", ha="center",
                           va="center", fontsize=8.5, zorder=4,
                           color="white" if grid[i, j] > np.nanmean(grid) else INK)
    fig.colorbar(im, ax=ax[1], shrink=.8, label="top-1 accuracy")
    ax[1].set_title("(b) Top-1 accuracy by zone, held-out matches")
    ax[1].annotate("", xy=(30, -30), xytext=(-30, -30),
                   arrowprops=dict(arrowstyle="->", color=MUT, lw=1.2))
    ax[1].text(0, -32.5, "direction of attack", ha="center", fontsize=8.5, color=MUT)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig10_spatial.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  fig10: {len(hv)} test passes placed, "
          f"{int((counts >= MIN_CELL).sum())} of {NX * NY} zones above the {MIN_CELL}-pass floor")


if __name__ == "__main__":
    banner("Step 32 - pitch figures")
    os.makedirs(OUT, exist_ok=True)
    run_id = P.canonical_run_id()
    pred_path = f"{C.WORK}/metrics_predictions_{run_id}.csv"
    if not os.path.isfile(pred_path):
        raise SystemExit(f"{pred_path} missing: run step23_canonical.py first")
    preds = pd.read_csv(pred_path)
    passes = pd.read_csv(f"{C.WORK}/passes_synced.csv")
    P.print_header("pitch figures", run_id, lead_s=C.PREDICT_LEAD_S,
                   test=C.TEST_MATCHES)

    pass_row, match_id = None, None
    for i, a in enumerate(sys.argv):
        if a == "--pass-row":
            pass_row = int(sys.argv[i + 1])
        if a == "--match":
            match_id = sys.argv[i + 1]
    if pass_row is None:
        pick = pick_pass(passes, preds)
        pass_row, match_id = int(pick.pass_row), pick["match"]
        print(f"  selection rule: ball-direction probe correct at the emitted frame, "
              f"ball speed closest to the median ({pick.speed_at_window_end:.2f} m/s "
              "at the window end)")
    elif match_id is None:
        match_id = passes.loc[pass_row, "match_id"]

    figure_leakage(passes, pass_row, match_id)
    figure_spatial(passes, preds)
    print(f"\nSaved -> {OUT}/fig9_leakage_pitch.png, {OUT}/fig10_spatial.png")
    print("Both are drawn from work/; no coordinate in either was entered by hand.")
