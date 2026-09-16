"""Step 16 - Tactical figures, built with floodlight.

Step 15 answers "is the method sound". This one answers "what does a situation
actually look like" - pitch layouts, trajectories, space control, team shape.

Everything here uses floodlight rather than a re-implementation, for two
reasons: it is the package the dataset authors used for their own paper, so a
reader recognises the output, and its space-control and centroid models are
already peer-reviewed rather than something we wrote the night before.

  fig9   trajectories        the 1.5 s run-up to a pass, both teams and the ball.
                             The same view the dataset paper uses for its shots.
  fig10  space control       Voronoi control at the moment of the pass, with the
                             receiver marked. Why an option was open.
  fig11  team shape          centroid and stretch index across the window.
  fig12  position density    where each playing position lives, from the
                             teamsheet's own position codes.

Run:  python step16_tactical.py [MATCH_ID]
Out:  figures/paper/fig9..12
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from floodlight.models.geometry import CentroidModel
from floodlight.models.space import DiscreteVoronoiModel

import config as C
from common import banner, coords, load_match

OUT = f"{C.FIGS}/paper"
os.makedirs(OUT, exist_ok=True)

INK, MUTED, RULE = "#14171A", "#6E7378", "#D8DAD5"
ATT, DEF, BALL_C, GOOD = "#2E6F9E", "#B3402F", "#14171A", "#1F7A4D"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "font.size": 12,
    "axes.titlesize": 13, "axes.titleweight": "600", "text.color": INK,
    "legend.frameon": False,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {OUT}/{name}.png  +  .pdf")


def pick_pass(passes, match_id):
    """A pass in open play with room either side, so the window is complete."""
    s = passes[(passes["match_id"] == match_id) &
               (passes["half"] == "firstHalf")]
    back = int((C.WINDOW_SECONDS + C.PREDICT_LEAD_S) * C.FRAMERATE)
    s = s[s["sync_frame"] > back + 100]
    if s.empty:
        return None
    return s.iloc[len(s) // 2]


# --------------------------------------------------------------------- fig 9
def fig9_trajectories(m, row):
    """floodlight's own trajectory view, applied to a pass instead of a shot."""
    half = row["half"]
    end = int(row["sync_frame"]) - int(C.PREDICT_LEAD_S * C.FRAMERATE)
    start = end - int(C.WINDOW_SECONDS * C.FRAMERATE)
    mine = "Home" if row["team_slot"] == "Home" else "Away"
    theirs = "Away" if mine == "Home" else "Home"

    fig, ax = plt.subplots(figsize=(10, 6.4))
    m["pitch"].plot(color_scheme="bw", ax=ax)

    # NOTE: floodlight sets linewidth and marker size itself, so passing the
    # short aliases (lw, s) raises "got both linewidth and lw". Use long names.
    m["xy"][half][mine].plot(t=(start, end), plot_type="trajectories",
                             color=ATT, alpha=0.75, ax=ax)
    m["xy"][half][theirs].plot(t=(start, end), plot_type="trajectories",
                               color=DEF, alpha=0.55, ax=ax)
    m["xy"][half]["Ball"].plot(t=(start, end), plot_type="trajectories",
                               color=BALL_C, ax=ax)

    # defending team first, so the attacking team and its markers sit on top
    m["xy"][half][theirs].plot(t=end, plot_type="positions",
                               color=DEF, zorder=4, ax=ax)
    m["xy"][half][mine].plot(t=end, plot_type="positions",
                             color=ATT, zorder=6, ax=ax)

    mine_xy = coords(m["xy"][half][mine])
    ax.scatter(*mine_xy[end, int(row["passer_slot"])], s=210, facecolor="none",
               edgecolor=GOOD, lw=2.8, zorder=9)
    ax.annotate("on the ball", mine_xy[end, int(row["passer_slot"])],
                xytext=(0, 16), textcoords="offset points", ha="center",
                color=GOOD, weight="600", fontsize=11)
    ax.scatter(*mine_xy[end, int(row["recipient_slot"])], s=210,
               facecolor="none", edgecolor=ATT, lw=2.8, ls=(0, (2, 1.6)),
               zorder=9)
    ax.annotate("receives it", mine_xy[end, int(row["recipient_slot"])],
                xytext=(0, 16), textcoords="offset points", ha="center",
                color=ATT, weight="600", fontsize=11)

    ax.set_title(f"The {C.WINDOW_SECONDS} s before the pass — "
                 f"attacking team in blue, ball in black\n"
                 f"window ends {C.PREDICT_LEAD_S} s before contact",
                 fontsize=12.5)
    save(fig, "fig9_trajectories")


# -------------------------------------------------------------------- fig 10
def fig10_space_control(m, row):
    """Voronoi control. Answers 'why was that option open' visually."""
    half = row["half"]
    end = int(row["sync_frame"]) - int(C.PREDICT_LEAD_S * C.FRAMERATE)
    mine = "Home" if row["team_slot"] == "Home" else "Away"
    theirs = "Away" if mine == "Home" else "Home"

    # one frame is enough and keeps the mesh cheap
    xy1 = m["xy"][half][mine].slice(end, end + 1)
    xy2 = m["xy"][half][theirs].slice(end, end + 1)

    dvm = DiscreteVoronoiModel(m["pitch"], mesh="square", xpoints=90)
    dvm.fit(xy1, xy2)

    fig, ax = plt.subplots(figsize=(10, 6.4))
    dvm.plot(t=0, team_colors=(ATT, DEF), ax=ax, alpha=0.30)
    m["pitch"].plot(color_scheme="bw", ax=ax)

    a = coords(xy1)[0]
    b = coords(xy2)[0]
    ax.scatter(a[:, 0], a[:, 1], c=ATT, s=95, ec="white", lw=1.1, zorder=5)
    ax.scatter(b[:, 0], b[:, 1], c=DEF, s=95, ec="white", lw=1.1, zorder=5)
    ax.scatter(*a[int(row["passer_slot"])], s=230, facecolor="none",
               edgecolor=GOOD, lw=2.6, zorder=7)
    ax.scatter(*a[int(row["recipient_slot"])], s=230, facecolor="none",
               edgecolor="white", lw=2.6, zorder=7)

    ctrl_a, _ = dvm.team_controls()
    # team_controls returns a TeamProperty per frame; ravel before indexing
    share = float(np.asarray(ctrl_a.property).ravel()[0]) / 100.0
    ax.set_title(f"Space control at the moment of decision\n"
                 f"attacking team holds {share:.0%} of the pitch — "
                 f"white ring is the player who receives it", fontsize=12.5)
    save(fig, "fig10_space_control")


# -------------------------------------------------------------------- fig 11
def fig11_team_shape(m, row):
    """How compact each team is through the run-up, using floodlight's models."""
    half = row["half"]
    end = int(row["sync_frame"]) - int(C.PREDICT_LEAD_S * C.FRAMERATE)
    start = end - int(C.WINDOW_SECONDS * C.FRAMERATE)
    mine = "Home" if row["team_slot"] == "Home" else "Away"
    theirs = "Away" if mine == "Home" else "Home"

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    t = np.arange(start, end) / C.FRAMERATE - end / C.FRAMERATE

    for side, col, lab in ((mine, ATT, "attacking"), (theirs, DEF, "defending")):
        seg = m["xy"][half][side].slice(start, end)
        cm = CentroidModel()
        cm.fit(seg)
        # stretch_index needs the positions again - it measures spread AROUND
        # the fitted centroid, so it cannot be computed from the centroid alone
        si = np.asarray(cm.stretch_index(seg).property).ravel()
        cen = np.asarray(cm.centroid().xy)
        ax[0].plot(t[:len(si)], si, color=col, lw=2.1, label=lab)
        ax[1].plot(cen[:, 0], cen[:, 1], color=col, lw=2.1, label=lab)
        ax[1].scatter(*cen[-1], color=col, s=70, zorder=4)

    ax[0].set_xlabel("seconds before the window ends")
    ax[0].set_ylabel("stretch index (m)")
    ax[0].set_title("How spread out each team is")
    ax[0].legend(fontsize=10.5)
    ax[0].grid(alpha=0.4)
    ax[0].spines[["top", "right"]].set_visible(False)

    m["pitch"].plot(color_scheme="bw", ax=ax[1])
    ax[1].set_title("Where the team centroids drift")
    ax[1].legend(fontsize=10.5)

    fig.suptitle("Team shape through the run-up to the pass",
                 fontsize=13, weight="600", y=1.03)
    save(fig, "fig11_team_shape")


# -------------------------------------------------------------------- fig 12
def fig12_position_density(m, side="Home", n_pos=4):
    """The dataset paper's own validation figure, reproduced.

    Uses the teamsheet's German position codes - TW keeper, IV centre back,
    ST striker and so on. If each position's density sits where it should, the
    tracking and the teamsheet agree with each other.
    """
    sheet = m["teamsheets"][side].teamsheet
    if "position" not in sheet.columns:
        print("  (no position column, skipping fig12)")
        return

    counts = sheet["position"].value_counts()
    picks = [p for p in counts.index if isinstance(p, str)][:n_pos]
    if not picks:
        print("  (no usable position codes, skipping fig12)")
        return

    fig, axes = plt.subplots(1, len(picks), figsize=(4.3 * len(picks), 3.4))
    axes = np.atleast_1d(axes)

    for ax, pos in zip(axes, picks):
        slots = sheet.loc[sheet["position"] == pos, "xID"].dropna().astype(int)
        pts = []
        for half in ("firstHalf", "secondHalf"):
            c = coords(m["xy"][half][side])
            sign = 1 if half == "firstHalf" else -1
            for s in slots:
                if s < c.shape[1]:
                    p = c[::25, s] * sign      # 1 Hz is plenty for a density
                    pts.append(p[~np.isnan(p[:, 0])])
        if not pts:
            continue
        pts = np.concatenate(pts)
        m["pitch"].plot(color_scheme="bw", ax=ax)
        ax.hexbin(pts[:, 0], pts[:, 1], gridsize=26, cmap="Blues",
                  mincnt=1, alpha=0.85, zorder=2)
        ax.set_title(f"{pos}   (n={len(slots)})", fontsize=12)

    fig.suptitle(f"Where each position actually plays — {side} team, "
                 "normalised to one attacking direction",
                 fontsize=12.5, weight="600", y=1.06)
    save(fig, "fig12_position_density")


if __name__ == "__main__":
    banner("Step 16 - tactical figures with floodlight")
    match_id = sys.argv[1] if len(sys.argv) > 1 else C.TEST_MATCHES[0]

    passes = pd.read_csv(f"{C.WORK}/passes_synced.csv")
    row = pick_pass(passes, match_id)
    if row is None:
        sys.exit(f"no usable pass found in {match_id}")

    print(f"match {match_id}, pass at frame {int(row['sync_frame'])} "
          f"({row['half']}, {row['team_slot']} in possession)\n")

    m = load_match(match_id)
    fig9_trajectories(m, row)
    fig10_space_control(m, row)
    fig11_team_shape(m, row)
    fig12_position_density(m)

    print("\nfig9 mirrors the trajectory view the dataset paper uses for shots,")
    print("so a reader of that paper will recognise it immediately.")
    print("fig10 is the one that explains a prediction: it shows the space the")
    print("receiver had, rather than asserting the model was right.")
