"""Step 15 - Figures for the presentation and the paper.

Each figure exists to carry one claim. Nothing here is decoration: if a figure
does not change what a reader believes, it is not in this file.

  fig1  leakage decay      the central result. How much the ball gives away,
                           as a function of how early you stop looking.
  fig2  timing diagram     what the model sees and when, in one picture.
  fig3  coordinate error   why the 63 m offset could not have been a timing bug.
  fig4  data funnel        11,137 events down to the samples we train on.
  fig5  split design       why the train/test split had to be what it is.
  fig6  pass origins       where passes come from, on a real pitch.
  fig7  results            baselines against the model, with intervals.
  fig8  ablation           which borrowed idea actually helped.

Every figure is written twice: PNG at 300 dpi for slides, PDF vector for the
paper. Fonts are sized for a projector, which is the harsher constraint.

Run:  python step15_figures.py
Out:  figures/paper/fig*.png and fig*.pdf
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from common import banner, coords, load_npz

OUT = f"{C.FIGS}/paper"
os.makedirs(OUT, exist_ok=True)

# one palette, used consistently, safe for the common colour deficiencies
INK, MUTED, RULE = "#14171A", "#6E7378", "#D8DAD5"
DATA, WARN, GOOD, ALT = "#2E6F9E", "#B3402F", "#1F7A4D", "#C98A2B"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300,
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.titleweight": "600", "axes.edgecolor": RULE, "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.labelcolor": INK, "grid.color": RULE, "grid.linewidth": 0.7,
    "legend.frameon": False,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  {OUT}/{name}.png  +  .pdf")


# --------------------------------------------------------------------- fig 1
def leakage_curve(passes, clean, leads_s):
    """Ball-direction probe accuracy at each lead time, in one pass over the data.

    PERFORMANCE NOTE, learned the hard way. `np.load` on an .npz returns a LAZY
    NpzFile: every `d["key"]` access re-reads and decompresses the whole array
    from the zip. The first version of this function did that lookup inside the
    per-pass loop, which meant decompressing a 23 MB array once per pass per
    lead - about 1.1 TB of pointless zlib work, roughly an hour of runtime.

    So: materialise each array exactly once per (match, half), then compute all
    leads for all passes with array operations. Same numbers, seconds instead of
    an hour.
    """
    leads_f = [int(l * C.FRAMERATE) for l in leads_s]
    hits = np.zeros(len(leads_f))
    total = np.zeros(len(leads_f))
    back = int(C.WINDOW_SECONDS * C.FRAMERATE)

    for mid, sub in passes.groupby("match_id"):
        if mid not in clean:
            continue
        d = clean[mid]() if callable(clean[mid]) else clean[mid]
        for half in ("firstHalf", "secondHalf"):
            s = sub[sub["half"] == half]
            if s.empty:
                continue

            # decompress once, not once per pass
            ball_p = np.asarray(d[f"{half}_Ball_pos"])[:, 0]
            ball_v = np.asarray(d[f"{half}_Ball_vel"])[:, 0]
            pos = {k: np.asarray(d[f"{half}_{k}_pos"]) for k in ("Home", "Away")}
            sign = {k: float(d[f"{half}_sign_{k.lower()}"]) for k in ("Home", "Away")}
            n_frames = len(ball_p)

            f0 = s["sync_frame"].to_numpy(dtype=int)
            passer = s["passer_slot"].to_numpy(dtype=int)
            recip = s["recipient_slot"].to_numpy(dtype=int)
            team = s["team_slot"].to_numpy()

            for li, lf in enumerate(leads_f):
                f = f0 - lf
                usable = (f - back >= 0) & (f < n_frames)

                for side in ("Home", "Away"):
                    m = usable & (team == side)
                    if not m.any():
                        continue
                    ff = f[m]
                    sg = sign[side]

                    b = ball_p[ff] * sg                      # (n, 2)
                    v = ball_v[ff] * sg
                    P = pos[side][ff] * sg                   # (n, slots, 2)

                    live = np.isfinite(b[:, 0]) & np.isfinite(v[:, 0])
                    if not live.any():
                        continue

                    ok = np.isfinite(P[:, :, 0])
                    ok[np.arange(len(ff)), passer[m]] = False   # not himself

                    nv = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
                    to = P - b[:, None, :]
                    to = to / (np.linalg.norm(to, axis=2, keepdims=True) + 1e-9)
                    cos = np.einsum("nsd,nd->ns", to, nv)
                    cos[~ok] = -2.0

                    pick = cos.argmax(axis=1)
                    r = recip[m]
                    # only count passes whose receiver was actually on the pitch
                    valid = live & ok[np.arange(len(ff)), np.clip(r, 0, ok.shape[1] - 1)]
                    hits[li] += np.sum((pick == r) & valid)
                    total[li] += np.sum(valid)

    return np.where(total > 0, hits / np.maximum(total, 1), np.nan), total


def fig1_leakage(passes, clean, yardstick=0.27):
    leads_s = np.arange(0, 1.05, 0.1)
    acc, counts = leakage_curve(passes, clean, leads_s)
    print(f"  computed on {int(counts[0]):,} passes per lead")

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axhspan(0, yardstick, color=RULE, alpha=0.35, zorder=0)
    ax.axhline(yardstick, color=MUTED, ls=(0, (4, 3)), lw=1.2, zorder=1)
    ax.plot(leads_s, acc, "-o", color=WARN, lw=2.4, ms=6.5, zorder=3)

    ax.axvline(C.PREDICT_LEAD_S, color=GOOD, lw=1.4, ls=(0, (2, 2)), zorder=2)
    ax.annotate("window ends here", (C.PREDICT_LEAD_S, max(acc) * 0.93),
                xytext=(14, 0), textcoords="offset points",
                color=GOOD, fontsize=11, weight="600", va="center")
    ax.annotate("nearest-teammate baseline", (0.92, yardstick),
                xytext=(0, -16), textcoords="offset points",
                color=MUTED, fontsize=10.5, ha="right")

    ax.set_xlabel("how early the window ends, before the ball is struck (s)")
    ax.set_ylabel("receiver predicted from\nball direction alone")
    ax.set_title("Ball-direction probe accuracy versus window offset")
    ax.set_ylim(0, max(acc) * 1.18)
    ax.set_xlim(-0.03, 1.03)
    ax.grid(axis="y", alpha=0.5)
    save(fig, "fig1_leakage_decay")
    return {round(float(k), 2): round(float(v), 3) for k, v in zip(leads_s, acc)}


# --------------------------------------------------------------------- fig 2
def fig2_timing():
    fig, ax = plt.subplots(figsize=(9.5, 2.9))
    kick = 0.0
    lead = -C.PREDICT_LEAD_S
    start = lead - C.WINDOW_SECONDS

    ax.axhline(0, color=RULE, lw=1.2)
    ax.add_patch(plt.Rectangle((start, -0.30), C.WINDOW_SECONDS, 0.60,
                               fc=DATA, alpha=0.16, ec=DATA, lw=1.3))
    ax.add_patch(plt.Rectangle((lead, -0.30), C.PREDICT_LEAD_S, 0.60,
                               fc=WARN, alpha=0.13, ec=WARN, lw=1.3,
                               ls=(0, (3, 2))))

    for t in np.arange(start, lead + 1e-9, C.STRIDE / C.FRAMERATE):
        ax.plot([t], [0], "|", color=DATA, ms=13, mew=1.7)

    ax.plot([kick], [0], "o", color=INK, ms=11, zorder=5)
    ax.annotate("ball struck", (kick, 0), xytext=(0, 26),
                textcoords="offset points", ha="center", weight="600")
    ax.text(start + C.WINDOW_SECONDS / 2, 0.40,
            f"model sees this  ({C.N_STEPS} frames, {C.WINDOW_SECONDS} s)",
            ha="center", color=DATA, weight="600", fontsize=11.5)
    ax.text(lead + C.PREDICT_LEAD_S / 2, -0.46,
            f"discarded  ({C.PREDICT_LEAD_S} s)\nball begins to move here",
            ha="center", va="top", color=WARN, fontsize=10.5)

    ax.set_xlim(start - 0.18, 0.42)
    ax.set_ylim(-0.95, 0.75)
    ax.set_yticks([])
    ax.set_xlabel("seconds relative to the moment the ball is struck")
    ax.spines["left"].set_visible(False)
    ax.set_title("Observation window relative to the emitted frame")
    save(fig, "fig2_timing")


# --------------------------------------------------------------------- fig 3
def fig3_coordinates(passes):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    ax[0].scatter(passes["at_x"], passes["at_y"], s=5, alpha=0.30, color=WARN,
                  ec="none")
    ax[0].add_patch(plt.Rectangle((-52.5, -34), 105, 68, fc="none",
                                  ec=DATA, lw=1.6))
    ax[0].set_title("as loaded: two different frames")
    ax[0].annotate("tracking pitch", (0, 34), xytext=(0, 8),
                   textcoords="offset points", ha="center", color=DATA,
                   fontsize=10.5, weight="600")
    ax[0].annotate("event coordinates", (52, 0), xytext=(6, 0),
                   textcoords="offset points", color=WARN, fontsize=10.5,
                   weight="600", va="center")

    if "ev_x" in passes:
        ax[1].scatter(passes["ev_x"], passes["ev_y"], s=5, alpha=0.30,
                      color=GOOD, ec="none")
    ax[1].add_patch(plt.Rectangle((-52.5, -34), 105, 68, fc="none",
                                  ec=DATA, lw=1.6))
    ax[1].set_title("after shifting by half the pitch")

    for a in ax:
        a.set_aspect("equal")
        a.set_xlim(-60, 115)
        a.set_ylim(-42, 76)
        a.axis("off")

    fig.suptitle("A constant 62.55 m error that no time shift could fix",
                 fontsize=13.5, weight="600", y=1.02)
    save(fig, "fig3_coordinates")


# --------------------------------------------------------------------- fig 4
def fig4_funnel(passes, total_samples):
    """Counts come from work/event_counts.csv (step2) and the arrays, never typed in."""
    ev_path = f"{C.WORK}/event_counts.csv"
    stages = []
    if os.path.isfile(ev_path):
        ev = pd.read_csv(ev_path)
        for col, label in (("events_all", "all events"),
                           ("open_play_passes", "open-play passes"),
                           ("successful_passes", "successfully completed")):
            if col in ev.columns:
                stages.append((label, int(ev[col].sum())))
    else:
        print("  (work/event_counts.csv missing: rerun step2_passes.py; the funnel"
              " starts at the synchronised passes)")
        stages.append(("synchronised passes", len(passes)))
    stages.append(("usable samples", total_samples))
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    top = stages[0][1]
    for i, (label, n) in enumerate(stages):
        w = n / top
        ax.add_patch(plt.Rectangle((0.5 - w / 2, -i), w, 0.62,
                                   fc=DATA, alpha=0.28 + 0.16 * i, ec="none"))
        ax.text(0.5, -i + 0.31, f"{n:,}", ha="center", va="center",
                weight="700", fontsize=13)
        ax.text(1.06, -i + 0.31, label, va="center", fontsize=11.5)
        if i:
            ax.text(-0.06, -i + 0.31, f"−{stages[i-1][1]-n:,}", va="center",
                    ha="right", fontsize=10, color=MUTED)

    ax.set_xlim(-0.30, 1.72)
    ax.set_ylim(-len(stages) + 0.2, 0.95)
    ax.axis("off")
    ax.set_title("From the raw event log to what the model trains on",
                 loc="left", fontsize=13)
    save(fig, "fig4_funnel")


# --------------------------------------------------------------------- fig 5
def fig5_split():
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    for i, (mid, (home, away, div)) in enumerate(C.MATCHES.items()):
        test = mid in C.TEST_MATCHES
        ax.barh(i, 1, color=WARN if test else DATA,
                alpha=0.85 if test else 0.42, height=0.66)
        ax.text(1.03, i, f"{home}  v  {away}", va="center", fontsize=11)
        ax.text(0.5, i, f"div {div}", va="center", ha="center",
                color="white", fontsize=10.5, weight="700")
    ax.set_yticks(range(len(C.MATCHES)))
    ax.set_yticklabels(C.MATCHES.keys(), fontsize=10.5)
    ax.set_xticks([])
    ax.set_xlim(0, 2.5)
    ax.spines["bottom"].set_visible(False)
    ax.set_title("Five of seven matches share a home team — and those five are\n"
                 "exactly the second-division matches", loc="left", fontsize=12.5)
    ax.text(0, len(C.MATCHES) - 0.1, "red = held out for testing",
            color=WARN, fontsize=10.5, weight="600")
    save(fig, "fig5_split")


# --------------------------------------------------------------------- fig 6
def fig6_pass_origins(passes):
    try:
        from mplsoccer import Pitch
    except ImportError:
        print("  (mplsoccer not installed, skipping fig6)")
        return
    if "ev_x" not in passes:
        print("  (no converted coordinates, skipping fig6)")
        return

    pitch = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68,
                  line_color=RULE, pitch_color="white", linewidth=1.2)
    fig, ax = pitch.draw(figsize=(9, 6))
    x = passes["ev_x"] + 52.5
    y = passes["ev_y"] + 34
    pitch.kdeplot(x, y, ax=ax, fill=True, levels=60, cmap="Blues",
                  alpha=0.85, thresh=0.02)
    ax.set_title(f"Where the {len(passes):,} open-play passes start",
                 fontsize=13, weight="600", pad=12)
    save(fig, "fig6_pass_origins")


# --------------------------------------------------------------------- fig 7
def fig7_results(base, model):
    if base is None or model is None:
        print("  (missing result tables, skipping fig7)")
        return
    order = ["random", "nearest teammate", "most advanced", "gradient boosting"]
    matches = list(base["test_match"].unique())

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    width = 0.36
    for k, mid in enumerate(matches):
        b = base[base.test_match == mid].set_index("model")
        vals = [b.loc[m, "top1"] if m in b.index else np.nan for m in order]
        errs = [b.loc[m, "ci95"] if m in b.index else 0 for m in order]
        mv = model[(model.test_match == mid) &
                   (model.setting.str.startswith("A"))]["top1"]
        vals.append(mv.mean())
        errs.append(mv.std())

        pos = np.arange(len(vals)) + (k - 0.5) * width
        cols = [MUTED] * len(order) + [DATA]
        ax.bar(pos, vals, width * 0.92, yerr=errs, capsize=3,
               color=cols, alpha=0.55 + 0.45 * k,
               error_kw=dict(lw=1, ecolor=INK, alpha=0.6), label=mid)

    ax.set_xticks(np.arange(len(order) + 1))
    ax.set_xticklabels(order + ["this model"], fontsize=11)
    ax.axhline(0.10, color=WARN, ls=(0, (4, 3)), lw=1.1)
    ax.annotate("chance", (len(order) + 0.35, 0.10), xytext=(0, 5),
                textcoords="offset points", color=WARN, fontsize=10.5)
    ax.set_ylabel("top-1 accuracy")
    ax.set_title("Top-1 accuracy by model and test match")
    ax.legend(fontsize=10.5)
    ax.grid(axis="y", alpha=0.45)
    save(fig, "fig7_results")


# --------------------------------------------------------------------- fig 8
def fig8_ablation(abl):
    if abl is None:
        print("  (no ablation table, skipping fig8)")
        return
    piv = abl.pivot_table(index="setting", columns="test_match",
                          values="top1", aggfunc="mean")
    if "plain" not in piv.index:
        print("  (no 'plain' row, skipping fig8)")
        return
    gain = (piv.mean(axis=1) - piv.loc["plain"].mean()) * 100
    gain = gain.drop("plain").sort_values()

    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    cols = [GOOD if v > 0 else WARN for v in gain.values]
    ax.barh(range(len(gain)), gain.values, color=cols, alpha=0.8, height=0.62)
    ax.set_yticks(range(len(gain)))
    ax.set_yticklabels(gain.index, fontsize=11.5)
    ax.axvline(0, color=INK, lw=1)
    ax.axvspan(-4, 4, color=RULE, alpha=0.4, zorder=0)
    # sits at the BOTTOM of the band, not the top — at the top it collided with
    # the axes title whenever the tallest bar was near the top of the plot
    ax.annotate("inside this band the change is\nsmaller than the uncertainty",
                (0, -0.75), xytext=(0, 0), textcoords="offset points",
                fontsize=9.5, color=MUTED, va="center", ha="center")
    ax.set_ylim(-1.3, len(gain) - 0.3)
    ax.set_xlabel("change in top-1 accuracy vs the plain model (percentage points)")
    ax.set_title("Change in top-1 accuracy per intervention")
    ax.grid(axis="x", alpha=0.45)
    save(fig, "fig8_ablation")


if __name__ == "__main__":
    banner("Step 15 - figures for the presentation and the paper")

    passes = pd.read_csv(f"{C.WORK}/passes_synced.csv")
    # loaders, not loaded arrays: all seven materialised at once is ~670 MB
    clean = {}
    for mid in C.MATCHES:
        if os.path.isfile(f"{C.WORK}/clean_{mid}.npz"):
            clean[mid] = (lambda m=mid: load_npz(f"clean_{m}"))
    total = 0
    for mid in C.MATCHES:
        try:
            total += len(load_npz(f"dataset_{mid}")["X"])
        except FileNotFoundError:
            pass

    def maybe(name):
        p = f"{C.WORK}/{name}"
        return pd.read_csv(p) if os.path.isfile(p) else None

    print("\nbuilding the decay curve (this one re-measures at each lead)...")
    curve = fig1_leakage(passes, clean)
    print("  lead (s):  " + "  ".join(f"{k:5.1f}" for k in curve))
    print("  accuracy:  " + "  ".join(f"{v:5.3f}" for v in curve.values()))
    print()

    fig2_timing()
    fig3_coordinates(passes)
    fig4_funnel(passes, total)
    fig5_split()
    fig6_pass_origins(passes)
    fig7_results(maybe("baseline_results.csv"), maybe("model_results.csv"))
    fig8_ablation(maybe("ablation_results.csv"))

    print("\nPNG at 300 dpi for slides, PDF vector for the paper.")
    print("fig1 is the one your contribution rests on. Lead with it.")
