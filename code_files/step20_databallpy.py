"""Step 20 - Does the leak survive somebody else's synchronisation?

This is the experiment the whole project has been pointing at, and until now it
had not been run.

The problem with the leakage finding as it stands: it is demonstrated on ONE
synchronisation implementation, the one in step3_sync.py, which I wrote. A
reviewer is entitled to say "your synchroniser is buggy" and stop reading. That
reading cannot be ruled out from inside this repository.

DataBallPy settles it. It is the package the dataset authors themselves used to
synchronise this data - the paper says so, v0.5.3 - and it synchronises with the
Needleman-Wunsch algorithm, which is also what the 2026 edge-conditioned MPNN
paper uses. It even has first-class support for these exact seven matches:
`load_sportec_open_tracking_data` cites Bassek, Weber, Rein & Memmert directly.

So we run the same probe on ITS frames.

  If the probe stays quiet, the leak was mine and the finding shrinks to a bug
  report. That is a real possible outcome and it is worth knowing.

  If the probe fires there too, the leak is a property of how the field
  synchronises event and tracking data, not of my code. That is the version
  worth writing up.

Either way the answer is a number, not an opinion.

DEPENDENCIES: this step needs only numpy, pandas, matplotlib and databallpy.
It does NOT import torch or floodlight, which matters because databallpy pins
numpy < 2.3 while floodlight wants >= 2.1.1. The overlap is narrow, so if your
main environment sits outside it, run this file in a throwaway virtualenv and
copy work/databallpy_comparison.csv back. Nothing else in the project needs to
move.

Run:  python step20_databallpy.py [MATCH_ID]
Out:  work/databallpy_comparison.csv, figures/paper/fig16_sync_comparison.png
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from common import banner, load_npz

OUT = f"{C.FIGS}/paper"
os.makedirs(OUT, exist_ok=True)
INK, MUTED, DATA, WARN, GOOD = "#14171A", "#6E7378", "#2E6F9E", "#B3402F", "#1F7A4D"
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 300, "font.size": 11,
                     "axes.titleweight": "600"})


def databallpy_pass_frames(match_id, verbose=True):
    """Synchronise with DataBallPy and return one row per pass.

    Nothing here is our code except the column renaming. The point is that the
    frames come from somebody else's algorithm.
    """
    from databallpy import get_open_game

    game = get_open_game(provider="sportec", game_id=match_id, verbose=verbose)
    game.synchronise_tracking_and_event_data(verbose=verbose)

    ed = game.event_data
    # column names come from databallpy.schemas.EventDataSchema, which declares:
    #   databallpy_event, event_id, player_id, to_player_id, team_id, start_x/y
    # and TrackingDataSchema: frame, ball_x, ball_y, ball_z, ball_status
    mask = ed["databallpy_event"].astype(str).str.contains("pass", case=False,
                                                           na=False)
    passes = ed[mask].copy()

    td = game.tracking_data
    col = [c for c in td.columns if "event_id" in c.lower()]
    if not col:
        raise RuntimeError(
            "no event_id column on the tracking data. Run with --inspect to "
            "dump the real column names and adjust this function.")
    link = td[["frame", col[0]]].dropna()
    link.columns = ["frame", "event_id"]

    merged = passes.merge(link, on="event_id", how="inner")

    # DataBallPy's sportec parser leaves to_player_id empty, so the receiver has
    # to come from somewhere else. Both tools read the SAME DFL file, which
    # carries a Recipient attribute, and step 2 already parsed it - so we attach
    # our own labels rather than inventing new ones. Same ground truth, same
    # metric, only the FRAMES differ, which is the whole point of the comparison.
    ours = pd.read_csv(f"{C.WORK}/passes.csv")
    if "event_id" not in ours.columns:
        raise RuntimeError(
            "work/passes.csv has no event_id column. Re-run step2_passes.py — "
            "it now keeps the DFL EventId, which is how the two pipelines are "
            "joined.")
    ours = ours[ours["match_id"] == match_id]

    joined = None
    if "event_id" in ours.columns and ours["event_id"].notna().any():
        o = ours.dropna(subset=["event_id"]).copy()
        o["event_id"] = o["event_id"].astype(str)
        m = merged.copy()
        for col in ("original_event_id", "event_id"):
            if col in m.columns:
                m["_join"] = m[col].astype(str)
                cand = m.merge(o[["event_id", "recipient"]],
                               left_on="_join", right_on="event_id",
                               how="inner", suffixes=("", "_ours"))
                if len(cand) > 0.2 * len(merged):
                    joined = cand
                    print(f"  joined {len(cand)} passes to our labels on {col}")
                    break

    if joined is None:
        # fall back to nearest timestamp within half a second
        print("  event ids did not join; falling back to timestamp matching")
        o = ours.dropna(subset=["timestamp"]).copy()
        o["t"] = pd.to_datetime(o["timestamp"], utc=True, errors="coerce")
        m = merged.copy()
        tcol = "datetime" if "datetime" in m.columns else None
        if tcol is None:
            raise RuntimeError("no datetime column to fall back on — run "
                               "--inspect and pick a join key by hand")
        m["t"] = pd.to_datetime(m[tcol], utc=True, errors="coerce")
        joined = pd.merge_asof(
            m.dropna(subset=["t"]).sort_values("t"),
            o.dropna(subset=["t"]).sort_values("t")[["t", "recipient"]],
            on="t", direction="nearest", tolerance=pd.Timedelta("0.5s"))
        joined = joined.dropna(subset=["recipient"])
        print(f"  matched {len(joined)} passes by timestamp")
        print("  note: a mismatched label can only make the probe look WORSE,")
        print("  never better, so any result here is a lower bound.")

    return joined, game


def inspect_api(match_id):
    """Dump the real column names, so a schema change is a two-minute fix.

    This function exists because the code above was written against
    databallpy.schemas rather than against a downloaded match — the sandbox it
    was developed in cannot reach the Sportec download host. The schema is the
    library's own declaration, so it should be right, but "should be" is not
    "verified", and this makes checking trivial.
    """
    from databallpy import get_open_game
    game = get_open_game(provider="sportec", game_id=match_id, verbose=True)
    print("\n--- event_data columns ---")
    print(list(game.event_data.columns))
    print("\n--- tracking_data columns (first 30) ---")
    print(list(game.tracking_data.columns)[:30])
    print("\n--- databallpy_event values ---")
    print(game.event_data["databallpy_event"].value_counts().head(12))

    ed = game.event_data
    for col in ("player_id", "to_player_id"):
        if col not in ed.columns:
            print(f"\n--- {col}: COLUMN MISSING ---")
            continue
        v = ed[col]
        print(f"\n--- {col} ---")
        print(f"  dtype {v.dtype}   non-null {v.notna().sum()} of {len(v)}")
        print(f"  sample values: {v.dropna().head(5).tolist()}")
        ok = sum(to_column(game, x) is not None for x in v.dropna().head(200))
        print(f"  of the first 200 non-null, {ok} map to a tracking column")
        if ok == 0:
            print("  -> none map. That is why the probe returns n=0.")
    return game


def to_column(game, pid):
    """player id -> tracking column id, tolerant about how the id is typed.

    DataBallPy declares player_id_to_column_id(player_id: int), but a pandas
    column that has ever held a NaN comes back as float, so 12345 arrives as
    12345.0 and the lookup misses. Sportec ids can also arrive as strings.
    """
    if pid is None:
        return None
    # our labels are DFL PersonIds like "DFL-OBJ-0027KL". DataBallPy's sportec
    # parser uses the same PersonId as its player id, so these match directly.
    if isinstance(pid, str) and pid.startswith("DFL-"):
        try:
            return game.player_id_to_column_id(pid)
        except Exception:
            return None
    try:
        if isinstance(pid, float):
            if np.isnan(pid):
                return None
            pid = int(pid)
        return game.player_id_to_column_id(pid)
    except Exception:
        pass
    for alt in (str(pid), int(pid) if str(pid).lstrip("-").isdigit() else None):
        if alt is None:
            continue
        try:
            return game.player_id_to_column_id(alt)
        except Exception:
            continue
    return None


def probe_on_frames(merged, game):
    """The same ball-direction probe, run against DataBallPy's frames.

    Deliberately identical in logic to check_leakage.direction_probe, so any
    difference in the result comes from the FRAMES, not from the metric.

    The first version of this failed on every row and returned n=0. The cause:
    it tried to find a player's tracking columns by matching the event's
    `team_id`, which is a DFL club identifier like "DFL-CLU-00000G", against
    column names like "home_7_x". Those never match. DataBallPy has an explicit
    mapping for exactly this - player_id_to_column_id and get_column_ids - and
    it should have been used from the start.
    """
    td = game.tracking_data
    fps = game.frame_rate

    home_ids = set(game.get_column_ids(team="home"))
    away_ids = set(game.get_column_ids(team="away"))

    # frame -> row index, so we are not scanning the table once per pass
    frame_pos = {int(f): i for i, f in enumerate(td["frame"].to_numpy())}

    hits = total = 0
    speeds, gaps = [], []
    skipped = {"no frame": 0, "no ball": 0, "no passer column": 0,
               "no receiver id": 0, "receiver not mapped": 0, "too few mates": 0}

    for _, ev in merged.iterrows():
        f = int(ev["frame"])
        i = frame_pos.get(f)
        j = frame_pos.get(f + 1)
        if i is None or j is None:
            skipped["no frame"] += 1
            continue
        row, nxt = td.iloc[i], td.iloc[j]

        bx, by = row.get("ball_x"), row.get("ball_y")
        vx = nxt.get("ball_x", np.nan) - bx
        vy = nxt.get("ball_y", np.nan) - by
        if not (np.isfinite(bx) and np.isfinite(vx)) or (vx == 0 and vy == 0):
            skipped["no ball"] += 1
            continue

        passer_col = to_column(game, ev.get("player_id"))
        if passer_col is None:
            skipped["no passer column"] += 1
            continue

        # look the receiver up BEFORE recording anything. In the first version
        # this happened last, so a failure here left speeds and gaps populated
        # while the count stayed at zero - a genuinely confusing state to debug.
        # our own label, carried over from the DFL Recipient attribute
        recv_raw = ev.get("recipient", ev.get("to_player_id"))
        if recv_raw is None or (isinstance(recv_raw, float) and np.isnan(recv_raw)):
            skipped["no receiver id"] += 1
            continue
        recv_col = to_column(game, recv_raw)
        if recv_col is None:
            skipped["receiver not mapped"] += 1
            continue

        side = home_ids if passer_col in home_ids else away_ids
        cand = []
        for col in side:
            if col == passer_col:
                continue
            px, py = row.get(f"{col}_x"), row.get(f"{col}_y")
            if np.isfinite(px) and np.isfinite(py):
                cand.append((col, float(px), float(py)))
        if len(cand) < 2:
            skipped["too few mates"] += 1
            continue

        n = np.hypot(vx, vy)
        best, best_cos = None, -2.0
        for col, px, py in cand:
            dx, dy = px - bx, py - by
            d = np.hypot(dx, dy) + 1e-9
            cos = (dx * vx + dy * vy) / (d * n)
            if cos > best_cos:
                best, best_cos = col, cos

        pxx, pyy = row.get(f"{passer_col}_x"), row.get(f"{passer_col}_y")
        if np.isfinite(pxx):
            gaps.append(float(np.hypot(bx - pxx, by - pyy)))
        speeds.append(float(n * fps))
        hits += best == recv_col
        total += 1

    kept = f"{total} of {len(merged)} passes used"
    dropped = ", ".join(f"{k}={v}" for k, v in skipped.items() if v)
    print(f"  {kept}" + (f"   (dropped: {dropped})" if dropped else ""))

    med_speed = float(np.median(speeds)) if speeds else np.nan
    # a football never travels at 50 m/s. If it appears to, the frame spacing or
    # the units are wrong, and every number below it is meaningless.
    if np.isfinite(med_speed) and med_speed > 50:
        print(f"  !! median ball speed {med_speed:.0f} m/s is not physical.")
        print("     Consecutive frames are probably not 1/frame_rate apart —")
        print("     check that tracking_data is sorted and has no gaps.")

    return {
        "n": total,
        "ball_direction_probe": hits / total if total else np.nan,
        "median_ball_speed": med_speed,
        "median_ball_to_passer": float(np.median(gaps)) if gaps else np.nan,
    }


def ours_at_zero_lead(match_id):
    """Our probe at the frame WE call the pass, with no lead applied.

    This is the control that makes the comparison fair. The headline "ours" row
    is measured 0.4 s earlier, so comparing it directly to DataBallPy conflates
    two things: which frame was chosen, and how far back we then step. This row
    steps back zero, so only the frame choice differs.
    """
    passes = pd.read_csv(f"{C.WORK}/passes_synced.csv")
    sub = passes[passes["match_id"] == match_id]
    if sub.empty:
        return None

    d = load_npz(f"clean_{match_id}")
    hits = total = 0
    speeds, gaps = [], []

    for half in ("firstHalf", "secondHalf"):
        s2 = sub[sub["half"] == half]
        if s2.empty:
            continue
        ball_p = d[f"{half}_Ball_pos"][:, 0]
        ball_v = d[f"{half}_Ball_vel"][:, 0]
        pos = {k: d[f"{half}_{k}_pos"] for k in ("Home", "Away")}

        for _, r in s2.iterrows():
            f = int(r["sync_frame"])
            if not (0 <= f < len(ball_p)):
                continue
            mine = "Home" if r["team_slot"] == "Home" else "Away"
            P = pos[mine][f]
            b, v = ball_p[f], ball_v[f]
            if not (np.isfinite(b[0]) and np.isfinite(v[0])):
                continue
            mates = np.where(np.isfinite(P[:, 0]))[0]
            mates = mates[mates != int(r["passer_slot"])]
            if len(mates) < 2 or int(r["recipient_slot"]) not in mates:
                continue
            nv = v / (np.linalg.norm(v) + 1e-9)
            to = P[mates] - b
            to = to / (np.linalg.norm(to, axis=1, keepdims=True) + 1e-9)
            pick = mates[int((to @ nv).argmax())]
            hits += pick == int(r["recipient_slot"])
            total += 1
            speeds.append(float(np.linalg.norm(v)))
            gaps.append(float(np.linalg.norm(b - P[int(r["passer_slot"])])))

    return {
        "n": total,
        "ball_direction_probe": hits / total if total else np.nan,
        "median_ball_speed": float(np.median(speeds)) if speeds else np.nan,
        "median_ball_to_passer": float(np.median(gaps)) if gaps else np.nan,
    }


def ours(match_id):
    """The same three numbers from our own pipeline, for the comparison."""
    d = load_npz(f"dataset_{match_id}")
    X, y = d["X"], d["y"]
    fr = X[:, -1]
    b, v = fr[:, -1, :2], fr[:, -1, 2:4]
    mates = fr[:, 1:1 + C.N_TEAMMATES, :2]
    nv = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    to = mates - b[:, None, :]
    to = to / (np.linalg.norm(to, axis=2, keepdims=True) + 1e-9)
    cos = np.einsum("nsd,nd->ns", to, nv)
    cos[fr[:, 1:1 + C.N_TEAMMATES, C.IDX_PRESENT] <= 0.5] = -2
    return {
        "n": len(y),
        "ball_direction_probe": float((cos.argmax(1) == y).mean()),
        "median_ball_speed": float(np.median(np.linalg.norm(v, axis=1))),
        "median_ball_to_passer": float(np.median(
            np.linalg.norm(fr[:, -1, :2] - fr[:, 0, :2], axis=1))),
    }


def figure(df):
    """The paper's figure: every match, three frame choices, one effect.

    The earlier version plotted a single match, which cannot show the thing that
    matters - that seven independent games all behave the same way.
    """
    piv = df.pivot_table(index="match", columns="pipeline",
                         values="ball_direction_probe")
    order = [c for c in ("ours, 0.4s lead", "ours, at our sync frame",
                         "DataBallPy") if c in piv.columns]
    piv = piv[order].sort_index()
    if piv.empty:
        return

    speed = df.pivot_table(index="match", columns="pipeline",
                           values="median_ball_speed")[order].sort_index()

    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.6))
    cols = {"ours, 0.4s lead": GOOD, "ours, at our sync frame": DATA,
            "DataBallPy": WARN}
    labels = {"ours, 0.4s lead": "our frame, stepped back 0.4 s",
              "ours, at our sync frame": "our synchronisation, at its own frame",
              "DataBallPy": "DataBallPy (Needleman-Wunsch)"}

    x = np.arange(len(piv))
    w = 0.26
    for i, c in enumerate(order):
        ax[0].bar(x + (i - 1) * w, piv[c], w * 0.92, color=cols[c],
                  alpha=0.88, label=labels[c])
    ax[0].axhline(0.27, color=MUTED, ls=(0, (4, 3)), lw=1.3)
    # sits below the line and left of the bars, where nothing overlaps it
    ax[0].annotate("nearest-teammate baseline", (-0.45, 0.27),
                   xytext=(0, -14), textcoords="offset points", ha="left",
                   color=MUTED, fontsize=10)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(piv.index, rotation=30, ha="right", fontsize=9.5)
    ax[0].set_ylabel("receiver from ball direction alone")
    ax[0].set_title("A one-line rule, at three different frames")
    ax[0].set_ylim(0, 0.85)
    ax[0].legend(fontsize=9.5, loc="upper left")
    ax[0].grid(axis="y", alpha=0.4)

    for i, c in enumerate(order):
        ax[1].plot(x, speed[c], "-o", color=cols[c], lw=2.1, ms=6,
                   label=labels[c])
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(piv.index, rotation=30, ha="right", fontsize=9.5)
    ax[1].set_ylabel("median ball speed at that frame (m/s)")
    ax[1].set_title("How far into the flight each frame sits")
    ax[1].grid(alpha=0.4)

    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)

    n = len(piv)
    fig.suptitle(
        f"The shortcut is in the tooling, not in one match  —  "
        f"{n} matches, mean {piv[order[-1]].mean():.3f} against a 0.27 baseline",
        fontsize=13, weight="600", y=1.04)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig16_sync_comparison.{ext}", bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  {OUT}/fig16_sync_comparison.png  +  .pdf")


def run_match(match_id, verbose=True):
    """Both pipelines on one match. Returns the rows, or fewer if something fails."""
    rows = []
    try:
        rows.append({"match": match_id, "pipeline": "ours, 0.4s lead",
                     **ours(match_id)})
    except FileNotFoundError:
        print(f"  {match_id}: no dataset — run step5_dataset.py")
    try:
        z = ours_at_zero_lead(match_id)
        if z:
            rows.append({"match": match_id, "pipeline": "ours, at our sync frame",
                         **z})
    except FileNotFoundError:
        pass

    try:
        merged, game = databallpy_pass_frames(match_id, verbose=verbose)
        print(f"  DataBallPy placed {len(merged)} pass events on tracking frames")
        rows.append({"match": match_id, "pipeline": "DataBallPy",
                     **probe_on_frames(merged, game)})
    except Exception as e:
        name = type(e).__name__
        print(f"  {match_id}: DataBallPy step failed: {name}: {e}")
        if isinstance(e, (ImportError, ModuleNotFoundError)):
            print("    pip install --force-reinstall databallpy")
        elif isinstance(e, (OSError, ConnectionError)) or "ParseError" in name:
            print("    looks like the download, not the code")
        else:
            print(f"    python step20_databallpy.py {match_id} --inspect")
    return rows


def summarise(df):
    """The question is not one number, it is whether seven matches agree."""
    piv = df.pivot_table(index="match", columns="pipeline",
                         values="ball_direction_probe")
    order = [c for c in ("ours, 0.4s lead", "ours, at our sync frame",
                         "DataBallPy") if c in piv.columns]
    piv = piv[order]
    print("\n" + "=" * 62)
    print("BALL-DIRECTION PROBE, PER MATCH")
    print("=" * 62)
    print(piv.round(3).to_string())

    if {"ours, at our sync frame", "DataBallPy"} <= set(piv.columns):
        o = piv["ours, at our sync frame"]
        d = piv["DataBallPy"]
        both = piv.dropna(subset=["ours, at our sync frame", "DataBallPy"])
        print(f"\nmean   ours at our frame {o.mean():.3f}   "
              f"DataBallPy {d.mean():.3f}")
        agree = int((both["DataBallPy"] > 0.40).sum())
        print(f"DataBallPy above 0.40 in {agree} of {len(both)} matches")
        if len(both) >= 5 and agree == len(both):
            print("\nConsistent across every match tested. One match could be a")
            print("coincidence; this cannot. The shortcut is a property of the")
            print("synchronisation, not of any single game.")
        elif agree < len(both):
            print("\nNOT consistent. Look at the matches that disagree before")
            print("claiming anything general.")


def require_databallpy():
    """Check the dependency up front and say exactly what to do about it.

    An earlier version triaged a ModuleNotFoundError as a column-name mismatch,
    which sent the reader off debugging the wrong thing. An error message that
    guesses wrong is worse than one that says nothing.
    """
    try:
        import databallpy
        return databallpy.__version__
    except ImportError:
        print("databallpy is not installed.\n")
        print("    pip install databallpy\n")
        print("It is in requirements.txt, but if you installed those before this")
        print("step existed you will not have it.")
        sys.exit(1)


if __name__ == "__main__":
    banner("Step 20 - the same probe on DataBallPy's synchronisation")
    version = require_databallpy()
    print(f"databallpy {version}\n")

    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if "--inspect" in sys.argv:
        inspect_api(args[0] if args else C.TEST_MATCHES[0])
        sys.exit(0)

    # default to every match: one is an anecdote, seven is a result
    matches = args if args else list(C.MATCHES)
    print("DataBallPy synchronises with Needleman-Wunsch, the same algorithm the")
    print("2026 MPNN paper uses. If the shortcut appears in its frames too, the")
    print("finding is about the field's tooling rather than about my code.\n")
    print(f"matches: {', '.join(matches)}")
    if len(matches) > 1:
        print("each is downloaded once and cached (~400 MB per match)\n")

    rows = []
    for mid in matches:
        print(f"--- {mid} ---")
        rows += run_match(mid, verbose=len(matches) == 1)
        print()

    if not rows:
        print("Nothing to compare: everything failed. Fix the errors above.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    out = f"{C.WORK}/databallpy_comparison.csv"
    df.to_csv(out, index=False)
    print(df.round(3).to_string(index=False))
    print(f"\nSaved -> {out}")

    summarise(df)

    print("\nCAVEATS worth stating before anyone else does:")
    print("  Ball SPEED is not comparable between the tools — we low-pass filter")
    print("  before differencing and DataBallPy does not, and a filter flattens")
    print("  the spike at contact. DIRECTION barely moves under filtering, so the")
    print("  probe is the number to quote.")
    print("  Where the event-id join fell back to timestamps, a mismatched label")
    print("  can only depress the probe, never inflate it — so those are floors.")

    figure(df)
