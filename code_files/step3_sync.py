"""Step 3 - Synchronise events with tracking.

BEFORE any timing work, the two files must be in the same coordinate frame.
The position file is CENTRE-based: x in (-52.5, 52.5), y in (-34, 34).
The event file is CORNER-based: x in [0, 105], y in [0, 68]. The paper's own
Box 2 shows the kickoff at X-Source-Position="52.50", Y-Source-Position="34.00"
- the centre spot, which is (0, 0) in tracking coords.

Ignoring this produces a constant 62.55 m error on every single event, which no
time shift can fix. floodlight does NOT convert this for you.

Then there are TWO separate timing problems, not one:

  1. A constant offset between the event clock and the position file. The
     dataset authors hardcode 1.6 s in their own visualisation script. We
     estimate it from the data instead of assuming it.
  2. Random error on each individual event, because a human tagged it live.
     The paper reports about +/- 1.8 s.

Stage 1 fixes (1) once per match and half. Stage 2 fixes (2) per event.

Validation: the paper reports the event-to-ball distance as about 9.4 m before
synchronisation and 2.6 m after. If we land near those numbers, the pipeline
is right. This is the best check in the whole project because there is a
published number to compare against.

Run:  python step3_sync.py
Out:  work/passes_synced.csv
"""

import numpy as np
import pandas as pd

import config as C
from common import banner, coords, load_match, velocity


def ball_track(m, half):
    """(T, 2) ball positions and (T, 2) ball velocity for one half."""
    b = coords(m["xy"][half]["Ball"])[:, 0, :]      # only one 'player': the ball
    return b, velocity(b[:, None, :])[:, 0, :]


def coord_variants(ex, ey, pitch):
    """Move event coordinates into the tracking frame.

    Adding the lower bound of the pitch maps [0, 105] onto (-52.5, 52.5).
    We test all four orientations instead of assuming one, because providers
    disagree about which way y increases and a silent flip would be invisible
    in the final accuracy while quietly ruining every feature.
    """
    sx = ex + pitch.xlim[0]
    sy = ey + pitch.ylim[0]
    return {
        "shift only":       (sx, sy),
        "shift + flip y":   (sx, -sy),
        "shift + flip x":   (-sx, sy),
        "shift + flip x,y": (-sx, -sy),
    }


def pick_orientation(ball, naive, ex, ey, pitch, verbose=True):
    """Whichever orientation puts the ball closest to the event wins."""
    scores = {}
    for name, (cx, cy) in coord_variants(ex, ey, pitch).items():
        scores[name] = np.nanmedian(event_to_ball_distance(ball, naive, cx, cy))
    best = min(scores, key=scores.get)
    if verbose:
        detail = "  ".join(f"{k}={v:.1f}" for k, v in scores.items())
        print(f"    orientation: {detail}   -> {best}")
    return best, coord_variants(ex, ey, pitch)[best]


def event_to_ball_distance(ball, frames, ex, ey):
    """Distance between where the event says it happened and where the ball is."""
    ok = (frames >= 0) & (frames < len(ball))
    d = np.full(len(frames), np.nan)
    bx = ball[np.clip(frames, 0, len(ball) - 1)]
    d[ok] = np.hypot(bx[ok, 0] - ex[ok], bx[ok, 1] - ey[ok])
    return d


# ------------------------------------------------------------------ stage 1
def constant_offset(ball, naive, ex, ey):
    """Find the single frame shift that best aligns all events in this half.

    We minimise the MEDIAN distance, not the mean, so a handful of badly
    tagged events cannot drag the estimate around.
    """
    span = int(C.SYNC_GLOBAL_RANGE_S * C.FRAMERATE)
    shifts = np.arange(-span, span + 1)
    scores = [np.nanmedian(event_to_ball_distance(ball, naive + s, ex, ey))
              for s in shifts]
    best = shifts[int(np.nanargmin(scores))]
    return int(best), float(np.nanmin(scores))


# ------------------------------------------------------------------ stage 2
def refine_one(ball, bvel, start, passer_xy, ev_x, ev_y):
    """Pick the frame that looks most like the moment of a kick.

    Five signals, matching the terms the paper describes for DataBallPy. All in
    metres or metres per second so the weights stay readable:
      e  ball is close to where the event says it happened -> want small
      a  ball is close to the passer                       -> want small
      b  ball moves away over the next 0.4 s               -> want large
      c  ball speeds up around this frame                  -> want large
      d  stay near the tagged timestamp                    -> penalise drifting

    Term `e` matters and was missing in the first version. Without it the search
    optimises ball-to-passer while the reported metric is ball-to-event, so it
    aims at a different target than the one being measured, and a third of the
    passes came out worse than before.

    Note the circularity this creates: `e` is also the number we report. So we
    additionally report ball-to-passer distance, which is independent of the
    human annotation and is the physically meaningful check - at the instant of
    a kick the ball is at the passer's foot.
    """
    span = int(C.SYNC_LOCAL_RANGE_S * C.FRAMERATE)
    lo, hi = max(0, start - span), min(len(ball) - 1, start + span)
    if hi - lo < 10:
        return start

    ahead = int(0.4 * C.FRAMERATE)
    best, best_score = start, -np.inf

    for f in range(lo, hi + 1):
        p = passer_xy[f]
        if np.isnan(p[0]):
            continue
        a = np.hypot(ball[f, 0] - p[0], ball[f, 1] - p[1])
        e = np.hypot(ball[f, 0] - ev_x, ball[f, 1] - ev_y)

        g = min(f + ahead, len(ball) - 1)
        pg = passer_xy[g]
        b = 0.0
        if not np.isnan(pg[0]):
            b = np.hypot(ball[g, 0] - pg[0], ball[g, 1] - pg[1]) - a

        s_before = np.linalg.norm(bvel[max(f - 5, 0)])
        s_after = np.linalg.norm(bvel[min(f + 5, len(ball) - 1)])
        c = s_after - s_before

        d = abs(f - start) / C.FRAMERATE

        score = -e / 5.0 - a / 5.0 + b / 5.0 + c / 5.0 - d
        if score > best_score:
            best, best_score = f, score

    return best


def ball_to_passer(ball, frames, team_xy, slots):
    """Independent check: at the moment of a kick the ball is at the passer's foot.

    This uses no event annotation at all, so unlike dist_after it cannot be
    gamed by the search that produced the frames.
    """
    out = np.full(len(frames), np.nan)
    for i, (f, s) in enumerate(zip(frames, slots)):
        if 0 <= f < len(ball):
            p = team_xy[f, int(s)]
            if not np.isnan(p[0]):
                out[i] = np.hypot(ball[f, 0] - p[0], ball[f, 1] - p[1])
    return out


def sync_match(passes, match_id):
    m = load_match(match_id)
    out = []

    for half in ("firstHalf", "secondHalf"):
        sub = passes[(passes["match_id"] == match_id) & (passes["half"] == half)].copy()
        if sub.empty:
            continue

        ball, bvel = ball_track(m, half)
        naive = np.floor(sub["gameclock"].values * C.FRAMERATE).astype(int)

        # STEP 0: get both files into the same coordinate frame
        orient, (ex, ey) = pick_orientation(
            ball, naive, sub["at_x"].values, sub["at_y"].values, m["pitch"])
        sub["orientation"] = orient
        sub["ev_x"], sub["ev_y"] = ex, ey

        d_before = event_to_ball_distance(ball, naive, ex, ey)
        shift, d_const = constant_offset(ball, naive, ex, ey)

        home = coords(m["xy"][half]["Home"])
        away = coords(m["xy"][half]["Away"])

        refined = []
        for i, (_, r) in enumerate(sub.iterrows()):
            team = home if r["team_slot"] == "Home" else away
            passer_xy = team[:, int(r["passer_slot"]), :]
            start = int(np.clip(naive[i] + shift, 0, len(ball) - 1))
            refined.append(refine_one(ball, bvel, start, passer_xy, ex[i], ey[i]))
        refined = np.array(refined)

        d_after = event_to_ball_distance(ball, refined, ex, ey)

        # independent check, per team
        bp = np.full(len(sub), np.nan)
        for side, team_arr in (("Home", home), ("Away", away)):
            mask = (sub["team_slot"] == side).values
            if mask.any():
                bp[mask] = ball_to_passer(ball, refined[mask], team_arr,
                                          sub.loc[mask, "passer_slot"].values)
        sub["ball_to_passer"] = bp
        sub["sync_frame"] = refined
        sub["naive_frame"] = naive
        sub["dist_before"] = d_before
        sub["dist_after"] = d_after
        out.append(sub)

        print(f"{match_id} {half:11s} n={len(sub):4d}  "
              f"shift {shift / C.FRAMERATE:+.2f} s  "
              f"event-ball {np.nanmedian(d_before):5.2f} -> {np.nanmedian(d_after):5.2f} m  "
              f"ball-passer {np.nanmedian(bp):5.2f} m")

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


if __name__ == "__main__":
    banner("Step 3 - synchronising events with tracking")
    passes = pd.read_csv(f"{C.WORK}/passes.csv")

    synced = pd.concat([sync_match(passes, mid) for mid in C.MATCHES],
                       ignore_index=True)

    out = f"{C.WORK}/passes_synced.csv"
    synced.to_csv(out, index=False)

    b, a, bp = synced["dist_before"], synced["dist_after"], synced["ball_to_passer"]
    print(f"\nEvent-to-ball  before: mean {b.mean():5.2f}  median {b.median():5.2f}")
    print(f"Event-to-ball  after : mean {a.mean():5.2f}  median {a.median():5.2f}")
    print("Paper: 9.37 +/- 8.39 before, 2.61 +/- 3.60 after")
    print(f"improved on {100*(a<b).mean():.1f}% of passes")
    print("\nBall-to-passer at the chosen frame (independent of the annotation):")
    print(f"  median {bp.median():.2f} m   under 2 m: {100*(bp<2).mean():.1f}%")
    print("  At a real kick the ball is at the passer's foot, so this should be small.")
    print("\nA long tail is expected and the paper says so: events far from the ball")
    print("come from mis-tagged data, e.g. the wrong player credited with the pass.")
    print("Report the median as well as the mean.")
    print(f"\nSaved -> {out}")
