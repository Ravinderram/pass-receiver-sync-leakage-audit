"""Self-test: fabricate fake matches and run the whole pipeline on them.

This does NOT validate the football. It validates the plumbing: array shapes,
slot mapping, index arithmetic, the leakage guard, and that every step runs.
Delete this file once the real data is flowing.

Run:  python _selftest.py
"""

import os
import pickle
import shutil

import numpy as np
import pandas as pd
from floodlight.core.code import Code
from floodlight.core.events import Events
from floodlight.core.pitch import Pitch
from floodlight.core.teamsheet import Teamsheet
from floodlight.core.xy import XY

import config as C

FRAMES = 6000          # 4 minutes per half, enough to exercise everything
SQUAD = 14             # 11 on the pitch + 3 unused subs (NaN throughout)
PASSES_PER_HALF = 90
TRUE_SHIFT = 40        # frames of constant offset we hide in the data (1.6 s)


def fake_half(rng, home_attacks_positive):
    """Make plausible player and ball tracks for one half."""
    t = np.arange(FRAMES)

    def team(defends_sign):
        pos = np.full((FRAMES, SQUAD, 2), np.nan)
        for p in range(11):
            if p == 0:                       # goalkeeper, pinned near his goal
                base_x = defends_sign * 45.0
                base_y = 0.0
                amp = 2.0
            else:
                base_x = defends_sign * rng.uniform(-5, 35)
                base_y = rng.uniform(-28, 28)
                amp = 8.0
            pos[:, p, 0] = base_x + amp * np.sin(t / 400 + p)
            pos[:, p, 1] = base_y + amp * np.cos(t / 350 + p)
        # a late substitute who plays only the last few minutes, parked wide.
        # He has a huge |mean x| over the frames he has, which is exactly how a
        # naive "furthest from centre" rule picks the wrong player.
        late = FRAMES - 600
        pos[late:, 11, 0] = -defends_sign * 49.0
        pos[late:, 11, 1] = 30.0
        return pos

    home_def = -1.0 if home_attacks_positive else 1.0
    home = team(home_def)
    away = team(-home_def)

    ball = np.zeros((FRAMES, 1, 2))
    ball[:, 0, 0] = 20 * np.sin(t / 120)
    ball[:, 0, 1] = 15 * np.cos(t / 90)
    return home, away, ball


def make_events(rng, half, home, away, ball, teamsheet_pids):
    """Passes whose gameclock is deliberately offset from the tracking frame.

    The receiver is NOT random. It is drawn from a softmax over a score built
    from geometry: close teammates are preferred, well-covered ones are avoided,
    forward options get a small bonus. Without learnable structure the self-test
    can only tell us that code runs, not whether a change made the model better.
    """
    rows = []
    frames = rng.choice(np.arange(200, FRAMES - 200), PASSES_PER_HALF, replace=False)
    for f in sorted(frames):
        # the passer is whoever is nearest the ball at that frame, so the fake
        # data reproduces the real physics: at a kick the ball is at his foot
        d = np.linalg.norm(home[f, :11, :] - ball[f, 0, :], axis=1)
        if np.isnan(d).all():
            continue
        passer = int(np.nanargmin(d))

        # only players actually on the pitch can receive; substitutes and a
        # sent-off player are NaN, exactly as in the real tracking data
        cands = [p for p in range(11)
                 if p != passer and not np.isnan(home[f, p, 0])]
        opp_now = away[f, :11]
        opp_now = opp_now[~np.isnan(opp_now[:, 0])]
        if len(cands) < 2 or len(opp_now) < 2:
            continue
        pxy = home[f, passer]
        opp = opp_now
        score = []
        for p in cands:
            q = home[f, p]
            dist = np.linalg.norm(q - pxy)
            open_ = np.linalg.norm(opp - q, axis=1).min()      # space around him
            fwd = q[0] - pxy[0]
            score.append(-dist / 12.0 + open_ / 8.0 + fwd / 25.0)
        score = np.array(score)
        prob = np.exp(score - score.max())
        prob /= prob.sum()
        recip = int(rng.choice(cands, p=prob))
        # the event's own location is roughly the ball, with annotation noise
        rows.append({
            "eID": "Play_Pass",
            # naive frame = gameclock * 25, which is TRUE_SHIFT frames too early
            "gameclock": (f - TRUE_SHIFT) / C.FRAMERATE,
            "timestamp": pd.Timestamp("2023-05-27") + pd.Timedelta(seconds=float(f) / 25),
            "minute": 0.0, "second": 0.0,
            # events are CORNER-based (0..105, 0..68) while tracking is
            # centre-based - exactly like the real DFL files
            "at_x": ball[f, 0, 0] + 52.5 + rng.normal(0, 1.5),
            "at_y": ball[f, 0, 1] + 34.0 + rng.normal(0, 1.5),
            "to_x": np.nan, "to_y": np.nan,
            "qualifier": {"Recipient": teamsheet_pids[recip],
                          "Evaluation": C.SUCCESS_LABEL,
                          "PlayAngle": "123.4", "Distance": "medium"},
            "outcome": 1,
            "tID": "T_HOME",
            "pID": teamsheet_pids[passer],
        })
    # a few junk rows the filter must remove
    for _ in range(20):
        rows.append({
            "eID": "Delete", "gameclock": 10.0,
            "timestamp": pd.Timestamp("2023-05-27"), "minute": 0.0, "second": 0.0,
            "at_x": np.nan, "at_y": np.nan, "to_x": np.nan, "to_y": np.nan,
            "qualifier": {}, "outcome": np.nan, "tID": "T_HOME", "pID": None,
        })
    return Events(events=pd.DataFrame(rows))


def build_match(match_id, seed):
    rng = np.random.default_rng(seed)
    pids = {side: [f"P_{side}_{i:02d}" for i in range(SQUAD)] for side in ("Home", "Away")}

    teamsheets = {}
    for side in ("Home", "Away"):
        teamsheets[side] = Teamsheet(teamsheet=pd.DataFrame({
            "player": pids[side],
            "position": ["TW"] + ["IV"] * 10 + ["TW"] + ["ST"] * (SQUAD - 12),
            "pID": pids[side],
            "jID": list(range(1, SQUAD + 1)),
            "tID": [f"T_{side.upper()}"] * SQUAD,
            "xID": list(range(SQUAD)),
        }))

    xy, events, possession, ballstatus = {}, {}, {}, {}
    for hi, half in enumerate(("firstHalf", "secondHalf")):
        # teams swap ends at half time - step 4 must detect this
        home_pos, away_pos, ball_pos = fake_half(rng, home_attacks_positive=(hi == 0))
        if match_id == "J03WN1":
            # reproduce the real thing: Leverkusen went down to ten in minute 8,
            # which cost 94% of this match before padding was added
            away_pos[int(0.08 * FRAMES):, 7, :] = np.nan
        xy[half] = {
            "Home": XY(xy=home_pos.reshape(FRAMES, -1), framerate=C.FRAMERATE),
            "Away": XY(xy=away_pos.reshape(FRAMES, -1), framerate=C.FRAMERATE),
            "Ball": XY(xy=ball_pos.reshape(FRAMES, -1), framerate=C.FRAMERATE),
        }
        events[half] = {
            "Home": make_events(rng, half, home_pos, away_pos, ball_pos, pids["Home"]),
            "Away": Events(events=pd.DataFrame(columns=[
                "eID", "gameclock", "timestamp", "minute", "second",
                "at_x", "at_y", "to_x", "to_y", "qualifier", "outcome", "tID", "pID"])),
        }
        possession[half] = Code(code=np.ones(FRAMES), name="possession",
                                definitions={1: "Home", 2: "Away"}, framerate=C.FRAMERATE)
        ballstatus[half] = Code(code=np.ones(FRAMES), name="ballstatus",
                                definitions={0: "dead", 1: "alive"}, framerate=C.FRAMERATE)

    return {
        "match_id": match_id, "events": events, "xy": xy,
        "possession": possession, "ballstatus": ballstatus,
        "teamsheets": teamsheets,
        "pitch": Pitch(xlim=(-52.5, 52.5), ylim=(-34, 34), unit="m",
                       boundaries="flexible", length=105, width=68, sport="football"),
    }


if __name__ == "__main__":
    if os.path.isdir(C.WORK):
        shutil.rmtree(C.WORK)
    os.makedirs(C.WORK, exist_ok=True)

    print("fabricating fake matches...")
    for i, mid in enumerate(C.MATCHES):
        with open(f"{C.WORK}/{mid}_parsed.pkl", "wb") as f:
            pickle.dump(build_match(mid, seed=i), f)
    print(f"  wrote {len(C.MATCHES)} fake matches to {C.WORK}/")
    print(f"  hidden constant offset: {TRUE_SHIFT} frames "
          f"= {TRUE_SHIFT / C.FRAMERATE:.2f} s")
    print("  event coords are corner-based, tracking is centre-based")
    print("\nStep 3 must pick 'shift only', recover roughly that offset, and")
    print("step 4 must report the attacking direction flipping between halves.\n")
