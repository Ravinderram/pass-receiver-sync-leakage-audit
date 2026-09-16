"""Write a tiny DFL-format match to disk, so the LOCAL loader can be tested.

This is not the dataset and not a scientific fixture. It is three small XML files
in the exact format the figshare release uses, so that
`common.load_match()`, `step0_inspect_local_data.py` and
`verify/data_provenance.py` can be exercised against floodlight's real parsers
without any network access and without the real data.

The frame counts are deliberately small, so `verify/data_provenance.py` will
correctly refuse this as a fixture. That refusal is the point of the test.

Run:  python _selftest_local_xml.py [OUTDIR] [MATCH_ID ...]
"""

import os
import random
import sys
from datetime import datetime, timedelta

FRAMERATE = 25
N_PLAYERS = 14              # 11 on the pitch plus substitutes
POSITIONS = ["TW", "IV", "IV", "LV", "RV", "DMI", "ZM", "ZM", "OLM", "ORM", "STZ",
             "IV", "ZM", "STZ"]


def _players(team_id, offset):
    out = []
    for i in range(N_PLAYERS):
        out.append(
            f'      <Player PersonId="DFL-OBJ-{offset + i:06d}" '
            f'FirstName="F{i}" LastName="L{i}" Shortname="P{offset + i}" '
            f'ShirtNumber="{i + 1}" PlayingPosition="{POSITIONS[i]}" '
            f'Starting="{"true" if i < 11 else "false"}" TeamId="{team_id}"/>')
    return "\n".join(out)


def match_information(match_id, home, away):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<matchInformation>
  <MatchInformation>
    <Environment PitchX="105.0" PitchY="68.0" PitchCircumference="0"/>
    <General MatchId="DFL-MAT-{match_id}" MatchTitle="{home} vs {away}"
             HomeTeamId="DFL-CLU-000001" HomeTeamName="{home}"
             AwayTeamId="DFL-CLU-000002" AwayTeamName="{away}"
             Season="2022/2023" MatchDay="1"/>
    <Teams>
      <Team TeamId="DFL-CLU-000001" TeamName="{home}" Role="home">
        <Players>
{_players("DFL-CLU-000001", 1000)}
        </Players>
      </Team>
      <Team TeamId="DFL-CLU-000002" TeamName="{away}" Role="guest">
        <Players>
{_players("DFL-CLU-000002", 2000)}
        </Players>
      </Team>
    </Teams>
  </MatchInformation>
</matchInformation>
"""


def simulate(n_frames, n_passes, seed=0):
    """A small consistent match: player tracks, a ball that is passed, and the
    pass events that describe it.

    Consistency is the point. The ball sits at the passer at the pass frame and
    then travels to the receiver, and each event's coordinates are the passer's
    own tracking position. Without that, synchronisation has nothing to lock on
    to and step3 fails, which is a property of the fixture rather than of the
    pipeline.
    """
    import math
    rng = random.Random(seed)
    base = {}
    for team, sign in (("home", -1.0), ("away", 1.0)):
        for i in range(N_PLAYERS):
            row, col = divmod(i, 4)
            base[(team, i)] = (sign * (8 + row * 12) + rng.uniform(-2, 2),
                               -24 + col * 16 + rng.uniform(-2, 2))

    tracks = {k: [] for k in base}
    for k, (bx, by) in base.items():
        phase = rng.uniform(0, 6.28)
        amp = rng.uniform(1.5, 4.0)
        for f in range(n_frames):
            t = f / FRAMERATE
            tracks[k].append((bx + amp * math.sin(0.35 * t + phase),
                              by + amp * math.cos(0.25 * t + phase)))

    # pass schedule: every `gap` frames, alternating teams
    gap = max(40, (n_frames - 120) // max(n_passes, 1))
    passes, ball = [], [None] * n_frames
    for k in range(n_passes):
        f = 60 + k * gap
        if f + 30 >= n_frames:
            break
        team = "home" if k % 2 == 0 else "away"
        passer = k % 11
        recip = (k + 3) % 11
        if recip == passer:
            recip = (recip + 1) % 11
        passes.append({"frame": f, "team": team, "passer": passer,
                       "recipient": recip})

    # ball: at the passer on the pass frame, then flying to the receiver
    for idx, p in enumerate(passes):
        f = p["frame"]
        src = tracks[(p["team"], p["passer"])][f]
        flight = 20
        dst_f = min(f + flight, n_frames - 1)
        dst = tracks[(p["team"], p["recipient"])][dst_f]
        for j in range(flight + 1):
            if f + j >= n_frames:
                break
            a = j / flight
            ball[f + j] = (src[0] + a * (dst[0] - src[0]),
                           src[1] + a * (dst[1] - src[1]))
        # between passes the ball is carried from the receiver to whoever plays
        # the next pass, so there is no teleport for the synchroniser to lock on
        nxt_i = idx + 1
        nxt = passes[nxt_i]["frame"] if nxt_i < len(passes) else n_frames
        carry_to = (tracks[(passes[nxt_i]["team"], passes[nxt_i]["passer"])]
                    if nxt_i < len(passes) else None)
        span = max(nxt - (f + flight + 1), 1)
        for g in range(f + flight + 1, min(nxt, n_frames)):
            here = tracks[(p["team"], p["recipient"])][g]
            if carry_to is None:
                ball[g] = here
            else:
                a = (g - (f + flight + 1)) / span
                there = carry_to[g]
                ball[g] = (here[0] + a * (there[0] - here[0]),
                           here[1] + a * (there[1] - here[1]))
    last = (0.0, 0.0)
    for f in range(n_frames):
        if ball[f] is None:
            ball[f] = last
        else:
            last = ball[f]
    return tracks, ball, passes


def _frame_rows(coords, start_n, t0, ball=False):
    rows = []
    for i, (x, y) in enumerate(coords):
        t = (t0 + timedelta(seconds=i / FRAMERATE)).strftime(
            "%Y-%m-%dT%H:%M:%S.%f")[:-3]
        extra = (' Z="0.10" BallStatus="1" BallPossession="1"' if ball
                 else ' S="3.0" D="0.0" A="0.0"')
        rows.append(f'      <Frame N="{start_n + i}" T="{t}+01:00" '
                    f'X="{x:.3f}" Y="{y:.3f}"{extra}/>')
    return "\n".join(rows)


def positions(match_id, sims):
    parts = ['<?xml version="1.0" encoding="UTF-8"?>', "<Positions>"]
    base = datetime(2023, 4, 1, 15, 30, 0)
    for half, (section, hoff) in enumerate((("firstHalf", 0), ("secondHalf", 3600))):
        tracks, ball, _ = sims[section]
        t0 = base + timedelta(seconds=hoff)
        start_n = 10000 + half * 100000
        parts.append(f'  <FrameSet MatchId="DFL-MAT-{match_id}" TeamId="BALL" '
                     f'GameSection="{section}" PersonId="DFL-OBJ-BALL">')
        parts.append(_frame_rows(ball, start_n, t0, ball=True))
        parts.append("  </FrameSet>")
        for team, team_id, offset in (("home", "DFL-CLU-000001", 1000),
                                      ("away", "DFL-CLU-000002", 2000)):
            # only the 11 starters are tracked; the substitutes appear on the
            # teamsheet but have no FrameSet, exactly as in the real release
            for i in range(11):
                parts.append(
                    f'  <FrameSet MatchId="DFL-MAT-{match_id}" TeamId="{team_id}" '
                    f'GameSection="{section}" PersonId="DFL-OBJ-{offset + i:06d}">')
                parts.append(_frame_rows(tracks[(team, i)], start_n, t0))
                parts.append("  </FrameSet>")
    parts.append("</Positions>")
    return "\n".join(parts)


def events(match_id, sims):
    """Pass events whose coordinates are the passer's own tracking position.

    Coordinates are written corner-based (0..105, 0..68) because that is what the
    DFL event feed uses, while tracking is centre-based.
    """
    base = datetime(2023, 4, 1, 15, 30, 0)
    parts = ['<?xml version="1.0" encoding="UTF-8"?>', "<Events>"]
    for section, hoff in (("firstHalf", 0), ("secondHalf", 3600)):
        tracks, _, passes = sims[section]
        t0 = base + timedelta(seconds=hoff)
        parts.append(f'  <Event EventId="k-{section}" EventTime='
                     f'"{t0.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]}+01:00">'
                     f'<KickoffWhistle GameSection="{section}"/></Event>')
        for i, p in enumerate(passes):
            t = t0 + timedelta(seconds=p["frame"] / FRAMERATE)
            team_id, offset = (("DFL-CLU-000001", 1000) if p["team"] == "home"
                               else ("DFL-CLU-000002", 2000))
            x, y = tracks[(p["team"], p["passer"])][p["frame"]]
            tx, ty = tracks[(p["team"], p["recipient"])][
                min(p["frame"] + 20, len(tracks[(p["team"], p["recipient"])]) - 1)]
            eid = "Play_Cross" if i % 12 == 11 else "Play_Pass"
            tag = "Cross" if eid == "Play_Cross" else "Pass"
            parts.append(
                f'  <Event EventId="e-{section}-{i}" '
                f'EventTime="{t.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]}+01:00" '
                f'X-Source-Position="{x + 52.5:.2f}" '
                f'Y-Source-Position="{y + 34.0:.2f}" '
                f'X-Position="{tx + 52.5:.2f}" Y-Position="{ty + 34.0:.2f}" '
                f'Team="{team_id}" Player="DFL-OBJ-{offset + p["passer"]:06d}">'
                f'<Play Evaluation="successfullyCompleted" '
                f'Recipient="DFL-OBJ-{offset + p["recipient"]:06d}" '
                f'Team="{team_id}" Player="DFL-OBJ-{offset + p["passer"]:06d}">'
                f'<{tag}/></Play></Event>')
        te = t0 + timedelta(seconds=(passes[-1]["frame"] + 40) / FRAMERATE)
        parts.append(f'  <Event EventId="f-{section}" EventTime='
                     f'"{te.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]}+01:00">'
                     f'<FinalWhistle GameSection="{section}"/></Event>')
    parts.append("</Events>")
    return "\n".join(parts)


def write_match(outdir, match_id, home="Test Home", away="Test Away",
                n_frames_per_half=4000, n_passes=70):
    os.makedirs(outdir, exist_ok=True)
    seed = abs(hash(match_id)) % 10000
    sims = {"firstHalf": simulate(n_frames_per_half, n_passes, seed),
            "secondHalf": simulate(n_frames_per_half, n_passes, seed + 1)}
    stem = f"DFL-COM-000002_DFL-MAT-{match_id}"
    files = {
        f"DFL_02_01_matchinformation_{stem}.xml": match_information(match_id, home, away),
        f"DFL_03_02_events_raw_{stem}.xml": events(match_id, sims),
        f"DFL_04_03_positions_raw_observed_{stem}.xml": positions(match_id, sims),
    }
    for name, text in files.items():
        with open(os.path.join(outdir, name), "w") as fh:
            fh.write(text)
    return list(files)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/idsse_local_fixture"
    ids = sys.argv[2:] or ["J03WMX"]
    for mid in ids:
        names = write_match(out, mid)
        print(f"{mid}: wrote {len(names)} files to {out}")
    print("\nThis is a FORMAT fixture, not the dataset. verify/data_provenance.py")
    print("must refuse it; that refusal is what the test checks.")
