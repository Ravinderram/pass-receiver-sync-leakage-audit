"""Step 2 - Find the passes.

Turns the nested events dictionary into one table of open-play passes with
the passer's slot and the receiver's slot in the tracking array.

Run:  python step2_passes.py
Out:  work/passes.csv
"""

import pandas as pd

import config as C
from common import banner, load_match


# per-match stage counts, written to work/event_counts.csv so the sample flow in
# the paper is counted from the data rather than typed in
COUNTS = []


def get_q(d, key):
    """Pull one attribute out of the qualifier dict floodlight builds from the XML."""
    return d.get(key) if isinstance(d, dict) else None


def passes_for_match(match_id):
    m = load_match(match_id)
    events, teamsheets = m["events"], m["teamsheets"]

    # pID -> slot index in the XY array. The slot order follows the teamsheet,
    # NOT the playing position, so this mapping must come from the teamsheet.
    # Guessing it would silently corrupt every feature downstream.
    links = {}
    for side in ("Home", "Away"):
        teamsheets[side].add_xIDs()
        links[side] = teamsheets[side].get_links("pID", "xID")

    rows = []
    for half in events:
        for side in events[half]:
            df = events[half][side].events.copy()
            df["half"] = half
            df["team_slot"] = side
            if len(df):                      # skip empty frames (pandas warns on those)
                rows.append(df)
    ev = pd.concat(rows, ignore_index=True)

    # floodlight gives team-less events (whistles etc.) to BOTH teams, so the
    # raw table double-counts them. Play events carry a team, but dedupe anyway.
    ev = ev.drop_duplicates(subset=["timestamp", "eID", "pID"])

    ev["recipient"] = ev["qualifier"].apply(lambda d: get_q(d, "Recipient"))
    ev["evaluation"] = ev["qualifier"].apply(lambda d: get_q(d, "Evaluation"))
    # the DFL EventId, kept so step 20 can join our passes to another tool's
    # synchronisation exactly rather than by matching timestamps
    ev["event_id"] = ev["qualifier"].apply(lambda d: get_q(d, "EventId"))

    n_all = len(ev)
    n_delete = int((ev["eID"] == "Delete").sum())

    # open play only
    p = ev[ev["eID"].isin(C.OPEN_PLAY_EIDS)].copy()
    n_open = len(p)

    # successful only -> the receiver is guaranteed to be a teammate.
    # Recipient records who ACTUALLY got the ball, so on a failed pass it can be
    # an opponent. Those become the stretch goal (RQ4), not the main experiment.
    p = p[p["evaluation"] == C.SUCCESS_LABEL]
    n_success = len(p)

    p = p.dropna(subset=["pID", "recipient"])

    # map both players to slots, and drop passes where the recipient is not in
    # the passer's own team (interceptions that were still tagged successful)
    def slot(row, col):
        return links[row["team_slot"]].get(row[col])

    p["passer_slot"] = p.apply(lambda r: slot(r, "pID"), axis=1)
    p["recipient_slot"] = p.apply(lambda r: slot(r, "recipient"), axis=1)
    n_before = len(p)
    p = p.dropna(subset=["passer_slot", "recipient_slot"])
    p = p[p["passer_slot"] != p["recipient_slot"]]

    p["match_id"] = match_id
    p["passer_slot"] = p["passer_slot"].astype(int)
    p["recipient_slot"] = p["recipient_slot"].astype(int)

    print(f"{match_id}: {n_all:5d} events ({n_delete} Delete) -> "
          f"{n_open:4d} open play -> {n_success:4d} successful -> "
          f"{len(p):4d} usable  (dropped {n_before - len(p)} non-teammate/unmapped)")
    COUNTS.append({"match_id": match_id, "events_all": int(n_all),
                   "events_delete": int(n_delete),
                   "events_excluding_delete": int(n_all - n_delete),
                   "open_play_passes": int(n_open),
                   "successful_passes": int(n_success),
                   "with_passer_and_recipient": int(n_before),
                   "mapped_teammate_passes": int(len(p))})

    keep = ["match_id", "half", "team_slot", "eID", "gameclock", "timestamp",
            "at_x", "at_y", "passer_slot", "recipient_slot", "pID", "recipient",
            "event_id"]
    return p[keep].reset_index(drop=True)


if __name__ == "__main__":
    banner("Step 2 - extracting passes")
    all_passes = pd.concat([passes_for_match(mid) for mid in C.MATCHES],
                           ignore_index=True)

    out = f"{C.WORK}/passes.csv"
    all_passes.to_csv(out, index=False)
    pd.DataFrame(COUNTS).to_csv(f"{C.WORK}/event_counts.csv", index=False)
    print(f"Saved -> {C.WORK}/event_counts.csv")

    print(f"\nTotal usable passes: {len(all_passes)}")
    print(f"Per match average  : {len(all_passes) / len(C.MATCHES):.0f}")
    print("\nExpected from the paper's own notebook: 5,241 Play_Pass + 140 Play_Cross")
    print("= 5,381 open play across 7 matches (about 769 per match).")
    print("After keeping successful passes only, expect roughly 600-650 per match.")
    print(f"\nSaved -> {out}")
