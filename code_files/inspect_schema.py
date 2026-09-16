"""Extract the exact shape of every table in the dataset.

Answers: how many rows, how many columns, and what are they called - for the
tracking arrays, the event tables, the teamsheets, and the raw XML attributes
that floodlight packs into the `qualifier` column.

Run:  python inspect_schema.py
Out:  work/schema.xlsx  and a printed summary
"""

import numpy as np
import pandas as pd

import config as C
from common import banner, load_match


def tracking_rows(match_id, m):
    rows = []
    for half in m["xy"]:
        for name in m["xy"][half]:
            o = m["xy"][half][name]
            rows.append({
                "match_id": match_id, "half": half, "object": name,
                "rows_frames": len(o), "slots": o.N,
                "columns_x_y": o.xy.shape[1], "framerate": o.framerate,
                "minutes": round(len(o) / o.framerate / 60, 1),
            })
    return rows


def event_rows(match_id, m):
    frames = []
    for half in m["events"]:
        for side in m["events"][half]:
            df = m["events"][half][side].events
            if len(df):
                d = df.copy()
                d["half"] = half
                d["team_slot"] = side
                frames.append(d)
    ev = pd.concat(frames, ignore_index=True)
    uniq = ev.drop_duplicates(subset=["timestamp", "eID", "pID"])
    return ev, uniq


def qualifier_keys(uniq):
    """Every raw XML attribute floodlight put into the qualifier dict, by event type."""
    out = []
    for eid, grp in uniq.groupby("eID"):
        counter = {}
        for q in grp["qualifier"]:
            if isinstance(q, dict):
                for k in q:
                    counter[k] = counter.get(k, 0) + 1
        for k, n in counter.items():
            out.append({"eID": eid, "attribute": k, "n_events_with_it": n,
                        "n_events_of_type": len(grp)})
    return out


if __name__ == "__main__":
    banner("Extracting the schema of every table")

    track, ev_summary, qual, ev_cols, ts_cols = [], [], [], None, None
    all_events = []

    for mid in C.MATCHES:
        m = load_match(mid)
        track += tracking_rows(mid, m)

        ev, uniq = event_rows(mid, m)
        ev_cols = list(ev.columns)
        ev_summary.append({
            "match_id": mid,
            "rows_raw": len(ev),
            "rows_deduplicated": len(uniq),
            "columns": len(ev.columns),
            "distinct_event_types": uniq["eID"].nunique(),
        })
        qual += qualifier_keys(uniq)
        all_events.append(uniq.assign(match_id=mid))

        ts = m["teamsheets"]["Home"].teamsheet
        ts_cols = list(ts.columns)

    track = pd.DataFrame(track)
    ev_summary = pd.DataFrame(ev_summary)
    qual = pd.DataFrame(qual)
    events = pd.concat(all_events, ignore_index=True)

    # ---------------------------------------------------------------- print
    print("\n--- TRACKING ARRAYS ---")
    print(track.to_string(index=False))
    totals = (track[track["object"] == "Home"]
              .groupby("match_id")["rows_frames"].sum())
    print("\nTotal frames per match (Home only, same as the paper):")
    for mid, n in totals.items():
        want = C.EXPECTED_FRAMES[mid]
        print(f"  {mid}: {n:>7,}  paper {want:>7,}  {'OK' if n == want else 'MISMATCH'}")
    print(f"  ALL:  {totals.sum():>7,}  paper 1,002,644")

    print("\nNOTE: floodlight reads only N, X, Y from each <Frame> element, plus")
    print("BallPossession and BallStatus. The Z (ball height), D (distance),")
    print("S (speed) and A (acceleration) attributes in the raw XML are DROPPED.")

    print("\n--- EVENT TABLES ---")
    print(ev_summary.to_string(index=False))
    print(f"\nEvent columns ({len(ev_cols)}): {ev_cols}")
    print(f"Teamsheet columns ({len(ts_cols)}): {ts_cols}")

    print("\n--- EVENT TYPES ACROSS ALL MATCHES ---")
    counts = events["eID"].value_counts()
    print(counts.to_string())
    print(f"\nTotal events: {counts.sum()}  (paper says 11,137)")

    print("\n--- QUALIFIER ATTRIBUTES (the raw XML fields) ---")
    wide = (qual.groupby("attribute")["n_events_with_it"].sum()
            .sort_values(ascending=False))
    print(f"{len(wide)} distinct attributes across all event types:")
    print(wide.to_string())

    print("\n--- IF THE QUALIFIER WERE FULLY EXPANDED ---")
    flat = len(ev_cols) - 1 + len(wide)
    print(f"one event table of {len(events)} rows x {flat} columns")

    # ----------------------------------------------------------------- save
    path = f"{C.WORK}/schema.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        track.to_excel(xl, sheet_name="tracking_shapes", index=False)
        ev_summary.to_excel(xl, sheet_name="event_shapes", index=False)
        counts.rename("count").to_excel(xl, sheet_name="event_types")
        qual.to_excel(xl, sheet_name="qualifier_by_type", index=False)
        pd.DataFrame({"event_columns": ev_cols}).to_excel(
            xl, sheet_name="column_names", index=False)
        pd.DataFrame({"teamsheet_columns": ts_cols}).to_excel(
            xl, sheet_name="column_names", index=False, startcol=2)

    print(f"\nSaved -> {path}")
