"""Export the EVENT data to Excel. Not the tracking data - see the note below.

The event data is small: 11,137 rows across all seven matches, about 1% of
Excel's row limit. Opening it in a spreadsheet is genuinely useful for seeing
what event types exist and what the qualifier attributes look like.

The tracking data is a different story:
  - long format  (one row per object per frame) = 41 million rows = 39x the limit
  - wide format  (one row per frame, 205 columns) technically fits at 96% of the
    limit, but that is 205 million cells. Excel will not open it usefully.
  - and every operation this project needs is array maths over time windows,
    which is exactly what a spreadsheet is worst at.

The tracking data already lives in a sensible format: step4_clean.py writes it
as compressed .npz, roughly 330 MB of actual numbers instead of 2.5 GB of XML.

Run:  python export_events_excel.py
Out:  work/events.xlsx
"""

import pandas as pd

import config as C
from common import banner, load_match

MAX_QUALIFIER_COLS = 40


def events_table(match_id):
    m = load_match(match_id)
    rows = []
    for half in m["events"]:
        for side in m["events"][half]:
            df = m["events"][half][side].events.copy()
            if not len(df):
                continue
            df["half"] = half
            df["team_slot"] = side
            rows.append(df)
    ev = pd.concat(rows, ignore_index=True)
    ev = ev.drop_duplicates(subset=["timestamp", "eID", "pID"])

    # the qualifier column holds a dict of the raw XML attributes. Excel cannot
    # show a dict, so spread the attributes into their own columns.
    q = pd.json_normalize(ev["qualifier"].apply(
        lambda d: d if isinstance(d, dict) else {}
    ))
    keep = q.notna().sum().sort_values(ascending=False).head(MAX_QUALIFIER_COLS).index
    q = q[keep]

    ev = ev.drop(columns=["qualifier"]).reset_index(drop=True)
    out = pd.concat([ev, q.reset_index(drop=True)], axis=1)
    out.insert(0, "match_id", match_id)
    return out


if __name__ == "__main__":
    banner("Exporting event data to Excel")

    tables = {mid: events_table(mid) for mid in C.MATCHES}
    everything = pd.concat(tables.values(), ignore_index=True)

    path = f"{C.WORK}/events.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        # one sheet with everything, plus a summary of what is in there
        everything.to_excel(xl, sheet_name="all_events", index=False)

        summary = (everything.groupby(["match_id", "eID"]).size()
                   .unstack(fill_value=0).T)
        summary["TOTAL"] = summary.sum(axis=1)
        summary.sort_values("TOTAL", ascending=False).to_excel(xl, sheet_name="event_counts")

        for mid, df in tables.items():
            df.to_excel(xl, sheet_name=mid, index=False)

    print(f"rows: {len(everything):,}  columns: {len(everything.columns)}")
    print("sheets: all_events, event_counts, and one per match")
    print(f"\nSaved -> {path}")
    print("\nCheck the event_counts sheet against the paper: Play_Pass should")
    print("total 5,241 and Play_Cross 140 across all seven matches.")
