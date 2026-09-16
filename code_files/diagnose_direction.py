"""Why did the attacking-direction check fail?

Dumps, per team and half, every slot's mean x, how many frames it has, and what
the teamsheet says its position is. The goalkeeper should be the slot with the
most extreme mean x. If some other slot beats him, this shows which one and why.

Run:  python diagnose_direction.py J03WR9
"""

import sys
import warnings

import numpy as np
import pandas as pd

import config as C
from common import banner, coords, goalkeeper_slot, load_match

match_id = sys.argv[1] if len(sys.argv) > 1 else "J03WR9"
m = load_match(match_id)

banner(f"Direction diagnostic for {match_id}")

for half in ("firstHalf", "secondHalf"):
    for side in ("Home", "Away"):
        c = coords(m["xy"][half][side])
        sheet = m["teamsheets"][side].teamsheet

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_x = np.nanmean(c[:, :, 0], axis=0)
            mean_y = np.nanmean(c[:, :, 1], axis=0)
        valid = (~np.isnan(c[:, :, 0])).sum(axis=0)

        rows = []
        for slot in range(c.shape[1]):
            pos = sheet.loc[sheet["xID"] == slot, "position"]
            rows.append({
                "slot": slot,
                "position": pos.iloc[0] if len(pos) else "?",
                "frames": int(valid[slot]),
                "pct_of_half": round(100 * valid[slot] / c.shape[0], 1),
                "mean_x": round(float(mean_x[slot]), 2) if valid[slot] else None,
                "mean_y": round(float(mean_y[slot]), 2) if valid[slot] else None,
                "abs_mean_x": round(abs(float(mean_x[slot])), 2) if valid[slot] else None,
            })
        df = pd.DataFrame(rows).sort_values("abs_mean_x", ascending=False, na_position="last")

        print(f"\n--- {half} / {side} ---")
        print(df.head(8).to_string(index=False))

        naive = int(df.iloc[0]["slot"])                      # old rule
        slot, source = goalkeeper_slot(m["xy"][half][side], sheet)   # new rule
        row = df[df["slot"] == slot].iloc[0]

        print(f"  old rule (furthest from centre) -> slot {naive}")
        print(f"  new rule ({source})              -> slot {slot} "
              f"(position {row['position']}, {row['frames']} frames, "
              f"mean_x {row['mean_x']})")
        if naive != slot:
            bad = df[df["slot"] == naive].iloc[0]
            print(f"  the old rule was fooled by slot {naive}: only "
                  f"{bad['pct_of_half']}% of the half but mean_x {bad['mean_x']}")
        print(f"  -> this team DEFENDS {'+x' if row['mean_x'] > 0 else '-x'}, "
              f"so it ATTACKS {'-x' if row['mean_x'] > 0 else '+x'}")

print("\nA team defends the side its goalkeeper sits on, so it ATTACKS the other")
print("way. The sign must be opposite in the two halves, and the two teams must")
print("always be on opposite sides of each other.")
