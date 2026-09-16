"""Step 1 - Load one match and check its shape.

Reads the local dataset when config.IDSSE_LOCAL_DATA_DIR is set, in which case
the network is blocked for the whole run and a missing match is an error rather
than a download.

Run:  python step1_load.py [MATCH_ID]
      python step1_load.py --all        every configured match
"""

import sys

import config as C
import netguard
from common import banner, load_match

netguard.arm()
netguard.banner()
print()

args = [a for a in sys.argv[1:] if not a.startswith("--")]
if "--all" in sys.argv:
    for mid in C.MATCHES:
        import subprocess
        r = subprocess.run([sys.executable, __file__, mid])
        if r.returncode != 0:
            sys.exit(r.returncode)
    sys.exit(0)

match_id = args[0] if args else "J03WMX"
m = load_match(match_id)
events, xy, pitch = m["events"], m["xy"], m["pitch"]

banner(f"{match_id}: {C.MATCHES[match_id][0]} vs {C.MATCHES[match_id][1]}")
print(f"Halves: {list(xy.keys())}")
print(f"Objects per half: {list(xy['firstHalf'].keys())}")

print("\n--- Tracking ---")
total = 0
for half in xy:
    for team in xy[half]:
        o = xy[half][team]
        print(f"{half:11s} {team:5s} frames={len(o):6d} slots={o.N:3d} fps={o.framerate}")
    total += len(xy[half]["Home"])          # count Home only, same as the paper

want = C.EXPECTED_FRAMES[match_id]
print(f"\nTotal frames: {total}  (paper says {want})")
print("MATCH" if total == want else "MISMATCH - check the loader")

print("\n--- Pitch ---")
print(f"x {pitch.xlim}   y {pitch.ylim}   unit {pitch.unit}   {pitch.length} x {pitch.width}")

print("\n--- Teams ---")
for side in m["teamsheets"]:
    sheet = m["teamsheets"][side].teamsheet
    print(f"{side}: {len(sheet)} players | columns: {list(sheet.columns)}")

print("\n--- Events ---")
n = sum(len(events[h][t].events) for h in events for t in events[h])
print(f"Event rows (teams counted separately): {n}")
print(f"Columns: {list(events['firstHalf']['Home'].events.columns)}")

bs = m["ballstatus"]["firstHalf"]
alive = int((bs.code == 1).sum())
print(f"\nBall alive (1st half): {alive} frames ({100 * alive / len(bs.code):.1f}%)")
