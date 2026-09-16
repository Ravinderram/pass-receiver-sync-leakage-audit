"""Step 0 - Inspect the local IDSSE dataset before anything else runs.

Reads config.IDSSE_LOCAL_DATA_DIR (or the IDSSE_DATA_DIR environment variable),
walks it recursively, and works out what is actually there by OPENING files
rather than trusting their names. Nothing is downloaded and nothing in that
folder is ever written to.

It reports, in order:

  1. the configured path, and whether it exists
  2. every file: extension, size, and the role its contents indicate
  3. the match identifiers found
  4. a per-match table of what each of the seven canonical matches has
  5. a parse of one representative match: halves, frames, objects, framerate,
     pitch, teamsheet columns and position codes, event columns and identifiers
  6. frame counts against config.EXPECTED_FRAMES, which come from the paper
  7. the fingerprint that will be tied into every run id

Exit codes
  0  all seven canonical matches are present and parse
  1  something is missing or does not parse; the message says which

Run:  python step0_inspect_local_data.py [--full] [--match J03WMX]
      --full   parse every match, not just one representative
Out:  work/local_idsse_index.json, work/local_idsse_fingerprint.json
"""

import sys

import pandas as pd

import config as C
import local_data as L
import netguard
import provenance as P


def parse_report(match_id):
    """Parse one match and describe it. Returns (row, problems)."""
    from common import load_match
    problems = []
    m = load_match(match_id, verbose=False)
    xy = m["xy"]
    halves = list(xy)
    frames = {h: len(xy[h]["Home"]) for h in halves}
    total = sum(frames.values())
    expected = C.EXPECTED_FRAMES.get(match_id)
    rates = sorted({float(xy[h][t].framerate) for h in xy for t in xy[h]})
    slots = {t: xy[halves[0]][t].N for t in xy[halves[0]]}

    sheets = {}
    codes = set()
    for side in m["teamsheets"]:
        sheet = m["teamsheets"][side].teamsheet
        sheets[side] = len(sheet)
        if "position" in sheet.columns:
            codes |= {str(v) for v in sheet["position"].dropna().unique()}

    ev_rows, eids, cols = 0, set(), []
    for half in m["events"]:
        for team in m["events"][half]:
            ev = m["events"][half][team].events
            ev_rows += len(ev)
            cols = list(ev.columns)
            if "eID" in ev.columns:
                eids |= {str(v) for v in ev["eID"].dropna().unique()}

    bs = m["ballstatus"][halves[0]]
    alive = int((bs.code == 1).sum())

    if expected is not None and total != expected:
        problems.append(f"{match_id}: {total} frames, paper says {expected}")
    if rates != [float(C.FRAMERATE)]:
        problems.append(f"{match_id}: framerate {rates}, expected {C.FRAMERATE}")
    missing_eid = [e for e in C.OPEN_PLAY_EIDS if e not in eids]
    if missing_eid:
        problems.append(f"{match_id}: event feed has no {missing_eid}")

    row = {"match": match_id, "halves": len(halves), "frames": total,
           "expected_frames": expected,
           "frames_match": expected is None or total == expected,
           "framerate": rates, "slots": slots, "teamsheets": sheets,
           "position_codes": sorted(codes)[:12], "event_rows": ev_rows,
           "n_event_ids": len(eids), "event_columns": cols,
           "ball_alive_first_half": alive}
    return row, problems


if __name__ == "__main__":
    full = "--full" in sys.argv
    one = None
    for i, a in enumerate(sys.argv):
        if a == "--match" and i + 1 < len(sys.argv):
            one = sys.argv[i + 1]

    netguard.arm()
    netguard.banner()
    print()

    if not C.LOCAL_ONLY:
        print("No local dataset configured.")
        print("  Set IDSSE_LOCAL_DATA_DIR in config.py to the absolute path of your")
        print("  downloaded IDSSE folder, for example:")
        print('     IDSSE_LOCAL_DATA_DIR = r"C:\\Users\\you\\Downloads\\IDSSE"')
        print("  or export IDSSE_DATA_DIR=/home/you/Downloads/IDSSE")
        sys.exit(1)

    index = L.index_local_data()
    L.save_index(index)
    print(L.describe(index))

    print("\nFiles (role determined by inspecting contents):")
    files = sorted(index["files"], key=lambda f: (f["match"] or "~", f["relpath"]))
    for f in files[:80]:
        print(f"  {f['size'] / 1e6:8.1f} MB  {f['role']:18s} "
              f"{f['match'] or '-':8s} {f['relpath']}")
    if len(files) > 80:
        print(f"  ... and {len(files) - 80} more")
    archives = [f for f in files if f["role"] == "archive"]
    if archives:
        print("\nArchives found. The loader does not unpack archives; extract them")
        print("into the folder and rerun this script:")
        for a in archives:
            print(f"  {a['relpath']}  ->  {len(a.get('archive_contents', []))} entries")

    print("\nCanonical matches:")
    rows = []
    for mid in C.MATCHES:
        st = L.match_status(index, mid)
        role = ("test" if mid in C.TEST_MATCHES else
                "validation" if mid == C.VAL_MATCH else "fit")
        rows.append({"match": mid, "role": role,
                     "found": all(st.values()),
                     "matchinformation": st["matchinformation"],
                     "events": st["events"], "positions": st["positions"]})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    missing = table[~table["found"]]["match"].tolist()
    problems = []
    if missing:
        for mid in missing:
            problems.append(f"ERROR: required real IDSSE match {mid} was not found "
                            f"under {index['root']}")

    to_parse = ([one] if one else
                (list(C.MATCHES) if full else
                 [m for m in C.MATCHES if m not in missing][:1]))
    parsed = []
    if to_parse:
        print(f"\nParsing {', '.join(to_parse)}"
              + ("" if full or one else "  (use --full for every match)"))
        for mid in to_parse:
            try:
                row, probs = parse_report(mid)
            except SystemExit as exc:
                problems.append(str(exc).splitlines()[0])
                continue
            except Exception as exc:
                problems.append(f"{mid}: failed to parse ({type(exc).__name__}: "
                                f"{str(exc)[:120]})")
                continue
            parsed.append(row)
            problems += probs
            print(f"\n  {mid}")
            print(f"    halves            {row['halves']}")
            print(f"    frames            {row['frames']} "
                  f"(paper: {row['expected_frames']})  "
                  f"{'MATCH' if row['frames_match'] else 'MISMATCH'}")
            print(f"    framerate         {row['framerate']}")
            print(f"    objects per half  {row['slots']}")
            print(f"    teamsheets        {row['teamsheets']}")
            print(f"    position codes    {', '.join(row['position_codes'])}")
            print(f"    event rows        {row['event_rows']} across "
                  f"{row['n_event_ids']} event ids")
            print(f"    event columns     {', '.join(row['event_columns'])}")
            print(f"    ball alive (1st)  {row['ball_alive_first_half']} frames")

    fp, payload = L.fingerprint(index)
    P.write_json(f"{C.WORK}/local_idsse_fingerprint.json", {
        "data_source": "local_idsse",
        "local_data_root_basename": index["root"].rstrip("/").split("/")[-1],
        "local_fingerprint": fp,
        "matches_present": sorted(m for m in C.MATCHES if payload.get(m)),
        "matches_missing": sorted(m for m in C.MATCHES if not payload.get(m)),
        "per_match_files": payload, "n_files_in_root": index["n_files"],
        "total_bytes": index["total_bytes"], "timestamp_utc": P.now_utc(),
        "network": netguard.status()})
    print(f"\nlocal data fingerprint: {fp}")
    print(f"Saved -> {C.WORK}/{C.LOCAL_INDEX}")
    print(f"Saved -> {C.WORK}/local_idsse_fingerprint.json")

    if problems:
        print("\nPROBLEMS")
        for p in problems:
            print("  - " + p)
        print("\nNothing was downloaded. Fix the folder contents and rerun.")
        sys.exit(1)
    print("\nAll seven canonical matches are present, parse, and match the frame")
    print("counts published in the dataset paper. Next: python step1_load.py J03WMX")
