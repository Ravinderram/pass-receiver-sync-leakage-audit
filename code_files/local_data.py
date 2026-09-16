"""Find and describe the local IDSSE dataset. Reads only; never downloads.

The figshare release has been distributed in more than one shape (flat folder,
one folder per match, zipped per match), so nothing here assumes a layout. The
directory is walked recursively and every candidate file is classified by
OPENING it: an XML file is identified by its root element and its attributes, not
by its name. Filenames are used only as a hint and as a fallback.

  index_local_data(root)   walk, classify, group by match id
  match_files(index, mid)  the three files one match needs, or None
  fingerprint(index)       a hash of what is on disk, for provenance
  describe(index)          a printable summary

Roles recognised
  matchinformation   <matchInformation> with MatchInformation/Environment,
                     General and Teams: pitch, teamsheets, positions, team ids
  positions          FrameSet elements with Frame X/Y, one per object per half
  events             Event elements with an event time and a nested action

Everything else is reported as "other" and ignored, which is how project caches,
documentation and stray archives stay out of the way.
"""

import hashlib
import json
import os
import re
import zipfile
from collections import defaultdict

import config as C

MATCH_RE = re.compile(r"(J0[0-9A-Z]{4})")
XML_HEAD_BYTES = 200_000        # enough to see the root and a few elements
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "node_modules"}


# ------------------------------------------------------------------ sniffing
def _head(path, n=XML_HEAD_BYTES):
    try:
        with open(path, "rb") as fh:
            return fh.read(n).decode("utf-8", "replace")
    except OSError:
        return ""


def classify_xml(path):
    """(role, match_id_or_None) from the file's CONTENT."""
    head = _head(path)
    if not head:
        return "unreadable", None
    mid = None
    m = MATCH_RE.search(head) or MATCH_RE.search(os.path.basename(path))
    if m:
        mid = m.group(1)

    low = head.lower()
    if "<matchinformation" in low and "<teams" in low:
        role = "matchinformation"
    elif "<frameset" in low or ("<frame " in low and "framerate" not in low):
        role = "positions"
    elif "<event" in low:
        role = "events"
    elif "<matchinformation" in low:
        role = "matchinformation"
    else:
        role = "other_xml"
    return role, mid


def classify(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xml":
        role, mid = classify_xml(path)
    elif ext == ".zip":
        role, mid = "archive", (MATCH_RE.search(os.path.basename(path)) or [None])
        mid = MATCH_RE.search(os.path.basename(path))
        mid = mid.group(1) if mid else None
    elif ext in (".csv", ".tsv", ".parquet", ".json", ".jsonl", ".pkl", ".npz", ".pt"):
        m = MATCH_RE.search(os.path.basename(path))
        role, mid = f"other{ext}", m.group(1) if m else None
    else:
        m = MATCH_RE.search(os.path.basename(path))
        role, mid = "other", m.group(1) if m else None
    return role, mid


def archive_contents(path, limit=50):
    try:
        with zipfile.ZipFile(path) as z:
            return z.namelist()[:limit]
    except Exception:
        return []


# -------------------------------------------------------------------- index
def index_local_data(root=None, verbose=False):
    """Walk `root` and return a description of everything in it."""
    root = root or C.IDSSE_LOCAL_DIR
    if not root:
        raise SystemExit(
            "No local dataset configured.\n"
            "  Set IDSSE_LOCAL_DATA_DIR in config.py to the absolute path of your\n"
            "  downloaded IDSSE folder, or export IDSSE_DATA_DIR.")
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        raise SystemExit(
            f"Local dataset directory does not exist.\n"
            f"  configured path : {root}\n"
            f"  exists          : no\n"
            f"  what to change  : set IDSSE_LOCAL_DATA_DIR in config.py (or the\n"
            f"                    IDSSE_DATA_DIR environment variable) to the\n"
            f"                    folder holding your downloaded IDSSE files.\n"
            f"  No download will be attempted.")

    files, by_match = [], defaultdict(lambda: defaultdict(list))
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            role, mid = classify(path)
            rec = {"path": path, "relpath": os.path.relpath(path, root),
                   "name": name, "ext": os.path.splitext(name)[1].lower(),
                   "size": st.st_size, "mtime": int(st.st_mtime),
                   "role": role, "match": mid}
            if role == "archive":
                rec["archive_contents"] = archive_contents(path)
            files.append(rec)
            if mid:
                by_match[mid][role].append(rec)
            if verbose:
                print(f"  {rec['relpath']}  [{role}]"
                      + (f"  match {mid}" if mid else ""))

    return {"root": root, "files": files,
            "by_match": {k: {r: v for r, v in d.items()} for k, d in by_match.items()},
            "n_files": len(files),
            "total_bytes": sum(f["size"] for f in files)}


def match_files(index, match_id):
    """(info, events, positions) absolute paths for one match, or None."""
    d = index["by_match"].get(match_id)
    if not d:
        return None
    def pick(role):
        got = d.get(role) or []
        return max(got, key=lambda r: r["size"])["path"] if got else None
    info, ev, pos = pick("matchinformation"), pick("events"), pick("positions")
    return (info, ev, pos) if all((info, ev, pos)) else None


def match_status(index, match_id):
    d = index["by_match"].get(match_id, {})
    return {role: bool(d.get(role)) for role in
            ("matchinformation", "events", "positions")}


# -------------------------------------------------------------- fingerprint
def fingerprint(index, match_ids=None):
    """Hash the files the pipeline will actually read.

    Covers name, size and modification time of each match's three files, so a
    re-download or an edited file changes the fingerprint and invalidates caches
    and run ids downstream.
    """
    match_ids = sorted(match_ids or C.MATCHES)
    payload = {}
    for mid in match_ids:
        got = match_files(index, mid)
        if not got:
            payload[mid] = None
            continue
        entries = []
        for path in got:
            st = os.stat(path)
            entries.append({"name": os.path.basename(path), "size": st.st_size,
                            "mtime": int(st.st_mtime)})
        payload[mid] = entries
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16], payload


def save_index(index, path=None):
    path = path or f"{C.WORK}/{C.LOCAL_INDEX}"
    os.makedirs(C.WORK, exist_ok=True)
    slim = dict(index)
    with open(path, "w") as fh:
        json.dump(slim, fh, indent=1)
    return path


def load_index(path=None):
    path = path or f"{C.WORK}/{C.LOCAL_INDEX}"
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            idx = json.load(fh)
    except Exception:
        return None
    return idx if idx.get("root") == C.IDSSE_LOCAL_DIR else None


def get_index(rebuild=False):
    """The index, from cache when it is still valid for the configured root."""
    if not rebuild:
        idx = load_index()
        if idx and all(os.path.isfile(f["path"]) for f in idx["files"][:50]):
            return idx
    idx = index_local_data()
    save_index(idx)
    return idx


# ---------------------------------------------------------------- describing
def describe(index, match_ids=None):
    match_ids = list(match_ids or C.MATCHES)
    by_ext, by_role = defaultdict(int), defaultdict(int)
    for f in index["files"]:
        by_ext[f["ext"] or "(none)"] += 1
        by_role[f["role"]] += 1
    lines = ["IDSSE LOCAL DATA INSPECTION", "=" * 27, "",
             f"Root:\n  {index['root']}", "",
             f"Total files:\n  {index['n_files']} "
             f"({index['total_bytes'] / 1e6:.1f} MB)", "", "File types:"]
    for ext, n in sorted(by_ext.items(), key=lambda x: -x[1]):
        lines.append(f"  {ext:10s} {n}")
    lines += ["", "Roles identified by inspecting file contents:"]
    for role, n in sorted(by_role.items(), key=lambda x: -x[1]):
        lines.append(f"  {role:18s} {n}")
    found = sorted(index["by_match"])
    lines += ["", f"Match identifiers found ({len(found)}):",
              "  " + (", ".join(found) if found else "none")]
    return "\n".join(lines)
