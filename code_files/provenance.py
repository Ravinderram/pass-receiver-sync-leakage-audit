"""Run identity and provenance.

The recurring defect in this project was numbers from different runs appearing
side by side as though they came from one model. This module makes that
detectable instead of relying on memory:

  config hash      hash of config.canonical_config(), the constants that define
                   the canonical run
  data fingerprint hash of the canonical sample arrays (X and y of every match)
  local fingerprint hash of the RAW local dataset files the samples came from,
                   so re-downloading or editing the source invalidates every
                   downstream number even if the arrays happen to look the same
  pretrain hash    hash of work/pretrained_encoder.pt, when pretraining is on
  run id           short hash of the four above, plus a quick-mode marker

Each run also records its data source: "local_idsse" when config.LOCAL_ONLY is
set, "figshare_download" when floodlight fetched the files, "synthetic" when the
fixture generator wrote them, or "unknown" when nothing is parsed yet.

Every reported CSV carries run_id (and seed, lead_s, model_variant, test_match
where they apply). Verification scripts recompute the run id from the current
config and arrays and refuse to report outputs whose run id differs.

Nothing here imports torch, so steps that never train stay fast.
"""

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np

import config as C


# ------------------------------------------------------------------ hashing
def _sha(data: bytes, n=12):
    return hashlib.sha256(data).hexdigest()[:n]


def hash_obj(obj, n=12):
    """Stable hash of a JSON-serialisable object."""
    return _sha(json.dumps(obj, sort_keys=True, default=str).encode(), n)


def hash_file(path, n=12):
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def hash_arrays(*arrays, n=12):
    """Hash raw array bytes plus shapes and dtypes."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.shape, str(a.dtype))).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:n]


def config_hash(cfg=None):
    return hash_obj(C.canonical_config() if cfg is None else cfg)


def data_fingerprint(match_ids=None, name_fmt="dataset_{mid}"):
    """Hash of the canonical sample arrays. None if any array is missing."""
    match_ids = list(C.MATCHES) if match_ids is None else match_ids
    h = hashlib.sha256()
    for mid in match_ids:
        path = f"{C.WORK}/{name_fmt.format(mid=mid)}.npz"
        if not os.path.isfile(path):
            return None
        with np.load(path, allow_pickle=True) as z:
            for key in ("X", "y"):
                a = np.ascontiguousarray(z[key])
                h.update(mid.encode())
                h.update(str((a.shape, str(a.dtype))).encode())
                h.update(a.tobytes())
    return h.hexdigest()[:12]


def data_source():
    """Where the parsed matches came from, read from the per-match cache meta."""
    if C.LOCAL_ONLY:
        return "local_idsse"
    sources = set()
    for mid in C.MATCHES:
        meta = read_json(f"{C.WORK}/{mid}_parsed.meta.json")
        if meta and meta.get("data_source"):
            sources.add(meta["data_source"])
    if len(sources) == 1:
        return sources.pop()
    if sources:
        return "mixed:" + ",".join(sorted(sources))
    if any(os.path.isfile(f"{C.WORK}/{mid}_parsed.pkl") for mid in C.MATCHES):
        return "unknown_cache"
    return "unknown"


def local_fingerprint():
    """Hash of the raw local files, or None when no local dataset is configured.

    Read from work/local_idsse_fingerprint.json when step0 has written it, so
    that every script need not re-walk a multi-gigabyte directory.
    """
    if not C.LOCAL_ONLY:
        return None
    rec = read_json(f"{C.WORK}/local_idsse_fingerprint.json")
    if rec and rec.get("local_fingerprint"):
        return rec["local_fingerprint"]
    try:
        import local_data as L
        return L.fingerprint(L.get_index())[0]
    except SystemExit:
        return "unavailable"


def pretrain_fingerprint():
    if not C.USE_PRETRAIN:
        return "off"
    return hash_file(f"{C.WORK}/pretrained_encoder.pt") or "missing"


def canonical_run_id(cfg=None, data_fp=None, pre_fp=None, local_fp=None):
    """The id every canonical output must carry.

    Computed from the config as it stands in config.py (call before flipping
    flags), the canonical arrays, the raw local dataset and the pretrained
    encoder. Changing any of them changes the id, so stale results cannot be
    reported against new inputs.
    """
    cfg = C.canonical_config() if cfg is None else cfg
    data_fp = data_fingerprint() if data_fp is None else data_fp
    pre_fp = pretrain_fingerprint() if pre_fp is None else pre_fp
    local_fp = local_fingerprint() if local_fp is None else local_fp
    rid = hash_obj({"config": cfg, "data": data_fp, "pretrain": pre_fp,
                    "local": local_fp, "source": data_source()}, n=10)
    return ("quick-" if cfg.get("QUICK_MODE") else "") + rid


# ---------------------------------------------------------------- metadata
def git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=5,
                           cwd=os.path.dirname(os.path.abspath(__file__)))
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
    except Exception:
        return None


def software_versions(include_torch=True):
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    mods = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "floodlight"]
    if include_torch:
        mods.append("torch")
    for m in mods:
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[m] = None
    return out


def _netguard_status():
    try:
        import netguard
        return netguard.status()
    except Exception:
        return {"local_only": C.LOCAL_ONLY, "armed": False}


def now_utc():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True, default=str)
    return path


def read_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def base_metadata(experiment, run_id, **extra):
    """Fields every experiment's metadata file carries."""
    meta = {
        "experiment": experiment,
        "run_id": run_id,
        "timestamp_utc": now_utc(),
        "git_commit": git_commit(),
        "config_hash": config_hash(),
        "canonical_config": C.canonical_config(),
        "quick_mode": C.QUICK_MODE,
        "anchor": C.ANCHOR,
        "data_source": data_source(),
        "local_data_fingerprint": local_fingerprint(),
        # the folder name only: the absolute path stays in
        # work/local_idsse_fingerprint.json and out of paper-facing files
        "local_data_root_basename": (os.path.basename(C.IDSSE_LOCAL_DIR.rstrip("/\\"))
                                     if C.LOCAL_ONLY else None),
        "network_guard": _netguard_status(),
        "software": software_versions(),
    }
    meta.update(extra)
    return meta


# ---------------------------------------------------------------- CSV stamps
PROVENANCE_COLUMNS = ["run_id", "seed", "lead_s", "anchor", "model_variant",
                      "test_match"]


def stamp(df, run_id, seed=None, lead_s=None, model_variant=None,
          test_match=None, anchor=None):
    """Put provenance columns first. Existing columns of the same name win."""
    df = df.copy()
    vals = {"run_id": run_id, "seed": seed, "lead_s": lead_s,
            "anchor": C.ANCHOR if anchor is None else anchor,
            "model_variant": model_variant, "test_match": test_match}
    for k in reversed(PROVENANCE_COLUMNS):
        if k not in df.columns:
            df.insert(0, k, vals[k])
    front = [c for c in PROVENANCE_COLUMNS if c in df.columns]
    return df[front + [c for c in df.columns if c not in front]]


def check_run_id(df, expected, what, allow_stale=False):
    """Refuse to report a table whose rows come from another run."""
    if "run_id" not in df.columns:
        msg = f"{what}: no run_id column, cannot confirm which run produced it"
        if allow_stale:
            print("  WARNING " + msg)
            return False
        raise SystemExit(msg + ". Regenerate it with the current scripts.")
    ids = sorted(set(df["run_id"].astype(str)))
    if ids != [expected]:
        msg = (f"{what}: run_id {ids} does not match the current canonical run "
               f"{expected}")
        if allow_stale:
            print("  WARNING " + msg)
            return False
        raise SystemExit(msg + ". Rerun the producing script, or pass "
                               "--allow-stale to inspect old results.")
    return True


def print_header(title, run_id=None, **fields):
    """Uniform experiment header: configuration, seeds, lead, outputs."""
    bar = "=" * 70
    print("\n" + bar)
    print(title)
    print(bar)
    if C.QUICK_MODE:
        print("QUICK MODE (PRX_QUICK=1): reduced epochs, seeds and grids.")
        print("These outputs exercise the plumbing and must NOT be reported.")
    if run_id is not None:
        print(f"run id        : {run_id}")
    print(f"anchor        : {C.ANCHOR} (offsets are relative to the emitted "
          "synchronised frame, not to true contact)")
    for k, v in fields.items():
        print(f"{k:14s}: {v}")
    print(bar)
