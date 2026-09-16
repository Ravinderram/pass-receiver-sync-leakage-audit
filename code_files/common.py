"""Small helpers used by more than one step. Nothing clever on purpose."""

import json
import os
import pickle
import warnings

import numpy as np

import config as C


# ------------------------------------------------------------------ loading
def _cache_meta_path(match_id):
    return f"{C.WORK}/{match_id}_parsed.meta.json"


def _read_cache_meta(match_id):
    path = _cache_meta_path(match_id)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def load_match(match_id, use_cache=True, verbose=True):
    """Load one match as {events, xy, possession, ballstatus, teamsheets, pitch}.

    Source order:

    1. work/<match>_parsed.pkl, but ONLY if the cache carries the same local
       data fingerprint as the files on disk now. A cache parsed from another
       copy of the dataset (or from the synthetic fixture) is ignored and the
       match is reparsed, so stale data cannot silently survive a re-download.
    2. config.IDSSE_LOCAL_DATA_DIR, read through floodlight.io.dfl. This is
       LOCAL-ONLY: no network call is made, and a missing match is an error.
    3. Only when no local directory is configured: floodlight's downloader.

    The local directory is never written to. Everything generated goes to work/.
    """
    os.makedirs(C.WORK, exist_ok=True)
    cache = f"{C.WORK}/{match_id}_parsed.pkl"

    want_fp = None
    if C.LOCAL_ONLY:
        import local_data as L
        idx = L.get_index()
        want_fp = L.fingerprint(idx, [match_id])[0]

    if use_cache and os.path.isfile(cache):
        meta = _read_cache_meta(match_id)
        have_fp = (meta or {}).get("local_fingerprint")
        if want_fp is None or have_fp == want_fp:
            with open(cache, "rb") as f:
                return pickle.load(f)
        if verbose:
            print(f"  {match_id}: cached parse came from different local files "
                  f"({have_fp} != {want_fp}); reparsing")

    if C.LOCAL_ONLY:
        import local_data as L
        idx = L.get_index()
        got = L.match_files(idx, match_id)
        if got is None:
            status = L.match_status(idx, match_id)
            missing = [k for k, v in status.items() if not v] or ["all three files"]
            raise SystemExit(
                f"ERROR: required real IDSSE match {match_id} was not found under "
                f"{C.IDSSE_LOCAL_DIR}.\n"
                f"  missing: {', '.join(missing)}\n"
                f"  found for this match: "
                f"{ {k: v for k, v in status.items()} }\n"
                f"  Add the missing file(s) to that folder and rerun. Nothing will "
                f"be downloaded.\n"
                f"  Run python step0_inspect_local_data.py to see what is there.")
        info, event_file, pos_file = got
        from floodlight.io.dfl import read_event_data_xml, read_position_data_xml
        if verbose:
            print(f"  {match_id}: parsing local XML from {C.IDSSE_LOCAL_DIR}")
        xy, possession, ballstatus, teamsheets, pitch = read_position_data_xml(
            pos_file, info)
        events, _, _ = read_event_data_xml(event_file, info)
        out = {"match_id": match_id, "events": events, "xy": xy,
               "possession": possession, "ballstatus": ballstatus,
               "teamsheets": teamsheets, "pitch": pitch}
        source = {"data_source": "local_idsse",
                  "files": [os.path.basename(p) for p in got]}
    else:
        from floodlight.io.datasets import IDSSEDataset
        ds = IDSSEDataset(match_id=match_id)
        events, xy, possession, ballstatus, teamsheets, pitch = ds.get(match_id)
        out = {"match_id": match_id, "events": events, "xy": xy,
               "possession": possession, "ballstatus": ballstatus,
               "teamsheets": teamsheets, "pitch": pitch}
        source = {"data_source": "figshare_download", "files": []}

    with open(cache, "wb") as f:
        pickle.dump(out, f)
    with open(_cache_meta_path(match_id), "w") as fh:
        json.dump({"match_id": match_id, "local_fingerprint": want_fp,
                   "local_root": C.IDSSE_LOCAL_DIR, **source}, fh, indent=1)
    return out


def coords(xy_obj):
    """(T, 2N) interleaved x0,y0,x1,y1,... -> (T, N, 2). Validated against XY.x/XY.y."""
    arr = xy_obj.xy
    return arr.reshape(arr.shape[0], arr.shape[1] // 2, 2)


# ------------------------------------------------------- attacking direction
def goalkeeper_slot(xy_team, teamsheet, min_frames_frac=0.20):
    """Which array slot is the goalkeeper in this half?

    Primary: the teamsheet position column, which floodlight copies verbatim
    from the DFL PlayingPosition attribute. "TW" is Torwart.
    If the keeper was substituted there are two TW entries, so we take whichever
    actually has tracking data in this half.

    Fallback: the slot furthest from the centre in x, but only among players who
    were on the pitch for a decent share of the half. Without that frames filter
    a substitute who came on late and stood wide can out-score the real keeper.
    """
    c = coords(xy_team)
    valid = (~np.isnan(c[:, :, 0])).sum(axis=0)

    if teamsheet is not None and "position" in teamsheet.columns:
        tw = teamsheet.loc[teamsheet["position"] == "TW", "xID"].dropna().astype(int)
        played = [s for s in tw if s < c.shape[1] and valid[s] > 0]
        if played:
            return int(max(played, key=lambda s: valid[s])), "teamsheet"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_x = np.nanmean(c[:, :, 0], axis=0)
    mean_x = np.where(np.isnan(mean_x), 0.0, mean_x)
    mean_x[valid < min_frames_frac * c.shape[0]] = 0.0   # ignore brief appearances
    return int(np.argmax(np.abs(mean_x))), "fallback"


def attack_sign(xy_team, teamsheet=None):
    """Return +1 if this team attacks towards +x in this half, else -1.

    A team defends the side its goalkeeper sits on, so it attacks the other way.
    """
    c = coords(xy_team)
    slot, _ = goalkeeper_slot(xy_team, teamsheet)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        gk_x = np.nanmean(c[:, slot, 0])
    return -1.0 if gk_x > 0 else 1.0


# ---------------------------------------------------------- players on pitch
def on_pitch(xy_team, frame):
    """Slot indices of players with valid coordinates at this frame.

    Substituted players are NaN for the part of the match they did not play,
    so this is how we get the eleven who are actually playing.
    """
    c = coords(xy_team)[frame]
    return np.where(~np.isnan(c[:, 0]))[0]


# --------------------------------------------------------------- velocities
def velocity(xy_arr, framerate=C.FRAMERATE):
    """Central-difference velocity in m/s from (T, N, 2) positions.

    We compute this ourselves rather than using floodlight's VelocityModel,
    which returns speed magnitude only. We need the x and y components.
    NaN in, NaN out - that is correct for a player who is not on the pitch.
    """
    v = np.full_like(xy_arr, np.nan)
    v[1:-1] = (xy_arr[2:] - xy_arr[:-2]) * framerate / 2.0
    v[0] = (xy_arr[1] - xy_arr[0]) * framerate
    v[-1] = (xy_arr[-1] - xy_arr[-2]) * framerate
    return v


# ------------------------------------------------------------------- saving
def mirror(X):
    """Reflect a batch of samples in the halfway line: y -> -y, vy -> -vy.

    The tactical situation is equivalent under this reflection, so the model
    should give the same answer for both. Rather than hoping it learns that from
    a few thousand samples, we hand it both versions.

    This is the cheap version of what TacticAI (Wang, Velickovic, Hennes et al.,
    Nat Commun 2024) does properly: they build D2 reflection equivariance into
    the architecture so the symmetry never has to be learnt. They report it as
    their main source of data efficiency, which is exactly our binding
    constraint. Augmentation gets part of that benefit for about ten lines.

    We only mirror in y. The x direction is already normalised in step 5 so the
    passing team always attacks +x, and flipping it would undo that.
    """
    out = X.copy()
    out[:, :, :, 1] *= -1.0     # y
    out[:, :, :, 3] *= -1.0     # vy
    return out


def save_npz(name, **arrays):
    os.makedirs(C.WORK, exist_ok=True)
    path = f"{C.WORK}/{name}.npz"
    np.savez_compressed(path, **arrays)
    return path


def load_npz(name, eager=True):
    """Load an .npz. Returns a plain dict of real arrays by default.

    np.load gives back a LAZY NpzFile: every `d["key"]` access re-reads and
    decompresses that array from the zip. Put such a lookup inside a per-row
    loop and you decompress a 23 MB array thousands of times - it cost this
    project about an hour of runtime in two separate places before anyone
    noticed, because nothing errors, it just crawls.

    Materialising up front makes that impossible. Pass eager=False only if you
    know you want one array out of a large file.
    """
    z = np.load(f"{C.WORK}/{name}.npz", allow_pickle=True)
    if not eager:
        return z
    with z:
        return {k: z[k] for k in z.files}


def banner(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)
