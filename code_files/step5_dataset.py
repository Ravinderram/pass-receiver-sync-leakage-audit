"""Step 5 - Build the sample table.

Each pass becomes one training example of shape (T, 23, 11):

  object 0      the passer
  objects 1-10  his ten teammates on the pitch   <- the model chooses among these
  objects 11-21 the eleven opponents
  object 22     the ball

  features: x, y, vx, vy, is_teammate, is_opponent, is_ball, is_passer,
            is_present, pressure, distance to ball

The window ends PREDICT_LEAD_S before the frame emitted by the synchroniser
(step3). That frame is an estimate of the pass moment, not ground-truth contact,
so the lead is relative to the emitted frame.

Each saved dataset also records pass_row (the row of work/passes_synced.csv the
sample came from), the anchor frame, the half, the lead and the anchor name, so
predictions can be traced back to individual passes and datasets built at
different leads can be intersected pass by pass.

The label is the index (0-9) of the receiving teammate.

LEAKAGE RULE: only information that existed BEFORE the ball was kicked may go
in here. Pass angle, pass distance, pass height, evaluation and the pass end
coordinates all describe the pass afterwards. Using any of them would let the
model read the answer off the input.

Run:  python step5_dataset.py
Out:  work/dataset_<match>.npz, work/dataset_counts.csv
"""

import numpy as np
import pandas as pd

import config as C
from common import banner, load_npz, save_npz

FORBIDDEN = ["PlayAngle", "Distance", "Height", "FlatCross", "Evaluation",
             "to_x", "to_y"]


def add_space_features(sample):
    """Fill features 9 and 10 in place, for every timestep.

    pressure  = distance to the nearest player of the other team
    ball_dist = distance to the ball

    Both are computed from coordinates already in the sample, so nothing new
    leaks in. They are here because the network has too little data to rediscover
    them, not because they add information.
    """
    xy = sample[:, :, :2]                                   # (T, 23, 2)
    mine = np.r_[0, np.arange(1, 1 + C.N_TEAMMATES)]        # passer + teammates
    theirs = np.arange(1 + C.N_TEAMMATES, C.N_OBJECTS - 1)
    present = sample[0, :, C.IDX_PRESENT] > 0

    def pairwise(a_idx, b_idx):
        b_idx = [j for j in b_idx if present[j]]
        if not b_idx:
            return None
        d = np.linalg.norm(xy[:, a_idx, None, :] - xy[:, None, b_idx, :], axis=3)
        return d.min(axis=2)                                # (T, len(a_idx))

    for group, other in ((mine, theirs), (theirs, mine)):
        g = [i for i in group if present[i]]
        if not g:
            continue
        d = pairwise(g, other)
        if d is not None:
            sample[:, g, C.IDX_PRESSURE] = d

    ball = xy[:, C.N_OBJECTS - 1]                           # (T, 2)
    sample[:, :, C.IDX_BALLDIST] = np.linalg.norm(xy - ball[:, None, :], axis=2)
    sample[:, ~present, C.IDX_PRESSURE] = 0.0
    sample[:, ~present, C.IDX_BALLDIST] = 0.0


def build_arrays(match_id, passes, lead_s=None, anchor_col="sync_frame",
                 clean=None, verbose=True):
    """Build the samples for one match at one lead. Returns a dict, saves nothing.

    lead_s      window end, in seconds BEFORE the anchor frame. Defaults to
                config.PREDICT_LEAD_S.
    anchor_col  column of `passes` holding the anchor frame. "sync_frame" is the
                frame emitted by step3; step27 passes an estimated kick proxy.
    clean       an already-loaded clean_<match>.npz dict, to avoid reloading it
                once per lead in the sweeps.
    """
    lead_s = C.PREDICT_LEAD_S if lead_s is None else float(lead_s)
    d = load_npz(f"clean_{match_id}") if clean is None else clean
    sub = passes[passes["match_id"] == match_id]

    back = int(C.WINDOW_SECONDS * C.FRAMERATE)     # 37 frames
    lead = int(round(lead_s * C.FRAMERATE))        # frames before the anchor
    offsets = np.arange(-back, 1, C.STRIDE) - lead  # 13 steps, ending at the lead

    # group the arrays by half ONCE. Even with an eager loader, doing these
    # lookups per pass builds thousands of throwaway references for nothing.
    by_half = {
        h: {
            "sign": {side: float(d[f"{h}_sign_{side.lower()}"])
                     for side in ("Home", "Away")},
            "pos": {k: d[f"{h}_{k}_pos"] for k in ("Home", "Away", "Ball")},
            "vel": {k: d[f"{h}_{k}_vel"] for k in ("Home", "Away", "Ball")},
        }
        for h in ("firstHalf", "secondHalf")
    }

    X, y, meta, pass_rows = [], [], [], []
    # teamsheet slot behind each teammate index, so downstream code can recover
    # who a prediction actually names. Without this the model's output classes
    # are anonymous indices and per-position metrics are impossible.
    mate_slots, team_side = [], []
    # separate counters: "not 11v11 at the pass" and "a NaN somewhere in the
    # 1.5 s window" are completely different failures and were being conflated
    skipped = {"early": 0, "not_11v11": 0, "nan_window": 0, "recipient_off": 0}
    lineup_counts = {}      # (n_mine_on_pitch, n_theirs_on_pitch) -> how often
    nan_source = {"passer": 0, "mates": 0, "opps": 0, "ball": 0}

    for row_idx, r in sub.iterrows():
        half = r["half"]
        if pd.isna(r[anchor_col]):
            skipped["no_anchor"] = skipped.get("no_anchor", 0) + 1
            continue
        f = int(r[anchor_col])
        if f - back - lead < 0:
            skipped["early"] += 1
            continue
        n_frames_half = by_half[half]["pos"]["Ball"].shape[0]
        if f - lead >= n_frames_half:
            skipped["late"] = skipped.get("late", 0) + 1
            continue

        mine = "Home" if r["team_slot"] == "Home" else "Away"
        theirs = "Away" if mine == "Home" else "Home"
        H = by_half[half]
        sign = H["sign"][mine]

        pos_m, vel_m = H["pos"][mine], H["vel"][mine]
        pos_o, vel_o = H["pos"][theirs], H["vel"][theirs]
        pos_b, vel_b = H["pos"]["Ball"], H["vel"]["Ball"]

        # who is actually on the pitch at the moment of the pass
        mates = np.where(~np.isnan(pos_m[f, :, 0]))[0]
        opps = np.where(~np.isnan(pos_o[f, :, 0]))[0]
        passer = int(r["passer_slot"])
        recipient = int(r["recipient_slot"])

        mates = mates[mates != passer]
        key = (len(mates) + 1, len(opps))
        lineup_counts[key] = lineup_counts.get(key, 0) + 1

        # A red card is a real football situation, not bad data. We pad the
        # missing slot and flag it as absent rather than dropping the pass.
        # More than a full side means a substitution is mid-swap and both
        # players are briefly tracked - that IS ambiguous, so we skip it.
        if not (C.MIN_TEAMMATES <= len(mates) <= C.N_TEAMMATES
                and C.MIN_OPPONENTS <= len(opps) <= C.N_OPPONENTS):
            skipped["not_11v11"] += 1
            continue
        if recipient not in mates:
            skipped["recipient_off"] += 1
            continue

        frames = f + offsets
        sample = np.zeros((len(offsets), C.N_OBJECTS, C.N_FEATURES), dtype=np.float32)

        def fill(idx, p, v, slot, flags):
            # sign flips x and y together, i.e. a 180 degree rotation, so that
            # the passing team always attacks towards +x
            sample[:, idx, 0] = p[frames, slot, 0] * sign
            sample[:, idx, 1] = p[frames, slot, 1] * sign
            sample[:, idx, 2] = v[frames, slot, 0] * sign
            sample[:, idx, 3] = v[frames, slot, 1] * sign
            sample[:, idx, 4:8] = flags
            sample[:, idx, C.IDX_PRESENT] = 1.0      # padded slots stay at 0

        fill(0, pos_m, vel_m, passer, [0, 0, 0, 1])
        for k, s in enumerate(mates):
            fill(1 + k, pos_m, vel_m, s, [1, 0, 0, 0])
        for k, s in enumerate(opps):
            fill(1 + C.N_TEAMMATES + k, pos_o, vel_o, s, [0, 1, 0, 0])
        fill(C.N_OBJECTS - 1, pos_b, vel_b, 0, [0, 0, 1, 0])

        if C.USE_SPACE_FEATURES:
            add_space_features(sample)

        # only real players can carry NaN; padded slots are all zeros already
        if np.isnan(sample[:, sample[0, :, C.IDX_PRESENT] > 0]).any():
            bad = np.isnan(sample).any(axis=(0, 2))          # which object
            bad &= sample[0, :, C.IDX_PRESENT] > 0
            if bad[0]:
                nan_source["passer"] += 1
            if bad[1:1 + C.N_TEAMMATES].any():
                nan_source["mates"] += 1
            if bad[1 + C.N_TEAMMATES:-1].any():
                nan_source["opps"] += 1
            if bad[-1]:
                nan_source["ball"] += 1
            skipped["nan_window"] += 1
            continue

        X.append(sample)
        y.append(int(np.where(mates == recipient)[0][0]))
        meta.append((f, r["team_slot"], half))
        pass_rows.append(int(row_idx))
        padded = np.full(C.N_TEAMMATES, -1, dtype=np.int16)
        padded[:len(mates)] = mates
        mate_slots.append(padded)
        team_side.append(0 if r["team_slot"] == "Home" else 1)

    X = np.stack(X) if X else np.zeros((0, len(offsets), C.N_OBJECTS, C.N_FEATURES), np.float32)
    y = np.array(y, dtype=np.int64)

    if verbose:
        print(f"{match_id}: {len(sub):4d} passes -> {len(X):4d} samples   "
              f"(skipped: {skipped['early']} early, {skipped['not_11v11']} not 11v11, "
              f"{skipped['nan_window']} NaN in window, "
              f"{skipped['recipient_off']} recipient off pitch)")

        odd = {k: v for k, v in sorted(lineup_counts.items(), key=lambda x: -x[1])
               if k != (11, 11)}
        if odd:
            print("    lineups other than 11v11: "
                  + ", ".join(f"{a}v{b}={n}" for (a, b), n in list(odd.items())[:6]))
            print("    10 = red card, kept and padded.  12+ = mid-substitution, skipped.")
        if skipped["nan_window"]:
            print("    NaN came from: "
                  + ", ".join(f"{k}={v}" for k, v in nan_source.items() if v))

    return {
        "X": X, "y": y,
        "mate_slots": (np.array(mate_slots, dtype=np.int16) if mate_slots
                       else np.zeros((0, C.N_TEAMMATES), np.int16)),
        "team_side": np.array(team_side, dtype=np.int8),
        "pass_row": np.array(pass_rows, dtype=np.int64),
        "anchor_frame": np.array([m[0] for m in meta], dtype=np.int64),
        "half_id": np.array([0 if m[2] == "firstHalf" else 1 for m in meta],
                            dtype=np.int8),
        "lead_s": np.float32(lead_s),
        "anchor": np.array(anchor_col),
        "_counts": {"match_id": match_id, "passes_in": int(len(sub)),
                    "samples": int(len(X)), **{f"skipped_{k}": int(v)
                                               for k, v in skipped.items()}},
    }


def save_arrays(name, arrays):
    """Save a build_arrays() result under work/<name>.npz."""
    return save_npz(name, **{k: v for k, v in arrays.items()
                             if not k.startswith("_")})


def build_match(match_id, passes, lead_s=None, name=None, verbose=True):
    """Build and save one match. Kept for backward compatibility.

    With the defaults this writes work/dataset_<match>.npz at the canonical
    lead, exactly as before. Sweeps must pass `name` so they never overwrite the
    canonical arrays (the old step17 did, which is how provenance got lost).
    """
    arr = build_arrays(match_id, passes, lead_s=lead_s, verbose=verbose)
    save_arrays(name or f"dataset_{match_id}", arr)
    return len(arr["X"])


if __name__ == "__main__":
    banner("Step 5 - building samples")
    print(f"Window {C.WINDOW_SECONDS} s, stride {C.STRIDE} -> {C.N_STEPS} time steps")
    print(f"Window ends {C.PREDICT_LEAD_S} s before the frame emitted by the")
    print("synchroniser (not before true contact, which is unobservable), so the")
    print("ball is not already flying towards the receiver. See check_leakage.py.")
    print(f"Shape per sample: ({C.N_STEPS}, {C.N_OBJECTS}, {C.N_FEATURES})")
    print(f"Never used as input: {', '.join(FORBIDDEN)}\n")

    passes = pd.read_csv(f"{C.WORK}/passes_synced.csv")
    counts = []
    for mid in C.MATCHES:
        arr = build_arrays(mid, passes)
        save_arrays(f"dataset_{mid}", arr)
        counts.append(arr["_counts"])
    total = sum(c["samples"] for c in counts)
    df = pd.DataFrame(counts).fillna(0)
    df.insert(1, "lead_s", C.PREDICT_LEAD_S)
    df.insert(2, "anchor", C.ANCHOR)
    df.to_csv(f"{C.WORK}/dataset_counts.csv", index=False)
    print(f"\nTotal samples: {total}")
    print(f"Saved -> {C.WORK}/dataset_counts.csv")
