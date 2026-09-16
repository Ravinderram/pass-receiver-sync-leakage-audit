"""Step 21 - Where every feature comes from.

A reviewer asked which features are in the dataset and which the pipeline
invents. The answer is worth writing down because it is not what most people
guess: of the eleven features the model sees, only two are values that exist in
a file. Everything else is computed or assigned.

Three categories are used, and the distinction matters:

  FROM DATA    the value exists as a field in one of the XML files and is read
               (possibly filtered or reoriented, but not computed from anything
               else)

  DERIVED      computed from fields that are in the data. No new information
               enters; the point is to hand the model something it would
               otherwise have to rediscover from a few thousand samples

  STRUCTURAL   encodes how this pipeline arranged the arrays. Nothing in the
               files corresponds to it

There is also a fourth list, and it is the interesting one: fields that exist in
the raw XML and are NOT used. Some are dropped by floodlight before we ever see
them; some are deliberately forbidden because they describe the pass after it
happened; and some are simply unused and available.

Run:  python step21_features.py
Out:  work/feature_provenance.csv
"""

import numpy as np
import pandas as pd

import config as C
from common import banner, load_npz

# (index, name, category, source, note)
MODEL_FEATURES = [
    (0, "x", "FROM DATA", "<Frame X=...>",
     "Butterworth low-pass filtered (step 4), then sign-flipped so the passing "
     "team always attacks +x (step 5)"),
    (1, "y", "FROM DATA", "<Frame Y=...>",
     "same filtering and flip"),
    (2, "vx", "DERIVED", "central difference on filtered x",
     "the raw XML has a speed field S, but floodlight drops it, so velocity is "
     "computed here"),
    (3, "vy", "DERIVED", "central difference on filtered y", "same"),
    (4, "is_mate", "DERIVED", "event Team + tracking array membership",
     "1 for the passer's ten teammates"),
    (5, "is_opp", "DERIVED", "event Team + tracking array membership",
     "1 for the eleven opponents"),
    (6, "is_ball", "STRUCTURAL", "which array the object came from",
     "the ball is always object index 22"),
    (7, "is_passer", "DERIVED", "<Play Player=...>",
     "the event names the player; we encode it as a flag on object index 0"),
    (8, "is_present", "DERIVED", "NaN pattern in the tracking array",
     "distinguishes a real player from a padded slot after a red card"),
    (9, "pressure", "DERIVED", "distance to nearest opponent",
     "computable from coordinates the model already has; given explicitly "
     "because a few thousand samples are not enough to rediscover it"),
    (10, "ball_dist", "DERIVED", "distance to the ball", "same reasoning"),
]

BASELINE_FEATURES = [
    ("dist", "passer to candidate"),
    ("angle", "bearing from passer to candidate"),
    ("d_opp", "candidate to his nearest opponent"),
    ("lane", "opponents inside the passer-candidate corridor"),
    ("speed", "candidate's speed"),
    ("moved", "how far the candidate moved across the window"),
    ("ahead", "candidate's x relative to the passer"),
    ("to_goal", "candidate's distance to the opposition goal"),
]

# raw fields the DFL files contain, and what happens to each
RAW_FIELDS = [
    ("Frame", "N", "used", "frame number, the index everything is joined on"),
    ("Frame", "T", "dropped by floodlight", "per-frame ISO timestamp"),
    ("Frame", "X, Y", "used", "the only raw numeric features that reach the model"),
    ("Frame", "Z", "dropped by floodlight",
     "ball height. Would be a genuine ball-only feature and is the one real "
     "argument for a separate ball encoder"),
    ("Frame", "D", "dropped by floodlight", "distance covered since last frame"),
    ("Frame", "S", "dropped by floodlight",
     "speed. This is why velocity is derived rather than read"),
    ("Frame", "A", "dropped by floodlight", "acceleration"),
    ("Frame", "M", "dropped by floodlight", "minute of play"),
    ("Frame", "BallPossession", "available, unused", "which team has the ball"),
    ("Frame", "BallStatus", "available, unused", "ball in play or dead"),
    ("Play", "Player", "used", "becomes is_passer"),
    ("Play", "Team", "used", "becomes is_mate / is_opp"),
    ("Play", "Recipient", "used as the LABEL", "the target, never an input"),
    ("Play", "PlayAngle", "FORBIDDEN", "describes the pass after it happened"),
    ("Play", "Distance", "FORBIDDEN", "same"),
    ("Play", "Height", "FORBIDDEN", "same"),
    ("Play", "FlatCross", "FORBIDDEN", "same"),
    ("Play", "Evaluation", "used as a FILTER", "keeps successful passes; never an input"),
    ("Play", "PlayOrigin", "available, unused", "own half or opposition half"),
    ("Play", "PenaltyBox", "available, unused", "whether the pass entered the box"),
    ("Play", "SemiField", "available, unused", ""),
    ("Play", "FromOpenPlay", "available, unused", "used implicitly via the eID filter"),
    ("Play", "BallPossessionPhase", "available, unused",
     "a possession sequence id — would group passes into attacks"),
    ("Event", "X-Source-Position", "used", "pass origin, converted from corner-based"),
    ("Event", "Y-Source-Position", "used", "same"),
    ("Event", "X-Position, Y-Position", "FORBIDDEN", "where the ball ended up"),
    ("Teamsheet", "position", "used, but not as a model input",
     "TW/IV/ST codes. Used to find the goalkeeper and to report per-position "
     "metrics. Never fed to the network — an obvious feature to try"),
    ("Teamsheet", "pID, xID", "used", "maps a player to his array slot"),
    ("Teamsheet", "jID, player, team", "available, unused", "shirt number, name"),
]


def verify():
    """Check the claims above against what is actually on disk."""
    print("--- verification ---")
    try:
        d = load_npz(f"dataset_{list(C.MATCHES)[0]}")
    except FileNotFoundError:
        print("  no dataset yet — run step5_dataset.py to verify shapes")
        return

    X = d["X"]
    print(f"  sample shape {X.shape[1:]} -> {X.shape[3]} features, "
          f"table below describes {len(MODEL_FEATURES)}")
    assert X.shape[3] == len(MODEL_FEATURES), "feature table is out of date"

    from floodlight.core.xy import XY
    import inspect
    import floodlight.io.dfl as dfl
    src = inspect.getsource(dfl.read_position_data_xml)
    dropped = [a for a in ("Z", "D", "S", "A", "M") if f'"{a}"' not in src]
    print(f"  floodlight drops these <Frame> attributes: {', '.join(dropped)}")
    print("  so speed and acceleration must be derived, and ball height is lost")


if __name__ == "__main__":
    banner("Step 21 - feature provenance")

    df = pd.DataFrame(MODEL_FEATURES,
                      columns=["idx", "feature", "category", "source", "note"])
    print("MODEL INPUT — what the network sees\n")
    print(df[["idx", "feature", "category", "source"]].to_string(index=False))

    counts = df["category"].value_counts()
    print(f"\n  {counts.get('FROM DATA', 0)} of {len(df)} features are values that "
          f"exist in a file.")
    print(f"  {counts.get('DERIVED', 0)} are computed from those values.")
    print(f"  {counts.get('STRUCTURAL', 0)} encode how the arrays are laid out.")
    print("\n  No feature adds information that is not already in the dataset.")
    print("  They add CONVENIENCE: at this sample size the model should not have")
    print("  to rediscover 'distance to nearest opponent' from raw coordinates.")

    print("\n\nGRADIENT BOOSTING BASELINE — all eight derived\n")
    for n, d in BASELINE_FEATURES:
        print(f"  {n:10s} {d}")

    raw = pd.DataFrame(RAW_FIELDS, columns=["file", "field", "status", "note"])
    print("\n\nRAW FIELDS IN THE DFL FILES, AND WHAT BECOMES OF THEM\n")
    for status in ["used", "used as the LABEL", "used as a FILTER",
                   "used, but not as a model input", "FORBIDDEN",
                   "dropped by floodlight", "available, unused"]:
        sub = raw[raw["status"] == status]
        if not len(sub):
            continue
        print(f"  {status.upper()}")
        for _, r in sub.iterrows():
            print(f"    {r['file']:10s} {r['field']:22s} {r['note']}")
        print()

    print("VERIFY\n")
    verify()

    out = f"{C.WORK}/feature_provenance.csv"
    pd.concat([
        df.assign(table="model_input"),
        pd.DataFrame(BASELINE_FEATURES, columns=["feature", "note"])
          .assign(table="gbm_baseline", category="DERIVED"),
        raw.rename(columns={"field": "feature", "status": "category"})
           .assign(table="raw_fields"),
    ]).to_csv(out, index=False)
    print(f"\nSaved -> {out}")
