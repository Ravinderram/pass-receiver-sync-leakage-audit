"""All settings in one place. Change things here, not inside the step files.

CANONICAL RUN. The block at the end of this file ("CANONICAL EXPERIMENT
DEFINITION") lists exactly which constants define the one run that every
reported number must come from. `canonical_config()` returns them as a
JSON-serialisable dict, and provenance.py hashes that dict into the run id that
is written next to every output. Change a canonical constant and the run id
changes, so verification scripts refuse to mix old and new results.

QUICK MODE. Setting the environment variable PRX_QUICK=1 shrinks epochs, seeds
and grids so the plumbing can be exercised on the synthetic self-test data in
minutes. Quick-mode outputs carry quick_mode=True in their run id and metadata
and must never be reported.
"""

import os as _os

# ---------------------------------------------------------------- matches
# NOTE: five of the seven matches are Fortuna Duesseldorf home games, and those
# five are exactly the 2. Bundesliga matches. So a random split would leak both
# team identity and division. We therefore test on 1. Bundesliga matches only.
MATCHES = {
    "J03WMX": ("1. FC Koeln", "FC Bayern Muenchen", 1),
    "J03WN1": ("VfL Bochum 1848", "Bayer 04 Leverkusen", 1),
    "J03WPY": ("Fortuna Duesseldorf", "1. FC Nuernberg", 2),
    "J03WOH": ("Fortuna Duesseldorf", "SSV Jahn Regensburg", 2),
    "J03WQQ": ("Fortuna Duesseldorf", "FC St. Pauli", 2),
    "J03WOY": ("Fortuna Duesseldorf", "F.C. Hansa Rostock", 2),
    "J03WR9": ("Fortuna Duesseldorf", "1. FC Kaiserslautern", 2),
}

# frame counts from Table 1 of the paper, used as a load check
EXPECTED_FRAMES = {
    "J03WMX": 145967, "J03WN1": 141561, "J03WPY": 146211, "J03WOH": 137214,
    "J03WQQ": 142345, "J03WOY": 142536, "J03WR9": 146810,
}

TRAIN_MATCHES = ["J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9"]
TEST_MATCHES = ["J03WMX", "J03WN1"]   # reported separately, never pooled

# ---------------------------------------------------------------- data
FRAMERATE = 25          # Hz, stated in the paper
WINDOW_SECONDS = 1.5    # how much history the model sees

# How long BEFORE the ball is struck the window ends.
#
# Measured BACKWARDS FROM THE FRAME THE SYNCHRONISER EMITS, not from ball
# contact: contact time is unobservable in this dataset, so no constant here can
# be expressed relative to it.
#
# This is not a tuning knob, it is a correctness fix. Step 3 picks the frame that
# best matches the pass event, and at that frame the ball is already moving, so
# its velocity vector points at the receiver: a one-line rule on it scores far
# above the nearest-teammate yardstick. The current numbers are in
# work/leakage_probes.csv (check_leakage.py) and work/leadsweep_summary.csv
# (step17_leadsweep.py); they are measured, not recorded here.
#
# 0.4 s is where the ball probe falls back to the yardstick. check_leakage.py
# re-measures it; do not lower this without re-running that and step24.
PREDICT_LEAD_S = 0.4
STRIDE = 3              # keep every 3rd frame -> about 8 Hz
N_STEPS = int(WINDOW_SECONDS * FRAMERATE) // STRIDE + 1   # = 13

N_TEAMMATES = 10        # players the passer can pass to
N_OPPONENTS = 11
N_OBJECTS = 1 + N_TEAMMATES + N_OPPONENTS + 1             # passer, mates, opps, ball = 23
# 0 x, 1 y, 2 vx, 3 vy, 4 is_mate, 5 is_opp, 6 is_ball, 7 is_passer,
# 8 is_present, 9 pressure (distance to nearest opponent), 10 distance to ball
N_FEATURES = 11
IDX_PRESENT = 8         # a red card means a slot is empty; the model must know
IDX_PRESSURE = 9
IDX_BALLDIST = 10

# Features 9 and 10 are derivable from the coordinates, so in principle the
# network could compute them itself. At this sample size it should not
# have to. Anzer & Bauer built their receiver model on pitch control, and the
# dataset paper's own xG model uses a pressure metric (Andrienko et al.).
USE_SPACE_FEATURES = True

# A team can legitimately be down to ten men. Requiring exactly 11 v 11 throws
# away the rest of the match, which cost 94% of J03WN1 (Adli sent off, minute 8).
MIN_TEAMMATES = 9       # 10 outfield teammates, or 9 after a red card
MIN_OPPONENTS = 10

# open play only. Set pieces are a different decision problem and the ball is
# not moving beforehand, so there is no useful movement history.
OPEN_PLAY_EIDS = ["Play_Pass", "Play_Cross"]
SUCCESS_LABEL = "successfullyCompleted"

# ---------------------------------------------------------------- sync
SYNC_GLOBAL_RANGE_S = 5.0    # stage 1: search a constant offset in +/- this range
SYNC_LOCAL_RANGE_S = 2.5     # stage 2: refine each event within +/- this window

# ---------------------------------------------------------------- filter
FILTER_ORDER = 3
FILTER_CUTOFF_HZ = 1.0       # floodlight butterworth_lowpass default

# ---------------------------------------------------------------- training
# Mirror the pitch in y during training and average both views at test time.
# The tactical situation is unchanged by the reflection, so this doubles the
# training data for free. Set False to measure how much it is worth.
# Normalise players and the ball with separate statistics.
#
# The pooled version computes one mean and sd over all 23 objects. Since 22 of
# them are players, those are player statistics, and the ball - which reaches
# ~30 m/s against a player's ~10 - lands about +2.7 sd out on average and +8.5
# at its 99th percentile. It is a permanent outlier in its own representation.
#
# This is the real problem behind the "the ball should have its own encoder"
# objection. A separate encoder is the expensive answer: it would see one object
# per sample instead of 23, so it gets 23x less gradient
# signal. Per-type normalisation fixes the scale without paying that.
NORMALISE_PER_TYPE = True

AUGMENT_MIRROR = True      # train on both reflections: doubles the samples
TTA_MIRROR = True          # average the two views at inference

# These are TWO different interventions that share a name, and the first version
# of this project reported them as one row in the ablation, which was wrong.
#   AUGMENT_MIRROR increases the training set. That is what augmentation is for.
#   TTA_MIRROR averages predictions over a transformation the task is invariant
#   to. Its purpose is variance reduction at inference, not more data.
# They are now separable, because if TTA helps it means the model did NOT learn
# the symmetry - which is an argument for building equivariance into the
# architecture, as TacticAI does, rather than patching it at test time.

# Plain self-attention sees only node embeddings; who is near whom has to be
# inferred from raw coordinates. TacticAI represents a situation as a graph where
# EDGES carry the relations between players, and reports that relations matter
# more than absolute positions. This adds a learned bias to each attention score
# based on the pair's relative geometry - the same idea, minimal code.
USE_EDGE_ATTENTION = True

# Warm-start the encoder from step11_pretrain.py, which learns to predict player
# movement from unlabelled frames. There are 1,002,644 frames in this dataset and
# only ~2,400 labelled passes, so this is the largest unused resource here.
USE_PRETRAIN = True

# Pretraining settings live here (previously hard-coded in step11) so that they
# are part of the canonical definition and of the run id.
PRETRAIN_WINDOWS = 20000     # unlabelled windows sampled per pretraining run
PRETRAIN_EPOCHS = 8
PRETRAIN_HORIZON_S = 1.0     # predict displacement this far ahead

# Early stopping, and the validation match it needs.
#
# Without this the epoch count is a guess, and a fixed count was costing
# accuracy: training accuracy keeps climbing after the validation peak. The
# per-epoch evidence for the current run is work/canonical_history.csv and
# figures/paper/fig18_learning_curves.png; the selected epoch per seed is in
# work/run_metadata.json.
#
# The validation match comes out of TRAIN_MATCHES, never out of the test set, so
# the test matches stay untouched by any decision the model makes about itself.
# All five training matches are Fortuna Duesseldorf home games, so validation is
# a same-distribution check - good enough to time early stopping, not a second
# generalisation estimate.
# NOTE ON SAMPLE COUNTS. No count is written down here or in the manuscript
# scripts: verify/sample_flow.py counts them from the arrays that exist, and
# work/run_metadata.json records the counts of the run that produced each number.
# EARLY_STOPPING must stay on for the canonical protocol; step23 refuses to run
# without it, because turning it off silently changes which matches are fitted.
EARLY_STOPPING = True
VAL_MATCH = "J03WR9"
PATIENCE = 8               # epochs without a validation improvement before stopping

SEEDS = [0, 1, 2]
EPOCHS = 60               # a cap now, not a target: early stopping decides
BATCH_SIZE = 64
LR = 1e-3
HIDDEN = 64

WORK = "work"
FIGS = "figures"


# ================================================================
# CANONICAL EXPERIMENT DEFINITION
# ================================================================
# Every reported number must come from the run these constants define.
# verify/reproduce_all.py writes them to work/canonical_config.json.

# Offsets (PREDICT_LEAD_S and every lead in the sweeps below) are measured
# BACKWARDS FROM THE FRAME THE SYNCHRONISER EMITS (step3_sync.py), not from true
# ball contact. No public dataset provides ground-truth contact times, so no
# script in this project may describe an offset as "before contact".
ANCHOR = "emitted_sync_frame"

# Matches that fit model weights. The validation match is carved out of the
# non-test matches and is used ONLY to pick the epoch (early stopping).
FIT_MATCHES = [m for m in TRAIN_MATCHES if m != VAL_MATCH]

# Baselines (gradient boosting and heuristics) make no model-selection decision,
# so they are fitted on every non-test match. This gives the strongest baseline
# the same total non-test data the neural model sees, which is the conservative
# choice for the paired comparison.
BASELINE_TRAIN_MATCHES = list(TRAIN_MATCHES)

# Unlabelled windows for self-supervised pretraining. The validation match is
# excluded so that nothing, not even unlabelled movement, flows from the match
# that selects the epoch into the encoder. Test matches are never used.
PRETRAIN_MATCHES = list(FIT_MATCHES)

MODEL_SELECTION_METRIC = "val_top1"   # what early stopping monitors
RESTORE_BEST_CHECKPOINT = True        # roll back to the best validation epoch
N_HEADS = 4
N_ATTENTION_LAYERS = 1
EXPECTED_PARAMS = 35909               # asserted by verify/architecture_checks.py

TOPK_REPORTED = [1, 2, 3, 5]
ECE_BINS = 10                         # equal-width confidence bins, top-label ECE
SEED_AGGREGATION = "mean_over_seeds"  # headline = mean of per-seed metrics;
                                      # the seed ensemble is reported as a
                                      # separate, labelled row, never mixed in

SOFTWARE_ASSUMPTIONS = {
    "python": ">=3.10",
    "torch": ">=2.1 (CPU or CUDA; GPU raises batch size 4x, see device.py)",
    "floodlight": ">=1.2.0",
    "databallpy": "0.8.1, step20 only, separate environment allowed",
}

# ---------------------------------------------------------------- experiments
# These grids are NOT part of the canonical run id: changing them changes which
# extra experiments are run, not what the canonical model is.
LEAD_SWEEP_LEADS = [0.0, 0.2, 0.4, 0.8, 1.2]
CROSS_OFFSET_LEADS = [0.0, 0.2, 0.4, 0.8, 1.2]
BALL_MASK_TRAIN_LEADS = [0.0, PREDICT_LEAD_S]

# Optional higher-seed robustness mode for the unstable ablation rows
# (python step12_ablation.py --robust). Only these settings get extra seeds.
ROBUST_SEEDS = list(range(10))
ROBUST_SETTINGS = ["plain", "+ pretrain", "+ train mirror", "+ both mirror",
                   "+ space", "everything"]

# Training-size curve (step30). Number of fit matches used, nested in config order.
SIZE_CURVE_N_FIT = [1, 2, 3, 4]
SIZE_CURVE_SETTINGS = ["plain", "+ space", "+ train mirror", "+ pretrain",
                       "everything"]

# Estimated kick proxy (step27). This is an ESTIMATE from ball kinematics, not a
# ground-truth contact time.
KICK_PROXY = {
    "search_back_s": 1.0,      # how far before the emitted frame to look
    "search_fwd_s": 0.5,       # how far after the emitted frame to look for the speed peak
    "onset_frac": 0.2,         # onset = last frame below pre + frac * (peak - pre)
    "min_peak_speed": 5.0,     # m/s; weaker peaks are not treated as a kick
    "max_ball_passer_m": 3.0,  # ball must be this close to the passer at onset
    "median_window": 5,        # frames, median filter on raw ball speed
}

# Masella-style passer-facing angle probe (step28). Below this ball-to-passer
# distance the facing vector is numerically undefined.
MASELLA_MIN_BALL_PASSER_M = 0.3

# ================================================================
# LOCAL DATASET  (the only place a data path is configured)
# ================================================================
# Absolute path to your downloaded copy of the IDSSE dataset. Set it here, or
# leave it None and export IDSSE_DATA_DIR instead. Examples:
#
#   IDSSE_LOCAL_DATA_DIR = r"C:\Users\ranjit\Downloads\IDSSE"
#   IDSSE_LOCAL_DATA_DIR = "/home/ranjit/Downloads/IDSSE"
#   IDSSE_LOCAL_DATA_DIR = "/Users/ranjit/Downloads/IDSSE"
#
# The folder is read recursively and is NEVER written to. Layout does not
# matter: step0_inspect_local_data.py finds the files by inspecting them.
#
# When this is set, the project is in LOCAL-ONLY mode: common.load_match() will
# not import floodlight's downloader, and netguard.py blocks outbound sockets
# during the run, so a download cannot happen even by accident. A missing match
# is an error, never a download.
IDSSE_LOCAL_DATA_DIR =r"C:\Users\USER\Downloads\IDSS"

# resolved path, environment variable included
IDSSE_LOCAL_DIR = (IDSSE_LOCAL_DATA_DIR or _os.environ.get("IDSSE_DATA_DIR")
                   or None)
if IDSSE_LOCAL_DIR:
    IDSSE_LOCAL_DIR = _os.path.abspath(_os.path.expanduser(IDSSE_LOCAL_DIR))

# True when a local dataset is configured: no network access is permitted.
LOCAL_ONLY = IDSSE_LOCAL_DIR is not None

# Where the inspector caches what it found, so the loader need not re-walk a
# large directory on every call.
LOCAL_INDEX = "local_idsse_index.json"

# Cross-provider probe (step29). Point this at a local copy of the PFF FC World
# Cup 2022 release, e.g.  export PFF_WC2022_DIR=/data/pff_wc2022
PFF_DATA_DIR = _os.environ.get("PFF_WC2022_DIR") or None

# ---------------------------------------------------------------- quick mode
QUICK_MODE = _os.environ.get("PRX_QUICK", "0") == "1"
if QUICK_MODE:
    EPOCHS = 8
    PATIENCE = 3
    SEEDS = [0, 1]
    ROBUST_SEEDS = [0, 1, 2]
    PRETRAIN_WINDOWS = 1500
    PRETRAIN_EPOCHS = 2
    LEAD_SWEEP_LEADS = [0.0, 0.4, 1.2]
    CROSS_OFFSET_LEADS = [0.0, 0.4, 1.2]
    SIZE_CURVE_N_FIT = [2, 4]
    SIZE_CURVE_SETTINGS = ["plain", "+ space", "everything"]

# The constants that define the canonical run, in a fixed order.
CANONICAL_KEYS = [
    "MATCHES", "TRAIN_MATCHES", "FIT_MATCHES", "VAL_MATCH", "TEST_MATCHES",
    "BASELINE_TRAIN_MATCHES", "PRETRAIN_MATCHES",
    "ANCHOR", "PREDICT_LEAD_S", "FRAMERATE", "WINDOW_SECONDS", "STRIDE",
    "N_STEPS", "N_TEAMMATES", "N_OPPONENTS", "N_OBJECTS", "N_FEATURES",
    "MIN_TEAMMATES", "MIN_OPPONENTS", "OPEN_PLAY_EIDS", "SUCCESS_LABEL",
    "SYNC_GLOBAL_RANGE_S", "SYNC_LOCAL_RANGE_S", "FILTER_ORDER",
    "FILTER_CUTOFF_HZ",
    "USE_SPACE_FEATURES", "NORMALISE_PER_TYPE", "AUGMENT_MIRROR", "TTA_MIRROR",
    "USE_EDGE_ATTENTION", "USE_PRETRAIN", "PRETRAIN_WINDOWS", "PRETRAIN_EPOCHS",
    "PRETRAIN_HORIZON_S",
    "EARLY_STOPPING", "PATIENCE", "MODEL_SELECTION_METRIC",
    "RESTORE_BEST_CHECKPOINT", "SEEDS", "EPOCHS", "BATCH_SIZE", "LR", "HIDDEN",
    "N_HEADS", "N_ATTENTION_LAYERS", "EXPECTED_PARAMS",
    "TOPK_REPORTED", "ECE_BINS", "SEED_AGGREGATION", "QUICK_MODE",
]


def canonical_config():
    """The canonical run definition as a plain, JSON-serialisable dict.

    Reads the CURRENT module values, so call it before any script flips flags
    (step12 does). provenance.canonical_run_id() hashes this dict.
    """
    g = globals()
    out = {}
    for k in CANONICAL_KEYS:
        v = g[k]
        if isinstance(v, dict):
            v = {kk: list(vv) if isinstance(vv, tuple) else vv for kk, vv in v.items()}
        elif isinstance(v, tuple):
            v = list(v)
        out[k] = v
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(canonical_config(), indent=1))
