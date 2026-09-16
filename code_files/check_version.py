"""Check that every file is the current version.

We have patched several files across several rounds. If a zip is extracted over
an existing folder and the overwrite prompt is skipped, you end up with a mix of
old and new files that still imports and still runs - and quietly produces wrong
numbers. This checks for a marker from each fix.

Run:  python check_version.py
"""

import hashlib
import json
import os
import sys

CHECKS = [
    ("common.py", "def goalkeeper_slot",
     "goalkeeper found from the teamsheet, not by furthest-from-centre"),
    ("common.py", "min_frames_frac",
     "brief substitute appearances ignored when finding the keeper"),
    ("step3_sync.py", "def coord_variants",
     "event coords converted from corner-based to centre-based"),
    ("step3_sync.py", "def pick_orientation",
     "the four orientations tested rather than assumed"),
    ("step3_sync.py", "def ball_to_passer",
     "independent ball-to-passer check that does not use the annotation"),
    ("step3_sync.py", "score = -e / 5.0",
     "event-to-ball distance included in the cost function"),
    ("step4_clean.py", "NO FLIP between halves",
     "attacking direction must flip between halves"),
    ("step5_dataset.py", "FORBIDDEN",
     "leakage guard on the post-pass event attributes"),
    ("step6_explore.py", 'p["ev_x"]',
     "figures use converted coordinates, not raw at_x"),
    ("step8_model.py", "permutation check",
     "permutation-invariance self-test"),
    ("config.py", "IDX_PRESENT",
     "presence feature so a red card does not drop the rest of the match"),
    ("step5_dataset.py", "MIN_TEAMMATES",
     "10-man lineups padded instead of skipped"),
    ("step8_model.py", "key_padding_mask",
     "padded slots masked in attention and in the softmax"),
    ("step7_baselines.py", "def present_mask",
     "baselines cannot pick a player who is off the pitch"),
    ("config.py", "PREDICT_LEAD_S",
     "window ends before ball contact, so the ball is not already flying at the receiver"),
    ("step5_dataset.py", "lead = int(round(lead_s * C.FRAMERATE))",
     "the window-end offset is applied when building samples, at any lead"),
    ("common.py", "def mirror",
     "pitch-reflection augmentation (TacticAI-style data efficiency)"),
    ("config.py", "USE_SPACE_FEATURES",
     "pressure and ball-distance node features"),
    ("step5_dataset.py", "def add_space_features",
     "space features actually computed"),
    ("step8_model.py", "def edge_bias",
     "geometric edge bias in the attention"),
    ("experiment.py", "pretrained_encoder.pt",
     "training warm-starts from the pretrained encoder"),
    ("_selftest.py", "softmax over a score built",
     "fixture has learnable structure, so ablations can be validated"),
    ("step13_inspect.py", "def frame_panel",
     "per-pass visual inspection"),
    ("step14_report.py", "validation ledger",
     "HTML report leading with the validation checks"),
    ("step10_analysis.py", "prob[i, :len(mates)]",
     "figures handle a 10-man team without crashing"),
    ("config.py", "EARLY_STOPPING",
     "epoch count decided by a validation match, not guessed"),
    ("step9_train.py", "def load_train_val",
     "validation match carved out of the training matches"),
    ("experiment.py", "RESTORE_BEST_CHECKPOINT",
     "weights rolled back to the best validation epoch"),
    ("experiment.py", "def fit_normaliser",
     "per-type normalisation statistics, excluding padded slots"),
    ("experiment.py", "weights_only=True",
     "checkpoints loaded without unpickling arbitrary objects"),
    ("experiment.py", "PIN_MEMORY",
     "pinned host memory for faster host-to-device copies"),
    ("config.py", "NORMALISE_PER_TYPE",
     "players and ball normalised on their own scales"),
    ("config.py", "TTA_MIRROR",
     "train-time augmentation separated from test-time averaging"),
    ("metrics_lib.py", "def ranking_metrics",
     "ranking, calibration and per-position metrics"),
    ("step18_metrics.py", "canonical_predictions",
     "metrics read the canonical predictions instead of retraining"),
    ("config.py", "CANONICAL_KEYS",
     "one canonical experiment definition, hashed into the run id"),
    ("config.py", "ANCHOR = \"emitted_sync_frame\"",
     "offsets stated relative to the emitted frame, never to true contact"),
    ("provenance.py", "def canonical_run_id",
     "run id from config, data and pretrained encoder"),
    ("experiment.py", "def get_or_train",
     "one shared trainer and checkpoint cache for every script"),
    ("experiment.py", "def ensure_datasets",
     "lead-specific datasets cached separately, canonical arrays never overwritten"),
    ("step5_dataset.py", "pass_rows.append",
     "per-pass identifiers stored so predictions can be traced back"),
    ("step23_canonical.py", "def prediction_rows",
     "per-pass predictions exported from the canonical run"),
    ("metrics_lib.py", "def mcnemar",
     "paired McNemar from the exported predictions"),
    ("probes.py", "def flatness_diagnostic",
     "flatness criterion with a stated unresolved case"),
    ("step24_cross_offset.py", "train_lead_s",
     "cross-offset dependence experiment"),
    ("step25_ball_mask.py", "def apply_mask",
     "ball masking at test time without retraining"),
    ("step30_size_curve.py", "SIZE_CURVE", "training-size curve"),
    ("verify/reproduce_all.py", "STAGES",
     "one entry point that regenerates everything"),
    ("verify/data_provenance.py", "EXPECTED_FRAMES",
     "real-versus-fixture gate, checked against the dataset paper"),
    ("verify/data_provenance.py", "def check_census",
     "dataset-wide totals checked against the authors' own published census"),
    ("verify/no_quick_results.py", "PAPER_FACING",
     "quick-mode results cannot reach a paper-facing file"),
    ("verify/real_run_summary.py", "Valid for paper reporting",
     "the run summary is generated, not typed"),
    ("config.py", "IDSSE_LOCAL_DATA_DIR",
     "one configured path to the local dataset"),
    ("common.py", "local_fingerprint",
     "parsed caches are invalidated when the raw local files change"),
    ("local_data.py", "def index_local_data",
     "the local folder is classified by opening files, not by their names"),
    ("netguard.py", "class NetworkBlocked",
     "no download can happen during a local-data run"),
    ("step0_inspect_local_data.py", "def parse_report",
     "the local dataset is inspected before anything trains"),
    ("step5_dataset.py", "mate_slots",
     "teammate index mapped back to the teamsheet, so classes mean something"),
    ("step20_databallpy.py", "def require_databallpy",
     "dependency check present (it was deleted once by a slice edit)"),
    ("check_symbols.py", "undefined name",
     "static undefined-name check"),
    ("verify/section8_results.py", "canonical_predictions",
     "Section VIII numbers derived from one exported prediction file"),
    ("verify/table4_ablation.py", "clears_2sd",
     "ablation judged against its own paired seed spread"),
    ("verify/table4_ablation.py", "def additivity",
     "composites compared with the sum of their own components"),
    ("verify/table3_sync.py", "def heterogeneity",
     "chi-square heterogeneity test replacing the sd-vs-CI error"),
    ("step22_curves.py", "def learning_curves",
     "train/validation/test curves, ROC and precision-recall"),
    ("step21_features.py", "MODEL_FEATURES",
     "feature provenance table, verified against the arrays"),
    ("step20_databallpy.py", "def probe_on_frames",
     "the leak tested against DataBallPy's own synchronisation"),
    ("step18_metrics.py", "def confusion_figure",
     "confusion matrix and per-class figure"),
    ("step19_architecture.py", "def build",
     "the architecture diagram"),
    ("step17_leadsweep.py", "LEAD_SWEEP_LEADS",
     "the offset sweep that replaces a single cutoff with a curve"),
    ("common.py", "eager=True",
     "load_npz materialises arrays, so a lazy lookup cannot sit in a hot loop"),
    ("step5_dataset.py", "by_half = {",
     "array lookups hoisted out of the per-pass loop"),
    ("step15_figures.py", "def leakage_curve",
     "decay curve vectorised - seconds instead of an hour"),
    ("device.py", "DEVICE = torch.device",
     "single device definition shared by every torch script"),
    ("experiment.py", "from device import DEVICE",
     "training uses the shared device, not its own"),
    ("step16_tactical.py", "def fig10_space_control",
     "floodlight Voronoi and trajectory figures"),
    ("step15_figures.py", "def fig1_leakage",
     "the decay curve the contribution rests on"),
    ("probes.py", "def flatness_diagnostic",
     "probes judged on level and temporal shape, not a fixed threshold"),
    ("check_leakage.py", "BALL_FAIL_EXCESS",
     "the check fails on the ball shortcut, not on an unresolved signal"),
    ("_selftest.py", "corner-based",
     "fixture reproduces the coordinate mismatch"),
    ("_selftest.py", "nearest the ball",
     "fixture puts the ball at the passer's feet"),
]

EXPECTED_FILES = [
    "config.py", "common.py", "run_all.py", "_selftest.py",
    "step1_load.py", "step2_passes.py", "step3_sync.py", "step4_clean.py",
    "step5_dataset.py", "step6_explore.py", "step7_baselines.py",
    "step8_model.py", "step9_train.py", "step10_analysis.py",
    "diagnose_direction.py", "inspect_schema.py", "export_events_excel.py",
    "check_leakage.py", "check_version.py", "device.py", "make_bundle.py",
    "check_symbols.py",
    "verify/section8_results.py", "verify/table4_ablation.py",
    "verify/table3_sync.py", "verify/sample_flow.py",
    "verify/architecture_checks.py", "verify/reproduce_all.py",
    "verify/vutil.py", "verify/README.md", "LICENSE",
    "verify/data_provenance.py", "verify/no_quick_results.py",
    "verify/real_run_summary.py", "REAL_RUN_STATUS.md",
    "reference/idsse_reference_counts.json", "local_data.py", "netguard.py",
    "step0_inspect_local_data.py", "verify/inspect_local_idsse.py",
    "_selftest_local_xml.py",
    "provenance.py", "experiment.py", "metrics_lib.py", "probes.py",
    "providers/__init__.py", "providers/base.py", "providers/idsse.py",
    "providers/pff.py", "UPDATE_NOTES.md",
    "step23_canonical.py", "step24_cross_offset.py", "step25_ball_mask.py",
    "step26_flatness_validation.py", "step27_kick_proxy.py",
    "step28_masella_probe.py", "step29_cross_provider.py",
    "step30_size_curve.py", "step31_paper_figures.py",
    "step11_pretrain.py", "step12_ablation.py",
    "step13_inspect.py", "step14_report.py", "step15_figures.py", "step16_tactical.py", "step17_leadsweep.py", "step18_metrics.py", "step19_architecture.py", "step20_databallpy.py", "step21_features.py", "step22_curves.py",
]

print("=" * 62)
print("Version check")
print("=" * 62)

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def hash_report():
    """Compare every shipped file against the manifest built with the zip.

    The marker checks below say WHICH FEATURE is missing, in words. This says
    WHICH FILES differ, exhaustively - which is what you want after copying only
    some files across from a newer download.
    """
    if not os.path.isfile("manifest.json"):
        return None
    with open("manifest.json") as fh:
        data = json.load(fh)
    # older manifests were a flat {file: hash} map
    want = data.get("_files", data)
    stamp = data.get("_release", "unknown")
    built = data.get("_built", "unknown")
    print(f"Release {stamp}   built {built}")
    if os.path.isfile("CHANGELOG.md"):
        print("See CHANGELOG.md for what changed in each release.")

    n_files = len(want)
    stale, missing, extra = [], [], []
    for name, digest in want.items():
        if not os.path.isfile(name):
            missing.append(name)
        elif sha(name) != digest:
            stale.append(name)
    for f in sorted(os.listdir(".")):
        if f.endswith(".py") and f not in want and f != "make_manifest.py":
            extra.append(f)
    return stale, missing, extra, n_files


bad = [c for c in CHECKS if len(c) != 3]
if bad:
    print("check_version.py is itself malformed:", bad)
    sys.exit(2)

report = hash_report()
if report is not None:
    stale, missing, extra, n_files = report
    if stale or missing:
        print("\nFILES THAT DO NOT MATCH THIS RELEASE:\n")
        for f in missing:
            print(f"  MISSING   {f}")
        for f in stale:
            print(f"  OLD COPY  {f}")
        if extra:
            print()
            for f in extra:
                print(f"  (not part of this release: {f})")
        print(f"\nCopy those {len(stale) + len(missing)} file(s) across, or "
              "replace the whole folder.")
        print("Keep work/ either way - it holds the parsed matches.")
        sys.exit(1)
    print(f"All {n_files} files match this release.")

missing_files = [f for f in EXPECTED_FILES if not os.path.isfile(f)]
if missing_files:
    print("\nMISSING FILES:")
    for f in missing_files:
        print(f"  {f}")

stale = []
for filename, marker, what in CHECKS:
    if not os.path.isfile(filename):
        stale.append((filename, what, "file missing"))
        continue
    with open(filename, encoding="utf-8") as fh:
        if marker not in fh.read():
            stale.append((filename, what, "OLD VERSION"))

if stale:
    print("\nOUT OF DATE:")
    for filename, what, why in stale:
        print(f"  {filename:22s} {why:12s} - {what}")
    print("\nFix: delete this whole folder and extract the zip fresh.")
    print("     Keep your work/ folder first - it holds the parsed matches and")
    print("     re-parsing all seven takes a long time:")
    print("       1. move work/ somewhere safe")
    print("       2. delete the project folder")
    print("       3. extract the zip to a clean location")
    print("       4. move work/ back in")
    sys.exit(1)

if missing_files:
    sys.exit(1)

print("\nAll files present and current.")
print("\nCached matches in work/:")
if os.path.isdir("work"):
    pk = sorted(f for f in os.listdir("work") if f.endswith("_parsed.pkl"))
    print(f"  {len(pk)} of 7 parsed" + (f": {', '.join(p[:6] for p in pk)}" if pk else ""))
else:
    print("  work/ does not exist yet")
