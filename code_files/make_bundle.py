"""Package the results that are worth sharing, and skip the ones that are not.

Not everything in work/ is worth uploading. The rule:

  Numbers you decided to record go in CSVs. They answer the questions you
  already asked.

  The sample arrays are the model's actual input. They answer questions nobody
  has thought of yet - which is how the ball-velocity leak in this project was
  found. It appeared in no table; it came out of running a new probe directly
  against dataset_*.npz.

So: all the small tables, the sample arrays, the figures, and the report. The
parsed pickles and the filtered tracking arrays are large, rebuildable, and tell
you nothing you cannot recompute, so they stay out.

Run:  python make_bundle.py
Out:  results_bundle.zip
"""

import os
import zipfile

import config as C

OUT = "results_bundle.zip"

# (pattern, why it is here)
INCLUDE = [
    ("work/run_metadata.json", "what produced the canonical numbers"),
    ("work/reproduce_all_report.json", "which stages actually ran"),
    ("work/passes.csv", "every pass, before synchronisation"),
    ("work/passes_synced.csv", "sync frames, distances, orientation, ball-to-passer"),
    ("work/event_counts.csv", "event stages per match"),
    ("work/dataset_counts.csv", "samples and skips per match"),
    ("work/sample_flow.csv", "the sample-flow table"),
    ("work/sample_flow_by_match.csv", "the same, per match"),
    ("work/baseline_results.csv", "the baselines"),
    ("work/model_results.csv", "history versus a single frame"),
    ("work/canonical_metrics_summary.csv", "the headline metrics"),
    ("work/per_position_metrics.csv", "receiving-position categories"),
    ("work/paired_tests.csv", "McNemar against the baselines"),
    ("work/strata_metrics.csv", "accuracy by lane crowding and distance"),
    ("work/ablation_results.csv", "the ablation, per seed"),
    ("work/ablation_summary.csv", "gains with the noise floor"),
    ("work/ablation_additivity.csv", "composites against their own components"),
    ("work/ablation_definitions.csv", "which factor each setting turns on"),
    ("work/leakage_probes.csv", "probe level and shape per match"),
    ("work/leadsweep_summary.csv", "accuracy and probes against the offset"),
    ("work/cross_offset_matrix.csv", "does the model depend on the shortcut"),
    ("work/ball_mask_summary.csv", "normal versus ball-masked"),
    ("work/size_curve_summary.csv", "training size against intervention gain"),
    ("work/flatness_validation.csv", "synthetic validation of the diagnostic"),
    ("work/flatness_sensitivity.csv", "where the criterion stops resolving"),
    ("work/masella_probe.csv", "passer-facing angle as a one-line probe"),
    ("work/kick_proxy_curves.csv", "curves under the estimated kick proxy"),
    ("work/cross_provider_status.json", "which providers were actually run"),
    ("work/metrics.csv", "ranking and calibration"),
    ("work/curves_summary.csv", "pair-level ROC and AP"),
    ("work/schema.xlsx", "table shapes and column names"),
    ("report.html", "everything assembled, self-contained"),
]

# the per-pass predictions every reported metric is derived from; the filename
# carries the run id, so it is matched by pattern rather than listed
import glob as _glob
INCLUDE += [(p, "per-pass predictions from the canonical run")
            for p in sorted(_glob.glob("work/metrics_predictions_*.csv"))]

# the sample arrays, which is where unplanned diagnostics happen
DATASETS = [f"work/dataset_{m}.npz" for m in C.MATCHES]

SKIP_REASON = {
    "_parsed.pkl": "hundreds of MB, and rebuildable by re-running step 1",
    "clean_": "~100 MB per match, rebuildable by re-running step 4",
    "pretrained_encoder.pt": "weights, not results",
    "events.xlsx": "large and only useful for browsing the raw events",
}


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


if __name__ == "__main__":
    print("Packaging results\n")
    total = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for path, why in INCLUDE:
            if os.path.isfile(path):
                z.write(path)
                size = os.path.getsize(path)
                total += size
                print(f"  {human(size):>9}  {path:<32} {why}")
            else:
                print(f"  {'—':>9}  {path:<32} (not produced yet)")

        print()
        for path in DATASETS:
            if os.path.isfile(path):
                z.write(path)
                size = os.path.getsize(path)
                total += size
                print(f"  {human(size):>9}  {path}")

        print()
        n_fig = 0
        for root, _, files in os.walk(C.FIGS):
            for f in sorted(files):
                if f.endswith(".png"):
                    p = os.path.join(root, f)
                    z.write(p)
                    total += os.path.getsize(p)
                    n_fig += 1
        print(f"  {n_fig} figures (PNG only — the PDFs are for the paper)")

    print(f"\n{OUT}  —  {human(os.path.getsize(OUT))} on disk, "
          f"{human(total)} of content")

    print("\nLeft out on purpose:")
    for k, why in SKIP_REASON.items():
        print(f"  {k:<24} {why}")

    print("\nWhy the .npz files are in here: the CSVs only contain what we")
    print("decided in advance to measure. The sample arrays let someone run a")
    print("check nobody thought of yet. That is exactly how the ball-velocity")
    print("leak was caught — it was in no table.")
