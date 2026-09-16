"""Run every step in order. Stops at the first failure.

This is the full pipeline including the exploratory and presentation steps. For
just the numbers that go in the paper, use verify/reproduce_all.py, which runs
the canonical run, the experiments and the verification scripts, and writes a
report of what actually executed.

Run:  python run_all.py
"""

import subprocess
import sys

STEPS = [
    ("check_symbols.py", "static check for undefined names"),
    ("step1_load.py", "load one match and check its shape"),
    ("step2_passes.py", "extract open-play passes and count the event stages"),
    ("step3_sync.py", "synchronise events with tracking"),
    ("step4_clean.py", "filter and derive velocities"),
    ("step5_dataset.py", "build model-ready samples at the canonical lead"),
    ("check_leakage.py", "probe for shortcuts before trusting any accuracy"),
    ("step6_explore.py", "exploration figures"),
    ("step7_baselines.py", "baselines"),
    ("step8_model.py", "model self-test"),
    ("step11_pretrain.py", "self-supervised pretraining on unlabelled frames"),
    ("step23_canonical.py", "THE canonical run: per-pass predictions and metadata"),
    ("step9_train.py", "movement history versus a single frame"),
    ("step10_analysis.py", "error analysis on the canonical model"),
    ("step12_ablation.py", "which factors actually help"),
    ("step13_inspect.py", "render individual passes so a human can check them"),
    ("step14_report.py", "assemble everything into one readable HTML report"),
    ("step15_figures.py", "figures for the presentation and the paper"),
    ("step16_tactical.py", "pitch, trajectory and space-control figures"),
    ("step17_leadsweep.py", "accuracy against the window-end offset"),
    ("step18_metrics.py", "ranking, calibration and per-position metrics"),
    ("step19_architecture.py", "the architecture diagram"),
    ("step20_databallpy.py", "the same probe on DataBallPy's synchronisation"),
    ("step21_features.py", "where every feature comes from"),
    ("step22_curves.py", "learning curves, ROC and precision-recall"),
    ("step24_cross_offset.py", "does the model DEPEND on the near-anchor shortcut"),
    ("step25_ball_mask.py", "ball masking at test time, no retraining"),
    ("step26_flatness_validation.py", "synthetic validation of the flatness criterion"),
    ("step27_kick_proxy.py", "estimated kick proxy as a second anchor"),
    ("step28_masella_probe.py", "passer-facing angle as a one-line probe"),
    ("step29_cross_provider.py", "the same probes on a second provider, if available"),
    ("step30_size_curve.py", "training size versus intervention gain"),
    ("step31_paper_figures.py", "figures A to F, from the results files"),
    ("verify/sample_flow.py", "sample flow counted from the arrays"),
    ("verify/section8_results.py", "every result-section metric, from one file"),
    ("verify/table4_ablation.py", "ablation bookkeeping and additivity"),
    ("verify/table3_sync.py", "sync comparison and heterogeneity"),
    ("verify/architecture_checks.py", "architecture assertions"),
]

for script, what in STEPS:
    print(f"\n{'#' * 60}\n# {script} - {what}\n{'#' * 60}")
    r = subprocess.run([sys.executable, script])
    if r.returncode != 0:
        print(f"\nFAILED at {script}. Fix this before continuing.")
        sys.exit(1)

print("\nAll steps finished. Figures are in figures/, tables in work/.")
print("Every reported table carries the run id in work/run_metadata.json.")
