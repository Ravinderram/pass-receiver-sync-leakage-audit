# Changelog

`check_version.py` prints the release id your copy came from. Find it below to
see what changed since, and which files to replace.

Always replace `manifest.json` alongside any file you copy across. It holds the
checksums, so an old manifest will flag a correctly updated file as wrong.

---

## latest  (the release check_version prints)

**New files: `provenance.py`, `experiment.py`, `metrics_lib.py`, `probes.py`,
`providers/` (`base.py`, `idsse.py`, `pff.py`), `step23_canonical.py`,
`step24_cross_offset.py`, `step25_ball_mask.py`,
`step26_flatness_validation.py`, `step27_kick_proxy.py`,
`step28_masella_probe.py`, `step29_cross_provider.py`, `step30_size_curve.py`,
`step31_paper_figures.py`, `verify/vutil.py`, `verify/sample_flow.py`,
`verify/architecture_checks.py`, `verify/reproduce_all.py`, `UPDATE_NOTES.md`.
Changed: `config.py`, `common.py`, `check_leakage.py`, `check_symbols.py`,
`check_version.py`, `make_manifest.py`, `run_all.py`, `step2_passes.py`,
`step5_dataset.py`, `step7_baselines.py`, `step9_train.py`, `step10_analysis.py`,
`step11_pretrain.py`, `step12_ablation.py`, `step13_inspect.py`,
`step14_report.py`, `step15_figures.py`, `step17_leadsweep.py`,
`step18_metrics.py`, `step19_architecture.py`, `step20_databallpy.py`,
`step21_features.py`, `step22_curves.py`, `verify/section8_results.py`,
`verify/table3_sync.py`, `verify/table4_ablation.py`, `verify/README.md`,
`README.md`, `requirements.txt`, `manifest.json`.**

The project is now organised around one canonical run whose identity is computed
rather than remembered, and it adds the experiments that separate "a shortcut is
available" from "the trained model uses it". `UPDATE_NOTES.md` is the full
account; the short version:

- `config.py` gained a CANONICAL EXPERIMENT DEFINITION and
  `canonical_config()`. `provenance.py` hashes it, together with the sample
  arrays and the pretrained encoder, into a run id that every CSV carries and
  that the verify scripts check before reporting anything.
- `experiment.py` is now the only place a model is trained. One model per seed
  is trained and cached, and every script reuses it, so the several coexisting
  headline accuracies are no longer possible. Early stopping on the validation
  match is back on for the canonical run, and the selected epoch is exported.
- `step23_canonical.py` exports per-pass predictions (logits, probabilities,
  top-k lists, rank, hit flags, positions) for the model per seed, the seed
  ensemble and every baseline on the same passes. Every result-section metric,
  including the McNemar test against gradient boosting, is derived from that one
  file by `verify/section8_results.py`.
- New experiments: the train-offset x test-offset matrix (step24), ball masking
  without retraining (step25), a synthetic validation of the flatness criterion
  (step26), an estimated kick proxy (step27), the Masella angle as a one-line
  probe (step28), the cross-provider probe interface (step29), and the
  training-size curve (step30).
- The ablation distinguishes individual factors from composite settings, so the
  aggregate row is never counted as an intervention and nested mirror conditions
  are never double-counted. The arithmetic is generated, not written down.
- Sample counts, ablation gains and every figure now come from generated files.
  Hard-coded measurement values were removed from the code comments and the
  README, which point at the file holding the current number instead.
- Claims are kept apart by construction: no script states a synchronisation
  offset measured against ground truth, offsets are anchored on the emitted
  frame, and a gradually rising probe is reported as unresolved rather than as
  leakage.

Caveat on this release: it was prepared in an environment without the real
dataset, so every run was on the synthetic fixture in quick mode. Those outputs
carry `quick-` in the run id. Rerun on the real data before quoting any number.

### Previous entry


**Files: `step20_databallpy.py`, `step18_metrics.py`, `step5_dataset.py`,
`check_symbols.py` (new), `run_all.py`, `check_version.py`, `requirements.txt`,
`manifest.json`**

`step20_databallpy.py` raised `NameError: require_databallpy is not defined`.
The function had been removed by an edit that replaced everything between two
other function definitions; it happened to sit between them. The file still
compiled, and the version checker still passed, because it looks for text
markers and the marker it checked was in a different function.

- `require_databallpy` restored.
- `check_symbols.py` added: runs pyflakes and fails on undefined names. It is
  now the first step in `run_all.py`. This is the check that would have caught
  the bug in one second.
- Fixed a real issue it found: `step18_metrics.py` re-imported `train_one`
  inside a loop, shadowing the module-level import.
- Cleaned the remaining lint warnings so a genuine one stays visible.

### Previous entry

**Files: `step22_curves.py` (new), `run_all.py`, `check_version.py`,
`README.md`, `manifest.json`**

Added the three diagnostics a machine learning write-up is expected to show and
this project did not have.

Learning curves are the important one — training, validation and test accuracy
per epoch on one plot, with the early-stopping epoch marked. On the real data
the training line keeps climbing past epoch 21 while test flattens, which is
exactly the overfitting that a fixed epoch count was hiding.

ROC and precision-recall are computed over every (pass, candidate) pair, since
the model's classes are arbitrary slot indices and a per-slot ROC would be
noise. Measured on J03WMX: ROC AUC 0.947 against random 0.500, and average
precision 0.725 against random 0.100. PR is the better of the two here because
the positive rate is 1 in 10, and ROC's false-positive rate is diluted by a
large true-negative pool.

### Previous entry

**Files: `config.py`, `step9_train.py`, `step12_ablation.py`,
`step18_metrics.py`, `check_version.py`, `manifest.json`**

Added a validation match and early stopping. The epoch count had been a guess,
and it was the wrong guess.

Measured on the real data, training accuracy climbs to 0.845 by epoch 40 while
test accuracy peaks around epoch 20 and then decays — 0.693 down to 0.654 on
J03WMX. Training to a fixed 40 epochs was discarding about four points.

- `VAL_MATCH` is carved out of `TRAIN_MATCHES`, never out of the test set, so
  nothing the model decides about itself touches the matches it is reported on.
- Training stops after `PATIENCE` epochs without a validation improvement, and
  the weights are rolled back to the best epoch. Without the rollback, early
  stopping only saves time.
- `EPOCHS` is now a cap of 60 rather than a target of 40.
- `train_one` records training accuracy per epoch, so the generalisation gap is
  visible in the history rather than something to go and measure separately.
- Steps 12 and 18 use the same split, so every reported number comes from the
  same procedure.

On J03WMX, one seed: validation picked epoch 21 and the restored model scored
0.672 on test, against a training accuracy of 0.72 at that point.

### Previous entry

**Files: `step9_train.py`, `step18_metrics.py`, `manifest.json`**

Fixed a real inconsistency: `step18_metrics.py` had its own hand-written
training loop that silently omitted mirror augmentation and test-time
averaging, so it reported a model six points weaker than `step9` and `step12`
did. The same project was publishing two different accuracies for the same
model.

It now calls `train_one` from `step9`, so there is one training path and one
number. `train_one` gained `return_probs=True` to make that possible, and
step 18 ensembles over seeds rather than trusting a single run.

Step 18 also prints a HEADLINE block first — one number, the top-1 accuracy —
before any of the diagnostics. Leading with eight metrics is how a working model
ends up looking unclear.

### Previous entry

**Files: `step20_databallpy.py`, `manifest.json`**

Runs all seven matches by default instead of one, and aggregates them. One match
is an anecdote; the question is whether they agree. The summary prints the probe
per match, the means, and how many matches exceed the threshold, and says
plainly when they do not agree.

Nothing else needs re-running: step 20 reads `passes.csv`, `passes_synced.csv`,
`clean_*.npz` and `dataset_*.npz`, all of which already exist for every match.
Only the DataBallPy download is per-match, and it caches.

    python step20_databallpy.py            # all seven
    python step20_databallpy.py J03WN1     # one

### Previous entry

**Files: `step20_databallpy.py`, `manifest.json`**

Added the control row the comparison needed. The headline "ours" number is
measured 0.4 s before contact while DataBallPy's is at its own chosen frame, so
comparing them directly conflates two things: which frame was chosen, and how
far back we then step. `ours, at our sync frame` steps back zero, so only the
frame choice differs.

Also stated two caveats in the output rather than leaving them for a reviewer:
ball SPEED is not comparable between the tools because we low-pass filter before
differencing and they do not, and the timestamp fallback can only depress the
probe, never inflate it.

### Previous entry

**Files: `step2_passes.py`, `step20_databallpy.py`, `manifest.json`**

The DataBallPy probe reported `no receiver id=981` — its Sportec parser leaves
`to_player_id` empty, so the receiver had to come from elsewhere.

Both tools read the same DFL event file, which carries a `Recipient` attribute,
and step 2 already parsed it. So our own labels are joined onto DataBallPy's
frames: same ground truth, same metric, only the frames differ, which is exactly
what the comparison is meant to isolate.

- `step2_passes.py` now keeps the DFL `EventId`, which is the exact join key.
- `step20` joins on it, falling back to nearest-timestamp matching within half a
  second if the ids do not line up.
- `to_column()` accepts DFL PersonIds (`DFL-OBJ-...`). DataBallPy's Sportec
  parser uses the same PersonId as its player id, so they map directly.

**Re-run `step2_passes.py` before `step20`**, or the join key will be missing.

## 9b0cf6f0

**Files: `step20_databallpy.py`, `manifest.json`**

Fixed the receiver lookup that made the DataBallPy probe report `n=0` while
still filling in the ball-speed columns — a confusing state to debug.

- The receiver id was resolved *after* the speed and distance were recorded, so
  a failure there left those populated and the count at zero. It now happens
  first, and has its own skip reason.
- `player_id_to_column_id` is declared as taking an `int`, but a pandas column
  that has ever held a NaN comes back as `float`, so ids arrived as `12345.0`
  and never matched. A tolerant `to_column()` helper handles float, int and
  string forms.
- `--inspect` now reports the dtype, null count and mapping success rate of
  `player_id` and `to_player_id`, which is what you need when the probe returns
  nothing.
- The ball-speed contrast between the two pipelines is printed in the summary.
  It is informative on its own, and it is available even when the probe cannot
  be computed.

## 3eef9cd3

**Files: `step20_databallpy.py`, `manifest.json`**

Rewrote the probe against DataBallPy's real API. It had been matching tracking
column names like `home_7_x` against the event table's `team_id`, which is a DFL
club identifier like `DFL-CLU-00000G`. Those never match, so every pass was
skipped. It now uses `game.get_column_ids()` and `game.player_id_to_column_id()`.

Added two diagnostics: the probe always reports how many passes it used and why
the rest were dropped, and it warns if the median ball speed is above 50 m/s,
which would mean the frame spacing or the units are wrong.

## 24f77d22

**Files: `step20_databallpy.py`, `check_version.py`, `make_manifest.py`,
`manifest.json`, `README.md`**

- `manifest.json` now carries a release id and build time. It ships inside the
  zip, so an old copy will happily confirm old files; the stamp is what makes
  that visible.
- Removed unused `load_match` and `coords` imports from `step20`, so it genuinely
  needs neither torch nor floodlight. That matters because databallpy pins
  `numpy < 2.3` while floodlight wants `>= 2.1.1`, and the overlap is narrow —
  step 20 can now be run in a throwaway environment.
- README documents the numpy conflict and the `python -m pip` requirement on
  Windows.

## a3124a34

**Files: `check_version.py`, `manifest.json`**

Repaired three malformed entries in `CHECKS`, where an earlier find-and-replace
across filename lists had leaked into the tuples and turned 3-tuples into
5-tuples. Caught by the self-guard added for exactly this.

## Earlier

Release ids were not stamped before `a3124a34`. If `check_version.py` reports
`unknown`, replace the whole folder — keep `work/`, which holds the parsed
matches and takes a long time to rebuild.
