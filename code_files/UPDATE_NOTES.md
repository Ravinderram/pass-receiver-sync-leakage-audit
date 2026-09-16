# Update notes

This release reorganises the project around **one canonical run** whose identity
is computed, not remembered, and adds the experiments that separate "a shortcut
is available" from "the trained model uses it".

Nothing about the model architecture changed. It is still 13 time steps, 23
objects, 11 features, a shared GRU with hidden size 64, one transformer encoder
block with four heads and an optional geometric edge bias, a two-layer teammate
head, masking for absent teammates, ten-way output, cross-entropy loss and
**35,909 parameters** (asserted by `verify/architecture_checks.py`).

---

## 1. What was changed

### One canonical configuration
`config.py` now ends with a CANONICAL EXPERIMENT DEFINITION listing the
constants that define the run: `FIT_MATCHES`, `VAL_MATCH`, `TEST_MATCHES`,
`BASELINE_TRAIN_MATCHES`, `PRETRAIN_MATCHES`, `PREDICT_LEAD_S`, `EPOCHS`,
`EARLY_STOPPING`, `PATIENCE`, `MODEL_SELECTION_METRIC`,
`RESTORE_BEST_CHECKPOINT`, `SEEDS`, the architecture constants, the
preprocessing and augmentation flags, and `SOFTWARE_ASSUMPTIONS`.
`config.canonical_config()` returns them as a dict; `python config.py` prints it.

`ANCHOR = "emitted_sync_frame"` is part of that definition. Every offset in this
project is measured backwards from the frame the synchroniser emits. No script
describes an offset as being "before contact".

### Run identity and provenance
New `provenance.py` hashes the canonical config, the sample arrays and the
pretrained encoder into a short **run id**. Every CSV now begins with
`run_id, seed, lead_s, anchor, model_variant, test_match`. Verification scripts
recompute the id and refuse to report a file that does not match; `--allow-stale`
downgrades that to a warning.

`work/run_metadata.json` (and a copy keyed by run id) records timestamp, git
commit, full config, training/validation/test matches, lead, seeds, epoch cap,
early-stopping settings, whether pretraining was used, the selected epoch per
seed, the parameter count, the sample counts and a sha of the prediction file.

### One trainer, one model
New `experiment.py` holds the only training function and a checkpoint cache keyed
on the training signature, lead, anchor, seed and the actual fit arrays. Previous
versions trained a separate model per test match on identical training data, and
step9, step12, step17, step18 and step22 each trained their own. Now each seed
trains once and every script reuses that checkpoint, so the canonical run, the
lead sweep, the cross-offset matrix and the ablation share the same model at the
canonical offset.

Validation and early stopping are back on for the canonical run: selection
monitors validation top-1 only, the best epoch is restored, and `stopped_at`,
`selected_epoch`, `epochs_run` and `best_val_top1` are exported per seed. Test
matches are monitored for learning curves and never touch a decision.
`step23_canonical.py` refuses to run if the protocol is violated (test match in
the fit set, validation disabled, missing encoder, wrong parameter count).

### Lead-specific datasets no longer overwrite the canonical ones
The old lead sweep rebuilt `work/dataset_*.npz` at every offset and restored them
at the end. Now non-canonical offsets go to `work/lead_cache/` with a source key,
and the canonical arrays are never touched. `step5_dataset.py` also stores
`pass_row`, `anchor_frame` and `half_id`, so predictions trace back to individual
passes and datasets built at different offsets can be intersected pass by pass.

### Metrics, tests and bookkeeping
- `metrics_lib.py` holds every definition (top-k, MRR, mean rank, Brier, ECE,
  pair-level ROC-AUC and AP, McNemar, per-position PRF). All are computed from
  the exported predictions.
- ROC-AUC and AP are labelled descriptive: one positive per pass, candidates
  within a pass are not independent.
- Per-position metrics are labelled as receiving-**position category**
  evaluation, distinct from receiver identity, with support, precision, recall,
  F1 and macro/weighted means, and the count of passes dropped for an unknown
  position code.
- The ablation now types each setting as reference, individual or composite.
  Composites are never counted as interventions and never added to a sum of
  individual gains; `verify/table4_ablation.py` compares each composite with the
  sum of its own components and reports the difference as an interaction. Gains
  use the paired per-seed difference, with its sd and a t-interval.
- Sample counts are computed from the data everywhere
  (`work/event_counts.csv`, `work/dataset_counts.csv`,
  `verify/sample_flow.py`), and the flow check fails if
  training + validation + test does not equal the usable samples.

### Claims kept conservative
`check_leakage.py`, the lead sweep and the figures now distinguish three claims
that were previously blurred: a shortcut being *available*, a trained model
*using* it, and an offset being *measured against ground truth* (which is not
claimed anywhere, because no public dataset provides contact times). The passer
signal is reported as unresolved, never as leakage.

Hard-coded measurement values were removed from `config.py`, `README.md`,
`step14_report.py`, `step15_figures.py`, `step19_architecture.py`,
`step21_features.py` and `common.py`, and replaced with pointers to the file that
holds the current measurement.

---

## 2. New experiments

| Script | Question it answers |
|---|---|
| `step23_canonical.py` | the canonical run; exports per-pass predictions and run metadata |
| `step24_cross_offset.py` | train-offset x test-offset matrix: does a model trained near the anchor **depend** on the shortcut, or merely have it available? |
| `step25_ball_mask.py` | the same trained model with ball information removed at test time, no retraining; two mask levels (ball token, and ball token plus the derived distance-to-ball feature) |
| `step26_flatness_validation.py` | synthetic validation of the flatness criterion, on cases whose truth is known by construction, plus a sensitivity sweep |
| `step27_kick_proxy.py` | an **estimated** kick proxy from ball-speed onset, compared with the emitted frame; never called ground truth |
| `step28_masella_probe.py` | the Masella passer-facing angle as a one-line probe (not a reproduction of their model) |
| `step29_cross_provider.py` | the same probes through `providers/`, on a second provider when its data are present |
| `step30_size_curve.py` | do intervention gains stabilise as training data grow? |
| `step31_paper_figures.py` | figures A to F, each drawn from a results file |
| `verify/architecture_checks.py` | the model is the one the paper describes |
| `verify/sample_flow.py` | the sample-flow table, counted from the arrays |
| `verify/reproduce_all.py` | one entry point, with a report of what actually ran |

---

## 3. New outputs

Predictions and metadata
`work/metrics_predictions_<run_id>.csv`, `work/run_metadata.json`,
`work/run_metadata_<run_id>.json`, `work/canonical_history.csv`,
`work/reproduce_all_report.json`, `work/checkpoints/`.

Summary tables
`work/canonical_metrics_summary.csv`, `work/per_position_metrics.csv`,
`work/paired_tests.csv`, `work/strata_metrics.csv`,
`work/ablation_summary.csv`, `work/ablation_additivity.csv`,
`work/ablation_definitions.csv`, `work/sample_flow.csv`,
`work/sample_flow_by_match.csv`, `work/event_counts.csv`,
`work/dataset_counts.csv`, `work/reliability.csv`.

Experiments
`work/cross_offset_results.csv`, `work/cross_offset_matrix.csv`,
`work/ball_mask_results.csv`, `work/ball_mask_summary.csv`,
`work/ball_mask_predictions.csv`, `work/leadsweep_summary.csv`,
`work/size_curve_results.csv`, `work/size_curve_summary.csv`,
`work/flatness_validation.csv`, `work/flatness_sensitivity.csv`,
`work/leakage_probes.csv`, `work/passes_kick_proxy.csv`,
`work/kick_proxy_curves.csv`, `work/masella_probe.csv`,
`work/cross_provider_status.json`, `work/table3_sync.csv`.

Figures
`figures/paper/figA_topk_by_match`, `figB_cross_offset`, `figC_ball_mask`,
`figD_leadsweep`, `figE_ablation`, `figF_size_curve`,
`fig_flatness_validation`, `fig_kick_proxy`, `fig_masella_probe`
(PNG and PDF), plus the existing fig1-fig19.

---

## 4. How to run each important script

```bash
# everything the paper depends on, with a report of what ran
python verify/reproduce_all.py --with-data --with-pretrain

# the canonical run on its own (needs steps 1-5 and step11 first)
python step23_canonical.py

# the two experiments that test dependence rather than availability
python step24_cross_offset.py
python step25_ball_mask.py

# ablation; add --robust for config.ROBUST_SEEDS on the unstable rows
python step12_ablation.py
python step12_ablation.py --robust
python verify/table4_ablation.py            # add --robust to read that file

# does the effect hold as data grow
python step30_size_curve.py

# diagnostics and probes
python check_leakage.py
python step17_leadsweep.py
python step26_flatness_validation.py
python step27_kick_proxy.py                  # --probes-only to skip the model stage
python step28_masella_probe.py
PFF_WC2022_DIR=/path/to/pff python step29_cross_provider.py

# tables and figures, all from results files
python verify/sample_flow.py
python verify/section8_results.py
python step18_metrics.py
python step22_curves.py
python step31_paper_figures.py
python verify/architecture_checks.py
```

Reproducibility command:

```bash
python verify/reproduce_all.py --with-data --with-pretrain
```

---

## 4a. Local dataset mode

`config.IDSSE_LOCAL_DATA_DIR` is the single place a data path is configured.
When it is set (or `IDSSE_DATA_DIR` is exported), the project is in local-only
mode:

- `common.load_match()` reads the folder through `floodlight.io.dfl` and never
  imports the downloader; a missing match is an error naming the match and the
  folder, never a download
- `netguard.py` patches `socket.connect` for the run, so an accidental download
  raises `NetworkBlocked` naming the host
- `local_data.py` indexes the folder by OPENING files: an XML is classified by
  its root element and attributes, so no layout is assumed and project caches or
  stray files in the same folder are ignored
- `step0_inspect_local_data.py` reports the whole folder, a per-match table for
  the seven canonical matches, a parse of a representative match (halves, frames,
  framerate, objects, pitch, teamsheets, position codes, event columns and ids,
  ball-alive frames) and frame counts against `config.EXPECTED_FRAMES`
- parsed caches carry the raw-file fingerprint in `work/<match>_parsed.meta.json`,
  so replacing a source file invalidates the cache and reparses
- the raw fingerprint feeds `provenance.canonical_run_id()`, so changing the
  local data changes the run id and old metrics cannot be reported against it
- every run records `data_source` (`local_idsse`, `figshare_download`,
  `synthetic`) and prints `DATA SOURCE` / `NETWORK DOWNLOAD` before anything else

The raw folder is only ever read. Everything generated stays in `work/` and
`figures/`.

## 4b. Real-data mode versus quick mode

`verify/reproduce_all.py --with-data --with-pretrain` is the canonical real-data
reproduction command. `--quick` (or `PRX_QUICK=1`) is for plumbing only.

Three gates now enforce the distinction, all failing closed:

- `verify/data_provenance.py` verifies each parsed match against the frame counts
  published in the dataset paper, the framerate, the DFL position codes and the
  event identifiers. Outside quick mode it exits 1 unless the verdict is `REAL`,
  and it runs before anything trains.
- `verify/no_quick_results.py` fails if any paper-facing file carries a `quick-`
  run id. It is a required stage of `reproduce_all.py`, so a contaminated run
  cannot exit zero.
- `verify/real_run_summary.py` writes `work/REAL_RUN_SUMMARY.md` from the
  generated files, stating `Valid for paper reporting: YES` or `NO` with the
  reason, so the judgement is recorded rather than remembered.

See `REAL_RUN_STATUS.md` for the state of the real-data run in this archive.

## 5. Which experiments were actually executed

**Important: the real dataset was NOT available in the environment where this
update was written**, and the real-data run was therefore never performed.
`REAL_RUN_STATUS.md` records the exact refusals (`x-deny-reason:
host_not_allowed` for figshare, the DOI, Nature and Zenodo) and the check of the
official GitHub companion repository, which holds no data.

Every run below used the synthetic fixture from `_selftest.py` in quick mode.
Those runs exercise the plumbing, not the football. **All of their outputs were
deleted from this archive** rather than shipped: `work/` and `figures/` are
empty. Nothing in this package is a result.

Executed end to end on the synthetic fixture (19 of 20 stages ok):
`step2`, `step3`, `step4`, `step5`, `step11`, `step23`, `step7`,
`check_leakage`, `step17`, `step24`, `step25`, `step12`, `step30`, `step26`,
`step27`, `step28`, `step29`, `verify/sample_flow`, `verify/section8_results`,
`verify/table4_ablation`, `verify/architecture_checks`, `step18`, `step22`,
`step31`, plus `step6`, `step9`, `step10`, `step13`, `step14`, `step15`,
`step16`, `step19`, `step21`.

Also verified:
- `verify/architecture_checks.py` passes all 11 assertions, including the 35,909
  parameter count, the -1e9 masked logit, the `(batch, 13, 23, 11)` input and
  `(batch, 10)` output shapes, and edge attention behaving differently when on
  and off.
- The stale-result guard fires: changing `LR` in `config.py` made
  `verify/section8_results.py` exit 1 rather than report the previous run.
- `check_symbols.py` (pyflakes) reports no undefined names across 59 files.

Not executed:
- **`step20_databallpy.py` and `verify/table3_sync.py`** — `databallpy` is not
  installed in that environment, so `work/databallpy_comparison.csv` does not
  exist. `reproduce_all.py` marks the stage optional and reports it as not run;
  no number from it is reported anywhere.
- **`step29_cross_provider.py` on PFF** — see below.
- **Any run on the real IDSSE data.**

---

## 6. External datasets that were unavailable

**PFF FC World Cup 2022.** The adapter is implemented in `providers/pff.py` and
documents the exact layout it expects (`metadata/`, `rosters/`, `tracking/`
as JSONL, `events/`). It is not present in this environment, so
`step29_cross_provider.py` records `"dataset not available - not run"` in
`work/cross_provider_status.json` and produces no numbers for it. No
cross-provider claim is made, and the script says so explicitly when fewer than
two providers ran. Point `PFF_WC2022_DIR` at a local copy to enable it. Because
PFF's event feed is frame-tagged against its own tracking rather than aligned
post hoc, its field names may differ between releases; the adapter reads
defensively and skips what it cannot resolve rather than guessing.

**Masella et al.** Their data and code are not public. `step28_masella_probe.py`
implements their angle definition as a one-line probe only, states that this is
not a reproduction of their model, and makes no claim about their reported
accuracy. Their random pass-level split is noted as a condition relevant to
interpretation, not as evidence against their result.

---

## 7. Which results are canonical

Canonical = produced by `step23_canonical.py` under `config.canonical_config()`
and carrying the current run id. In practice:

- `work/metrics_predictions_<run_id>.csv` is the source of truth for every
  result-section number. Everything in `canonical_metrics_summary.csv`,
  `per_position_metrics.csv`, `paired_tests.csv`, `strata_metrics.csv`,
  `metrics.csv` and `curves_summary.csv` is derived from it.
- The **headline** is `model = canonical, aggregation = mean_over_seeds` in
  `canonical_metrics_summary.csv`. The seed ensemble is a separate, labelled row
  (`canonical_seed_ensemble`) and must never be mixed into the headline.
- The canonical offset is `config.PREDICT_LEAD_S` before the emitted frame.
  Lead-sweep and cross-offset rows at other offsets are experiments, not the
  headline.
- Ablation rows of kind `composite` are configurations, not interventions.
  `everything` is the canonical configuration and reuses its checkpoints.
- Anything with `quick-` in the run id is a plumbing test, not a result.

---

## 8. Known limitations

- **The anchor is not contact.** Every offset is relative to the frame the
  synchroniser emits, which sits after contact by a match-varying amount, so a
  fixed offset is not the same moment in every match. The kick proxy in step27 is
  an estimate from ball kinematics: the low-pass filter in step4 biases the onset
  late, and a ball already moving before the kick has no clean onset, so its
  acceptance rate is reported alongside it.
- **Ball masking is not a causal proof.** A masked input is out of distribution
  for any model, so some drop is expected even without dependence. The evidence
  is the contrast between the model trained at the emitted frame and the one
  trained at the canonical offset. A retrained ball-free model would be the
  stronger control and is not run.
- **The flatness criterion has a stated blind spot.** It separates a step-like
  leak from a gradual ramp, but it cannot resolve a gradual one: a player turning
  towards his target and a soft leak produce the same curve. Step26 measures
  where the boundary falls on synthetic data and labels the ambiguous case
  unresolved rather than guessing.
- **Small samples.** Most interventions do not clear twice their paired seed
  spread, and some change sign between runs. `--robust` raises the seed count for
  the unstable rows, and step30 asks whether the sign is stable across training
  sizes; neither turns a few thousand passes into many.
- **The size curve is not a scaling law.** Four nested subsets of one dataset
  support a stability check and nothing more.
- **One dataset, one league, one provider, men's professional football.** The
  synchronisation behaviour measured may reflect this source's timestamping
  rather than a general property; the cross-provider machinery exists precisely
  because that cannot be settled from inside this repository.
- **Ten candidates.** A pass played into space rather than to a player cannot be
  represented, and the roughly 20% of passes whose recipient can be an opponent
  are still dropped.
- **Checkpoint cache.** `work/checkpoints/` grows across configurations. It is
  keyed on content, so stale entries are never reused wrongly, but delete the
  directory if you want a clean retrain.
