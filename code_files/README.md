# Pass Receiver Prediction — Bundesliga tracking data

Given the 1.5 seconds before a pass, predict which of the passer's ten teammates
receives the ball.

Data: Bassek, Rein, Weber & Memmert (2025), *An integrated dataset of spatiotemporal
and event data in elite soccer*, Sci Data 12:195. CC-BY 4.0.

## Install and run

```bash
pip install -r requirements.txt
python check_version.py             # confirms no file is left over from an older copy
python verify/reproduce_all.py --with-data --with-pretrain
```

`verify/reproduce_all.py` is the entry point for everything the paper reports:
it builds the samples, trains the canonical run, runs every experiment, then
runs the verification scripts that derive the tables. It writes
`work/reproduce_all_report.json` naming each stage, its exit code and how long
it took, so a stage that did not run is never presented as having run.

`python run_all.py` still exists and runs the full pipeline including the
exploratory and presentation steps.

## Two modes, and how they are kept apart

| | real-data production run | quick / fixture mode |
|---|---|---|
| command | `python verify/reproduce_all.py --with-data --with-pretrain` | `python verify/reproduce_all.py --quick`, or `PRX_QUICK=1` |
| data | the seven IDSSE matches from figshare | the synthetic fixture from `_selftest.py`, or any partial copy |
| seeds and epochs | `config.SEEDS`, full epoch cap | reduced |
| run id | plain, e.g. `4f2ab9c113` | prefixed `quick-` |
| valid for the paper | yes, once the three gates below pass | **never** |

Quick mode exists to exercise the plumbing in minutes. It is not a smaller
experiment; its numbers mean nothing about football.

Three gates keep the two apart, and all three fail closed:

1. `verify/data_provenance.py` checks each parsed match against the frame counts
   published in the dataset paper, plus framerate, DFL position codes and event
   identifiers. Outside quick mode it exits 1 unless the verdict is `REAL`, and
   `reproduce_all.py` runs it before anything trains.
2. `verify/no_quick_results.py` scans `work/` and fails if any paper-facing file
   carries a `quick-` run id. It is a required stage, so a contaminated run
   cannot exit zero.
3. The run id itself: change a canonical constant and every verification script
   refuses the old results.

`verify/real_run_summary.py` then writes `work/REAL_RUN_SUMMARY.md` from the
generated files, with **Valid for paper reporting: YES/NO** on its second line.

## Local data setup

Point the project at your downloaded copy of the dataset. One variable, in
`config.py`, and nothing else:

```python
IDSSE_LOCAL_DATA_DIR = r"C:\Users\ranjit\Downloads\IDSSE"   # Windows
IDSSE_LOCAL_DATA_DIR = "/home/ranjit/Downloads/IDSSE"          # Linux
IDSSE_LOCAL_DATA_DIR = "/Users/ranjit/Downloads/IDSSE"         # macOS
```

or, without editing the file, `export IDSSE_DATA_DIR=/path/to/IDSSE`.

Then:

```bash
python step0_inspect_local_data.py          # what is actually in that folder
python step1_load.py J03WMX                 # prove one match parses
python verify/reproduce_all.py --with-data --with-pretrain
```

The last command reads that folder and downloads nothing. With the variable set
the project is in **local-only mode**:

- `common.load_match()` parses the local XML through `floodlight.io.dfl` and
  never imports floodlight's downloader
- `netguard.py` patches `socket.connect` for the whole run, so an accidental
  download raises `NetworkBlocked` naming the host it tried
- a missing match is an error that names it, never a download:
  `ERROR: required real IDSSE match J03WMX was not found under <path>`
- every run prints its two source lines before anything else:

```text
DATA SOURCE: LOCAL IDSSE (/home/ranjit/Downloads/IDSSE)
NETWORK DOWNLOAD: DISABLED
```

The folder is read recursively and never written to. Its layout does not matter:
`step0_inspect_local_data.py` classifies files by opening them, so a flat folder,
one folder per match, or anything else works, and project caches or stray
documents in the same folder are ignored. Everything the project generates stays
under `work/` and `figures/`.

Parsed matches are cached as `work/<match>_parsed.pkl` next to a
`.meta.json` holding the fingerprint of the raw files they came from. Replace or
re-download a file and the fingerprint changes, the cache is discarded, the match
is reparsed, and the run id changes so earlier numbers cannot be quoted against
the new data.

## Checked against the authors' own numbers

`reference/idsse_reference_counts.json` holds the descriptive statistics the
dataset authors published as executed output in `data_summary.ipynb` in their
companion repository (`github.com/spoho-datascience/idsse-data`): 1,002,644
position frames, 11,137 event rows, 5,241 `Play_Pass` and 140 `Play_Cross`, 207
players across 10 teams. `verify/data_provenance.py` checks a parsed `work/`
against them, so a fixture or a partial download cannot pass for the real thing.

Three of this project's definitions were confirmed against that file rather than
assumed:

- `config.EXPECTED_FRAMES` sums to exactly 1,002,644.
- `config.OPEN_PLAY_EIDS = ["Play_Pass", "Play_Cross"]` gives 5,381 open-play
  passes, and correctly leaves out the 681 set-piece pass-like events
  (`ThrowIn_Play_Pass`, `FreeKick_Play_Pass`, `GoalKick_Play_Pass`,
  `KickOff_Play_Pass`, `CornerKick_Play_Cross` and the rest).
- The authors' own KDE example reads the goalkeeper from column 22, i.e. slot 11,
  which is why this project maps slots from the teamsheet rather than by
  position.

Their plotting script offsets the event clock by 1.6 s when it picks a frame for
one goal in one match, with the comment *"offset event clock, pos data"*. That
is a hand-set constant for that figure, not a dataset-wide alignment, which is
why `step3_sync.py` estimates the offset per half and then refines each event.

## One canonical run

Every reported number comes from one run, and the code enforces it rather than
asking you to remember.

`config.py` ends with a CANONICAL EXPERIMENT DEFINITION: the constants that fix
the fit matches, the validation match, the test matches, the lead time, the
epoch cap, early stopping and patience, the seeds, the architecture, the
preprocessing, the augmentation, the normalisation, and whether pretraining is
used. `config.canonical_config()` returns them as a dict.

`provenance.py` hashes that dict, together with the sample arrays and the
pretrained encoder, into a short **run id**. Every CSV this project writes
carries `run_id, seed, lead_s, anchor, model_variant, test_match` as its first
columns, and `work/run_metadata.json` records what produced them: timestamp, git
commit, splits, seed list, selected epoch per seed, parameter count and sample
counts.

Change a canonical constant and the run id changes, so the verification scripts
refuse to report results from the old one:

```
no predictions for run id 4f2ab9c113. Run step23_canonical.py first.
```

## Three claims that are not the same claim

The code keeps these apart, and so should anything written from it.

1. **A shortcut is available.** A one-line probe scores far above the
   nearest-teammate yardstick on the frames a pipeline emits. `check_leakage.py`,
   `step17_leadsweep.py`.
2. **A trained model uses it.** The model trained on those frames loses its
   advantage when the near-anchor information is taken away.
   `step24_cross_offset.py`, `step25_ball_mask.py`.
3. **A synchronisation offset was measured against ground truth.** This one is
   NOT made anywhere: no public dataset provides ball-contact times. Every offset
   in this project is measured backwards from the frame the synchroniser emits
   (`config.ANCHOR`), and `step27_kick_proxy.py` computes an *estimated* kick
   proxy that is compared with that frame, never substituted for the truth.

`check_version.py` compares every file against `manifest.json`, which ships with
the release, and names any file of yours that is missing or out of date. It also
prints a release id; `CHANGELOG.md` lists what changed in each one and which
files to copy across. Run it after every download.

Always replace `manifest.json` alongside any file you copy. It holds the
checksums, so an old manifest will flag a correctly updated file as wrong.

**Extract the zip to a clean folder.** Unpacking it over an existing copy can
leave a mix of old and new files that still imports and still runs, but produces
wrong numbers. `check_version.py` catches that. If you already have parsed
matches in `work/`, move that folder aside first and put it back afterwards -
re-parsing all seven matches takes a long time.

Or one step at a time:

| File | What it does | Writes |
|---|---|---|
| `step1_load.py` | load one match, check frame count against the paper | `work/<id>_parsed.pkl` |
| `step2_passes.py` | extract open-play passes, map players to array slots | `work/passes.csv` |
| `step3_sync.py` | two-stage event/tracking synchronisation | `work/passes_synced.csv` |
| `step4_clean.py` | Butterworth low-pass filter, velocities, attack direction | `work/clean_<id>.npz` |
| `step5_dataset.py` | build `(13, 23, 11)` samples with labels and pass identifiers | `work/dataset_<id>.npz`, `work/dataset_counts.csv` |
| `step6_explore.py` | six exploration and check figures | `figures/01..06` |
| `step7_baselines.py` | random, nearest, most advanced, gradient boosting | `work/baseline_results.csv` |
| `step8_model.py` | model definition + permutation-invariance self-test | — |
| `step9_train.py` | movement history versus a single frame (setting A is the canonical model) | `work/model_results.csv` |
| `check_leakage.py` | probes for shortcuts before you trust any accuracy | `work/leakage_probes.csv` |
| `step10_analysis.py` | breakdown, calibration, confident mistakes | `figures/07..10` |
| `step11_pretrain.py` | self-supervised pretraining on unlabelled frames | `work/pretrained_encoder.pt` |
| `step12_ablation.py` | measures which improvement actually helps | `work/ablation_results.csv` |
| `step13_inspect.py` | renders single passes across the window, with probabilities | `figures/inspect_*.png` |
| `step14_report.py` | assembles everything into one shareable HTML file | `report.html` |
| `step15_figures.py` | eight figures, PNG for slides and PDF vector for the paper | `figures/paper/` |
| `step16_tactical.py` | trajectories, Voronoi space control, team shape, position density | `figures/paper/` |
| `step17_leadsweep.py` | accuracy versus how early you predict — the trade-off curve | `work/leadsweep_results.csv` |
| `step18_metrics.py` | ranking, calibration and per-position metrics | `work/metrics.csv` |
| `step19_architecture.py` | the architecture diagram | `figures/paper/fig14_*` |
| `step20_databallpy.py` | runs the leakage probe on DataBallPy's synchronisation, all seven matches | `work/databallpy_comparison.csv` |
| `step21_features.py` | which features come from the data and which are derived | `work/feature_provenance.csv` |
| `step22_curves.py` | learning curves, ROC and precision-recall, from the canonical run | `figures/paper/fig18-19` |
| `step23_canonical.py` | **the canonical run**: per-pass predictions and run metadata | `work/metrics_predictions_<run_id>.csv`, `work/run_metadata.json` |
| `step24_cross_offset.py` | train-offset x test-offset matrix: does the model *depend* on the shortcut | `work/cross_offset_matrix.csv` |
| `step25_ball_mask.py` | same trained model, ball removed at test time | `work/ball_mask_summary.csv` |
| `step26_flatness_validation.py` | synthetic validation of the flatness criterion | `work/flatness_validation.csv` |
| `step27_kick_proxy.py` | estimated kick proxy as a second anchor (not ground truth) | `work/passes_kick_proxy.csv` |
| `step28_masella_probe.py` | passer-facing angle as a one-line probe | `work/masella_probe.csv` |
| `step29_cross_provider.py` | the same probes on a second provider, if its data are present | `work/cross_provider_status.json` |
| `step30_size_curve.py` | training size versus intervention gain | `work/size_curve_summary.csv` |
| `step31_paper_figures.py` | figures A to F, each drawn from a results file | `figures/paper/fig[A-F]_*` |
| `verify/reproduce_all.py` | runs everything above that the paper depends on, and reports what ran | `work/reproduce_all_report.json` |
| `step0_inspect_local_data.py` | what is in the local IDSSE folder, and does it parse? | `work/local_idsse_index.json`, `work/local_idsse_fingerprint.json` |
| `verify/data_provenance.py` | is `work/` the real dataset, or a fixture? | `work/data_provenance.json` |
| `verify/no_quick_results.py` | fails if a quick-mode result reaches a paper-facing file | `work/contamination_check.json` |
| `verify/real_run_summary.py` | the run summary, generated from the result files | `work/REAL_RUN_SUMMARY.md` |
| `make_bundle.py` | packages the shareable results, skipping the large intermediates | `results_bundle.zip` |
| `export_events_excel.py` | event data to Excel (optional, for eyeballing) | `work/events.xlsx` |

`config.py` holds every constant. `common.py` holds the shared helpers.
`device.py` decides where tensors live and is imported by every torch script.

## GPU

`device.py` picks CUDA when it is there and CPU otherwise, and every training
script prints which it got before doing any work:

```
Using Device: GPU (NVIDIA GeForce RTX 4060 Laptop GPU)
  8.0 GB VRAM · CUDA 12.4 · torch 2.9.0+cu124
```

It is deliberately not in `config.py`: steps 2 through 7 import config and never
touch torch, and importing torch costs a few seconds of startup each time.

On GPU the batch size is raised 4x automatically. The model is only ~36k
parameters, so at batch 64 the card spends more time launching kernels than
computing — expect a modest speedup on a single training run, and a large one on
`step12_ablation.py`. Note that the ablation, the lead sweep and the cross-offset
matrix share the checkpoint cache in `work/checkpoints/`, so a setting already
trained under the same configuration is loaded rather than retrained.

The first run downloads ~350 MB per match from figshare and parsing takes a few
minutes. Results are cached in `work/`, so later runs are fast.

## Three checks that tell you the pipeline is right

1. **Step 1** — total frames must equal the value in `config.EXPECTED_FRAMES`,
   which comes from Table 1 of the paper.
2. **Step 3** — the orientation line must report **"shift only"** as the winner,
   with the other three variants clearly worse. The mean event-to-ball distance
   before synchronisation should land near the paper's **9.4 m** — that is the
   single strongest sign the coordinate conversion is right. After
   synchronisation the **median** should be near 2.6 m; the mean will be higher
   because of a tail of mis-tagged events, which the paper also reports. That is the value
   the dataset authors hardcode in their own visualisation script. The mean
   event-to-ball distance should go from about 9.4 m to about 2.6 m, matching
   the paper.
3. **`check_leakage.py`** — probes are judged on **level and shape**, not on a
   fixed threshold. A probe is compared with the nearest-teammate yardstick at
   the frame closest to the anchor, and its excess is tracked across the whole
   window. A rise concentrated in the last frames is the profile of an outcome
   entering the input near the anchor. A rise spread over the window is reported
   as UNRESOLVED, because a player legitimately turning towards his target looks
   the same; `step26_flatness_validation.py` demonstrates that limit on synthetic
   data. The check fails only on the ball shortcut, never on an unresolved
   passer signal.
4. **Step 5** — the `lineups other than 11v11` line should show `11v10` for
   J03WN1 in the hundreds. Those are kept. If a match loses most of its passes,
   something is wrong.
5. **Step 4** — the attacking direction must **flip between the two halves** for
   every match, and the two teams must always be on opposite sides. Teams always
   swap ends at half time, so a non-flip is always a detection failure, never a
   real result. If it happens, run `python diagnose_direction.py <match_id>`.

   The keeper is identified from the teamsheet `position` column (`TW`), not by
   finding the player furthest from the centre. That naive rule is fooled by a
   substitute who comes on late and stands wide: he can have a larger mean |x|
   than the real keeper off a handful of frames.

## Design decisions

**Split by match, test on 1. Bundesliga only.** Five of the seven matches are
Fortuna Düsseldorf home games and those five are exactly the 2. Bundesliga
matches. Any other split leaks both team identity and division.

**No leakage from the event attributes.** `PlayAngle`, `Distance`, `Height`,
`FlatCross`, `Evaluation` and the pass end coordinates all describe the pass
*after* it happened. Feeding them in would let the model read the answer off the
input. They are listed in `step5_dataset.FORBIDDEN` and never used.

**Only successful passes.** The DFL `Recipient` attribute records who *actually*
received the ball, so on a failed pass it can be an opponent. Keeping only
`Evaluation = successfullyCompleted` guarantees the receiver is a teammate.

**The model must not be handed a moving ball.** Step 3 finds the frame that best
matches the pass event, and at that frame the ball is already travelling, so its
velocity vector points at the receiver: a one-line rule on it scores far above
the nearest-teammate yardstick. An early model trained on those frames scored
well and was worthless, and the give-away was that one frame scored almost as
well as 1.5 s of history.

The window therefore ends `PREDICT_LEAD_S = 0.4 s` before **the frame the
synchroniser emits**, which is an estimate of the pass moment, not ball contact:
this dataset records no contact time, so no constant here can be expressed
relative to one. Expect accuracy well below what the emitted frame yields; the
lower number is the honest one.

No measured value is written down in this README or in the code comments. The
current ones live in `work/leakage_probes.csv` (level and shape per match),
`work/leadsweep_summary.csv` (across offsets) and `work/cross_offset_matrix.csv`
(whether the trained model depends on the shortcut). `check_leakage.py`
re-measures and fails loudly if the ball shortcut returns.

**The ball and the players share an encoder, but not a scale.** A reviewer asked
why the ball goes through the same GRU as the players when it has no tactical
role. The scoring head already prevents the ball from being predicted, and the
encoder's job is kinematic summarisation, which is the same problem for both —
so a shared encoder plus a type flag is the standard design (baller2vec does the
same). A separate ball encoder would also see one object per sample instead of
23, which at this sample size is a large loss of gradient signal.

But the objection found a real defect. Normalisation pooled all 23 objects, and
since 22 are players those were player statistics: the ball, three times faster
at the top end, sat about +2.7 standard deviations out on average and +8.5 at
its 99th percentile. `NORMALISE_PER_TYPE` gives players and the ball their own
statistics, and `step12_ablation.py` measures what that is worth.

**A red card is football, not bad data.** Requiring exactly 11 v 11 cost 94% of
J03WN1, where Leverkusen's Amine Adli was sent off in minute 8 — and J03WN1 is one
of the two test matches. Instead the missing slot is padded and flagged with the
`is_present` feature, and the model masks padded slots both in the attention and
in the final softmax, so it can never predict a player who is not on the pitch.
Lineups above eleven mean a substitution is mid-swap and both players are briefly
tracked; those are genuinely ambiguous and are still skipped.

**Permutation invariance.** The order of players in the array is arbitrary, so the
answer must not change when it is shuffled. `step8_model.py` asserts this.

**Slot mapping from the teamsheet.** Array slot order follows the teamsheet, not
playing position — in the authors' own example the goalkeeper sits at slot 11.
The mapping comes from `Teamsheet.get_links("pID", "xID")`, never from guessing.

**The two files are in different coordinate frames.** The position data is
centre-based, x in (-52.5, 52.5) and y in (-34, 34). The event data is
corner-based, x in [0, 105] and y in [0, 68] — the paper's Box 2 shows the
kickoff at `X-Source-Position="52.50"`, which is the centre spot. floodlight does
**not** convert this. Ignoring it puts a constant `hypot(52.5, 34) = 62.55 m`
error on every event, and no time shift can remove it. Step 3 converts the event
coordinates first and reports which of the four orientations fits.

**Timing.** Step 3 estimates a constant offset per half and then refines each
event individually. Measured on the real data the constant offsets come out near
zero, so floodlight's `gameclock` is already roughly aligned — the 1.6 s that the
dataset authors hardcode in their plotting script is specific to that script, not
a general offset. The per-event correction is what does the work.

**The reported metric is partly circular.** The cost function includes the
event-to-ball distance, and event-to-ball distance is also what we report. So
step 3 additionally reports **ball-to-passer distance** at the chosen frame,
which uses no annotation at all: at the instant of a kick the ball is at the
passer's foot. That is the honest check.

## Self-test

`_selftest.py` fabricates seven fake matches with a known 1.6 s offset planted in
them and a known half-time direction switch, so the whole pipeline can be run
without the real data.

```bash
python _selftest.py && python run_all.py
```

On fake data every baseline should score about 10%, because the fake labels are
random. **If any baseline scores well above 10% there, something is leaking.**

Delete `_selftest.py` and clear `work/` before running on the real data.

## Why not just convert everything to Excel?

The XML is 2.5 GB, but that is XML verbosity, not data volume — each `<Frame/>`
element spends about 150 bytes storing roughly 8 numbers. The same coordinates
as float32 are about **330 MB**, which is what `step4_clean.py` stores as `.npz`.

| Data | Rows if converted | Excel limit is 1,048,576 |
|---|---|---|
| Events, all 7 matches | 11,137 | 1% — **fine, use `export_events_excel.py`** |
| Tracking, long format, 1 match | 5,984,647 | 5.7x over |
| Tracking, long format, all 7 | 41,108,404 | 39x over |
| Tracking, wide format, all 7 | 1,002,644 rows x 205 cols | fits at 96%, but 205M cells |

Wide format technically fits. It is still a bad idea: Excel cannot do array
maths over sliding time windows, which is the only thing this project does with
the tracking data.

## Improvements, and where they come from

The binding constraint is data: a few thousand labelled passes (the exact count
for your build is in `work/sample_flow.csv`). Every idea below targets that, and
each is measured separately by `step12_ablation.py`, which distinguishes
individual factors from composite settings so the gains are never double-counted;
`verify/table4_ablation.py` does the arithmetic and
`step30_size_curve.py` asks whether the effects hold as training data grow.

**Mirror augmentation** (`AUGMENT_MIRROR`, and `TTA_MIRROR` for the separate
test-time averaging). Reflect the pitch in y during training; averaging the two
views at inference is a different intervention and is measured apart. TacticAI (Wang, Veličković, Hennes
et al., *Nature Communications* 2024) builds D2 reflection equivariance into its
architecture and reports it as its main source of data efficiency, on the same
receiver-prediction task. Augmentation buys part of that in ten lines. Only y is
mirrored — step 5 already normalises the attacking direction in x.

**Space features** (`USE_SPACE_FEATURES`). Pressure — distance to the nearest
opponent — and distance to the ball, as node features. Anzer & Bauer built their
receiver model on pitch control, and the dataset paper's own xG model uses a
pressure metric. Both are derivable from the coordinates the model already sees;
they are here because at this sample size it should not have to rediscover them.

**Edge attention** (`USE_EDGE_ATTENTION`). A learned per-head attention bias from
each pair's relative geometry (dx, dy, distance). TacticAI represents a situation
as a graph whose *edges* carry player relations and argues those matter more than
absolute positions. Their ablation also found attentional GNNs (GATv2) struck the
best expressivity/data-efficiency trade-off, with message-passing networks
overfitting — which supports keeping this attention-based rather than going
deeper.

**Self-supervised pretraining** (`USE_PRETRAIN`, step 11). The largest unused
resource in the project: the dataset holds **1,002,644 frames** and steps 5–10
touch about 0.4% of them, because the rest carry no pass label. Step 11 invents a
label that needs no annotation — given 1.5 s of play, predict where every player
is 1 second later — pretrains the encoder on that, and step 9 fine-tunes it on
the few thousand passes. Windows are drawn from `config.PRETRAIN_MATCHES`, which is the fit matches only:
no test match (that would leak its movement patterns into the encoder) and not
the validation match either, since that match selects the epoch.

Directly comparable published work on this exact task: Rahimian, Kim & Toka,
*Pass Receiver and Outcome Prediction in Soccer Using Temporal Graph Networks*
(MLSA workshop, 2023), which splits the problem into receiver-selection and
receiver-prediction probabilities and derives pass success from them.

### Not implemented, and why

- **Full D2 equivariance** rather than augmentation. Correct but a rewrite of the
  encoder; augmentation captures most of the benefit.
- **SoccerMap-style pitch surfaces** (Fernández & Bornn) — a fully convolutional
  network estimating a probability surface over the whole pitch. This fixes a
  real limitation of the 10-candidate framing: a pass played into space, rather
  than to a player, cannot be represented at all here.
- **Failed passes as extra training signal.** Roughly 20% of passes are dropped
  because `Recipient` can be an opponent. Predicting over all 21 outfield players
  would recover them and give a pass-success model for free.

## Installing databallpy without breaking the rest

`step20_databallpy.py` is the only file that needs `databallpy`, and it needs
nothing else from this project's stack — no torch, no floodlight. That matters
because the two libraries disagree about numpy:

| | numpy requirement |
|---|---|
| floodlight | `>= 2.1.1, < 3.0.0` |
| databallpy | `>= 1.26.4, < 2.3.0` |
| overlap | `>= 2.1.1, < 2.3.0` |

The window is narrow. If your environment sits outside it, installing databallpy
will downgrade numpy, and a compiled extension built against a different numpy
ABI (torch, most likely) can start failing with `_ARRAY_API not found`.

Check first:

```bash
python -c "import numpy; print(numpy.__version__)"
```

Inside `2.1.1 - 2.2.x`: install normally, in the same environment.

```bash
python -m pip install databallpy
```

Outside it: use a throwaway environment and copy the one CSV back.

```bash
python -m venv .venv-dbp
.venv-dbp\Scripts\activate          # Windows
python -m pip install databallpy pandas matplotlib
python step20_databallpy.py
```

On Windows always use `python -m pip`, never `pip` directly. Pip cannot modify
itself while running, which is what produces
`ERROR: To modify pip, please run the following command`.

## A performance trap worth knowing about

`np.load` on an `.npz` returns a **lazy** `NpzFile`. Every `d["key"]` access
re-reads and decompresses that array out of the zip. Put such a lookup inside a
per-row loop over 4,000 passes and you decompress a 23 MB array thousands of
times — about 1.1 TB of pointless work, roughly an hour per script.

It happened twice here, in `step5_dataset.py` and `step15_figures.py`, and
neither errored. They just crawled, which is much harder to notice than a crash.

`common.load_npz` now materialises eagerly and returns a plain dict, so the
mistake is not available any more. Array lookups are also hoisted out of the hot
loops, and the decay curve is computed for all leads in one vectorised pass.
Step 15 went from over an hour to about five seconds.


## Shared modules

| File | What it holds |
|---|---|
| `provenance.py` | run ids, fingerprints, run metadata, CSV stamping, the stale-result guard |
| `experiment.py` | the ONE training function, the normaliser, the checkpoint cache, lead-specific datasets |
| `metrics_lib.py` | every metric definition, computed from exported predictions only |
| `probes.py` | the one-line probes, the provider-independent Snapshot, the flatness diagnostic |
| `providers/` | adapters so the same probes can run on another dataset (`idsse`, `pff`) |

## What the flatness criterion can and cannot do

`step26_flatness_validation.py` tests the criterion on synthetic cases whose
truth is known by construction. On that evidence it separates a step-like leak
from a gradual ramp, and reports the ramp as **unresolved** rather than as
leakage, because a legitimate ramp (a player turning towards his target) and a
soft leak produce the same curve. The rule this project used before that test
(rise > 0.10 or excess > 0.12 means "leak") labelled every legitimate synthetic
case a leak, which is why it was replaced.

## Known limitations of the tooling

- The lead sweep, the cross-offset matrix and the kick proxy are all anchored on
  the emitted synchronised frame, which sits after contact by a match-varying
  amount. A fixed offset is therefore not the same moment in every match.
- Ball masking evaluates a model on an input distribution it never saw. The
  evidence is the contrast between the two training offsets, not either drop on
  its own; a retrained ball-free model would be the stronger control and is not
  run here.
- The cross-provider comparison is implemented but needs data that are not part
  of this repository. Without them it records "dataset not available", and no
  cross-provider claim may be made.
