# What Synchronisation Costs a Pass-Receiver Model

A leakage audit of event–tracking synchronisation in soccer, on the seven-match
public IDSSE dataset.

This repository contains the full analysis pipeline for the paper *What
Synchronisation Costs a Pass-Receiver Model: A Leakage Audit on a Seven-Match
Public Dataset*. One command regenerates every table and figure in the paper from
the raw data.

---

## What this is, and what it is not

Pass-receiver prediction is normally evaluated on event and tracking data joined
by an automatic synchronisation algorithm, and that join is treated as settled
preprocessing. This project asks what the remaining temporal placement costs a
downstream model.

**It is not a state-of-the-art receiver model.** The model here is deliberately
small (35,909 parameters). It exists to be audited; a larger architecture would
make the timing question harder to isolate, not easier. If you want a strong
receiver model, look at the graph and message-passing work cited in the paper.

**It does not claim to have discovered synchronisation error.** That has been
reported before, notably by Biermann et al. (2023). The contribution is the
downstream consequence: what the emitted frame does to a model trained on it,
and diagnostics for detecting that.

**It does not measure a true synchronisation offset.** The IDSSE release carries
no annotated ball-contact timestamp, so every offset in this repository is
defined relative to the frame the synchroniser emits. Nothing here is expressed
as "before ball contact", and the code will not let you write it that way either.

---

## Headline results

All from run `a95b7a72bb`, seven matches, 4,229 usable open-play passes, three
seeds.

**A shortcut is available at the emitted frame.** A parameter-free rule that
follows the ball's velocity direction identifies the receiver on 70.9% of passes
at the frames DataBallPy emits, against a 27.6% nearest-teammate yardstick. The
ball is already moving at a median 15.49 m/s there.

**A model trained there depends on it.** Cross-offset evaluation, top-1 accuracy:

| trained at | tested 0.0 s | tested 0.4 s | tested 1.2 s |
|---|---|---|---|
| 0.0 s | **0.831** | 0.441 | 0.391 |
| 0.4 s (canonical) | 0.445 | **0.657** | 0.438 |

The emitted-frame model loses 0.390 when moved 0.4 s earlier, falling below the
model trained at that offset. Masking the ball at inference, with no retraining,
costs it 0.356 top-1 against 0.142 for the canonical model.

**The audited model.** Trained 0.4 s before the emitted frame: pooled top-1
0.656 (SD 0.007 over seeds), against 0.458 for gradient boosting and 0.276 for
nearest teammate. McNemar on identical passes gives exact p < 10⁻³⁰ against both.

**Most interventions do not survive their own error bars.** At three seeds, no
individual ablation intervention has a 95% interval excluding zero, and four of
six change sign between runs. Self-supervised pretraining on ~10⁶ unlabelled
frames comes out at −1.25 pp.

---

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.10+, PyTorch 2.x (CPU or CUDA), floodlight ≥1.2.0, numpy, pandas,
scipy, scikit-learn, matplotlib, mplsoccer.

`databallpy` is deliberately **not** in `requirements.txt`: it pins numpy
incompatibly with floodlight. It is needed only by `step20_databallpy.py`.
Install it in a separate environment, run that one script there, and copy
`work/databallpy_comparison.csv` back.

---

## Getting the data

The dataset is Bassek et al. (2025), *An integrated dataset of spatiotemporal
and event data in elite soccer*, Scientific Data 12:195, CC-BY, on figshare at
`doi:10.6084/m9.figshare.28196177`. It is not redistributed here.

**Recommended: local mode.** Download the archive, then set one variable in
`config.py`:

```python
IDSSE_LOCAL_DATA_DIR = r"C:\Users\you\Downloads\IDSSE"   # or your path
```

or export `IDSSE_DATA_DIR` instead of editing the file. With this set the
project reads that folder and makes no network call: `netguard.py` patches
`socket.connect` for the run, a missing match is an error naming it rather than
a download, and every run prints

```
DATA SOURCE: LOCAL IDSSE (/path/to/IDSSE)
NETWORK DOWNLOAD: DISABLED
```

The folder is read recursively and never written to. Its layout does not matter:
`step0_inspect_local_data.py` classifies files by opening them, so a flat folder,
one folder per match, or anything else works, and unrelated files in the same
folder are ignored.

**Alternative: let floodlight fetch it.** Leave `IDSSE_LOCAL_DATA_DIR` as `None`
and `floodlight.io.datasets.IDSSEDataset` downloads ~2.4 GB from figshare on
first use.

---

## Running it

```bash
python step0_inspect_local_data.py --full     # what is in the data folder
python step1_load.py --all                    # prove every match parses
python verify/reproduce_all.py --with-data --with-pretrain
```

The last command runs the whole pipeline and writes
`work/reproduce_all_report.json` naming each stage, its exit code and duration,
so a stage that did not run is never presented as having run.

Useful flags: `--only=step23,section8`, `--skip=size_curve`, `--continue` (do not
stop at the first failure).

For the ten-seed robustness mode on the unstable ablation rows:

```bash
python step12_ablation.py --robust
python verify/table4_ablation.py --robust
```

**Quick mode** (`--quick`, or `PRX_QUICK=1`) shrinks epochs, seeds and grids so
the plumbing can be exercised in minutes on the synthetic fixture. It is not a
smaller experiment; its numbers mean nothing about football, and its outputs
carry `quick-` in the run id so they cannot be mistaken for results.

---

## How the results are kept honest

The recurring failure mode in this project's history was numbers from different
training runs appearing side by side as though they came from one. Four
mechanisms now prevent that, and all of them fail closed.

**One canonical configuration.** `config.py` ends with a block listing the
constants that define the run: fit matches, validation match, test matches, lead
time, epoch cap, early stopping, seeds, architecture, preprocessing,
augmentation, pretraining. `canonical_config()` returns them as a dict.

**A run id.** `provenance.py` hashes that config, the sample arrays, the raw
local files and the pretrained encoder into a short id. Every results CSV carries
`run_id, seed, lead_s, anchor, model_variant, test_match` as its first columns,
and the verification scripts recompute the id and refuse to report a file that
does not match:

```
no predictions for run id 4f2ab9c113. Run step23_canonical.py first.
```

**One trainer.** `experiment.py` holds the only training function and a
checkpoint cache keyed on config, lead, anchor, seed and the actual fit arrays.
The canonical run, the lead sweep, the cross-offset matrix and the ablation share
the same checkpoint at the canonical offset rather than each training their own.

**Three gates.** `verify/data_provenance.py` checks the parsed matches against
the frame counts published in the dataset paper *and* against the authors' own
census in `reference/idsse_reference_counts.json` (1,002,644 frames, 11,137 event
rows, 5,241 `Play_Pass`, 140 `Play_Cross`); it exits 1 outside quick mode unless
the verdict is `REAL`. `verify/no_quick_results.py` fails if a quick-mode result
reaches a paper-facing file. `verify/real_run_summary.py` writes
`work/REAL_RUN_SUMMARY.md` headed `Valid for paper reporting: YES/NO` with the
reason.

---

## Repository layout

**Shared modules**

| file | holds |
|---|---|
| `config.py` | every setting, including the canonical run definition |
| `provenance.py` | run ids, fingerprints, run metadata, the stale-result guard |
| `experiment.py` | the one training function, normaliser, checkpoint cache |
| `metrics_lib.py` | every metric definition, computed from exported predictions |
| `probes.py` | the parameter-free probes and the temporal-shape diagnostic |
| `local_data.py`, `netguard.py` | local-only data loading, no-network enforcement |
| `providers/` | adapters so the probes can run on another dataset |

**Pipeline**

| step | does |
|---|---|
| `step0_inspect_local_data.py` | inspect the local data folder before anything trains |
| `step1_load.py` – `step5_dataset.py` | load, extract passes, synchronise, filter, build samples |
| `step7_baselines.py` | random, most-advanced, nearest teammate, gradient boosting |
| `step8_model.py`, `step9_train.py` | the model; history versus single frame |
| `step11_pretrain.py` | self-supervised encoder on unlabelled frames |
| `step12_ablation.py` | seven interventions, typed individual vs composite |
| `step23_canonical.py` | **the canonical run**: per-pass predictions and metadata |
| `step24_cross_offset.py` | train-offset × test-offset matrix |
| `step25_ball_mask.py` | ball masking at inference, no retraining |
| `step26_flatness_validation.py` | synthetic validation of the diagnostic |
| `step27_kick_proxy.py` | estimated kick proxy as a second anchor |
| `step28_masella_probe.py` | passer-facing angle as a one-line probe |
| `step29_cross_provider.py` | the same probes on a second provider |
| `step30_size_curve.py` | training size versus intervention gain |
| `step31_paper_figures.py`, `step32_pitch_figures.py` | figures, all from results files |

**Verification** (`verify/`): `reproduce_all.py`, `data_provenance.py`,
`sample_flow.py`, `section8_results.py`, `table3_sync.py`, `table4_ablation.py`,
`architecture_checks.py`, `no_quick_results.py`, `real_run_summary.py`.

**Outputs**: everything generated lands in `work/` (CSVs, per-pass predictions,
run metadata, checkpoints) and `figures/`. Neither is tracked.

---

## Key outputs

| file | contents |
|---|---|
| `work/metrics_predictions_<run_id>.csv` | per-pass logits, probabilities, top-k, rank, hit flags, receiver position — the source of every result-section number |
| `work/run_metadata.json` | what produced it: config, splits, seeds, selected epoch per seed, parameter count, sample counts |
| `work/canonical_metrics_summary.csv` | top-1/2/3/5, MRR, mean rank, Brier, ECE, pair-level AUC and AP |
| `work/paired_tests.csv` | McNemar against each baseline on identical passes |
| `work/cross_offset_matrix.csv`, `ball_mask_summary.csv` | the dependence experiments |
| `work/ablation_summary.csv`, `size_curve_summary.csv` | gains with paired seed spread and intervals |
| `work/REAL_RUN_SUMMARY.md` | the whole run written up, headed valid or not valid for reporting |

---

## Known limitations

These are properties of the study, not bugs.

- **One dataset, one league, one family of synchronisation methods.** The
  cross-provider comparison is implemented (`providers/pff.py`) but was not run,
  because the PFF FC World Cup 2022 release was unavailable. This is the most
  important open question for the argument.
- **No ground-truth contact time.** Every offset is relative to the emitted
  frame. The kick proxy in step 27 is an estimate with a filter-induced late bias
  that fails on 20.5% of passes.
- **Ball masking is out-of-distribution.** The contrast between two models is
  interpretable; the absolute drops are not. A retrained ball-free model would be
  a stronger control and was not run.
- **Three seeds.** The `--robust` ten-seed mode is implemented but was not
  executed, so the reported sign instabilities should be confirmed at higher seed
  counts.
- **The diagnostic has a stated blind spot.** It separates a step-like leak from
  a gradual ramp but cannot resolve a gradual rise; step 26 measures where that
  boundary falls. The earlier version of the rule labelled all three legitimate
  synthetic cases as leaks, which is why it was replaced.

---

## Citation

If you use this code or its findings:

```bibtex
@article{singh2026synchronisation,
  author  = {Singh, Ranjit},
  title   = {What Synchronisation Costs a Pass-Receiver Model:
             A Leakage Audit on a Seven-Match Public Dataset},
  journal = {[under review]},
  year    = {2026}
}
```

Please also cite the dataset:

```bibtex
@article{bassek2025idsse,
  author  = {Bassek, Manuel and Rein, Robert and Weber, Hendrik and Memmert, Daniel},
  title   = {An integrated dataset of spatiotemporal and event data in elite soccer},
  journal = {Scientific Data},
  volume  = {12},
  pages   = {195},
  year    = {2025},
  doi     = {10.1038/s41597-025-04505-y}
}
```

---

## Licence

Code: [MIT](LICENSE). The IDSSE dataset is CC-BY 4.0 and is not redistributed
here; obtain it from figshare and cite the authors.

---

## Acknowledgements

Thanks to the IDSSE authors for releasing full-match tracking and event data
under an open licence, without which this audit would not have been possible,
and for publishing the companion code whose descriptive statistics are used here
as an independent check on parsing.

A large language model was used for language editing, drafting assistance and
code review during this project. All experimental design, analysis and reported
results are the author's own and were verified against the underlying result
files.
