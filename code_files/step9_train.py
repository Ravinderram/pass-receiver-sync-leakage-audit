"""Step 9 - Train, and run the movement-history experiment.

The question: does movement help, or is a single frozen frame enough?

  Setting A   full 1.5 s of history   (13 time steps)   = the canonical model
  Setting B   only the last frame     (1 time step)

Same architecture, same data, same training protocol: fit on config.FIT_MATCHES,
early stopping on config.VAL_MATCH, weights restored to the best validation
epoch. Each seed trains ONE model and that model is evaluated on every test
match. (The old version trained a separate model per test match on identical
training data, which doubled compute and produced two slightly different
"canonical" models.)

Setting A is loaded from the shared checkpoint cache (experiment.get_or_train),
so it is the very same model the canonical run (step23) reports.

This module also keeps the helper API other scripts import: load_split,
normalise, load_train_val and train_one.

Run:  python step9_train.py
Out:  work/model_results.csv
"""

import numpy as np
import pandas as pd

import config as C
import experiment as E
import provenance as P
from common import banner, load_npz
from step8_model import PassReceiverNet, count_params

from device import DEVICE, banner as device_banner, batch_size  # noqa: F401

DEV = DEVICE          # kept so other steps importing DEV keep working


# ------------------------------------------------------------ helper API
def load_split(match_ids):
    """(X, y) at the canonical lead, matches concatenated in the given order."""
    Xs, ys = [], []
    for mid in match_ids:
        d = load_npz(f"dataset_{mid}")
        if len(d["X"]):
            Xs.append(d["X"])
            ys.append(d["y"])
    return np.concatenate(Xs), np.concatenate(ys)


def normalise(Xtr, *others):
    """Scale positions and velocities using TRAINING statistics only.

    Thin wrapper over experiment.fit_normaliser / apply_normaliser, which also
    let the statistics be stored with a checkpoint and reused at other leads.
    """
    stats = E.fit_normaliser(Xtr)
    return (E.apply_normaliser(Xtr, stats),) + tuple(
        E.apply_normaliser(o, stats) for o in others)


def load_train_val():
    """Fit matches, plus the validation match carved out of the non-test matches."""
    Xtr, ytr = load_split(C.FIT_MATCHES)
    if not C.EARLY_STOPPING or not C.VAL_MATCH:
        return Xtr, ytr, None, None
    Xval, yval = load_split([C.VAL_MATCH])
    return Xtr, ytr, Xval, yval


def train_one(Xtr, ytr, Xte, yte, seed, epochs=None, augment=None,
              tta=None, return_probs=False, Xval=None, yval=None):
    """Backward-compatible wrapper around experiment.fit_model.

    The test set is only monitored for the learning curve; selection uses the
    validation set when one is given. Returns (acc, top3, loss, history[, probs])
    with history rows (epoch, train_loss, test_loss, test_acc, train_acc, val_acc).
    """
    tta = C.TTA_MIRROR if tta is None else tta
    model, info = E.fit_model(Xtr, ytr, Xval, yval, seed, epochs=epochs,
                              augment=augment, tta=tta,
                              monitor={"test": (Xte, yte)})
    logits = E.predict_logits(model, Xte, tta=tta)
    prob = E.softmax(logits)
    acc = float((logits.argmax(1) == yte).mean())
    top3 = float((np.argsort(-logits, axis=1)[:, :3] == yte[:, None]).any(1).mean())
    loss = float(-np.log(prob[np.arange(len(yte)), yte] + 1e-12).mean())
    history = [(h["epoch"], h["train_loss"], h.get("test_loss", np.nan),
                h.get("test_top1", np.nan), h["train_top1"],
                h.get("val_top1", np.nan)) for h in info["history"]]
    if return_probs:
        return acc, top3, loss, history, prob
    return acc, top3, loss, history


# ------------------------------------------------------------ experiment
def _scores(logits, y):
    order = np.argsort(-logits, axis=1)
    return (float((order[:, 0] == y).mean()),
            float((order[:, :3] == y[:, None]).any(1).mean()))


if __name__ == "__main__":
    banner("Step 9 - training and the movement-history experiment")
    device_banner()
    run_id = P.canonical_run_id()
    P.print_header("movement history vs single frame", run_id,
                   fit=C.FIT_MATCHES, validation=C.VAL_MATCH,
                   test=C.TEST_MATCHES, lead_s=C.PREDICT_LEAD_S, seeds=C.SEEDS,
                   parameters=f"{count_params(PassReceiverNet()):,}",
                   batch=batch_size(),
                   early_stopping=f"{C.EARLY_STOPPING}, patience {C.PATIENCE}, "
                                  f"cap {C.EPOCHS}")
    if not C.EARLY_STOPPING:
        raise SystemExit("EARLY_STOPPING is off; the canonical protocol needs "
                         "the validation match. Turn it back on in config.py.")

    fit = E.load_split(C.FIT_MATCHES)
    val = E.load_split([C.VAL_MATCH])
    tests = {m: E.load_split([m]) for m in C.TEST_MATCHES}
    rows = []

    for seed in C.SEEDS:
        # A: the canonical model, shared with step23 through the cache
        model, stats, info = E.get_or_train(C.PREDICT_LEAD_S, seed)
        for m, te in tests.items():
            t1, t3 = _scores(E.evaluate(model, stats, te), te["y"])
            rows.append({"test_match": m, "setting": "A: 1.5s of history",
                         "seed": seed, "top1": t1, "top3": t3, "steps": C.N_STEPS,
                         "selected_epoch": info["selected_epoch"],
                         "epochs_run": info["epochs_run"]})
            print(f"  seed {seed}  A  {m}: top1 {t1:.3f}  top3 {t3:.3f}")

        # B: last frame only, same protocol, trained here (not a cached variant)
        stats_b = E.fit_normaliser(fit["X"][:, -1:])
        mb, info_b = E.fit_model(E.apply_normaliser(fit["X"][:, -1:], stats_b),
                                 fit["y"],
                                 E.apply_normaliser(val["X"][:, -1:], stats_b),
                                 val["y"], seed)
        for m, te in tests.items():
            lg = E.predict_logits(mb, E.apply_normaliser(te["X"][:, -1:], stats_b))
            t1, t3 = _scores(lg, te["y"])
            rows.append({"test_match": m, "setting": "B: single frame",
                         "seed": seed, "top1": t1, "top3": t3, "steps": 1,
                         "selected_epoch": info_b["selected_epoch"],
                         "epochs_run": info_b["epochs_run"]})
            print(f"  seed {seed}  B  {m}: top1 {t1:.3f}  top3 {t3:.3f}")

    df = pd.DataFrame(rows)
    df = P.stamp(df, run_id, lead_s=C.PREDICT_LEAD_S, model_variant="history_ablation")
    out = f"{C.WORK}/model_results.csv"
    df.to_csv(out, index=False)

    print("\nsummary (mean over seeds, range):")
    for (m, s), g in df.groupby(["test_match", "setting"]):
        print(f"  {m}  {s:22s} {g.top1.mean():.3f}  ({g.top1.min():.3f}-{g.top1.max():.3f})")
    print(f"\nSaved -> {out}")
    print("Read the gap only together with the seed ranges: if they overlap,")
    print("history and a single frame are not distinguishable on that match.")
