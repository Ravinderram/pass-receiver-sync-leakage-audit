"""Shared machinery for every script that trains or evaluates the receiver model.

Why this exists
---------------
Before this module, step9, step12, step17, step18 and step22 each trained their
own models, on different splits and schedules, and each trained a separate model
per test match even though the training data was identical. That is how several
headline accuracies came to coexist.

Now there is one training function and one checkpoint cache:

  load_split()       samples at any lead and anchor, with pass identifiers
  fit_normaliser()   training-set statistics, stored with the checkpoint so a
  apply_normaliser() model trained at one lead can be applied at another
  fit_model()        one training run. Model selection sees ONLY the validation
                     match. Test matches may be monitored for learning curves,
                     but nothing about them reaches a decision.
  predict_logits()   logits with the configured test-time averaging
  get_or_train()     trains once per (config, lead, anchor, seed, data) and
                     caches the checkpoint, so the canonical run, the lead sweep
                     and the cross-offset experiment literally share the model
                     at the canonical lead

Offsets are measured back from the anchor frame. The default anchor is the frame
emitted by step3_sync.py, which is NOT ground-truth ball contact.
"""

import contextlib
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config as C
import provenance as P
from common import load_npz, mirror
from device import DEVICE, PIN_MEMORY, batch_size, seed_everything, to_numpy
from step8_model import PassReceiverNet, count_params

LEAD_CACHE = "lead_cache"          # under work/
CKPT_DIR = f"{C.WORK}/checkpoints"
SYNC_ANCHOR = "sync_frame"


# ================================================================== datasets
def lead_tag(lead_s):
    return f"{float(lead_s):.2f}".replace(".", "p")


def is_canonical_lead(lead_s, anchor=SYNC_ANCHOR):
    return anchor == SYNC_ANCHOR and abs(float(lead_s) - C.PREDICT_LEAD_S) < 1e-9


def dataset_name(match_id, lead_s, anchor=SYNC_ANCHOR):
    """work/dataset_<mid>.npz for the canonical lead; a cache file otherwise."""
    if is_canonical_lead(lead_s, anchor):
        return f"dataset_{match_id}"
    return f"{LEAD_CACHE}/{anchor}_lead{lead_tag(lead_s)}_{match_id}"


_FP_MEMO = {}


def _canonical_fp_cached():
    """P.data_fingerprint() re-reads every canonical array; memoise it on the
    files' size and modification time so sweeps do not pay for it per call."""
    stamp = []
    for m in C.MATCHES:
        path = f"{C.WORK}/dataset_{m}.npz"
        st = os.stat(path) if os.path.isfile(path) else None
        stamp.append((path, st.st_mtime_ns if st else None, st.st_size if st else None))
    stamp = tuple(stamp)
    if stamp not in _FP_MEMO:
        _FP_MEMO.clear()
        _FP_MEMO[stamp] = P.data_fingerprint()
    return _FP_MEMO[stamp]


def _source_key(passes, anchor):
    """What a cached lead dataset was built from: the anchor frame and slots of
    every pass, and the canonical arrays (which change whenever clean data do)."""
    cols = ["match_id", "half", anchor, "passer_slot", "recipient_slot"]
    sub = passes[[c for c in cols if c in passes.columns]]
    hashed = pd.util.hash_pandas_object(sub, index=True).to_numpy()
    return P.hash_obj({"passes": P.hash_arrays(hashed), "anchor": anchor,
                       "canonical": _canonical_fp_cached()})


def ensure_datasets(lead_s, anchor=SYNC_ANCHOR, passes=None, match_ids=None,
                    verbose=False):
    """Make sure samples exist at this lead and anchor; build what is missing.

    The canonical arrays (step5 output) are never rebuilt here. Everything else
    goes to work/lead_cache/ and is rebuilt if its source key is stale.
    """
    import step5_dataset as s5

    match_ids = list(C.MATCHES) if match_ids is None else list(match_ids)
    if is_canonical_lead(lead_s, anchor):
        missing = [m for m in match_ids
                   if not os.path.isfile(f"{C.WORK}/dataset_{m}.npz")]
        if missing:
            raise SystemExit(f"canonical datasets missing for {missing}: run "
                             "step5_dataset.py")
        for m in match_ids:
            d = load_npz(f"dataset_{m}", eager=False)
            with d:
                if "pass_row" not in d.files:
                    raise SystemExit(f"work/dataset_{m}.npz predates pass "
                                     "identifiers: rerun step5_dataset.py")
        return

    if passes is None:
        passes = pd.read_csv(f"{C.WORK}/passes_synced.csv")
    key = _source_key(passes, anchor)
    os.makedirs(f"{C.WORK}/{LEAD_CACHE}", exist_ok=True)
    for m in match_ids:
        name = dataset_name(m, lead_s, anchor)
        path = f"{C.WORK}/{name}.npz"
        if os.path.isfile(path):
            with np.load(path, allow_pickle=True) as z:
                if "source_key" in z.files and str(z["source_key"]) == key:
                    continue
        arr = s5.build_arrays(m, passes, lead_s=lead_s, anchor_col=anchor,
                              verbose=verbose)
        arr["source_key"] = np.array(key)
        s5.save_arrays(name, arr)


def load_match_arrays(match_id, lead_s=None, anchor=SYNC_ANCHOR):
    lead_s = C.PREDICT_LEAD_S if lead_s is None else lead_s
    return load_npz(dataset_name(match_id, lead_s, anchor))


def load_split(match_ids, lead_s=None, anchor=SYNC_ANCHOR, keep_space=True,
               passes=None):
    """Concatenate several matches. Returns a dict of aligned arrays.

    keys: X, y, pass_row, match, sample_idx, mate_slots, team_side
    `sample_idx` is the row inside that match's dataset file, `pass_row` the row
    of work/passes_synced.csv. keep_space=False zeroes the two space features,
    which is how the ablation removes them without rebuilding.
    """
    lead_s = C.PREDICT_LEAD_S if lead_s is None else float(lead_s)
    ensure_datasets(lead_s, anchor, passes=passes, match_ids=match_ids)
    parts = {k: [] for k in ("X", "y", "pass_row", "match", "sample_idx",
                             "mate_slots", "team_side")}
    for m in match_ids:
        d = load_match_arrays(m, lead_s, anchor)
        n = len(d["X"])
        if n == 0:
            continue
        parts["X"].append(d["X"])
        parts["y"].append(d["y"])
        parts["pass_row"].append(d["pass_row"])
        parts["match"].append(np.array([m] * n, dtype=object))
        parts["sample_idx"].append(np.arange(n))
        parts["mate_slots"].append(d["mate_slots"])
        parts["team_side"].append(d["team_side"])
    if not parts["X"]:
        raise SystemExit(f"no samples for {match_ids} at lead {lead_s}")
    out = {k: np.concatenate(v) for k, v in parts.items()}
    if not keep_space:
        out["X"] = out["X"].copy()
        out["X"][:, :, :, C.IDX_PRESSURE] = 0.0
        out["X"][:, :, :, C.IDX_BALLDIST] = 0.0
    return out


def subset(split, mask):
    return {k: v[mask] for k, v in split.items()}


# ================================================================ normaliser
def _stats(A, mask):
    """Mean and sd of the first four features over present objects only."""
    sel = A[:, :, mask, :4]
    present = A[:, :, mask, C.IDX_PRESENT] > 0.5
    if present.sum() == 0:
        return np.zeros((1, 1, 1, 4), np.float32), np.ones((1, 1, 1, 4), np.float32)
    flat = sel[present]
    return (flat.mean(0).reshape(1, 1, 1, 4).astype(np.float32),
            (flat.std(0) + 1e-6).reshape(1, 1, 1, 4).astype(np.float32))


def fit_normaliser(Xtr, per_type=None):
    """Statistics from TRAINING arrays only. Returned as a JSON-able dict."""
    per_type = C.NORMALISE_PER_TYPE if per_type is None else per_type
    players = np.arange(0, C.N_OBJECTS - 1)
    ball = np.array([C.N_OBJECTS - 1])
    if per_type:
        mu_p, sd_p = _stats(Xtr, players)
        mu_b, sd_b = _stats(Xtr, ball)
    else:
        mu_p, sd_p = _stats(Xtr, np.arange(C.N_OBJECTS))
        mu_b, sd_b = mu_p, sd_p
    return {"per_type": bool(per_type),
            "mu_p": mu_p.ravel().tolist(), "sd_p": sd_p.ravel().tolist(),
            "mu_b": mu_b.ravel().tolist(), "sd_b": sd_b.ravel().tolist()}


def apply_normaliser(X, stats):
    players = np.arange(0, C.N_OBJECTS - 1)
    ball = np.array([C.N_OBJECTS - 1])
    r = lambda k: np.asarray(stats[k], np.float32).reshape(1, 1, 1, 4)
    X = X.copy()
    X[:, :, players, :4] = (X[:, :, players, :4] - r("mu_p")) / r("sd_p")
    X[:, :, ball, :4] = (X[:, :, ball, :4] - r("mu_b")) / r("sd_b")
    absent = X[:, :, :, C.IDX_PRESENT] <= 0.5       # padded slots stay zero
    X[:, :, :, :4][absent] = 0.0
    return X


# ================================================================== training
def _pretrained_state():
    path = f"{C.WORK}/pretrained_encoder.pt"
    if not os.path.isfile(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


def _acc_loss(model, X_t, X_mir, y_np, tta, lossf):
    with torch.no_grad():
        lg = model(X_t)
        if tta:
            lg = (lg + model(X_mir)) / 2.0
        y_t = torch.tensor(y_np).to(DEVICE)
        return (float((lg.argmax(1) == y_t).float().mean().item()),
                float(lossf(lg, y_t).item()))


def fit_model(Xtr, ytr, Xval, yval, seed, epochs=None, augment=None, tta=None,
              use_pretrain=None, monitor=None, verbose=False):
    """Train one model. Returns (model, info).

    Xtr, Xval must already be normalised. Selection uses ONLY (Xval, yval):
    the best validation top-1 epoch is restored when RESTORE_BEST_CHECKPOINT is
    on. `monitor` maps a name to (X, y) arrays that are scored every epoch for
    learning curves; they never influence training or selection.

    With Xval=None the model trains for a fixed `epochs` and the final weights
    are kept. That mode exists for legacy scripts; the canonical run requires a
    validation match.
    """
    epochs = C.EPOCHS if epochs is None else epochs
    augment = C.AUGMENT_MIRROR if augment is None else augment
    tta = C.TTA_MIRROR if tta is None else tta
    use_pretrain = C.USE_PRETRAIN if use_pretrain is None else use_pretrain
    monitor = monitor or {}

    seed_everything(seed)
    np.random.seed(seed)
    n_fit = len(Xtr)
    if augment:
        Xtr = np.concatenate([Xtr, mirror(Xtr)])
        ytr = np.concatenate([ytr, ytr])

    model = PassReceiverNet()
    pretrained = False
    if use_pretrain:
        state = _pretrained_state()
        if state is not None:
            model.load_state_dict(state, strict=False)
            pretrained = True
        elif verbose:
            print("    (no pretrained encoder found - run step11_pretrain.py)")
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR)
    lossf = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                        batch_size=batch_size(), shuffle=True,
                        pin_memory=PIN_MEMORY)

    val_t = torch.tensor(Xval).to(DEVICE) if Xval is not None else None
    val_mir = torch.tensor(mirror(Xval)).to(DEVICE) if Xval is not None else None
    tr_eval = torch.tensor(Xtr[:n_fit]).to(DEVICE)
    mon = {k: (torch.tensor(Xm).to(DEVICE), torch.tensor(mirror(Xm)).to(DEVICE), ym)
           for k, (Xm, ym) in monitor.items()}

    best = {"val": -1.0, "epoch": -1, "state": None}
    since, history, stopped_early = 0, [], False
    for ep in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=PIN_MEMORY)
            yb = yb.to(DEVICE, non_blocking=PIN_MEMORY)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)

        model.eval()
        row = {"epoch": ep, "train_loss": total / len(Xtr)}
        with torch.no_grad():
            row["train_top1"] = float((model(tr_eval).argmax(1).cpu().numpy()
                                       == ytr[:n_fit]).mean())
        for k, (Xm_t, Xm_mir, ym) in mon.items():
            row[f"{k}_top1"], row[f"{k}_loss"] = _acc_loss(model, Xm_t, Xm_mir,
                                                           ym, tta, lossf)
        if val_t is not None:
            row["val_top1"], row["val_loss"] = _acc_loss(model, val_t, val_mir,
                                                         yval, tta, lossf)
            if row["val_top1"] > best["val"]:
                best = {"val": row["val_top1"], "epoch": ep,
                        "state": {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}}
                since = 0
            else:
                since += 1
        history.append(row)
        if verbose:
            extra = "".join(f"  {k} {v:.3f}" for k, v in row.items()
                            if k.endswith("_top1"))
            print(f"      epoch {ep:3d}  loss {row['train_loss']:.3f}{extra}")
        if val_t is not None and C.EARLY_STOPPING and since >= C.PATIENCE:
            stopped_early = True
            break

    if best["state"] is not None and C.RESTORE_BEST_CHECKPOINT:
        model.load_state_dict(best["state"])
    model.eval()
    info = {
        "seed": int(seed),
        "epochs_run": len(history),
        "epoch_cap": int(epochs),
        "stopped_early": bool(stopped_early),
        "selected_epoch": int(best["epoch"]) if best["state"] is not None
        else len(history) - 1,
        "stopped_at": len(history) - 1,
        "best_val_top1": float(best["val"]) if best["state"] is not None else None,
        "selection": ("best validation top-1, weights restored"
                      if best["state"] is not None and C.RESTORE_BEST_CHECKPOINT
                      else "final epoch (no validation set)"),
        "augment_mirror": bool(augment), "tta_mirror": bool(tta),
        "pretrained_encoder_loaded": pretrained,
        "n_fit_samples": int(n_fit),
        "n_val_samples": int(len(Xval)) if Xval is not None else 0,
        "parameters": count_params(model),
        "history": history,
    }
    return model, info


def predict_logits(model, X, tta=None, chunk=2048):
    """Logits for normalised X, averaged over the mirror view if tta is on."""
    tta = C.TTA_MIRROR if tta is None else tta
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            xb = X[i:i + chunk]
            lg = model(torch.tensor(xb).to(DEVICE))
            if tta:
                lg = (lg + model(torch.tensor(mirror(xb)).to(DEVICE))) / 2.0
            out.append(to_numpy(lg))
    return (np.concatenate(out) if out
            else np.zeros((0, C.N_TEAMMATES), np.float32))


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ============================================================ config override
TRAINING_FLAGS = ["AUGMENT_MIRROR", "TTA_MIRROR", "USE_EDGE_ATTENTION",
                  "USE_PRETRAIN", "NORMALISE_PER_TYPE"]


@contextlib.contextmanager
def overrides(**flags):
    """Temporarily set config flags, e.g. overrides(AUGMENT_MIRROR=False).

    The model reads USE_EDGE_ATTENTION at construction, the normaliser reads
    NORMALISE_PER_TYPE at call time, fit_model reads the rest, so setting them
    on the config module for the duration of a block is sufficient.
    """
    old = {k: getattr(C, k) for k in flags}
    try:
        for k, v in flags.items():
            setattr(C, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(C, k, v)


# ========================================================== checkpoint cache
def training_signature():
    """Everything in config that changes what fit_model produces."""
    keys = TRAINING_FLAGS + ["USE_SPACE_FEATURES", "EARLY_STOPPING", "PATIENCE",
                             "EPOCHS", "BATCH_SIZE", "LR", "HIDDEN", "N_HEADS",
                             "WINDOW_SECONDS", "STRIDE", "FILTER_ORDER",
                             "FILTER_CUTOFF_HZ", "RESTORE_BEST_CHECKPOINT",
                             "MODEL_SELECTION_METRIC", "QUICK_MODE"]
    sig = {k: getattr(C, k) for k in keys}
    sig["pretrain_fp"] = P.pretrain_fingerprint() if C.USE_PRETRAIN else "off"
    return sig


def get_or_train(lead_s=None, seed=0, anchor=SYNC_ANCHOR, fit_matches=None,
                 val_match=None, keep_space=True, fit_mask_fn=None,
                 subset_tag=None, passes=None, force=False, verbose=True,
                 monitor_matches=None):
    """Load the model for this exact setting from the cache, or train it.

    Returns (model, normaliser_stats, info). The cache key covers the training
    signature, lead, anchor, seed, split, the fit and validation arrays
    themselves, and any subset. Monitoring matches are not part of the key
    because they cannot influence the trained weights.

    fit_mask_fn(split) -> boolean mask lets step30 subsample training passes;
    pass a matching `subset_tag` so different subsets never share a cache entry.
    """
    lead_s = C.PREDICT_LEAD_S if lead_s is None else float(lead_s)
    fit_matches = list(C.FIT_MATCHES if fit_matches is None else fit_matches)
    val_match = C.VAL_MATCH if val_match is None else val_match
    monitor_matches = (list(C.TEST_MATCHES) if monitor_matches is None
                       else list(monitor_matches))

    fit = load_split(fit_matches, lead_s, anchor, keep_space, passes=passes)
    if fit_mask_fn is not None:
        fit = subset(fit, fit_mask_fn(fit))
    val = (load_split([val_match], lead_s, anchor, keep_space, passes=passes)
           if (C.EARLY_STOPPING and val_match) else None)

    key_obj = {"sig": training_signature(), "lead_s": round(lead_s, 4),
               "anchor": anchor, "seed": int(seed), "fit": fit_matches,
               "val": val_match if val is not None else None,
               "keep_space": bool(keep_space), "subset": subset_tag,
               "fit_fp": P.hash_arrays(fit["X"], fit["y"]),
               "val_fp": P.hash_arrays(val["X"], val["y"]) if val is not None else None}
    key = P.hash_obj(key_obj, n=16)
    path = f"{CKPT_DIR}/{key}.pt"

    if os.path.isfile(path) and not force:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        model = PassReceiverNet()
        model.load_state_dict(blob["state_dict"])
        model = model.to(DEVICE).eval()
        info = blob["info"]
        info["cache"] = "loaded"
        info["checkpoint"] = path
        if verbose:
            print(f"    seed {seed} lead {lead_s:.2f}: loaded checkpoint "
                  f"{os.path.basename(path)} (selected epoch "
                  f"{info['selected_epoch']})")
        return model, blob["normaliser"], info

    stats = fit_normaliser(fit["X"])
    Xfit = apply_normaliser(fit["X"], stats)
    Xval = apply_normaliser(val["X"], stats) if val is not None else None
    monitor = {}
    for m in monitor_matches:
        te = load_split([m], lead_s, anchor, keep_space, passes=passes)
        monitor[f"test_{m}"] = (apply_normaliser(te["X"], stats), te["y"])

    if verbose:
        print(f"    seed {seed} lead {lead_s:.2f}: training on {fit_matches} "
              f"({len(Xfit)} passes), validation {val_match if val is not None else 'none'}"
              f" ({0 if val is None else len(Xval)}), cap {C.EPOCHS}, "
              f"patience {C.PATIENCE}")
    model, info = fit_model(Xfit, fit["y"], Xval,
                            val["y"] if val is not None else None, seed,
                            monitor=monitor)
    info.update({"lead_s": lead_s, "anchor": anchor, "fit_matches": fit_matches,
                 "val_match": val_match if val is not None else None,
                 "keep_space": bool(keep_space), "subset": subset_tag,
                 "cache_key": key, "key_contents": key_obj, "cache": "trained",
                 "checkpoint": path,
                 "fit_pass_rows_fp": P.hash_arrays(fit["pass_row"])})
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in
                               model.state_dict().items()},
                "normaliser": stats, "info": info}, path)
    if verbose:
        print(f"      selected epoch {info['selected_epoch']} of "
              f"{info['epochs_run']} run (val top-1 {info['best_val_top1']}), "
              f"saved {os.path.basename(path)}")
    return model, stats, info


def evaluate(model, stats, split, tta=None):
    """Normalise with the model's own statistics and return logits."""
    return predict_logits(model, apply_normaliser(split["X"], stats), tta=tta)


def seed_list(robust=False):
    return list(C.ROBUST_SEEDS if robust else C.SEEDS)
