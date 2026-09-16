"""Step 23 - The canonical run. Every reported number comes from here.

Protocol (all of it read from config.py, see canonical_config()):
  fit          config.FIT_MATCHES
  validation   config.VAL_MATCH, top-1 accuracy monitored every epoch,
               patience config.PATIENCE, cap config.EPOCHS, best epoch restored
  test         config.TEST_MATCHES, never used for any decision
  lead         config.PREDICT_LEAD_S before the EMITTED synchronised frame
  seeds        config.SEEDS; one model per seed, evaluated on every test match

Outputs
  work/metrics_predictions_<run_id>.csv   one row per (predictor, test pass):
      the canonical model per seed, the seed ensemble (mean of softmax
      probabilities, a separate labelled variant), and every baseline on the
      same passes. Logits, probabilities, top-k lists, rank of the true receiver,
      hit flags and receiver positions.
  work/canonical_history.csv              per-epoch curves per seed (test curves
      are monitored only; they never influence selection)
  work/run_metadata.json                  what produced the file above
      (also copied to work/run_metadata_<run_id>.json)

Nothing here computes the paper's summary metrics. verify/section8_results.py
derives all of them from the prediction file, so they cannot drift apart.

Run:  python step23_canonical.py [--force]
"""

import os
import sys

import numpy as np
import pandas as pd

import config as C
import experiment as E
import provenance as P
from device import ON_GPU, banner as device_banner, batch_size
from step7_baselines import baseline_scores
from step8_model import PassReceiverNet, count_params

K = C.N_TEAMMATES


def check_protocol():
    problems = []
    if not C.EARLY_STOPPING or not C.VAL_MATCH:
        problems.append("EARLY_STOPPING must be on with a VAL_MATCH")
    if C.VAL_MATCH in C.TEST_MATCHES:
        problems.append("VAL_MATCH is a test match")
    if set(C.FIT_MATCHES) & set(C.TEST_MATCHES):
        problems.append("a test match is in FIT_MATCHES")
    if C.VAL_MATCH in C.FIT_MATCHES:
        problems.append("VAL_MATCH is also a fit match")
    if set(C.PRETRAIN_MATCHES) & (set(C.TEST_MATCHES) | {C.VAL_MATCH}):
        problems.append("PRETRAIN_MATCHES includes a test or validation match")
    if C.USE_PRETRAIN and not os.path.isfile(f"{C.WORK}/pretrained_encoder.pt"):
        problems.append("USE_PRETRAIN is on but work/pretrained_encoder.pt is "
                        "missing: run step11_pretrain.py first")
    if count_params(PassReceiverNet()) != C.EXPECTED_PARAMS:
        problems.append(f"parameter count {count_params(PassReceiverNet())} != "
                        f"EXPECTED_PARAMS {C.EXPECTED_PARAMS}")
    if problems:
        print("Canonical protocol violated:")
        for p in problems:
            print("  - " + p)
        sys.exit(1)


def position_lookup(match_id):
    """(side, teamsheet slot) -> playing position code, from the parsed match."""
    try:
        from common import load_match
        m = load_match(match_id)
    except Exception as exc:                        # parsed pickle unavailable
        print(f"  (positions unavailable for {match_id}: {exc})")
        return {}
    out = {}
    for side in ("Home", "Away"):
        sheet = m["teamsheets"][side].teamsheet
        if "position" not in sheet.columns:
            continue
        for _, r in sheet.iterrows():
            if pd.notna(r.get("xID")):
                out[(side, int(r["xID"]))] = str(r["position"])
    return out


def positions_of(split, idx, lookup):
    """Position code of teammate index idx[i] for every pass i."""
    out = np.empty(len(idx), dtype=object)
    for i, k in enumerate(idx):
        slot = int(split["mate_slots"][i, int(k)])
        side = "Home" if split["team_side"][i] == 0 else "Away"
        out[i] = lookup.get((side, slot), "?") if slot >= 0 else "?"
    return out


def prediction_rows(split, match_id, variant, seed, score, prob, logits,
                    lookup):
    """One DataFrame row per pass for one predictor."""
    y = split["y"]
    n = len(y)
    present = split["X"][:, -1, 1:1 + K, C.IDX_PRESENT] > 0.5
    order = np.argsort(-score, axis=1, kind="stable")
    rank = np.argmax(order == y[:, None], axis=1) + 1
    pred = order[:, 0]
    df = pd.DataFrame({
        "model_variant": variant, "seed": seed, "test_match": match_id,
        "sample_idx": split["sample_idx"], "pass_row": split["pass_row"],
        "n_present": present.sum(1),
        "true_receiver": y,
        "true_receiver_slot": split["mate_slots"][np.arange(n), y],
        "pred_top1": pred,
        "top2": ["|".join(map(str, r[:2])) for r in order],
        "top3": ["|".join(map(str, r[:3])) for r in order],
        "top5": ["|".join(map(str, r[:5])) for r in order],
        "rank_true": rank,
    })
    for k in C.TOPK_REPORTED:
        df[f"hit_top{k}"] = (rank <= k).astype(int)
    df["true_position"] = positions_of(split, y, lookup)
    df["pred_position"] = positions_of(split, pred, lookup)
    for j in range(K):
        df[f"present_{j}"] = present[:, j].astype(int)
    for j in range(K):
        df[f"logit_{j}"] = logits[:, j] if logits is not None else np.nan
    for j in range(K):
        df[f"prob_{j}"] = prob[:, j] if prob is not None else np.nan
    for j in range(K):
        df[f"score_{j}"] = np.where(np.isfinite(score[:, j]), score[:, j], np.nan)
    return df


if __name__ == "__main__":
    force = "--force" in sys.argv
    device_banner()
    run_id = P.canonical_run_id()
    cfg = C.canonical_config()
    P.print_header(
        "Step 23 - canonical run", run_id,
        fit=C.FIT_MATCHES, validation=C.VAL_MATCH, test=C.TEST_MATCHES,
        lead_s=C.PREDICT_LEAD_S, seeds=C.SEEDS,
        training=f"cap {C.EPOCHS}, early stopping on {C.MODEL_SELECTION_METRIC}, "
                 f"patience {C.PATIENCE}, restore best {C.RESTORE_BEST_CHECKPOINT}",
        pretraining=f"{C.USE_PRETRAIN} (encoder from {C.PRETRAIN_MATCHES})",
        augmentation=f"mirror train {C.AUGMENT_MIRROR}, test-time avg {C.TTA_MIRROR}",
        features=f"space {C.USE_SPACE_FEATURES}, per-type norm {C.NORMALISE_PER_TYPE}, "
                 f"edge attention {C.USE_EDGE_ATTENTION}",
        batch=batch_size())
    check_protocol()

    fit = E.load_split(C.FIT_MATCHES)
    val = E.load_split([C.VAL_MATCH])
    tests = {m: E.load_split([m]) for m in C.TEST_MATCHES}
    counts = {"fit": {m: int((fit["match"] == m).sum()) for m in C.FIT_MATCHES},
              "validation": {C.VAL_MATCH: int(len(val["y"]))},
              "test": {m: int(len(t["y"])) for m, t in tests.items()}}
    print(f"samples  fit {len(fit['y'])} {counts['fit']}")
    print(f"         validation {len(val['y'])}   test {counts['test']}\n")

    lookups = {m: position_lookup(m) for m in C.TEST_MATCHES}
    frames, hist_rows, seed_info = [], [], []
    probs_by_match = {m: [] for m in C.TEST_MATCHES}

    for seed in C.SEEDS:
        model, stats, info = E.get_or_train(C.PREDICT_LEAD_S, seed, force=force)
        seed_info.append({k: info[k] for k in (
            "seed", "selected_epoch", "stopped_at", "epochs_run", "epoch_cap",
            "stopped_early", "best_val_top1", "pretrained_encoder_loaded",
            "parameters", "n_fit_samples", "n_val_samples", "checkpoint",
            "cache_key", "cache", "selection")})
        for h in info["history"]:
            hist_rows.append({"seed": seed, **h})
        for m, te in tests.items():
            logits = E.evaluate(model, stats, te)
            prob = E.softmax(logits)
            probs_by_match[m].append(prob)
            frames.append(prediction_rows(te, m, "canonical", seed, logits,
                                          prob, logits, lookups[m]))
            print(f"  seed {seed}  {m}: top-1 {(logits.argmax(1) == te['y']).mean():.3f}"
                  f"   (selected epoch {info['selected_epoch']}, "
                  f"val top-1 {info['best_val_top1']:.3f})")

    for m, te in tests.items():
        ens = np.mean(probs_by_match[m], axis=0)
        frames.append(prediction_rows(te, m, "canonical_seed_ensemble", -1, ens,
                                      ens, None, lookups[m]))

    Xb, yb = [], []
    for mid in C.BASELINE_TRAIN_MATCHES:
        s = E.load_split([mid])
        Xb.append(s["X"])
        yb.append(s["y"])
    Xb, yb = np.concatenate(Xb), np.concatenate(yb)
    for m, te in tests.items():
        for name, (score, prob) in baseline_scores(Xb, yb, te["X"]).items():
            frames.append(prediction_rows(te, m, f"baseline:{name}", -1, score,
                                          prob, None, lookups[m]))

    # every frame has the same columns, but the baseline frames have all-NA logit
    # columns; reindex first so pandas does not warn about inferring dtypes
    cols = frames[0].columns
    frames = [f.reindex(columns=cols) for f in frames]
    pred = pd.concat(frames, ignore_index=True)
    pred = P.stamp(pred, run_id, lead_s=C.PREDICT_LEAD_S)
    pred_path = f"{C.WORK}/metrics_predictions_{run_id}.csv"
    pred.to_csv(pred_path, index=False)

    hist = P.stamp(pd.DataFrame(hist_rows), run_id, lead_s=C.PREDICT_LEAD_S,
                   model_variant="canonical")
    hist.to_csv(f"{C.WORK}/canonical_history.csv", index=False)

    meta = P.base_metadata(
        "canonical", run_id,
        training_matches=C.FIT_MATCHES, validation_match=C.VAL_MATCH,
        test_matches=C.TEST_MATCHES, baseline_training_matches=C.BASELINE_TRAIN_MATCHES,
        pretrain_matches=C.PRETRAIN_MATCHES, lead_s=C.PREDICT_LEAD_S,
        n_seeds=len(C.SEEDS), seeds=C.SEEDS, epoch_cap=C.EPOCHS,
        early_stopping={"enabled": C.EARLY_STOPPING, "monitor": C.MODEL_SELECTION_METRIC,
                        "patience": C.PATIENCE,
                        "restore_best": C.RESTORE_BEST_CHECKPOINT},
        pretraining_used=C.USE_PRETRAIN,
        pretrained_encoder_loaded=all(s["pretrained_encoder_loaded"] for s in seed_info),
        model_parameters=count_params(PassReceiverNet()),
        sample_counts=counts,
        per_seed=seed_info,
        data_fingerprint=P.data_fingerprint(),
        pretrain_fingerprint=P.pretrain_fingerprint(),
        device="cuda" if ON_GPU else "cpu", effective_batch_size=batch_size(),
        predictions_file=pred_path, predictions_sha=P.hash_file(pred_path, n=16),
        history_file=f"{C.WORK}/canonical_history.csv",
        variants=sorted(pred["model_variant"].unique().tolist()),
    )
    P.write_json(f"{C.WORK}/run_metadata.json", meta)
    P.write_json(f"{C.WORK}/run_metadata_{run_id}.json", meta)

    print("\nsummary (top-1, mean over seeds):")
    can = pred[pred.model_variant == "canonical"]
    for m, g in can.groupby("test_match"):
        per_seed = g.groupby("seed")["hit_top1"].mean()
        print(f"  {m}: {per_seed.mean():.3f}  (seeds {', '.join(f'{v:.3f}' for v in per_seed)})")
    print(f"\nSaved -> {pred_path}")
    print(f"Saved -> {C.WORK}/canonical_history.csv")
    print(f"Saved -> {C.WORK}/run_metadata.json")
    print("\nNext: python verify/section8_results.py derives every Section VIII")
    print("number from the prediction file above.")
