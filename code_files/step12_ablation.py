"""Step 12 - Which improvements actually help?

Six factors were added on top of the plain model, each borrowed from published
work. This measures them, with the bookkeeping made explicit.

  factor         config flag            what it is
  mirror_train   AUGMENT_MIRROR         reflect the pitch in y during training
                                        (cheap stand-in for TacticAI's D2
                                        equivariance)
  tta            TTA_MIRROR             average the two reflections at inference
  space          (arrays, features 9-10) pressure and ball distance as node features
  edge           USE_EDGE_ATTENTION     geometric bias in the attention
  pretrain       USE_PRETRAIN           encoder warm-started from step11
  pertype        NORMALISE_PER_TYPE     players and ball on separate scales

Settings are of three KINDS, and only one kind may be summed:
  reference    "plain": all six factors off
  individual   exactly ONE factor on. Individual gains are non-overlapping and
               may be summed to an additive prediction.
  composite    several factors on at once ("+ both mirror" = mirror_train + tta,
               "everything" = all six). A composite is never counted as an
               intervention and is never added to a sum of individual gains;
               verify/table4_ablation.py compares it with the sum of its own
               components and reports the difference as an interaction.

Every run uses the canonical protocol (fit matches, early stopping on the
validation match, best epoch restored). Each seed trains one model per setting,
evaluated on every test match. "everything" is the canonical configuration, so it
reuses the canonical checkpoints.

Run:  python step12_ablation.py              seeds = config.SEEDS
      python step12_ablation.py --robust     config.ROBUST_SEEDS for the
                                             config.ROBUST_SETTINGS rows
Out:  work/ablation_results.csv (or ablation_results_robust.csv),
      work/ablation_definitions.csv
"""

import os
import sys

import numpy as np
import pandas as pd

import config as C
import experiment as E
import provenance as P

FACTORS = ["mirror_train", "tta", "space", "edge", "pretrain", "pertype"]
FLAG = {"mirror_train": "AUGMENT_MIRROR", "tta": "TTA_MIRROR",
        "edge": "USE_EDGE_ATTENTION", "pretrain": "USE_PRETRAIN",
        "pertype": "NORMALISE_PER_TYPE"}

SETTINGS = [
    ("plain", "reference", []),
    ("+ train mirror", "individual", ["mirror_train"]),
    ("+ test-time avg", "individual", ["tta"]),
    ("+ both mirror", "composite", ["mirror_train", "tta"]),
    ("+ space", "individual", ["space"]),
    ("+ edge", "individual", ["edge"]),
    ("+ pretrain", "individual", ["pretrain"]),
    ("+ per-type norm", "individual", ["pertype"]),
    ("everything", "composite", list(FACTORS)),
]
KIND = {name: kind for name, kind, _ in SETTINGS}
COMPONENTS = {name: comps for name, _, comps in SETTINGS}


def definitions():
    rows = []
    for name, kind, comps in SETTINGS:
        rows.append({"setting": name, "kind": kind,
                     "components": "+".join(comps) if comps else "",
                     **{f: int(f in comps) for f in FACTORS}})
    return pd.DataFrame(rows)


def flags_for(comps):
    return {FLAG[f]: (f in comps) for f in FLAG}


def run_setting(name, seeds, fit_matches=None, fit_mask_fn=None, subset_tag=None,
                tests=None, verbose=True):
    """Train (or load) one model per seed for this setting; score every test match."""
    comps = COMPONENTS[name]
    keep_space = "space" in comps
    tests = tests or {m: E.load_split([m], keep_space=keep_space)
                      for m in C.TEST_MATCHES}
    rows = []
    with E.overrides(**flags_for(comps)):
        for seed in seeds:
            model, stats, info = E.get_or_train(
                C.PREDICT_LEAD_S, seed, fit_matches=fit_matches,
                keep_space=keep_space, fit_mask_fn=fit_mask_fn,
                subset_tag=subset_tag, verbose=verbose)
            for m, te in tests.items():
                lg = E.evaluate(model, stats, te)
                order = np.argsort(-lg, axis=1)
                rows.append({"setting": name, "kind": KIND[name],
                             "components": "+".join(comps), "seed": seed,
                             "test_match": m, "n": len(te["y"]),
                             "top1": float((order[:, 0] == te["y"]).mean()),
                             "top3": float((order[:, :3] == te["y"][:, None]).any(1).mean()),
                             "selected_epoch": info["selected_epoch"],
                             "epochs_run": info["epochs_run"],
                             "n_fit_samples": info["n_fit_samples"],
                             "pretrained_encoder_loaded": info["pretrained_encoder_loaded"],
                             **{f: int(f in comps) for f in FACTORS}})
    return rows


if __name__ == "__main__":
    robust = "--robust" in sys.argv
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--settings="):
            only = a.split("=", 1)[1].split(",")
    run_id = P.canonical_run_id()
    names = [n for n, _, _ in SETTINGS if only is None or n in only]
    seeds_for = {n: (E.seed_list(True) if robust and n in C.ROBUST_SETTINGS
                     else E.seed_list(False)) for n in names}
    P.print_header("Step 12 - ablation", run_id,
                   mode="robust (extra seeds)" if robust else "standard",
                   settings=len(names), fit=C.FIT_MATCHES, validation=C.VAL_MATCH,
                   test=C.TEST_MATCHES, lead_s=C.PREDICT_LEAD_S,
                   seeds={n: len(s) for n, s in seeds_for.items()})
    if not os.path.isfile(f"{C.WORK}/pretrained_encoder.pt"):
        print("No pretrained encoder yet: the pretrain rows will run WITHOUT one.")
        print("Run step11_pretrain.py first for meaningful pretraining rows.\n")

    rows = []
    for name in names:
        print(f"--- {name} ({KIND[name]}) ---")
        rows += run_setting(name, seeds_for[name])
        df_n = pd.DataFrame([r for r in rows if r["setting"] == name])
        print("  " + "   ".join(f"{m} {g.top1.mean():.3f} (sd {g.top1.std(ddof=1):.3f})"
                                for m, g in df_n.groupby("test_match")))

    df = P.stamp(pd.DataFrame(rows), run_id, lead_s=C.PREDICT_LEAD_S)
    df["model_variant"] = df["setting"]
    out = f"{C.WORK}/ablation_results{'_robust' if robust else ''}.csv"
    df.to_csv(out, index=False)
    P.stamp(definitions(), run_id, model_variant="ablation_definitions").to_csv(
        f"{C.WORK}/ablation_definitions.csv", index=False)
    print(f"\nSaved -> {out}")
    print(f"Saved -> {C.WORK}/ablation_definitions.csv")
    print("Next: python verify/table4_ablation.py computes gains, noise floors and")
    print("the non-overlapping additive arithmetic from this file.")
