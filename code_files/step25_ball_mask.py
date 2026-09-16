"""Step 25 - Ball masking: does the trained model use the ball?

The SAME trained model is evaluated twice on the same test passes: once normally,
once with the ball's information removed at test time. Nothing is retrained, so
the difference isolates how much the trained model's predictions depend on ball
information.

Mask variants (applied to the raw arrays, before the model's own normaliser):
  none                     unchanged
  ball_token               the ball object is removed: all its features zeroed,
                           including is_present, so the attention mask excludes
                           it exactly like a padded player
  ball_token_and_balldist  additionally, every player's distance-to-ball feature
                           (index 10) is replaced by its training-set mean at that
                           offset, so ball position cannot leak through a derived
                           feature either

Models: those trained at config.BALL_MASK_TRAIN_LEADS, by default 0.0 s (the
emitted frame, where the ball shortcut is available) and the canonical offset
(the control).

How to read it. A masked input is out of distribution for any model, so some
drop is expected even for a model that never relied on the ball. The evidence
is therefore the CONTRAST: a drop at 0.0 s much larger than the drop of the
canonical-offset model under the same mask. This is still not a causal proof;
a retrained ball-free model would be the stronger control and is not run here.

Run:  python step25_ball_mask.py
Out:  work/ball_mask_results.csv (per seed and match), work/ball_mask_summary.csv,
      work/ball_mask_predictions.csv (per pass), figures/paper/figC_ball_mask.png
"""

import numpy as np
import pandas as pd

import config as C
import experiment as E
import provenance as P
from device import banner as device_banner

BALL = C.N_OBJECTS - 1
MASKS = ["none", "ball_token", "ball_token_and_balldist"]


def balldist_mean(lead):
    fit = E.load_split(C.FIT_MATCHES, lead)
    X = fit["X"]
    present = X[:, :, :BALL, C.IDX_PRESENT] > 0.5
    return float(X[:, :, :BALL, C.IDX_BALLDIST][present].mean())


def apply_mask(X, mask, bd_mean):
    X = X.copy()
    if mask == "none":
        return X
    X[:, :, BALL, :] = 0.0
    if mask == "ball_token_and_balldist":
        present = X[:, :, :, C.IDX_PRESENT] > 0.5
        X[:, :, :, C.IDX_BALLDIST] = np.where(present, bd_mean, 0.0)
    return X


if __name__ == "__main__":
    device_banner()
    run_id = P.canonical_run_id()
    P.print_header("Step 25 - ball masking at test time (no retraining)", run_id,
                   trained_at_s=C.BALL_MASK_TRAIN_LEADS, masks=MASKS, seeds=C.SEEDS,
                   test=C.TEST_MATCHES)

    rows, pp = [], []
    for lead in C.BALL_MASK_TRAIN_LEADS:
        bd = balldist_mean(lead)
        tests = {m: E.load_split([m], lead) for m in C.TEST_MATCHES}
        print(f"\n--- models trained and tested at {lead:.2f} s ---")
        for seed in C.SEEDS:
            model, stats, info = E.get_or_train(lead, seed)
            for m, te in tests.items():
                accs = {}
                for mask in MASKS:
                    Xm = apply_mask(te["X"], mask, bd)
                    lg = E.predict_logits(model, E.apply_normaliser(Xm, stats))
                    pred = lg.argmax(1)
                    hit = pred == te["y"]
                    accs[mask] = float(hit.mean())
                    pp.append(pd.DataFrame({
                        "train_lead_s": lead, "seed": seed, "test_match": m,
                        "mask": mask, "pass_row": te["pass_row"],
                        "true_receiver": te["y"], "pred_top1": pred,
                        "hit_top1": hit.astype(int)}))
                for mask in MASKS[1:]:
                    rows.append({"train_lead_s": lead, "seed": seed, "test_match": m,
                                 "n": len(te["y"]), "mask": mask,
                                 "normal_top1": accs["none"], "masked_top1": accs[mask],
                                 "abs_drop": accs["none"] - accs[mask],
                                 "rel_drop": (accs["none"] - accs[mask]) / accs["none"]
                                 if accs["none"] > 0 else np.nan})
                print(f"  seed {seed} {m}: normal {accs['none']:.3f}   "
                      + "   ".join(f"{k} {accs[k]:.3f}" for k in MASKS[1:]))

    res = pd.DataFrame(rows)
    res.insert(0, "lead_s", res["train_lead_s"])
    res = P.stamp(res, run_id, model_variant="ball_mask")
    res.to_csv(f"{C.WORK}/ball_mask_results.csv", index=False)
    pred = pd.concat(pp, ignore_index=True)
    pred.insert(0, "lead_s", pred["train_lead_s"])
    P.stamp(pred, run_id, model_variant="ball_mask").to_csv(
        f"{C.WORK}/ball_mask_predictions.csv", index=False)

    summ = (res.groupby(["train_lead_s", "mask", "test_match"])
            .agg(n=("n", "first"), normal_mean=("normal_top1", "mean"),
                 masked_mean=("masked_top1", "mean"),
                 abs_drop_mean=("abs_drop", "mean"),
                 abs_drop_sd=("abs_drop", lambda v: v.std(ddof=1)),
                 rel_drop_mean=("rel_drop", "mean"), n_seeds=("seed", "nunique"))
            .reset_index())
    P.stamp(summ, run_id, model_variant="ball_mask_summary").to_csv(
        f"{C.WORK}/ball_mask_summary.csv", index=False)
    print("\n" + summ.round(3).to_string(index=False))

    d = summ.groupby(["train_lead_s", "mask"])["abs_drop_mean"].mean().unstack()
    if 0.0 in d.index and C.PREDICT_LEAD_S in d.index:
        for mask in MASKS[1:]:
            print(f"\n{mask}: drop {d.loc[0.0, mask]:+.3f} for the 0.00 s model vs "
                  f"{d.loc[C.PREDICT_LEAD_S, mask]:+.3f} for the {C.PREDICT_LEAD_S:.2f} s "
                  f"model (difference {d.loc[0.0, mask] - d.loc[C.PREDICT_LEAD_S, mask]:+.3f})")
    print("\nThe contrast between the two models is the evidence; a masked input is")
    print("out of distribution for both, so neither drop alone is.")
    print(f"\nSaved -> {C.WORK}/ball_mask_results.csv, ball_mask_summary.csv, "
          "ball_mask_predictions.csv")
    try:
        from step31_paper_figures import figure_c_ball_mask
        figure_c_ball_mask()
    except Exception as exc:
        print(f"(figure C not drawn: {exc})")
