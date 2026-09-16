"""Step 7 - Baselines.

A deep learning model that cannot beat a simple rule is a failed project, and
it is better to find that out now than on day seven.

Four baselines, in increasing order of effort:
  1. random          pick one of ten
  2. nearest         the closest teammate
  3. most advanced   the teammate furthest up the pitch
  4. gradient boosting on hand-made features

All features are computed from the last frame of the window (which ends
PREDICT_LEAD_S before the emitted synchronised frame), using nothing that
describes the pass itself.

Gradient boosting is fitted on config.BASELINE_TRAIN_MATCHES (every non-test
match). It makes no model-selection decision, so giving it the validation match
as well is the conservative choice for the paired comparison with the model.

`baseline_scores()` returns per-candidate scores for every baseline on the same
passes, which step23 exports next to the model's predictions so the McNemar test
compares predictors on identical passes.

Run:  python step7_baselines.py
Out:  work/baseline_results.csv
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import config as C
import provenance as P
from common import banner, load_npz


def load_split(match_ids):
    Xs, ys = [], []
    for mid in match_ids:
        d = load_npz(f"dataset_{mid}")
        if len(d["X"]):
            Xs.append(d["X"])
            ys.append(d["y"])
    return np.concatenate(Xs), np.concatenate(ys)


def ci95(acc, n):
    """Half-width of the 95% interval. With ~600 test passes this is ~4 points,
    so two models within 4 points of each other are not distinguishable."""
    return 1.96 * np.sqrt(acc * (1 - acc) / n)


# ---------------------------------------------------------------- geometry
def present_mask(X):
    """(n, 10) - which teammate slots hold a real player, not padding."""
    return X[:, -1, 1:1 + C.N_TEAMMATES, C.IDX_PRESENT] > 0.5


def candidate_features(X):
    """One feature row per (pass, candidate teammate). Shape (n, 10, F)."""
    now = X[:, -1]                       # (n, 23, 8) at the moment of the pass
    prev = X[:, 0]                       # start of the window

    passer = now[:, 0:1, :]              # (n, 1, 8)
    mates = now[:, 1:1 + C.N_TEAMMATES]  # (n, 10, 8)
    opps = now[:, 1 + C.N_TEAMMATES:1 + C.N_TEAMMATES + C.N_OPPONENTS]

    d_xy = mates[:, :, :2] - passer[:, :, :2]
    dist = np.linalg.norm(d_xy, axis=2)                    # (n, 10)
    angle = np.arctan2(d_xy[:, :, 1], d_xy[:, :, 0])

    # distance from each candidate to his nearest opponent
    diff = mates[:, :, None, :2] - opps[:, None, :, :2]    # (n, 10, 11, 2)
    d_opp = np.linalg.norm(diff, axis=3).min(axis=2)       # (n, 10)

    # opponents roughly inside the passing lane: project each opponent onto the
    # passer->candidate line and count those that are close to it and in front
    lane = np.zeros_like(dist)
    for k in range(C.N_TEAMMATES):
        v = d_xy[:, k]                                     # (n, 2)
        L = np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
        u = v / L
        w = opps[:, :, :2] - passer[:, :, :2]              # (n, 11, 2)
        along = (w * u[:, None, :]).sum(axis=2)
        across = np.abs(w[:, :, 0] * u[:, None, 1] - w[:, :, 1] * u[:, None, 0])
        lane[:, k] = ((along > 0) & (along < L) & (across < 3.0)).sum(axis=1)

    speed = np.linalg.norm(mates[:, :, 2:4], axis=2)
    moved = np.linalg.norm(mates[:, :, :2] - prev[:, 1:1 + C.N_TEAMMATES, :2], axis=2)
    ahead = mates[:, :, 0] - passer[:, :, 0]               # + = further upfield
    to_goal = 52.5 - mates[:, :, 0]                        # attacking +x

    return np.stack([dist, angle, d_opp, lane, speed, moved, ahead, to_goal], axis=2)


# --------------------------------------------------------------- baselines
def bl_random(X, rng):
    """Random among the players actually on the pitch, not among ten slots."""
    m = present_mask(X)
    out = np.zeros(len(X), dtype=int)
    for i in range(len(X)):
        out[i] = rng.choice(np.where(m[i])[0])
    return out


def bl_nearest(X):
    d = candidate_features(X)[:, :, 0].copy()
    d[~present_mask(X)] = np.inf          # never pick an empty slot
    return d.argmin(axis=1)               # smallest distance


def bl_advanced(X):
    a = candidate_features(X)[:, :, 6].copy()
    a[~present_mask(X)] = -np.inf
    return a.argmax(axis=1)               # furthest upfield


def bl_gbm(Xtr, ytr, Xte):
    """Binary 'is this candidate the receiver?', then pick the highest score."""
    Ftr, Fte = candidate_features(Xtr), candidate_features(Xte)
    n, k, f = Ftr.shape

    flat = Ftr.reshape(n * k, f)
    lab = np.zeros(n * k, dtype=int)
    lab[np.arange(n) * k + ytr] = 1
    keep = present_mask(Xtr).reshape(-1)      # drop padded candidates
    flat, lab = flat[keep], lab[keep]

    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         random_state=0)
    clf.fit(flat, lab)

    p = clf.predict_proba(Fte.reshape(-1, f))[:, 1].reshape(len(Xte), k)
    p[~present_mask(Xte)] = -1.0
    return p.argmax(axis=1), p


def baseline_scores(Xtr, ytr, Xte, rng_seed=0):
    """Per-candidate scores for every baseline, on exactly the passes in Xte.

    Returns {name: (score (n, 10) with -inf on absent slots, prob or None)}.
    Only gradient boosting yields probabilities; its per-candidate binary
    probabilities are renormalised over the present teammates so each pass sums
    to one. Heuristics yield a ranking but no calibrated probability, so their
    Brier score and calibration error are not defined.
    """
    present = present_mask(Xte)
    feats = candidate_features(Xte)
    rng = np.random.default_rng(rng_seed)

    out = {}
    s_rand = rng.random(present.shape)
    out["random"] = (np.where(present, s_rand, -np.inf), None)
    out["nearest teammate"] = (np.where(present, -feats[:, :, 0], -np.inf), None)
    out["most advanced"] = (np.where(present, feats[:, :, 6], -np.inf), None)
    _, p = bl_gbm(Xtr, ytr, Xte)
    p = np.where(present, np.clip(p, 0.0, None), 0.0)
    prob = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    out["gradient boosting"] = (np.where(present, prob, -np.inf), prob)
    return out


def topk(prob, y, k=3):
    order = np.argsort(-prob, axis=1)[:, :k]
    return float((order == y[:, None]).any(axis=1).mean())


if __name__ == "__main__":
    banner("Step 7 - baselines")
    rng = np.random.default_rng(0)

    run_id = P.canonical_run_id()
    P.print_header("baselines", run_id, fitted_on=C.BASELINE_TRAIN_MATCHES,
                   test=C.TEST_MATCHES, lead_s=C.PREDICT_LEAD_S)
    Xtr, ytr = load_split(C.BASELINE_TRAIN_MATCHES)
    print(f"train: {len(Xtr)} passes from {len(C.BASELINE_TRAIN_MATCHES)} matches\n")

    rows = []
    for test_id in C.TEST_MATCHES:
        Xte, yte = load_split([test_id])
        n = len(Xte)
        print(f"--- test on {test_id} ({n} passes) ---")

        preds = {
            "random": bl_random(Xte, rng),
            "nearest teammate": bl_nearest(Xte),
            "most advanced": bl_advanced(Xte),
        }
        gbm_pred, gbm_prob = bl_gbm(Xtr, ytr, Xte)
        preds["gradient boosting"] = gbm_pred

        for name, pred in preds.items():
            acc = float((pred == yte).mean())
            t3 = topk(gbm_prob, yte) if name == "gradient boosting" else np.nan
            print(f"  {name:20s} top1 = {acc:.3f} +/- {ci95(acc, n):.3f}"
                  + (f"   top3 = {t3:.3f}" if name == "gradient boosting" else ""))
            rows.append({"test_match": test_id, "model": name, "n": n,
                         "top1": acc, "ci95": ci95(acc, n), "top3": t3})
        print()

    out = f"{C.WORK}/baseline_results.csv"
    P.stamp(pd.DataFrame(rows), run_id, lead_s=C.PREDICT_LEAD_S,
            model_variant="baselines").to_csv(out, index=False)
    print(f"Saved -> {out}")
    print("\nThe neural network in step 9 must beat gradient boosting by more")
    print("than the confidence interval to count as an improvement.")
