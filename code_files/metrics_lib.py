"""Metrics computed from per-pass predictions. No torch, no training.

Every function takes arrays that come out of the exported prediction files, so
every reported metric can be recomputed from those files alone.

Definitions (stated once, here, and quoted by the verification scripts)
-----------------------------------------------------------------------
top-k        share of passes whose true receiver is among the k highest-probability
             teammates. Absent teammates (red card padding) have probability 0 and
             can never be ranked above a present one.
MRR          mean of 1 / rank of the true receiver.
mean rank    mean rank of the true receiver, 1 = best, 10 = worst.
Brier        multi-class Brier score: sum over the ten candidates of
             (p - onehot)^2, averaged over passes. Range 0 (perfect) to 2.
ECE          top-label expected calibration error with config.ECE_BINS equal-width
             bins on the top-1 probability: sum_b (n_b / n) |acc_b - conf_b|.
ROC-AUC, AP  computed over (pass, candidate) pairs, one binary decision per present
             candidate ("is this player the receiver?"). DESCRIPTIVE ONLY: each pass
             contributes exactly one positive, and candidates of the same pass are
             not independent, so these support no significance test.
McNemar      paired comparison of two predictors on the same passes. Reports the
             full 2x2 table, the continuity-corrected chi-square with its p-value,
             and the exact binomial p-value on the discordant pairs.
per position receiver POSITION CATEGORY evaluation, not identity: the true and the
             predicted receiver are mapped to their teamsheet playing position and
             precision / recall / F1 are computed over those categories. A
             prediction of the wrong centre back counts as a correct "IV". These
             numbers are therefore NOT ten-way teammate precision or recall.
"""

import numpy as np
import pandas as pd
from scipy import stats

import config as C


# ------------------------------------------------------------------ ranking
def ranks_from_probs(prob, y):
    """Rank (1 = top) of the true receiver. Ties broken by candidate index."""
    order = np.argsort(-prob, axis=1, kind="stable")
    return np.argmax(order == y[:, None], axis=1) + 1


def ranking_metrics(prob, y, ks=None):
    ks = C.TOPK_REPORTED if ks is None else ks
    rank = ranks_from_probs(prob, y)
    out = {f"top{k}": float((rank <= k).mean()) for k in ks}
    out["mrr"] = float((1.0 / rank).mean())
    out["mean_rank"] = float(rank.mean())
    return out


def binomial_ci95(p, n):
    """Normal-approximation half-width, as used in the manuscript tables."""
    return float(1.96 * np.sqrt(max(p * (1 - p), 0.0) / n)) if n else float("nan")


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(centre - half), float(centre + half)


# -------------------------------------------------------------- calibration
def brier(prob, y):
    onehot = np.zeros_like(prob)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((prob - onehot) ** 2).sum(1).mean())


def ece(prob, y, bins=None):
    bins = C.ECE_BINS if bins is None else bins
    conf = prob.max(1)
    correct = (prob.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def reliability_table(prob, y, bins=None):
    bins = C.ECE_BINS if bins is None else bins
    conf = prob.max(1)
    correct = (prob.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        rows.append({"bin_lo": lo, "bin_hi": hi, "n": int(m.sum()),
                     "mean_confidence": float(conf[m].mean()) if m.any() else np.nan,
                     "accuracy": float(correct[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------- candidate-level
def pair_arrays(prob, y, present=None):
    """Flatten to (score, label) over present candidates only."""
    n, k = prob.shape
    labels = np.zeros((n, k), dtype=int)
    labels[np.arange(n), y] = 1
    if present is None:
        present = prob > 0
    present = present.astype(bool)
    present[np.arange(n), y] = True
    return prob[present], labels[present]


def pair_auc_ap(prob, y, present=None):
    from sklearn.metrics import average_precision_score, roc_auc_score
    s, l = pair_arrays(prob, y, present)
    if l.min() == l.max():
        return float("nan"), float("nan")
    return float(roc_auc_score(l, s)), float(average_precision_score(l, s))


def full_metrics(prob, y, present=None):
    """Everything in canonical_metrics_summary.csv for one prediction set."""
    out = {"n": int(len(y))}
    out.update(ranking_metrics(prob, y))
    out["top1_ci95"] = binomial_ci95(out["top1"], len(y))
    out["brier"] = brier(prob, y)
    out["ece"] = ece(prob, y)
    out["roc_auc_pairs"], out["ap_pairs"] = pair_auc_ap(prob, y, present)
    return out


# --------------------------------------------------------------- McNemar
def mcnemar(a_hit, b_hit):
    """Paired test on the same passes. a_hit, b_hit: boolean arrays."""
    a_hit = np.asarray(a_hit, bool)
    b_hit = np.asarray(b_hit, bool)
    both = int((a_hit & b_hit).sum())
    a_only = int((a_hit & ~b_hit).sum())
    b_only = int((~a_hit & b_hit).sum())
    neither = int((~a_hit & ~b_hit).sum())
    disc = a_only + b_only
    if disc == 0:
        chi2, p_chi2, p_exact = 0.0, 1.0, 1.0
    else:
        chi2 = (abs(a_only - b_only) - 1) ** 2 / disc
        p_chi2 = float(stats.chi2.sf(chi2, 1))
        p_exact = float(stats.binomtest(min(a_only, b_only), disc, 0.5).pvalue)
    return {"n": int(len(a_hit)), "both_correct": both, "model_only": a_only,
            "baseline_only": b_only, "both_wrong": neither,
            "chi2_cc": float(chi2), "p_chi2_cc": float(p_chi2),
            "p_exact_binomial": float(p_exact)}


# -------------------------------------------------------------- positions
def per_position(true_pos, pred_pos, exclude=("?",)):
    """Precision, recall, F1 and support per receiving-position category.

    Rows whose true OR predicted position is unknown are dropped, and the number
    dropped is returned so it can be reported.
    """
    true_pos = np.asarray(true_pos, dtype=object)
    pred_pos = np.asarray(pred_pos, dtype=object)
    keep = ~np.isin(true_pos, exclude) & ~np.isin(pred_pos, exclude)
    t, p = true_pos[keep], pred_pos[keep]
    classes = sorted(set(t) | set(p))
    rows = []
    for c in classes:
        tp = int(((p == c) & (t == c)).sum())
        fp = int(((p == c) & (t != c)).sum())
        fn = int(((p != c) & (t == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"position": c, "support": int((t == c).sum()),
                     "predicted": int((p == c).sum()), "precision": prec,
                     "recall": rec, "f1": f1})
    df = pd.DataFrame(rows)
    if len(df):
        sup = df["support"].to_numpy(float)
        extra = []
        for name, w in (("macro", np.ones(len(df))), ("weighted", sup)):
            if w.sum() <= 0:
                continue
            extra.append({"position": f"_{name}_avg", "support": int(sup.sum()),
                          "predicted": int(df["predicted"].sum()),
                          "precision": float(np.average(df["precision"], weights=w)),
                          "recall": float(np.average(df["recall"], weights=w)),
                          "f1": float(np.average(df["f1"], weights=w))})
        if extra:
            df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    return df, int((~keep).sum())


# -------------------------------------------------------------- summaries
def mean_sd_ci(values):
    """Mean, sample sd and a t-based 95% interval over seeds."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    m = float(v.mean())
    if n == 1:
        return m, float("nan"), float("nan"), float("nan")
    sd = float(v.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return m, sd, m - half, m + half
