"""Step 26 - Does the flatness criterion actually work? A synthetic validation.

METHODOLOGICAL VALIDATION OF THE DIAGNOSTIC. Nothing here is evidence about real
soccer data; it is a test of the criterion itself on cases whose truth is known
by construction.

The criterion (probes.flatness_diagnostic) looks at how a probe's excess over the
nearest-teammate yardstick evolves across the observation window and labels it
flat, localised_rise, gradual_rise or falling.

Four synthetic generators, each producing windows with a known ground truth:

  A  leak_step       a planted shortcut that appears only in the final frames:
                     the signal points at the receiver from the anchor backwards
                     for a short time, and is uninformative before that.
                     TRUTH: leakage.
  B  info_ramp       a legitimate ramp: the passer turns towards his chosen
                     receiver gradually over the whole window, so the signal is
                     informative early and MORE informative late. No information
                     from after the decision is used. TRUTH: information.
  C  flat_info       a constant signal, informative equally at every step.
                     TRUTH: information.
  D  none            no signal at all. TRUTH: nothing.

What is being measured: whether the criterion separates A from B. The honest
expected outcome is that it separates A from C and D, and that B is reported as
"gradual_rise", i.e. UNRESOLVED, rather than being called leakage. A ramp that is
steep enough is genuinely indistinguishable from a soft leak by temporal shape
alone: the failure mode is real and is what the label records.

The legacy rule the project used before (rise > 0.10 or excess > 0.12 = "leak")
is applied to the same cases, so its false-positive rate on legitimate ramps is
measured rather than assumed.

Run:  python step26_flatness_validation.py
Out:  work/flatness_validation.csv, work/flatness_validation_curves.csv,
      figures/paper/fig_flatness_validation.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
import probes as PR
import provenance as P

OUT = f"{C.FIGS}/paper"
K = C.N_TEAMMATES
T = C.N_STEPS
INK, MUTED, DATA, WARN, GOOD = "#14171A", "#6E7378", "#2E6F9E", "#B3402F", "#1F7A4D"

# Cases are defined by the ACCURACY PROFILE they plant, not by a raw signal
# strength: the criterion's claim is about the shape of the probe's accuracy
# curve, so the generator controls that curve directly. w_for_accuracy() inverts
# the (signal weight -> accuracy) mapping empirically, once.
CASES = {
    "A_leak_step": dict(
        profile="step", desc="planted shortcut, final 2 frames only (TRUTH: leakage)",
        truth="leakage"),
    "B_info_ramp": dict(
        profile="linear", desc="legitimate turn, accuracy rises linearly across the "
                               "whole window (TRUTH: information)", truth="information"),
    "C_flat_info": dict(
        profile="flat", desc="constant informative signal (TRUTH: information)",
        truth="information"),
    "D_none": dict(profile="none", desc="no signal (TRUTH: nothing)", truth="nothing"),
}
LOW, HIGH = 0.30, 0.80          # accuracy floor and ceiling of the planted profiles


def accuracy_profile(kind, T=T, low=LOW, high=HIGH, leak_frames=2):
    """Target probe accuracy at each step, earliest first."""
    if kind == "step":
        a = np.full(T, low)
        a[-leak_frames:] = high
    elif kind == "linear":
        a = np.linspace(low, high, T)
    elif kind == "flat":
        a = np.full(T, (low + high) / 2)
    else:
        raise ValueError(f"no accuracy profile for {kind!r}")
    return a


def _mapping(n=4000, seed=7):
    """Empirical (signal weight -> probe accuracy) curve for this geometry."""
    ws = np.linspace(0.0, 1.0, 21)
    accs = []
    for w in ws:
        X, y = _make(np.full(T, w), n=n, rng=seed)
        accs.append(PR.temporal_curve(X, y, "passer_direction")[-1])
    return ws, np.array(accs)


_MAP = None


def w_for_accuracy(target):
    """Signal weights that produce the target accuracy profile."""
    global _MAP
    if _MAP is None:
        _MAP = _mapping()
    ws, accs = _MAP
    return np.interp(np.clip(target, accs.min(), accs.max()), accs, ws)


def _make(weights, n=3000, rng=None):
    """Samples whose passer-velocity points at the receiver with weight w per step.

    Object 0 is the passer, 1..10 the teammates. Teammates sit at random angles
    and distances, so the nearest-teammate yardstick stays weak and flat.
    """
    rng = np.random.default_rng(0 if rng is None else rng)
    X = np.zeros((n, T, C.N_OBJECTS, C.N_FEATURES), np.float32)
    X[:, :, :, C.IDX_PRESENT] = 1.0
    X[:, :, 1:1 + K, 4] = 1.0

    ang = rng.uniform(-np.pi, np.pi, (n, K))
    rad = rng.uniform(8.0, 30.0, (n, K))
    mates = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=2)
    X[:, :, 1:1 + K, :2] = mates[:, None, :, :]
    y = rng.integers(0, K, n)
    to_true = mates[np.arange(n), y]
    to_true = to_true / np.linalg.norm(to_true, axis=1, keepdims=True)

    for t, w in enumerate(weights):
        noise = rng.normal(size=(n, 2))
        noise /= np.linalg.norm(noise, axis=1, keepdims=True)
        v = w * to_true + (1 - w) * noise
        X[:, t, 0, 2:4] = v / np.linalg.norm(v, axis=1, keepdims=True)
    return X, y


def make_case(case, n=3000, rng=None, **kw):
    spec = CASES[case]
    if spec["profile"] == "none":
        return _make(np.zeros(T), n=n, rng=rng)      # no signal at all
    target = accuracy_profile(spec["profile"], **kw)
    return _make(w_for_accuracy(target), n=n, rng=rng)


def curves(X, y):
    return (PR.temporal_curve(X, y, "passer_direction"),
            PR.temporal_curve(X, y, "nearest_teammate"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    run_id = P.canonical_run_id()
    P.print_header("Step 26 - synthetic validation of the flatness criterion",
                   run_id, cases=list(CASES), steps=T, note="synthetic data only")
    print("This validates the DIAGNOSTIC. It says nothing about real soccer data.\n")

    rows, cdata = [], []
    for case, spec in CASES.items():
        X, y = make_case(case, rng=abs(hash(case)) % 1000)
        probe, yard = curves(X, y)
        d = PR.flatness_diagnostic(probe, yard)
        truth = spec["truth"]
        not_mislabelled = not (truth != "leakage" and d["label"] == "localised_rise")
        rows.append({"case": case, "description": spec["desc"], "truth": truth,
                     "label": d["label"],
                     "resolved": d["label"] != "gradual_rise",
                     "not_mislabelled": not_mislabelled,
                     "legacy_verdict": d["legacy_verdict"],
                     "legacy_correct": (d["legacy_verdict"] == "leak") == (truth == "leakage"),
                     **{k: v for k, v in d.items() if k not in ("label", "legacy_verdict")}})
        for t, (p, yv) in enumerate(zip(probe, yard)):
            cdata.append({"case": case, "step": t,
                          "offset_steps_before_anchor": T - 1 - t,
                          "probe": p, "yardstick": yv, "excess": p - yv})
        print(f"{case:14s} {d['label']:22s} excess {d['excess_start']:+.3f} -> "
              f"{d['excess_end']:+.3f}, rise {d['rise']:+.3f}, "
              f"localisation {d['localisation']:.2f}   legacy: {d['legacy_verdict']}")

    # How sharp must a rise be before the criterion calls it localised? Plant the
    # same total rise concentrated in the final `k` frames, k = 1 .. T.
    print("\nsensitivity: the same total rise, concentrated in the final k frames")
    sens = []
    for k in range(1, T + 1):
        target = np.full(T, LOW)
        target[T - k:] = np.linspace(LOW, HIGH, k + 1)[1:]
        X, y = _make(w_for_accuracy(target), rng=123)
        probe, yard = curves(X, y)
        d = PR.flatness_diagnostic(probe, yard)
        sens.append({"rise_over_final_k_frames": k,
                     "rise_duration_s": k * C.STRIDE / C.FRAMERATE,
                     "label": d["label"], "localisation": d["localisation"],
                     "rise": d["rise"], "legacy_verdict": d["legacy_verdict"]})
        print(f"  k={k:2d} ({k * C.STRIDE / C.FRAMERATE:.2f} s): "
              f"localisation {d['localisation']:.2f}  -> {d['label']}")
    sdf = pd.DataFrame(sens)
    P.stamp(sdf, run_id, model_variant="flatness_sensitivity").to_csv(
        f"{C.WORK}/flatness_sensitivity.csv", index=False)
    loc_rise = sdf[sdf.label == "localised_rise"]["rise_duration_s"]
    if len(loc_rise):
        print(f"  the criterion calls a rise localised when it happens within about "
              f"{loc_rise.max():.2f} s of the anchor or less")

    df = pd.DataFrame(rows)
    P.stamp(df, run_id, model_variant="flatness_validation").to_csv(
        f"{C.WORK}/flatness_validation.csv", index=False)
    cdf = pd.DataFrame(cdata)
    P.stamp(cdf, run_id, model_variant="flatness_validation").to_csv(
        f"{C.WORK}/flatness_validation_curves.csv", index=False)

    a = df[df.case == "A_leak_step"].iloc[0]
    b = df[df.case == "B_info_ramp"].iloc[0]
    print("\nWhat this establishes")
    print(f"  planted step leak  -> {a['label']}")
    print(f"  legitimate ramp    -> {b['label']}")
    if a["label"] == "localised_rise" and b["label"] == "gradual_rise":
        print("  The criterion separates a step-like leak from a gradual ramp, and")
        print("  reports the ramp as UNRESOLVED rather than as leakage. It cannot")
        print("  certify a gradual signal either way: that is its stated limit.")
    elif a["label"] == b["label"]:
        print("  The criterion does NOT separate these two cases. On this evidence a")
        print("  rising probe cannot be read as leakage at all.")
    legacy_fp = df[(df.truth != "leakage") & (df.legacy_verdict == "leak")]
    print(f"  legacy rule: {len(legacy_fp)} of {int((df.truth != 'leakage').sum())} "
          f"non-leak cases labelled 'leak'"
          + (f" ({', '.join(legacy_fp.case)})" if len(legacy_fp) else ""))

    fig, axes = plt.subplots(1, len(CASES), figsize=(4.0 * len(CASES), 3.6),
                             sharey=True)
    for ax, case in zip(np.atleast_1d(axes), CASES):
        s = cdf[cdf.case == case]
        x = -s["offset_steps_before_anchor"] * C.STRIDE / C.FRAMERATE
        ax.plot(x, s["probe"], "-o", color=DATA, lw=2, ms=4, label="probe")
        ax.plot(x, s["yardstick"], "--", color=MUTED, lw=1.4, label="yardstick")
        lab = df.loc[df.case == case, "label"].iloc[0]
        ax.set_title(f"{case}\n{lab}", fontsize=10)
        ax.set_xlabel("time relative to anchor (s)")
        ax.grid(alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("probe accuracy")
    np.atleast_1d(axes)[0].legend(fontsize=9)
    fig.suptitle("Synthetic validation of the flatness criterion "
                 "(synthetic data, not soccer)", fontsize=12, weight="600", y=1.04)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig_flatness_validation.{ext}", bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"\nSaved -> {C.WORK}/flatness_validation.csv, "
          f"flatness_sensitivity.csv, {OUT}/fig_flatness_validation.png")
