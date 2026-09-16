"""Leakage probe. Run this before believing any accuracy number.

A model can score well for the wrong reason. The one that bit this project: the
synchroniser emits a frame at which the ball is already travelling towards the
receiver, so a window ending there hands the model the answer.

Each probe (see probes.py) guesses the receiver from ONE trivial signal, at every
time step of the window. Two things are reported per probe:

  level   how far above the nearest-teammate yardstick it sits at the frame
          closest to the anchor
  shape   how that excess evolves across the window (probes.flatness_diagnostic)

What the shape can and cannot tell you
--------------------------------------
A LOCALISED rise, concentrated in the last quarter of the window, is the profile
of an outcome entering the input near the anchor (a moving ball). A GRADUAL rise
is unresolved: a player legitimately turning towards his target and a gradual
leak produce the same curve. step26_flatness_validation.py demonstrates both on
synthetic data. So a rising passer signal is reported as unresolved, never as
proven leakage, and it does not fail this check.

Offsets are relative to the frame emitted by the synchroniser, not to true
contact, which no public dataset provides.

Run:  python check_leakage.py
Out:  work/leakage_probes.csv   (exit code 1 if the ball shortcut is available)
"""

import sys

import pandas as pd

import config as C
import probes as PR
import provenance as P
from common import banner, load_npz

BALL = PR.BALL
nearest_probe = PR.nearest_probe          # kept: other scripts import these
direction_probe = PR.direction_probe

BALL_FAIL_EXCESS = 0.12    # ball probe this far above the yardstick = shortcut available


def ball_state(X):
    import numpy as np
    fr = X[:, -1]
    speed = np.linalg.norm(fr[:, BALL, 2:4], axis=1)
    gap = np.linalg.norm(fr[:, BALL, :2] - fr[:, 0, :2], axis=1)
    return float(np.median(speed)), float(np.median(gap))


if __name__ == "__main__":
    banner("Leakage probes")
    run_id = P.canonical_run_id()
    print(f"run id {run_id}")
    print(f"Window ends {C.PREDICT_LEAD_S} s before the emitted synchronised frame.\n")

    failed, rows = False, []
    for mid in C.MATCHES:
        try:
            d = load_npz(f"dataset_{mid}")
        except FileNotFoundError:
            print(f"{mid}: no dataset, run step5_dataset.py first")
            continue
        X, y = d["X"], d["y"]
        if len(X) == 0:
            continue

        yard = PR.temporal_curve(X, y, "nearest_teammate")
        speed, gap = ball_state(X)
        print(f"--- {mid}  (n={len(X)}) ---")
        print(f"  ball speed at the last frame : {speed:5.2f} m/s   "
              f"distance from passer: {gap:.2f} m")
        print(f"  nearest teammate (yardstick) : {yard[-1]:.3f}")

        for name in ("ball_direction", "passer_direction"):
            curve = PR.temporal_curve(X, y, name)
            diag = PR.flatness_diagnostic(curve, yard)
            note = PR.LABEL_TEXT[diag["label"]]
            if name == "ball_direction" and (
                    diag["excess_end"] > BALL_FAIL_EXCESS
                    or (diag["label"] == "localised_rise"
                        and diag["excess_end"] > PR.MARGIN)):
                note += "   *** ball shortcut AVAILABLE ***"
                failed = True
            print(f"  {name:22s}: {curve[0]:.3f} -> {curve[-1]:.3f}   "
                  f"excess {diag['excess_start']:+.3f} -> {diag['excess_end']:+.3f}"
                  f"   {note}")
            rows.append({"match": mid, "n": len(X), "probe": name,
                         "acc_first_step": curve[0], "acc_last_step": curve[-1],
                         "yardstick_last_step": yard[-1],
                         "ball_speed_median": speed, "ball_passer_gap_median": gap,
                         **diag})
        print()

    if rows:
        out = f"{C.WORK}/leakage_probes.csv"
        P.stamp(pd.DataFrame(rows), run_id, lead_s=C.PREDICT_LEAD_S,
                model_variant="probe").to_csv(out, index=False)
        print(f"Saved -> {out}\n")

    print("Reminder: a probe above the yardstick shows a shortcut is AVAILABLE in")
    print("the inputs. Whether the trained model USES it is tested by step24")
    print("(cross-offset) and step25 (ball masking).")
    if failed:
        print("\nThe ball-direction probe scores far above the yardstick near the")
        print("anchor. Raise PREDICT_LEAD_S in config.py and rebuild from step 5.")
        sys.exit(1)
