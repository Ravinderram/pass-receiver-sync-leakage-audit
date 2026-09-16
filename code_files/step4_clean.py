"""Step 4 - Clean the tracking data.

The paper warns that camera-based positions are noisy and that a low-pass
filter should be applied before deriving speed, especially in fast situations.
floodlight has the Butterworth filter built in and it handles the NaN gaps of
substituted players correctly (it filters each continuous stretch separately).

Run:  python step4_clean.py
Out:  work/clean_<match>.npz  with smoothed positions and velocities
"""

import numpy as np
from floodlight.transforms.filter import butterworth_lowpass

import config as C
from common import (attack_sign, banner, coords, goalkeeper_slot, load_match,
                    save_npz, velocity)


def clean_match(match_id):
    m = load_match(match_id)
    arrays = {}
    signs = {}

    for half in ("firstHalf", "secondHalf"):
        # which way does each team attack in this half?
        ts_home = m["teamsheets"]["Home"].teamsheet
        ts_away = m["teamsheets"]["Away"].teamsheet
        sign_home = attack_sign(m["xy"][half]["Home"], ts_home)
        sign_away_own = attack_sign(m["xy"][half]["Away"], ts_away)

        # cross-check: the two keepers must be at opposite ends
        if sign_home == sign_away_own:
            gk_h, src_h = goalkeeper_slot(m["xy"][half]["Home"], ts_home)
            gk_a, src_a = goalkeeper_slot(m["xy"][half]["Away"], ts_away)
            print(f"  !! {match_id} {half}: both teams appear to attack the same "
                  f"way. Home GK slot {gk_h} ({src_h}), Away GK slot {gk_a} "
                  f"({src_a}). Run diagnose_direction.py {match_id}")
        sign_away = -sign_home
        signs[half] = sign_home

        for name in ("Home", "Away", "Ball"):
            smooth = butterworth_lowpass(
                m["xy"][half][name], order=C.FILTER_ORDER, Wn=C.FILTER_CUTOFF_HZ
            )
            pos = coords(smooth)
            vel = velocity(pos)
            arrays[f"{half}_{name}_pos"] = pos.astype(np.float32)
            arrays[f"{half}_{name}_vel"] = vel.astype(np.float32)

        arrays[f"{half}_sign_home"] = np.float32(sign_home)
        arrays[f"{half}_sign_away"] = np.float32(sign_away)

        print(f"{match_id} {half:11s} "
              f"home attacks {'+x' if sign_home > 0 else '-x'}  "
              f"frames={arrays[f'{half}_Home_pos'].shape[0]}")

    # teams always swap ends at half time, so a non-flip is always an error
    if signs["firstHalf"] == signs["secondHalf"]:
        print(f"  *** {match_id}: NO FLIP between halves. Direction detection "
              f"failed. Run: python diagnose_direction.py {match_id}")

    return save_npz(f"clean_{match_id}", **arrays)


if __name__ == "__main__":
    banner("Step 4 - filtering and velocities")
    print("The attacking direction must flip between the two halves for each")
    print("team. If it does not, the direction detection is wrong.\n")
    for mid in C.MATCHES:
        path = clean_match(mid)
        print(f"  saved -> {path}\n")
