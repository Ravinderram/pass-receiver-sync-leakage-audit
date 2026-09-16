"""One-line receiver probes, provider-independent, plus the flatness diagnostic.

A probe guesses the receiver from ONE trivial signal. Probes are defined on a
provider-independent Snapshot (positions and velocities of the passer, his
teammates and the ball at a single frame), so the same code runs on the IDSSE
samples, on another provider through providers/, and on synthetic data.

Probes
------
nearest_teammate   the teammate closest to the passer. A legitimate baseline and
                   the yardstick the others are compared against.
ball_direction     the teammate the ball's velocity vector points at.
passer_direction   the teammate the passer's velocity vector points at (a
                   velocity-based stand-in for body orientation).
masella_min_abs    Masella et al. (arXiv:2605.25696, eq. 4) define the signed
masella_max_abs    angle between the passer's facing direction u and the vector w
                   to a teammate, theta = atan2(u_x w_y - u_y w_x, u . w), with u
                   the normalised vector "from ball to passer". As a one-line
                   rule the receiver is the teammate with the smallest |theta|
                   (min_abs) or the largest (max_abs). The two rules are also the
                   two possible readings of the sign convention of u, so both are
                   always reported, and which one is applied to test matches is
                   chosen on non-test matches only. This is a probe built from one
                   of their edge features, not a reproduction of their model.

Available versus used
---------------------
A probe scoring far above the yardstick shows that a shortcut is AVAILABLE in
the inputs. It does not show that a trained model USES it (that is what the
cross-offset and ball-masking experiments test), and it is not a measurement of
the synchronisation offset against ground truth (no public dataset allows that).
"""

from dataclasses import dataclass

import numpy as np

import config as C

BALL = C.N_OBJECTS - 1


# ================================================================ snapshot
@dataclass
class Snapshot:
    passer_xy: np.ndarray      # (n, 2)
    passer_v: np.ndarray       # (n, 2)
    mates_xy: np.ndarray       # (n, 10, 2)
    mates_present: np.ndarray  # (n, 10) bool
    ball_xy: np.ndarray        # (n, 2)
    ball_v: np.ndarray         # (n, 2)
    y: np.ndarray              # (n,) index of the receiver among the teammates

    def __len__(self):
        return len(self.y)


def snapshot_from_samples(X, y, t=-1):
    """Build a Snapshot from RAW (un-normalised) IDSSE sample arrays at step t."""
    fr = X[:, t]
    mates = fr[:, 1:1 + C.N_TEAMMATES]
    return Snapshot(passer_xy=fr[:, 0, :2], passer_v=fr[:, 0, 2:4],
                    mates_xy=mates[:, :, :2],
                    mates_present=mates[:, :, C.IDX_PRESENT] > 0.5,
                    ball_xy=fr[:, BALL, :2], ball_v=fr[:, BALL, 2:4],
                    y=np.asarray(y))


# ================================================================== probes
def _masked_argmax(score, present):
    score = np.where(present, score, -np.inf)
    return score.argmax(1)


def nearest_teammate(s):
    d = np.linalg.norm(s.mates_xy - s.passer_xy[:, None, :], axis=2)
    return _masked_argmax(-d, s.mates_present), np.ones(len(s), bool)


def _direction(origin, vel, s):
    v = vel / (np.linalg.norm(vel, axis=1, keepdims=True) + 1e-9)
    to = s.mates_xy - origin[:, None, :]
    to = to / (np.linalg.norm(to, axis=2, keepdims=True) + 1e-9)
    cos = (to * v[:, None, :]).sum(axis=2)
    return _masked_argmax(cos, s.mates_present), np.linalg.norm(vel, axis=1) > 1e-6


def ball_direction(s):
    return _direction(s.ball_xy, s.ball_v, s)


def passer_direction(s):
    return _direction(s.passer_xy, s.passer_v, s)


def masella_theta(s):
    """Signed angle (radians) per teammate, eq. 4 of Masella et al., and a mask
    of passes where the facing vector is defined."""
    u = s.passer_xy - s.ball_xy                      # "from ball to passer"
    norm = np.linalg.norm(u, axis=1, keepdims=True)
    defined = norm[:, 0] >= C.MASELLA_MIN_BALL_PASSER_M
    u = u / (norm + 1e-9)
    w = s.mates_xy - s.passer_xy[:, None, :]
    cross = u[:, None, 0] * w[:, :, 1] - u[:, None, 1] * w[:, :, 0]
    dot = u[:, None, 0] * w[:, :, 0] + u[:, None, 1] * w[:, :, 1]
    return np.arctan2(cross, dot), defined


def masella_min_abs(s):
    theta, defined = masella_theta(s)
    return _masked_argmax(-np.abs(theta), s.mates_present), defined


def masella_max_abs(s):
    theta, defined = masella_theta(s)
    return _masked_argmax(np.abs(theta), s.mates_present), defined


PROBES = {
    "nearest_teammate": nearest_teammate,
    "ball_direction": ball_direction,
    "passer_direction": passer_direction,
    "masella_min_abs": masella_min_abs,
    "masella_max_abs": masella_max_abs,
}


def run_probe(name, s, undefined="count_wrong"):
    """Accuracy of one probe on one snapshot.

    Returns dict(acc, n, n_defined, acc_defined). Passes where the probe's
    signal is undefined (zero velocity, ball at the passer's foot) are counted
    as misses in `acc` and excluded from `acc_defined`, so neither number can
    flatter the probe by silently dropping hard cases.
    """
    pred, defined = PROBES[name](s)
    hit = pred == s.y
    n_def = int(defined.sum())
    return {"acc": float(hit[defined].sum() / max(len(s), 1)) if undefined == "count_wrong"
            else float(hit.mean()),
            "n": int(len(s)), "n_defined": n_def,
            "acc_defined": float(hit[defined].mean()) if n_def else float("nan")}


# ============================================ backward-compatible helpers
def direction_probe(X, y, obj, t=-1):
    """Accuracy of 'the teammate object obj's velocity points at' (old API)."""
    s = snapshot_from_samples(X, y, t)
    fr = X[:, t]
    pred, _ = _direction(fr[:, obj, :2], fr[:, obj, 2:4], s)
    return float((pred == s.y).mean())


def nearest_probe(X, y, t=-1):
    s = snapshot_from_samples(X, y, t)
    return float((nearest_teammate(s)[0] == s.y).mean())


def temporal_curve(X, y, name):
    """Probe accuracy at every time step of the window (earliest first)."""
    return np.array([float((PROBES[name](snapshot_from_samples(X, y, t))[0] == y).mean())
                     for t in range(X.shape[1])])


# ======================================================= flatness diagnostic
LOCALISATION_THRESHOLD = 0.7   # share of the rise inside the final quarter
RISE_TOL = 0.03                # smaller rises are treated as flat
MARGIN = 0.04                  # excess over the yardstick that counts as "above"
TAIL_FRAC = 0.25


def flatness_diagnostic(curve, yardstick, tail_frac=TAIL_FRAC, rise_tol=RISE_TOL,
                        margin=MARGIN, loc_threshold=LOCALISATION_THRESHOLD):
    """Classify how a probe's excess over the yardstick evolves across a window.

    curve, yardstick : accuracies per time step, earliest first, last = closest
                       to the anchor frame.

    Returns a dict with the numbers and a label:
      flat_at_yardstick    no signal
      flat_above_yardstick persistent signal; consistent with information that
                           exists throughout the window
      localised_rise       most of the rise happens in the final quarter; the
                           step-like profile expected when the outcome enters the
                           input near the anchor (e.g. a moving ball)
      gradual_rise         UNRESOLVED. Legitimate decision execution (a player
                           turning towards his target) and a gradual leak produce
                           this same profile; temporal shape alone cannot tell
                           them apart. step26 demonstrates this on synthetic data.
      falling              signal decreases towards the anchor

    `legacy_verdict` reproduces the rule the project used before (rise > 0.10 or
    excess > 0.12 => "leak"), so its false-positive rate can be measured.
    """
    curve = np.asarray(curve, float)
    yardstick = np.asarray(yardstick, float)
    ex = curve - yardstick
    T = len(ex)
    k = max(1, min(3, T // 4))
    head = float(ex[:k].mean())
    end = float(ex[-1])
    tail_start = int(round((1 - tail_frac) * (T - 1)))
    lo = max(0, tail_start - (k - 1))
    pre_tail = float(ex[lo:tail_start + 1].mean())
    rise = end - head
    tail_rise = end - pre_tail
    loc = tail_rise / rise if rise > rise_tol else float("nan")

    if rise > rise_tol:
        label = "localised_rise" if loc >= loc_threshold else "gradual_rise"
    elif rise < -rise_tol:
        label = "falling"
    elif float(ex.mean()) > margin:
        label = "flat_above_yardstick"
    else:
        label = "flat_at_yardstick"

    legacy = ("leak" if (curve[-1] - curve[0] > 0.10 or ex[-1] > 0.12)
              else "ok")
    return {"excess_start": head, "excess_end": end, "rise": rise,
            "tail_rise": tail_rise, "localisation": loc, "mean_excess": float(ex.mean()),
            "label": label, "legacy_verdict": legacy}


LABEL_TEXT = {
    "flat_at_yardstick": "flat at the yardstick: no signal",
    "flat_above_yardstick": "flat above the yardstick: persistent, consistent with information",
    "localised_rise": "localised rise near the anchor: consistent with leakage entering near the anchor",
    "gradual_rise": "gradual rise: UNRESOLVED (legitimate ramp and gradual leak look the same)",
    "falling": "falls towards the anchor",
}
