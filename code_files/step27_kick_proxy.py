"""Step 27 - An estimated kick proxy, and what the curves look like anchored on it.

THIS IS NOT GROUND TRUTH. No public dataset provides ball-contact times. What is
computed here is an ESTIMATED KICK PROXY: the onset of the ball's acceleration
towards its post-pass speed, found from the tracking data alone and constrained
to a frame where the ball is still close to the passer.

Definition (config.KICK_PROXY holds the constants)
  1. take the ball's speed around the emitted synchronised frame, from
     search_back_s before to search_fwd_s after, median-filtered over
     median_window frames
  2. peak   = the frame of maximum speed in that window
     pre    = the minimum speed in the frames before the peak
  3. onset  = the LAST frame at or before the peak whose speed is still below
     pre + onset_frac * (peak - pre)
  4. accept the onset only if the peak speed exceeds min_peak_speed and the ball
     is within max_ball_passer_m of the passer at the onset frame; otherwise the
     proxy is undefined for that pass and it is excluded rather than guessed

Two sources of error remain and cannot be removed from inside this dataset: the
filter in step4 attenuates the acceleration spike (so the onset is biased late),
and a ball already moving before the kick has no clean onset at all. The proxy is
therefore reported with its acceptance rate and compared with, not substituted
for, the emitted frame.

What this produces
  * work/passes_kick_proxy.csv, the synchronised pass table with a kick_proxy
    frame column and the offset from the emitted frame
  * the same probe and model curves as step17, anchored on the proxy instead, so
    the two anchorings can be compared
  * figures/paper/fig_kick_proxy.png

Run:  python step27_kick_proxy.py [--probes-only]
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
import experiment as E
import probes as PR
import provenance as P
from common import banner, load_npz

KP = C.KICK_PROXY
ANCHOR = "kick_proxy"
OUT = f"{C.FIGS}/paper"
INK, MUTED, DATA, WARN, GOOD = "#14171A", "#6E7378", "#2E6F9E", "#B3402F", "#1F7A4D"


def median_filter(x, k):
    if k <= 1:
        return x
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.median(np.stack([xp[i:i + len(x)] for i in range(k)]), axis=0)


def proxy_for_half(ball_pos, ball_vel, team_pos, frames, passer_slots):
    """Estimated kick frame per pass, plus diagnostics. NaN where undefined."""
    speed = median_filter(np.linalg.norm(ball_vel[:, 0], axis=1), KP["median_window"])
    back = int(KP["search_back_s"] * C.FRAMERATE)
    fwd = int(KP["search_fwd_s"] * C.FRAMERATE)
    out = np.full(len(frames), np.nan)
    peak_speed = np.full(len(frames), np.nan)
    gap = np.full(len(frames), np.nan)
    for i, (f, slot) in enumerate(zip(frames, passer_slots)):
        lo, hi = max(0, f - back), min(len(speed) - 1, f + fwd)
        if hi - lo < 5:
            continue
        seg = speed[lo:hi + 1]
        pk = int(np.argmax(seg))
        peak_speed[i] = seg[pk]
        if pk == 0 or seg[pk] < KP["min_peak_speed"]:
            continue
        pre = float(seg[:pk].min())
        thr = pre + KP["onset_frac"] * (seg[pk] - pre)
        below = np.where(seg[:pk + 1] <= thr)[0]
        if not len(below):
            continue
        onset = lo + int(below[-1])
        p = team_pos[onset, int(slot)]
        if np.isnan(p[0]):
            continue
        d = float(np.hypot(ball_pos[onset, 0, 0] - p[0], ball_pos[onset, 0, 1] - p[1]))
        gap[i] = d
        if d > KP["max_ball_passer_m"]:
            continue
        out[i] = onset
    return out, peak_speed, gap


def build_proxy_table(passes):
    rows = []
    for mid in C.MATCHES:
        d = load_npz(f"clean_{mid}")
        sub = passes[passes.match_id == mid]
        for half in ("firstHalf", "secondHalf"):
            s = sub[sub.half == half]
            if s.empty:
                continue
            ball_p = d[f"{half}_Ball_pos"]
            ball_v = d[f"{half}_Ball_vel"]
            for side in ("Home", "Away"):
                m = (s.team_slot == side).to_numpy()
                if not m.any():
                    continue
                ss = s[m]
                pr, pk, gap = proxy_for_half(
                    ball_p, ball_v, d[f"{half}_{side}_pos"],
                    ss.sync_frame.to_numpy(int), ss.passer_slot.to_numpy(int))
                rows.append(pd.DataFrame({
                    "index": ss.index, "kick_proxy": pr, "proxy_peak_speed": pk,
                    "proxy_ball_passer_m": gap,
                    "proxy_offset_frames": pr - ss.sync_frame.to_numpy(int),
                    "proxy_offset_s": (pr - ss.sync_frame.to_numpy(int)) / C.FRAMERATE}))
    tab = pd.concat(rows).set_index("index").sort_index()
    out = passes.join(tab)
    return out


MIN_PASSES = 30      # below this a match cannot support a probe or a model


def usable_matches(passes, match_ids):
    """Matches with enough proxy-defined passes to be worth anchoring on."""
    ok = passes[passes.kick_proxy.notna()]
    return [m for m in match_ids if (ok.match_id == m).sum() >= MIN_PASSES]


def probe_curves(anchor, lead, passes, match_ids=None):
    rows = []
    for m in (match_ids or C.TEST_MATCHES):
        sp = E.load_split([m], lead, anchor, passes=passes)
        s = PR.snapshot_from_samples(sp["X"], sp["y"], t=-1)
        r = {"anchor": anchor, "lead_s": lead, "test_match": m, "n": len(sp["y"])}
        for name in ("nearest_teammate", "ball_direction", "passer_direction"):
            r[name] = PR.run_probe(name, s)["acc"]
        rows.append(r)
    return rows


if __name__ == "__main__":
    banner("Step 27 - estimated kick proxy")
    run_id = P.canonical_run_id()
    P.print_header("kick proxy (ESTIMATE, not ground truth)", run_id,
                   definition="ball speed onset near the emitted frame",
                   constants=KP, offsets_s=C.LEAD_SWEEP_LEADS)

    passes = pd.read_csv(f"{C.WORK}/passes_synced.csv")
    tab = build_proxy_table(passes)
    ok = tab.kick_proxy.notna()
    tab.to_csv(f"{C.WORK}/passes_kick_proxy.csv", index=False)
    print(f"\nproxy defined for {ok.sum()} of {len(tab)} passes "
          f"({100 * ok.mean():.1f}%)")
    off = tab.loc[ok, "proxy_offset_s"]
    print(f"proxy minus emitted frame: median {off.median():+.3f} s, "
          f"IQR [{off.quantile(.25):+.3f}, {off.quantile(.75):+.3f}] s")
    print("A negative value means the estimated kick is EARLIER than the frame the")
    print("synchroniser emits, which is the direction the leakage argument predicts.")
    print("It is an estimate: the low-pass filter biases the onset late, and a ball")
    print("already moving before the kick has no clean onset.")
    per_match = (tab.assign(ok=ok).groupby("match_id")
                 .agg(n=("ok", "size"), defined=("ok", "sum"),
                      median_offset_s=("proxy_offset_s", "median")))
    print("\n" + per_match.round(3).to_string())

    fit_ok = usable_matches(tab, C.FIT_MATCHES)
    val_ok = usable_matches(tab, [C.VAL_MATCH])
    test_ok = usable_matches(tab, C.TEST_MATCHES)
    print(f"\nmatches with at least {MIN_PASSES} proxy-defined passes: "
          f"fit {fit_ok}, validation {val_ok}, test {test_ok}")

    rows = []
    for lead in C.LEAD_SWEEP_LEADS:
        rows += probe_curves(E.SYNC_ANCHOR, lead, passes)
        rows += probe_curves(ANCHOR, lead, tab, test_ok)
    pr = pd.DataFrame(rows)

    model_rows = []
    can_train = bool(fit_ok and val_ok and test_ok)
    if "--probes-only" in sys.argv:
        print("\n--probes-only: the proxy-anchored model stage was not run.")
    elif not can_train:
        print("\nThe proxy is defined for too few passes to train on this anchor:")
        print(f"  fit {fit_ok or 'none'}, validation {val_ok or 'none'}, "
              f"test {test_ok or 'none'} (threshold {MIN_PASSES} passes)")
        print("The model stage was NOT run, and no proxy-anchored accuracy is")
        print("reported. The probe curves above are unaffected.")
    if "--probes-only" not in sys.argv and can_train:
        for lead in C.LEAD_SWEEP_LEADS:
            for seed in C.SEEDS:
                model, stats, info = E.get_or_train(
                    lead, seed, anchor=ANCHOR, passes=tab, fit_matches=fit_ok,
                    val_match=val_ok[0], monitor_matches=test_ok, verbose=False)
                for m in test_ok:
                    te = E.load_split([m], lead, ANCHOR, passes=tab)
                    acc = float((E.evaluate(model, stats, te).argmax(1) == te["y"]).mean())
                    model_rows.append({"anchor": ANCHOR, "lead_s": lead,
                                       "test_match": m, "seed": seed,
                                       "n": len(te["y"]), "model_top1": acc})
        md = pd.DataFrame(model_rows)
        pr = pd.concat([pr, md], ignore_index=True)
        print("\nmodel top-1 anchored on the proxy (mean over seeds):")
        print(md.groupby(["lead_s", "test_match"]).model_top1.mean().round(3).to_string())
        print("Compare with work/leadsweep_summary.csv, which is anchored on the")
        print("emitted frame. The proxy sits earlier, so a given offset from it")
        print("reaches further back in the play.")

    P.stamp(pr, run_id, model_variant="kick_proxy", anchor=ANCHOR).to_csv(
        f"{C.WORK}/kick_proxy_curves.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.2))
    ax[0].hist(off, bins=40, color=DATA, alpha=0.85)
    ax[0].axvline(0, color=INK, lw=1)
    ax[0].set_xlabel("estimated kick proxy minus emitted frame (s)")
    ax[0].set_ylabel("passes")
    ax[0].set_title(f"(a) Proxy offset, {ok.sum()} of {len(tab)} passes")
    last = None
    for anc, col, mk in ((E.SYNC_ANCHOR, DATA, "-o"), (ANCHOR, WARN, "--s")):
        g = pr[(pr.anchor == anc) & pr.ball_direction.notna()]
        if g.empty:
            continue
        g = g.groupby("lead_s")[["ball_direction", "nearest_teammate"]].mean()
        ax[1].plot(g.index, g.ball_direction, mk, color=col, lw=2, ms=5,
                   label=f"ball probe, anchored on {anc}")
        last = g
    if last is not None:
        ax[1].plot(last.index, last.nearest_teammate, ":", color=MUTED, lw=1.5,
                   label="nearest teammate")
    ax[1].set_xlabel("window-end offset before the anchor (s)")
    ax[1].set_ylabel("probe accuracy")
    ax[1].set_title("(b) Ball shortcut under two anchorings")
    ax[1].legend(fontsize=9)
    for a in ax:
        a.grid(alpha=0.4)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Estimated kick proxy: an estimate from ball kinematics, "
                 "not a ground-truth contact time", fontsize=12, weight="600", y=1.03)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig_kick_proxy.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved -> {C.WORK}/passes_kick_proxy.csv, kick_proxy_curves.csv, "
          f"{OUT}/fig_kick_proxy.png")
