"""Step 11 - Self-supervised pretraining.

The binding constraint on this project is data: about 2,400 labelled passes.
But the dataset holds 1,002,644 frames, and steps 5 to 10 touch roughly 0.4% of
them. Everything else is thrown away because it has no pass label.

So we invent a label that needs no annotation: given 1.5 s of play, predict where
every player will be 1 second later. That task needs the same understanding -
who is moving where, who is covered, how the shape is shifting - so the encoder
learns it from unlimited data, and the receiver head is then fine-tuned on the
few thousand passes we actually have.

This is the standard answer to a small labelled set, and it is the reason
TacticAI leans on geometric priors: high-quality tracking data is scarce. Here we
have the frames, just not the labels.

IMPORTANT: windows are sampled from config.PRETRAIN_MATCHES only, which are the
fit matches: no test match (that would leak its movement patterns into the
encoder) and, since the canonical protocol was tightened, not the validation
match either (it selects the epoch, so nothing else should flow from it).

Window count, epochs and horizon are config constants (PRETRAIN_WINDOWS,
PRETRAIN_EPOCHS, PRETRAIN_HORIZON_S), so they are part of the canonical run id.

Run:  python step11_pretrain.py
Out:  work/pretrained_encoder.pt
"""

import numpy as np
import torch
import torch.nn as nn

import config as C
from common import banner, load_npz
from step8_model import PassReceiverNet

from device import DEVICE, PIN_MEMORY, banner as device_banner, \
    seed_everything

DEV = DEVICE

N_WINDOWS = C.PRETRAIN_WINDOWS
HORIZON_S = C.PRETRAIN_HORIZON_S
PRE_EPOCHS = C.PRETRAIN_EPOCHS


def build_windows(match_ids, n_windows, rng):
    """Sample random in-play windows and format them exactly like step 5.

    The input layout must match the fine-tuning layout or the pretrained weights
    are useless: object 0 is the on-ball player, 1-10 his team, 11-21 the
    opponents, 22 the ball.
    """
    back = int(C.WINDOW_SECONDS * C.FRAMERATE)
    ahead = int(HORIZON_S * C.FRAMERATE)
    offsets = np.arange(-back, 1, C.STRIDE)

    X, Y = [], []
    per_match = max(1, n_windows // len(match_ids))

    for mid in match_ids:
        d = load_npz(f"clean_{mid}")
        for half in ("firstHalf", "secondHalf"):
            pos = {k: d[f"{half}_{k}_pos"] for k in ("Home", "Away", "Ball")}
            vel = {k: d[f"{half}_{k}_vel"] for k in ("Home", "Away", "Ball")}
            T = pos["Home"].shape[0]
            sign = {"Home": float(d[f"{half}_sign_home"]),
                    "Away": float(d[f"{half}_sign_away"])}

            lo, hi = back + 1, T - ahead - 1
            if hi <= lo:
                continue
            frames = rng.integers(lo, hi, per_match // 2)

            for f in frames:
                ball_now = pos["Ball"][f, 0]
                if np.isnan(ball_now[0]):
                    continue

                # whichever team has a player closest to the ball is "in possession"
                best, mine = np.inf, None
                for side in ("Home", "Away"):
                    dd = np.linalg.norm(pos[side][f] - ball_now, axis=1)
                    if np.isfinite(dd).any() and np.nanmin(dd) < best:
                        best, mine = np.nanmin(dd), side
                if mine is None or best > 5.0:
                    continue
                theirs = "Away" if mine == "Home" else "Home"

                on_ball = int(np.nanargmin(
                    np.linalg.norm(pos[mine][f] - ball_now, axis=1)))
                mates = np.where(~np.isnan(pos[mine][f, :, 0]))[0]
                mates = mates[mates != on_ball]
                opps = np.where(~np.isnan(pos[theirs][f, :, 0]))[0]
                if not (C.MIN_TEAMMATES <= len(mates) <= C.N_TEAMMATES
                        and C.MIN_OPPONENTS <= len(opps) <= C.N_OPPONENTS):
                    continue

                s = sign[mine]
                w = np.zeros((len(offsets), C.N_OBJECTS, C.N_FEATURES), np.float32)
                target = np.zeros((C.N_OBJECTS, 2), np.float32)
                fr = f + offsets

                def put(idx, side, slot, flags):
                    w[:, idx, 0] = pos[side][fr, slot, 0] * s
                    w[:, idx, 1] = pos[side][fr, slot, 1] * s
                    w[:, idx, 2] = vel[side][fr, slot, 0] * s
                    w[:, idx, 3] = vel[side][fr, slot, 1] * s
                    w[:, idx, 4:8] = flags
                    w[:, idx, C.IDX_PRESENT] = 1.0
                    # displacement over the horizon, which is what we predict
                    target[idx] = (pos[side][f + ahead, slot] - pos[side][f, slot]) * s

                put(0, mine, on_ball, [0, 0, 0, 1])
                for k, sl in enumerate(mates):
                    put(1 + k, mine, sl, [1, 0, 0, 0])
                for k, sl in enumerate(opps):
                    put(1 + C.N_TEAMMATES + k, theirs, sl, [0, 1, 0, 0])
                put(C.N_OBJECTS - 1, "Ball", 0, [0, 0, 1, 0])

                if np.isnan(w).any() or np.isnan(target).any():
                    continue

                if C.USE_SPACE_FEATURES:
                    from step5_dataset import add_space_features
                    add_space_features(w)

                X.append(w)
                Y.append(target)

    return np.stack(X), np.stack(Y)


class TrajectoryHead(nn.Module):
    """The same encoder as the pass model, with a displacement head on top."""

    def __init__(self):
        super().__init__()
        self.net = PassReceiverNet()
        self.out = nn.Sequential(
            nn.Linear(C.HIDDEN, C.HIDDEN), nn.ReLU(), nn.Linear(C.HIDDEN, 2)
        )

    def forward(self, x):
        n = self.net
        B, T, P, F = x.shape
        present = x[:, -1, :, C.IDX_PRESENT] > 0.5

        h = x.permute(0, 2, 1, 3).reshape(B * P, T, F)
        _, h = n.gru(h)
        h = h.squeeze(0).reshape(B, P, -1)

        pad = torch.zeros(B, 1, P, device=x.device, dtype=h.dtype)
        pad = pad.masked_fill(~present[:, None, :], -1e9)
        pad = pad.expand(B, P, P).repeat_interleave(n.n_heads, dim=0)
        bias = pad + n.edge_bias(x) if n.edge_attention else pad
        a, _ = n.attn(h, h, h, attn_mask=bias)
        h = n.norm(h + a)
        return self.out(h), present


if __name__ == "__main__":
    banner("Step 11 - self-supervised pretraining")
    rng = np.random.default_rng(0)

    print(f"Sampling windows from PRETRAIN_MATCHES only: {C.PRETRAIN_MATCHES}")
    print(f"(no test match, and not the validation match {C.VAL_MATCH})")
    print(f"windows {N_WINDOWS}, epochs {PRE_EPOCHS}, horizon {HORIZON_S} s\n")

    X, Y = build_windows(C.PRETRAIN_MATCHES, N_WINDOWS, rng)
    print(f"windows: {len(X)}   input {X.shape[1:]}   target {Y.shape[1:]}")
    print("(for comparison, see work/dataset_counts.csv for the labelled passes)\n")

    mu = X[:, :, :, :4].mean(axis=(0, 1, 2), keepdims=True)
    sd = X[:, :, :, :4].std(axis=(0, 1, 2), keepdims=True) + 1e-6
    X[:, :, :, :4] = (X[:, :, :, :4] - mu) / sd

    device_banner()
    print()
    seed_everything(0)
    model = TrajectoryHead().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    Xt = torch.tensor(X)
    Yt = torch.tensor(Y)
    n_val = len(Xt) // 10
    Xtr, Ytr = Xt[n_val:], Yt[n_val:]
    Xva, Yva = Xt[:n_val].to(DEVICE), Yt[:n_val].to(DEVICE)

    from device import batch_size
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr, Ytr),
        batch_size=batch_size() * 2, shuffle=True, pin_memory=PIN_MEMORY)

    for ep in range(PRE_EPOCHS):
        model.train()
        tot = 0.0
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=PIN_MEMORY)
            yb = yb.to(DEVICE, non_blocking=PIN_MEMORY)
            opt.zero_grad()
            pred, pres = model(xb)
            loss = (((pred - yb) ** 2).sum(-1) * pres).sum() / pres.sum()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            pv, pp = model(Xva)
            err = ((pv - Yva).norm(dim=-1) * pp).sum() / pp.sum()
            # constant-velocity is the baseline any trajectory model must beat
            cv = Xva[:, -1, :, 2:4] * HORIZON_S
            cv_err = ((cv - Yva).norm(dim=-1) * pp).sum() / pp.sum()
        print(f"  epoch {ep + 1}/{PRE_EPOCHS}  train {tot / len(Xtr):7.3f}   "
              f"val mean error {err.item():.3f} m   "
              f"(constant velocity: {cv_err.item():.3f} m)")

    torch.save(model.net.state_dict(), f"{C.WORK}/pretrained_encoder.pt")
    import provenance as P
    P.write_json(f"{C.WORK}/pretrained_encoder.json", {
        "timestamp_utc": P.now_utc(), "pretrain_matches": C.PRETRAIN_MATCHES,
        "windows": int(len(X)), "epochs": PRE_EPOCHS, "horizon_s": HORIZON_S,
        "sha": P.hash_file(f"{C.WORK}/pretrained_encoder.pt", n=16),
        "quick_mode": C.QUICK_MODE})
    print(f"\nSaved -> {C.WORK}/pretrained_encoder.pt (+ .json)")
    print("step9_train.py loads it when USE_PRETRAIN is True in config.py.")
    print("\nNote: the encoder is what transfers. The displacement head is thrown")
    print("away. If the val error is not clearly below constant velocity, the")
    print("encoder has learnt little worth transferring.")
