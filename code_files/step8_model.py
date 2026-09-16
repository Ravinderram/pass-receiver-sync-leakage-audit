"""Step 8 - The model.

Three stages:

  1. Time encoder   The same small GRU runs over every object's 13 time steps.
                    Shared weights, so there are no per-player parameters and
                    the model cannot learn anything about "player number 4".

  2. Context layer  One self-attention layer over the 23 objects, so each
                    player's vector is updated using everyone else. This is
                    where "he is open because the defender went the other way"
                    can be represented. A graph network would do the same job;
                    attention over a fully connected graph is the same idea
                    with less code.

  3. Scoring head   One number per teammate, softmax over the ten.

The key property is permutation invariance: the order of players in the array
is arbitrary, so shuffling it must not change the answer. Stages 1 and 2 both
treat objects as a set, so this holds by construction. A plain fully connected
network on a flattened array would not have it and would waste capacity
learning that the order does not matter.

About 35k parameters, deliberately small - there are only ~2,400 training
passes, so a bigger model would just memorise them.
"""

import torch
import torch.nn as nn

import config as C


class PassReceiverNet(nn.Module):
    def __init__(self, n_features=C.N_FEATURES, hidden=C.HIDDEN, n_heads=4,
                 edge_attention=None):
        super().__init__()
        self.n_heads = n_heads
        self.edge_attention = (C.USE_EDGE_ATTENTION if edge_attention is None
                               else edge_attention)
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        # turns the geometry of a PAIR of objects into a per-head attention bias
        self.edge = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, n_heads)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def edge_bias(self, x):
        """(B*heads, 23, 23) bias built from relative position and distance.

        Edge input is dx, dy and distance, all relative - so it is translation
        invariant, and it flips sign consistently under the mirror augmentation.
        """
        pos = x[:, -1, :, :2]                                # (B, 23, 2)
        d = pos[:, :, None, :] - pos[:, None, :, :]          # (B, 23, 23, 2)
        dist = d.norm(dim=-1, keepdim=True)
        e = torch.cat([d / 50.0, dist / 50.0], dim=-1)       # scaled to ~O(1)
        b = self.edge(e)                                     # (B, 23, 23, heads)
        B, P, _, H = b.shape
        return b.permute(0, 3, 1, 2).reshape(B * H, P, P)

    def forward(self, x):
        """x: (B, T, 23, F) -> logits (B, 10) over the teammates.

        Slots can be empty: after a red card a team has one player fewer, and we
        pad that slot rather than throw the pass away. Feature IDX_PRESENT says
        which slots are real. Padded slots must be invisible in two places - the
        attention, so they do not pollute anyone else's context, and the softmax,
        so the model can never predict a player who is not on the pitch.
        """
        B, T, P, F = x.shape
        present = x[:, -1, :, C.IDX_PRESENT] > 0.5      # (B, 23)

        # every object through the same GRU
        h = x.permute(0, 2, 1, 3).reshape(B * P, T, F)
        _, h = self.gru(h)                     # (1, B*P, hidden)
        h = h.squeeze(0).reshape(B, P, -1)     # (B, 23, hidden)

        # let objects see each other, but not the padded ones.
        # torch deprecates mixing a bool key_padding_mask with a float
        # attn_mask, so we fold the padding into the float bias instead.
        pad = torch.zeros(B, 1, P, device=x.device, dtype=h.dtype)
        pad = pad.masked_fill(~present[:, None, :], -1e9)
        pad = pad.expand(B, P, P).repeat_interleave(self.n_heads, dim=0)
        bias = pad + self.edge_bias(x) if self.edge_attention else pad
        a, _ = self.attn(h, h, h, attn_mask=bias)
        h = self.norm(h + a)

        scores = self.head(h).squeeze(-1)                  # (B, 23)
        logits = scores[:, 1:1 + C.N_TEAMMATES]
        mate_present = present[:, 1:1 + C.N_TEAMMATES]
        # a large negative number rather than -inf, which would make softmax NaN
        # if a whole row were ever masked
        return logits.masked_fill(~mate_present, -1e9)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from device import DEVICE, banner as device_banner
    device_banner()
    print()

    m = PassReceiverNet().to(DEVICE)
    print(f"edge attention: {m.edge_attention}")
    x = torch.randn(4, C.N_STEPS, C.N_OBJECTS, C.N_FEATURES, device=DEVICE)
    x[:, :, :, C.IDX_PRESENT] = 1.0
    x[2, :, 5, C.IDX_PRESENT] = 0.0      # pretend sample 2 is a man down
    out = m(x)
    print(f"masked teammate logit: {out[2, 4].item():.0f}  (must be about -1e9)")
    print(f"input  {tuple(x.shape)}")
    print(f"output {tuple(out.shape)}   (should be (4, {C.N_TEAMMATES}))")
    print(f"parameters: {count_params(m):,}")

    # permutation invariance check: shuffling the teammates must permute the
    # outputs the same way, not change them. If this fails, the model is wrong.
    x[:, :, :, C.IDX_PRESENT] = 1.0      # full lineup for the permutation test
    perm = torch.randperm(C.N_TEAMMATES, device=DEVICE)
    x2 = x.clone()
    x2[:, :, 1:1 + C.N_TEAMMATES] = x[:, :, 1:1 + C.N_TEAMMATES][:, :, perm]
    m.eval()
    with torch.no_grad():
        o1, o2 = m(x), m(x2)
    diff = (o1[:, perm] - o2).abs().max().item()
    print(f"\npermutation check: max difference = {diff:.2e}")
    print("PASS" if diff < 1e-4 else "FAIL - the model depends on player order")
