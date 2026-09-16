"""Step 19 - The architecture diagram.

Review feedback: there were thirteen figures about the data and the results and
not one about the model. The reviewer asked what the context layer actually is,
and how each object is represented. Both answers existed in the code and in no
picture, which is a presentation failure rather than a modelling one.

Two questions this figure is built to answer without being asked:

  What is the context layer?  A single transformer encoder block -
  nn.MultiheadAttention, four heads, one layer - over 23 tokens, plus a learned
  bias computed from each pair's relative geometry.

  How are the ball and the players represented?  Identically. They pass through
  the SAME GRU and the SAME attention, separated only by a one-hot role flag in
  the input. That is a deliberate choice (shared encoder plus type flag) and it
  has an obvious alternative - a separate encoder per object type - which was
  not tried. The figure says so rather than hiding it.

Run:  python step19_architecture.py
Out:  figures/paper/fig14_architecture.png and .pdf
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from common import banner, load_npz
from step8_model import PassReceiverNet, count_params


def _n_training_samples():
    """Fit-match samples, counted from the arrays rather than remembered."""
    n = 0
    for m in C.FIT_MATCHES:
        try:
            n += len(load_npz(f"dataset_{m}")["X"])
        except FileNotFoundError:
            return 0
    return n

OUT = f"{C.FIGS}/paper"
os.makedirs(OUT, exist_ok=True)

INK, MUTED, RULE = "#14171A", "#6E7378", "#D8DAD5"
DATA, WARN, GOOD, ALT = "#2E6F9E", "#B3402F", "#1F7A4D", "#C98A2B"


def box(ax, x, y, w, h, title, lines, colour, fill):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=fill, ec=colour, lw=1.7, zorder=2))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top",
            fontsize=12, weight="bold", color=colour, zorder=3)
    ax.text(x + w / 2, y + h - 0.62, "\n".join(lines), ha="center", va="top",
            fontsize=9.2, color="#3C4A59", zorder=3, linespacing=1.65)


def arrow(ax, x1, x2, y, label):
    ax.annotate("", (x2, y), (x1, y),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))
    ax.text((x1 + x2) / 2, y + 0.14, label, ha="center", fontsize=8.6,
            color=INK, family="monospace")


def build():
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    # ---------------------------------------------------------------- input
    ax.add_patch(plt.Rectangle((0.15, 2.1), 2.55, 2.45, fc="#EAF1F7",
                               ec=DATA, lw=1.7, zorder=2))
    ax.text(1.42, 4.33, "Input", ha="center", va="top", fontsize=12,
            weight="bold", color=DATA, zorder=3)
    ax.text(1.42, 3.95,
            f"{C.N_OBJECTS} objects × {C.N_STEPS} steps\n"
            f"× {C.N_FEATURES} features",
            ha="center", va="top", fontsize=9.2, color="#3C4A59",
            zorder=3, linespacing=1.6)
    for i, (lab, col) in enumerate([
            ("passer     ×  1", ALT), ("teammates  × 10", DATA),
            ("opponents  × 11", WARN), ("ball       ×  1", INK)]):
        ax.text(1.42, 3.36 - i * 0.235, lab, ha="center", va="center",
                fontsize=8.2, color=col, weight="bold", family="monospace",
                zorder=3)
    ax.text(1.42, 2.22, "told apart only by\na one-hot role flag",
            ha="center", va="bottom", fontsize=7.6, color=MUTED,
            style="italic", linespacing=1.5, zorder=3)

    arrow(ax, 2.72, 3.25, 3.3, "per object")

    # ------------------------------------------------------------- stage 1
    box(ax, 3.25, 2.1, 2.5, 2.45, "1  Time encoder",
        ["one shared GRU", "hidden = 64", "", "the SAME weights run",
         "over every object, so", "no per-player", "parameters exist"],
        GOOD, "#EAF4EE")

    arrow(ax, 5.75, 6.6, 3.3, f"(B, {C.N_OBJECTS}, 64)")

    # ------------------------------------------------------------- stage 2
    box(ax, 6.6, 2.1, 2.95, 2.45, "2  Context layer",
        ["a single TRANSFORMER", "encoder block", "nn.MultiheadAttention",
         "4 heads, 1 layer", "", "+ learned bias from each",
         "pair's dx, dy, distance"], WARN, "#FBEDEA")

    arrow(ax, 9.55, 10.4, 3.3, f"(B, {C.N_OBJECTS}, 64)")

    # ------------------------------------------------------------- stage 3
    box(ax, 10.4, 2.1, 2.45, 2.45, "3  Scoring head",
        ["MLP → one score", "per object", "", "keep the 10 teammates",
         "mask absent players", "softmax"], DATA, "#EAF1F7")
    ax.text(11.62, 1.82, "P(receiver = k)", ha="center", fontsize=10.5,
            weight="bold", color=INK)

    # -------------------------------------------------------- what it buys
    ax.add_patch(plt.Rectangle((0.15, 0.25), 12.7, 1.4, fc="#F6F8FA",
                               ec=RULE, lw=1))
    ax.text(0.45, 1.36, "Two properties, and one choice left untested",
            fontsize=10.5, weight="bold", color=INK)
    ax.text(0.45, 1.0,
            "Permutation invariance — stages 1 and 2 treat the objects as a SET, "
            "so shuffling the player order cannot change the answer. Asserted by a test.",
            fontsize=9.3, color="#3C4A59")
    ax.text(0.45, 0.62,
            "Shared representation — the ball goes through the same encoder as the "
            "players, separated only by its role flag. A per-type encoder is the "
            "obvious alternative and was not tried.",
            fontsize=9.3, color="#3C4A59")

    ax.set_title(f"Architecture — {count_params(PassReceiverNet()):,} parameters "
                 f"for {_n_training_samples():,} training samples",
                 fontsize=13.5, weight="600", loc="left", x=0.012, y=0.96)

    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig14_architecture.{ext}", bbox_inches="tight",
                    facecolor="white", dpi=300)
    plt.close(fig)
    print(f"  {OUT}/fig14_architecture.png  +  .pdf")


if __name__ == "__main__":
    banner("Step 19 - architecture diagram")
    build()
    print("\nIt names the context layer as a transformer encoder block, and says")
    print("plainly that ball and players share one encoder — the two things the")
    print("review asked about and no figure answered.")
