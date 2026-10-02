"""Draw the two-panel overview figure used in the README: how a slider acts and how it is trained."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SURFACE, INK, MUTED, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#c9c6bd"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def box(ax, x, y, w, h, title, body="", color=LINE, fill="#ffffff"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", linewidth=1.6,
                                edgecolor=color, facecolor=fill))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=10.5, fontweight="bold", color=INK)
    if body:
        drop = 0.62 + 0.27 * title.count("\n")
        ax.text(x + w / 2, y + h - drop, body, ha="center", va="top", fontsize=9, color=MUTED, linespacing=1.35)


def arrow(ax, a, b, color=MUTED):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, linewidth=1.4, color=color,
                                 shrinkA=2, shrinkB=2))


plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 7.2), dpi=150, facecolor=SURFACE,
                                  gridspec_kw=dict(height_ratios=[1, 1.25], hspace=0.08))
for ax in (top, bottom):
    ax.set_xlim(0, 11)
    ax.axis("off")
top.set_ylim(0, 3)
bottom.set_ylim(0, 3.8)

top.text(0, 2.85, "Using a slider", fontsize=13, fontweight="bold", color=INK, va="top")
box(top, 0.0, 0.55, 1.9, 1.5, "Prompt + seed", "\"mellow jazz\npiano trio\"")
box(top, 2.6, 0.55, 3.0, 1.5, "Frozen music model", "every linear layer computes\n" + r"$Wx + s\cdot\frac{\alpha}{r}BAx$",
    color=BLUE)
box(top, 6.3, 0.55, 1.9, 1.5, "Same piece,\nmoved along\none axis")
box(top, 8.9, 0.55, 2.1, 1.5, "Measured", "descriptor follows?\nstill music?\nsame piece?", color=AQUA)
arrow(top, (1.9, 1.3), (2.6, 1.3))
arrow(top, (5.6, 1.3), (6.3, 1.3))
arrow(top, (8.2, 1.3), (8.9, 1.3))
top.plot([3.1, 5.1], [0.18, 0.18], color=LINE, linewidth=5, solid_capstyle="round")
top.scatter([4.6], [0.18], s=170, color=ORANGE, zorder=3, edgecolor=SURFACE, linewidth=1.5)
top.text(2.95, 0.18, "sad", ha="right", va="center", fontsize=9, color=INK)
top.text(5.25, 0.18, "happy     slider position $s$", ha="left", va="center", fontsize=9, color=INK)
arrow(top, (4.1, 0.3), (4.1, 0.55), ORANGE)

bottom.text(0, 3.7, "Two ways to train one", fontsize=13, fontweight="bold", color=INK, va="top")
box(bottom, 0.0, 1.75, 5.2, 1.45, "From a prompt pair",
    "slider at ±1 learns the frozen model's prediction\nshifted by the difference between\n\"prompt, happy\" and \"prompt, sad\"",
    color=BLUE)
box(bottom, 5.8, 1.75, 5.2, 1.45, "From two sets of clips, no text",
    "slider at +1 denoises the \"high\" set, at -1 the \"low\" set;\none update with opposite signs, so shared\ncontent cancels and the difference remains",
    color=ORANGE)
box(bottom, 5.8, 0.0, 1.55, 1.45, "Sorted by a\nmeasurement", "centroid, onsets,\nkey clarity...")
box(bottom, 7.62, 0.0, 1.55, 1.45, "Sorted by\nquality", "aesthetics\nscore")
box(bottom, 9.45, 0.0, 1.55, 1.45, "Sorted along a\ndiscovered axis", "PCA of CLAP\nper concept")
for x in (6.57, 8.4, 10.22):
    arrow(bottom, (x, 1.45), (x, 1.75), ORANGE)
box(bottom, 0.0, 0.0, 5.2, 1.45, "The model's own output is the training data",
    "3,072 clips from 48 prompts,\nplus 1,024 clips for each of five concepts")
arrow(bottom, (5.2, 0.72), (5.8, 0.72))

out = Path("results/figures/pipeline.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
print("wrote", out)
