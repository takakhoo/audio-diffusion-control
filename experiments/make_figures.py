"""Draw the README's diagrams and graphs from saved runs.

    python experiments/make_figures.py --out results/figures

architecture.png      where the slider sits inside the transformer
training_curves.png   how much of the guidance target each slider has learned, by iteration
gating.png            what switching the slider on later buys and costs
discovery.png         variance carried by the leading discovered axes of each concept
quality_reference.png enjoyment scores of unsteered model output against real recordings
coverage.png          how much of each real-music axis the model's output spans
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SURFACE, INK, MUTED, GRID, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e4de", "#c9c6bd"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
FROZEN_FILL, TRAIN_FILL = "#eaf1fb", "#fdeee6"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2, "lines.markersize": 5,
        "font.family": "DejaVu Sans", "figure.dpi": 150,
    })


def box(ax, x, y, w, h, title, body="", edge=LINE, fill="#ffffff", size=10, weight="bold"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", linewidth=1.5,
                                edgecolor=edge, facecolor=fill))
    ty = y + h / 2 + (0.16 if body else 0)
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=size, fontweight=weight, color=INK)
    if body:
        ax.text(x + w / 2, ty - 0.33, body, ha="center", va="center", fontsize=8.5, color=MUTED)


def arrow(ax, a, b, color=MUTED, style="-|>"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12, linewidth=1.3, color=color,
                                 shrinkA=1, shrinkB=1))


def architecture(out):
    fig, ax = plt.subplots(figsize=(11.5, 6.4), dpi=150, facecolor=SURFACE)
    ax.set_xlim(0, 11.5); ax.set_ylim(0, 6.4); ax.axis("off")
    ax.text(0, 6.25, "Where the slider lives", fontsize=14, fontweight="bold", va="top")
    ax.text(0, 5.85, "Blue is frozen. Orange is the slider: the only trained weights, and the only thing that changes when you drag.",
            fontsize=9.5, color=MUTED, va="top")

    # main path
    box(ax, 0.0, 4.0, 1.5, 0.9, "Prompt", "\"jazz piano trio\"")
    box(ax, 0.0, 2.3, 1.5, 0.9, "Noise", "seed", )
    box(ax, 2.0, 4.0, 1.7, 0.9, "Text encoder", "frozen", BLUE, FROZEN_FILL)
    arrow(ax, (1.5, 4.45), (2.0, 4.45))
    ax.add_patch(FancyBboxPatch((4.2, 1.2), 3.5, 4.0, boxstyle="round,pad=0.02,rounding_size=0.12", linewidth=1.5,
                                edgecolor=BLUE, facecolor="#f6f9fe"))
    ax.text(5.95, 5.02, "Transformer block  × 24 to 32", ha="center", fontsize=10, fontweight="bold")
    for k, (name, note) in enumerate([("Self-attention", "q, k, v, out"), ("Cross-attention", "q, k, v, out"),
                                      ("Feed-forward", "2 to 3 linears")]):
        y = 3.75 - k * 1.1
        box(ax, 4.5, y, 2.9, 0.85, name, note, BLUE, FROZEN_FILL, size=9.5)
        ax.add_patch(FancyBboxPatch((6.95, y + 0.52), 0.36, 0.24, boxstyle="round,pad=0.01,rounding_size=0.05",
                                    linewidth=1.2, edgecolor=ORANGE, facecolor=TRAIN_FILL))
        ax.text(7.13, y + 0.64, "BA", ha="center", va="center", fontsize=7, color=INK)
        if k:
            arrow(ax, (5.95, y + 1.1), (5.95, y + 0.85))
    arrow(ax, (3.7, 4.45), (4.5, 3.05), MUTED)
    ax.text(3.55, 3.45, "condition", fontsize=8, color=MUTED, ha="right")
    arrow(ax, (1.5, 2.75), (4.2, 2.75))
    ax.text(2.85, 2.86, "latent $z_t$, time $t$", ha="center", fontsize=8.5, color=MUTED)
    box(ax, 8.2, 2.3, 1.35, 0.9, "Sampler", "8 or 50 steps", BLUE, FROZEN_FILL)
    arrow(ax, (7.7, 2.75), (8.2, 2.75))
    box(ax, 10.0, 2.3, 1.5, 0.9, "Decoder", "to stereo audio", BLUE, FROZEN_FILL)
    arrow(ax, (9.55, 2.75), (10.0, 2.75))
    ax.add_patch(FancyArrowPatch((8.87, 2.3), (5.95, 1.2), connectionstyle="arc3,rad=-0.35", arrowstyle="-|>",
                                 mutation_scale=12, linewidth=1.3, color=MUTED, linestyle=(0, (4, 3))))
    ax.text(8.2, 1.25, "next, less noisy $z_t$", fontsize=8.5, color=MUTED)

    # slider control
    ax.plot([8.2, 10.6], [5.0, 5.0], color=LINE, linewidth=6, solid_capstyle="round")
    ax.scatter([9.8], [5.0], s=220, color=ORANGE, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.text(8.1, 5.0, "sad", ha="right", va="center", fontsize=9.5)
    ax.text(10.7, 5.0, "happy", ha="left", va="center", fontsize=9.5)
    ax.text(9.4, 5.38, "slider position $s$", ha="center", fontsize=9.5, fontweight="bold")
    ax.add_patch(FancyArrowPatch((9.4, 4.82), (7.36, 4.4), connectionstyle="arc3,rad=-0.15", arrowstyle="-|>",
                                 mutation_scale=12, linewidth=1.4, color=ORANGE))
    ax.text(8.75, 4.33, "one number,\nread by every BA", fontsize=8.5, color=MUTED, ha="center", va="top")

    # inset: one linear layer
    ax.add_patch(FancyBboxPatch((0.0, 0.0), 6.9, 0.85, boxstyle="round,pad=0.02,rounding_size=0.1", linewidth=1.2,
                                edgecolor=LINE, facecolor="#ffffff"))
    ax.text(4.0, 0.72, "Inside every linear layer", fontsize=9.5, fontweight="bold", va="center")
    ax.text(0.15, 0.36, "$x$", fontsize=12, va="center")
    box(ax, 0.75, 0.42, 1.2, 0.34, "W  (frozen)", edge=BLUE, fill=FROZEN_FILL, size=8.5, weight="normal")
    box(ax, 0.75, 0.06, 0.55, 0.3, "A", edge=ORANGE, fill=TRAIN_FILL, size=8.5, weight="normal")
    box(ax, 1.4, 0.06, 0.55, 0.3, "B", edge=ORANGE, fill=TRAIN_FILL, size=8.5, weight="normal")
    arrow(ax, (0.38, 0.4), (0.75, 0.59)); arrow(ax, (0.38, 0.32), (0.75, 0.21))
    ax.text(2.1, 0.21, r"$\times\; s \cdot \alpha / r$", fontsize=10, va="center", color=INK)
    arrow(ax, (1.95, 0.59), (3.25, 0.45)); arrow(ax, (2.95, 0.21), (3.25, 0.37))
    ax.text(3.35, 0.4, "+", fontsize=14, va="center", fontweight="bold")
    arrow(ax, (3.58, 0.4), (3.9, 0.4))
    ax.text(4.0, 0.32, "$y = Wx + s\\,\\frac{\\alpha}{r}\\,BAx$", fontsize=11, va="center")
    ax.text(0.15, -0.28, "rank $r$ = 4.   ACE-Step: 352 layers, 10.0 M slider weights on a 4.17 B model (0.24%).   "
            "Stable Audio Open: 240 layers, 4.1 M on 1.06 B (0.39%).", fontsize=8.5, color=MUTED, va="center")
    fig.savefig(out / "architecture.png", bbox_inches="tight")
    plt.close(fig)


def training_curves(out, runs):
    sets = [(name, sorted(Path(d).glob("*.json"))) for name, d in runs if Path(d).exists()]
    sets = [(n, f) for n, f in sets if f]
    if not sets:
        return
    fig, axes = plt.subplots(1, len(sets), figsize=(5.2 * len(sets), 3.4), squeeze=False, sharey=True)
    for ax, (name, files) in zip(axes[0], sets):
        curves = []
        for f in files:
            h = json.loads(f.read_text())["history"]
            if not h or "baseline" not in h[0]:
                continue
            loss, base = np.array([r["loss"] for r in h]), np.array([r["baseline"] for r in h])
            win = 50
            ratio = np.convolve(loss, np.ones(win), "valid") / np.convolve(base, np.ones(win), "valid")
            x = np.arange(win - 1, len(loss))
            ax.plot(x, ratio, color=BLUE, alpha=0.22, linewidth=1)
            curves.append(ratio)
        mean = np.mean(curves, 0)
        ax.plot(x, mean, color=BLUE, linewidth=2.4)
        ax.annotate(f"mean of {len(curves)} sliders: {mean[-1]:.2f}", (x[-1], mean[-1]), xytext=(-8, 14),
                    textcoords="offset points", ha="right", fontsize=9, color=INK)
        ax.set_title(name)
        ax.set_xlabel("training iteration")
        ax.set_ylim(0, 1.1)
    axes[0, 0].set_ylabel("share of the target shift\nnot yet reproduced")
    fig.suptitle("How much of its guidance target each slider has learned", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "training_curves.png", bbox_inches="tight")
    plt.close(fig)


def gating(out, table):
    import csv

    if not Path(table).exists():
        return
    rows = list(csv.DictReader(open(table)))
    sliders = list(dict.fromkeys(r["slider"] for r in rows))
    colors = dict(zip(sliders, [BLUE, ORANGE, AQUA, YELLOW]))
    panels = [("range_in_std", "descriptor moved between -1 and +1\n(std of unsteered clips)"),
              ("clap_keep", "CLAP similarity to the\nunsteered clip at ±1"),
              ("ce_at_ends", "content enjoyment at ±1")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, (key, label) in zip(axes, panels):
        for s in sliders:
            pts = sorted(((int(r["skipped_steps"]), float(r[key])) for r in rows if r["slider"] == s))
            ax.plot(*zip(*pts), marker="o", color=colors[s], label=s)
        if key == "ce_at_ends":
            zero = float(rows[0]["ce_at_zero"])
            ax.axhline(zero, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
            ax.annotate("unsteered", (0, zero), xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=MUTED)
        ax.set_xlabel("sampling steps before the slider turns on (of 50)")
        ax.set_ylabel(label)
        ax.set_xlim(-1, 36)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Turning the slider on later keeps the piece and its quality, and costs range", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(out / "gating.png", bbox_inches="tight")
    plt.close(fig)


def discovery(out, pca_json):
    if not Path(pca_json).exists():
        return
    pca = json.loads(Path(pca_json).read_text())
    fig, axes = plt.subplots(1, len(pca), figsize=(3.1 * len(pca), 3.6), sharey=True)
    for ax, (concept, info) in zip(axes, pca.items()):
        comps = info["components"][:6]
        share = [100 * c["variance_share"] for c in comps]
        ax.bar(range(1, len(share) + 1), share, color=BLUE, width=0.62)
        ax.set_title(concept)
        ax.set_xlabel("component")
        ax.set_xticks(range(1, len(share) + 1))
        ax.grid(axis="x", visible=False)
        top = comps[0]
        ax.text(0.98, 0.97, f"axis 1:\n{', '.join(top['toward'][:2])}\nvs\n{', '.join(top['away'][:2])}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=MUTED, linespacing=1.3)
    axes[0].set_ylabel("share of variance within the concept (%)")
    fig.suptitle("A few axes carry much of how clips of one concept differ (ACE-Step, 1,024 clips each)", x=0.01,
                 ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "discovery.png", bbox_inches="tight")
    plt.close(fig)


def quality_reference(out, sources):
    data = []
    for label, path, color in sources:
        files = sorted(Path(path).glob("rows*.jsonl")) if Path(path).is_dir() else []
        vals = [json.loads(line).get("ce") for f in files for line in f.read_text().splitlines() if line]
        vals = [v for v in vals if v is not None]
        if vals:
            data.append((label, np.array(vals), color))
    if len(data) < 2:
        return
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bins = np.linspace(1, 9.5, 46)
    for label, vals, color in data:
        hist, edges = np.histogram(vals, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, hist, color=color, label=f"{label} (mean {vals.mean():.2f}, n = {len(vals):,})")
        ax.fill_between(centers, hist, color=color, alpha=0.1, linewidth=0)
    ax.set_xlabel("Audiobox Aesthetics content enjoyment (1 to 10)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title("Unsteered model output against real recordings", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "quality_reference.png", bbox_inches="tight")
    plt.close(fig)


def coverage(out, table, model="ace"):
    if not Path(table).exists():
        return
    rep = json.loads(Path(table).read_text())
    m = rep["models"][model]
    order = np.argsort(m["total"])
    fig, ax = plt.subplots(figsize=(9.2, 0.42 * len(order) + 2.0))
    for row, i in enumerate(order):
        a = rep["axes"][i]
        ax.plot([m["within"][i], m["total"][i]], [row, row], color=LINE, linewidth=2, zorder=1)
        ax.scatter(m["total"][i], row, s=64, color=BLUE, zorder=2, edgecolor=SURFACE, linewidth=2,
                   label="all 648 prompts" if row == 0 else None)
        ax.scatter(m["within"][i], row, s=64, color=ORANGE, zorder=2, edgecolor=SURFACE, linewidth=2,
                   label="one prompt, many seeds (real music: one genre)" if row == 0 else None)
        ax.text(-0.03, row, f"{', '.join(a['toward'][:2])}  /  {', '.join(a['away'][:2])}", ha="right", va="center",
                fontsize=8.5, color=INK, transform=ax.get_yaxis_transform())
    ax.axvline(1, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(0.985, len(order) + 0.6, "real music", ha="right", va="center", fontsize=8.5, color=MUTED)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.7, len(order) + 1.1)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("spread of generated clips along the axis, as a fraction of the spread of real recordings")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.suptitle("The model explores less of each musical axis than real music does", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.text(0.01, 0.905, f"Axes: independent components of MuQ-MuLan embeddings of {rep['real_clips']:,} FMA recordings. "
             f"Model: ACE-Step 1.5 XL turbo, {m['clips']:,} clips.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0.2, 0, 1, 0.9))
    fig.savefig(out / "coverage.png", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--only", default=None, help="draw a single figure, by name")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    if args.only in (None, "coverage"):
        coverage(out, "results/coverage/muq_ica/coverage.json")
    if args.only == "coverage":
        return
    architecture(out)
    training_curves(out, [("ACE-Step 1.5 XL turbo", "runs/sliders/ace"), ("Stable Audio Open 1.0", "runs/sliders/v1")])
    gating(out, "results/gating/summary.csv")
    discovery(out, "results/discovery/ace/pca.json")
    quality_reference(out, [("Real recordings (FMA)", "runs/reference/fma", MUTED),
                            ("ACE-Step 1.5 XL turbo", "runs/corpus/ace_train", BLUE),
                            ("Stable Audio Open 1.0", "runs/corpus/train", ORANGE)])
    print("wrote", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
