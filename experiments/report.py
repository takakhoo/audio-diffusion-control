"""Turn evaluation runs into tables and figures.

    python experiments/report.py --eval runs/eval/main --out results/main

Reads <eval>/<method>_<slider>/rows.jsonl for every method and slider it finds and writes
summary.csv, summary.md, summary.json, leakage.csv, and PNG figures.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from audiosliders import metrics as M

# Categorical palette in fixed order: blue, orange, aqua, yellow, magenta.
COLORS = dict(lora="#2a78d6", guidance="#eb6834", embed="#1baf7a", dsp="#eda100", contrast="#e87ba4", caa="#008300",
              community="#4a3aa7")
LABELS = dict(lora="LoRA slider", guidance="Prompt-pair guidance", embed="Prompt interpolation",
              dsp="Signal processing", contrast="Descriptor slider", caa="Activation steering",
              community="Community slider")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e4de"
LEAK_KEYS = ["centroid_oct", "rolloff_oct", "bass_ratio", "flatness", "flux", "rms_db", "onset_rate",
             "pulse_bpm", "percussive_ratio", "decay_s", "side_ratio", "majorness"]
SHORT = dict(centroid_oct="centroid", rolloff_oct="rolloff", bass_ratio="bass", flatness="flatness", flux="flux",
             rms_db="loudness", onset_rate="onsets", pulse_bpm="tempo", percussive_ratio="percussive",
             decay_s="decay", side_ratio="width", majorness="major")


def style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2, "lines.markersize": 5,
        "font.family": "DejaVu Sans", "figure.dpi": 150,
    })


def legend(fig, axes, y):
    """One legend for the figure, collecting every method that appears in any panel."""
    seen = {}
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    order = [LABELS[m] for m in COLORS if LABELS[m] in seen]
    fig.legend([seen[l] for l in order], order, loc="lower center", ncol=len(order), frameon=False, bbox_to_anchor=(0.5, y))


def load(root: Path):
    runs = {}
    for d in sorted(root.iterdir()):
        if (d / "rows.jsonl").exists() and "_" in d.name:
            method, _, slider = d.name.partition("_")
            runs[(method, slider)] = M.load_rows(d)
    return runs


def response_figure(runs, spec, out, plt, file="response.png"):
    sliders = [s for s in spec if spec[s].get("measure") and any(k[1] == s for k in runs)]
    cols = min(4, len(sliders))
    rows_n = int(np.ceil(len(sliders) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.3 * cols, 2.7 * rows_n), squeeze=False)
    for ax, name in zip(axes.flat, sliders):
        key, sign = spec[name]["measure"], spec[name].get("measure_sign", 1)
        for method in COLORS:
            if (method, name) not in runs:
                continue
            rows = runs[(method, name)]
            std = M.natural_std(rows, key)
            r = M.response(rows, key)
            y, ci = sign * r["mean"] / std, r["ci"] / std
            ax.fill_between(r["scales"], y - ci, y + ci, color=COLORS[method], alpha=0.15, linewidth=0)
            ax.plot(r["scales"], y, color=COLORS[method], marker="o", label=LABELS[method])
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_title(f"{name}  ({SHORT.get(key, key)})")
        ax.set_xlabel("slider position")
    for ax in axes.flat[len(sliders):]:
        ax.axis("off")
    axes[0, 0].set_ylabel("change in descriptor\n(std of unsteered clips)")
    legend(fig, axes, -0.02)
    fig.suptitle("Does the measured descriptor follow the slider?", x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out / file, bbox_inches="tight")
    plt.close(fig)


def tradeoff_figure(runs, spec, out, plt):
    """Content kept against descriptor moved, one point per scale: up and to the right is better."""
    sliders = [s for s in spec if spec[s].get("measure") and any(k[1] == s for k in runs)]
    cols = min(4, len(sliders))
    rows_n = int(np.ceil(len(sliders) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.3 * cols, 2.7 * rows_n), squeeze=False, sharey=True)
    for ax, name in zip(axes.flat, sliders):
        key, sign = spec[name]["measure"], spec[name].get("measure_sign", 1)
        for method in COLORS:
            if (method, name) not in runs:
                continue
            rows = runs[(method, name)]
            std = M.natural_std(rows, key)
            move = sign * M.response(rows, key)["mean"] / std
            keep = M.response(rows, "clap_keep")["level"]
            ax.plot(move, keep, color=COLORS[method], marker="o", label=LABELS[method])
        ax.set_title(name)
        ax.set_xlabel("change in descriptor (std)")
    for ax in axes.flat[len(sliders):]:
        ax.axis("off")
    axes[0, 0].set_ylabel("CLAP similarity to unsteered clip")
    legend(fig, axes, -0.02)
    fig.suptitle("How much of the clip survives a given amount of change?", x=0.01, ha="left", fontweight="bold",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out / "tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def leakage_figure(runs, spec, method, out, plt):
    names = [s for s in spec if (method, s) in runs]
    if not names:
        return None
    table = np.array([[M.leakage(runs[(method, n)], LEAK_KEYS)[k] for k in LEAK_KEYS] for n in names])
    fig, ax = plt.subplots(figsize=(0.75 * len(LEAK_KEYS) + 2, 0.45 * len(names) + 1.6))
    lim = np.nanmax(np.abs(table))
    im = ax.imshow(table, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(LEAK_KEYS)), [SHORT[k] for k in LEAK_KEYS], rotation=40, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    for i, n in enumerate(names):
        for j, k in enumerate(LEAK_KEYS):
            v = table[i, j]
            if np.isfinite(v):
                target = spec[n].get("measure") == k
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.6 * lim else INK, fontweight="bold" if target else "normal")
    fig.colorbar(im, ax=ax, shrink=0.8, label="std per unit of slider")
    ax.set_title(f"What each {LABELS[method].lower()} moves (bold = its own descriptor)")
    fig.tight_layout()
    fig.savefig(out / f"leakage_{method}.png", bbox_inches="tight")
    plt.close(fig)
    return names, table


def quality_figure(runs, spec, out, plt, real_mean=None, file="quality.png"):
    """Aesthetics content-enjoyment at each slider position: a flat line means the music survives."""
    sliders = [s for s in spec if any(k[1] == s and any("ce" in r for r in v) for k, v in runs.items())]
    if not sliders:
        return
    cols = min(5, len(sliders))
    rows_n = int(np.ceil(len(sliders) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(2.9 * cols, 2.4 * rows_n), squeeze=False, sharey=True)
    for ax, name in zip(axes.flat, sliders):
        for method in COLORS:
            if (method, name) in runs and any("ce" in r for r in runs[(method, name)]):
                r = M.response(runs[(method, name)], "ce")
                ax.fill_between(r["scales"], r["level"] - r["ci"], r["level"] + r["ci"], color=COLORS[method],
                                alpha=0.15, linewidth=0)
                ax.plot(r["scales"], r["level"], color=COLORS[method], marker="o", label=LABELS[method])
        if real_mean is not None:
            ax.axhline(real_mean, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_title(name)
        ax.set_xlabel("slider position")
    for ax in axes.flat[len(sliders):]:
        ax.axis("off")
    axes[0, 0].set_ylabel("content enjoyment (1-10)")
    legend(fig, axes, -0.03)
    note = "  Dashed: mean of 2,000 real recordings (FMA)." if real_mean is not None else ""
    fig.suptitle("Does it still sound like music as the slider moves?" + note, x=0.01, ha="left", fontweight="bold",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out / file, bbox_inches="tight")
    plt.close(fig)


def tag_table(root, runs, spec, vocab_path, out):
    """Which musical tags rise and fall between the two ends of each slider."""
    from audiosliders.tags import tag_shift

    text = np.load(vocab_path)["text"]
    lines = ["| Slider | Method | Rises toward + | Falls toward + |", "|---|---|---|---|"]
    for (method, name), rows in runs.items():
        emb_path = root / f"{method}_{name}" / "clap.npy"
        if not emb_path.exists():
            continue
        all_rows = M.load_rows(root / f"{method}_{name}")
        emb = np.load(emb_path)
        scales = sorted({r["scale"] for r in rows})
        hi = np.array([r["scale"] == scales[-1] for r in all_rows])
        lo = np.array([r["scale"] == scales[0] for r in all_rows])
        up, down = tag_shift(emb[hi], emb[lo], text)
        fmt = lambda items: ", ".join(f"{t} ({d:+.3f})" for t, d in items)
        lines.append(f"| {name} | {LABELS.get(method, method)} | {fmt(up)} | {fmt(down)} |")
    (out / "tags.md").write_text("\n".join(lines) + "\n")


def second_opinion(runs, spec, out, real_path=None):
    """Compare Audiobox content enjoyment with SongEval musicality, a predictor trained on different data."""
    from scipy.stats import spearmanr

    lines = ["| Slider | Method | Enjoyment at 0 / low end / high end | Musicality at 0 / low end / high end | "
             "Usable span by enjoyment | Usable span by musicality | Agreement between the two (Spearman over clips) |",
             "|---|---|---|---|---|---|---:|"]
    for (method, name), rows in runs.items():
        if name not in spec or not all("se_musicality" in r and "ce" in r for r in rows):
            continue
        scales = sorted({r["scale"] for r in rows})
        mean = lambda k, x: float(np.mean([r[k] for r in rows if r["scale"] == x]))
        a, b = M.usable_span(rows), M.usable_span(rows, "se_musicality", 0.25)
        rho = spearmanr([r["ce"] for r in rows], [r["se_musicality"] for r in rows]).statistic
        lines.append(f"| {name} | {LABELS.get(method, method)} | "
                     f"{mean('ce', 0.0):.2f} / {mean('ce', scales[0]):.2f} / {mean('ce', scales[-1]):.2f} | "
                     f"{mean('se_musicality', 0.0):.2f} / {mean('se_musicality', scales[0]):.2f} / {mean('se_musicality', scales[-1]):.2f} | "
                     f"{a[0]:+.1f} to {a[1]:+.1f} | {b[0]:+.1f} to {b[1]:+.1f} | {rho:.2f} |")
    if len(lines) == 2:
        return
    note = ""
    if real_path and (Path(real_path) / "songeval.jsonl").exists():
        vals = [json.loads(line)["se_musicality"] for line in (Path(real_path) / "songeval.jsonl").read_text().splitlines() if line]
        note = f"\nReal recordings (FMA, {len(vals):,} clips): mean musicality {np.mean(vals):.2f}.\n"
    (out / "quality_check.md").write_text(
        "Audiobox Aesthetics content enjoyment (1 to 10) next to SongEval musicality (1 to 5). The usable span by "
        "musicality uses a tolerance of 0.25, half the tolerance used for enjoyment on a scale twice as wide.\n"
        + note + "\n" + "\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default=None, help="base run whose CLAP embeddings anchor the kernel distance")
    ap.add_argument("--max-scale", type=float, default=None, help="ignore scales beyond this magnitude")
    ap.add_argument("--real", default=None, help="directory with clap.npy and rows.jsonl of real recordings")
    ap.add_argument("--vocab", default=None, help="vocab.npz with tag text embeddings")
    ap.add_argument("--figure-methods", nargs="+", default=None,
                    help="methods drawn in response.png, tradeoff.png and quality.png; default all")
    ap.add_argument("--compare", nargs="+", default=None,
                    help="also draw compare_response.png and compare_quality.png for these methods, on sliders that have them all")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style(plt)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = yaml.safe_load(Path("configs/sliders.yaml").read_text())
    runs = load(Path(args.eval))
    if args.max_scale is not None:
        runs = {k: [r for r in v if abs(r["scale"]) <= args.max_scale] for k, v in runs.items()}
    reference = np.load(Path(args.reference) / "clap.npy") if args.reference else None
    real = np.load(Path(args.real) / "clap.npy") if args.real else None
    real_ce = float(np.mean([r["ce"] for r in M.load_rows(args.real)])) if args.real else None

    table = []
    for (method, name), rows in runs.items():
        if name not in spec:
            continue
        s = spec[name]
        row = dict(slider=name, method=LABELS.get(method, method), measure=s.get("measure") or "",
                   n=len({(r["prompt_index"], r["seed"]) for r in rows}))
        row.update(M.summarize(rows, s.get("measure"), s.get("measure_sign", 1)))
        if s.get("measure"):
            others = [k for k in LEAK_KEYS if k != s["measure"] and all(k in r for r in rows)]
            row["selectivity"] = M.selectivity(rows, s["measure"], others, s.get("measure_sign", 1))
            row["monotone_share"] = M.monotone_share(rows, s["measure"], s.get("measure_sign", 1))
        if reference is not None and (Path(args.eval) / f"{method}_{name}" / "clap.npy").exists():
            emb = np.load(Path(args.eval) / f"{method}_{name}" / "clap.npy")
            all_rows = M.load_rows(Path(args.eval) / f"{method}_{name}")
            scales = sorted({r["scale"] for r in rows})
            ends = np.array([r["scale"] in (scales[0], scales[-1]) for r in all_rows])
            zero = np.array([r["scale"] == 0 for r in all_rows])
            row["kad_at_ends"] = M.kernel_distance(emb[ends], reference)
            row["kad_at_zero"] = M.kernel_distance(emb[zero], reference)
        if real is not None and (Path(args.eval) / f"{method}_{name}" / "clap.npy").exists():
            emb = np.load(Path(args.eval) / f"{method}_{name}" / "clap.npy")
            all_rows = M.load_rows(Path(args.eval) / f"{method}_{name}")
            scales = sorted({r["scale"] for r in rows})
            for label, pick in (("lo", scales[0]), ("zero", 0.0), ("hi", scales[-1])):
                mask = np.array([r["scale"] == pick for r in all_rows])
                row[f"kad_real_{label}"] = M.kernel_distance(emb[mask], real)
        table.append(row)
    table.sort(key=lambda r: (list(spec).index(r["slider"]), list(LABELS.values()).index(r["method"])))

    fields = list(dict.fromkeys(k for r in table for k in r))
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(table)
    cols = [("slider", "Slider", None), ("method", "Method", None), ("rho", "Monotonicity ρ", 2),
            ("consistent", "Ends ordered", 2), ("muq_rho", "MuQ ρ", 2), ("muq_ordered", "MuQ ordered", 2),
            ("usable_lo", "Usable from", 1), ("usable_hi", "to", 1),
            ("usable_range_in_std", "Descriptor moved (std)", 2), ("usable_clap_range", "CLAP moved", 3),
            ("usable_clap_keep", "Piece kept", 2), ("ce_at_zero", "Quality at 0", 2),
            ("ce_at_ends", "Quality at ends", 2)]
    lines = ["| " + " | ".join(c[1] for c in cols) + " |", "|" + "|".join("---" if c[2] is None else "---:" for c in cols) + "|"]
    for r in table:
        cells = []
        for key, _, digits in cols:
            v = r.get(key)
            cell = "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (
                f"{v:.{digits}f}" if digits is not None else str(v))
            ci = r.get({"rho": "rho_ci", "usable_range_in_std": "usable_range_ci", "muq_rho": "muq_rho_ci"}.get(key, ""))
            if cell and ci is not None and np.isfinite(ci):
                cell += f" ± {ci:.{digits}f}"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps(dict(
        summary=table,
        summary_columns=[dict(key=k, label=l, digits=d) for k, l, d in cols],
    ), default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else o))

    shown = {k: v for k, v in runs.items() if args.figure_methods is None or k[0] in args.figure_methods}
    response_figure(shown, spec, out, plt)
    tradeoff_figure(shown, spec, out, plt)
    quality_figure(shown, spec, out, plt, real_ce)
    if args.compare:
        both = {k: v for k, v in runs.items() if k[0] in args.compare and all((m, k[1]) in runs for m in args.compare)}
        response_figure(both, spec, out, plt, "compare_response.png")
        quality_figure(both, spec, out, plt, real_ce, "compare_quality.png")
    if args.vocab:
        tag_table(Path(args.eval), runs, spec, args.vocab, out)
    second_opinion(runs, spec, out, args.real)
    with (out / "leakage.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "slider"] + LEAK_KEYS)
        for method in COLORS:
            got = leakage_figure(runs, spec, method, out, plt)
            if got:
                for n, vals in zip(*got):
                    w.writerow([method, n] + [f"{v:.4f}" for v in vals])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
