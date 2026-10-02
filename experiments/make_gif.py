"""Animate slider sweeps for the README: spectrogram, slider knob, and measured values.

    python experiments/make_gif.py --out results/demo/sliders.gif \
        runs/eval/ace/lora_mood:3 runs/eval/ace/lora_ensemble:0 ...

Each argument is an evaluation run saved with --save-audio and a prompt index. One panel
per argument; all panels move through the slider positions together.
"""

import argparse
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import yaml
from matplotlib.animation import FuncAnimation, PillowWriter

from audiosliders import metrics

SURFACE, INK, MUTED, TRACK, ACCENT = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d5cc", "#eb6834"

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("panels", nargs="+", help="RUN_DIR:PROMPT_INDEX[:LOW,HIGH]")
ap.add_argument("--out", required=True)
ap.add_argument("--cols", type=int, default=2)
ap.add_argument("--fps", type=float, default=1.6)
ap.add_argument("--max-scale", type=float, default=None)
args = ap.parse_args()

spec = yaml.safe_load(Path("configs/sliders.yaml").read_text())
panels = []
for item in args.panels:
    parts = item.split(":")
    run, pid = Path(parts[0]), int(parts[1])
    name = run.name.split("_", 1)[1]
    rows = [r for r in metrics.load_rows(run) if r["prompt_index"] == pid]
    seed = min(r["seed"] for r in rows)
    rows = sorted((r for r in rows if r["seed"] == seed), key=lambda r: r["scale"])
    if args.max_scale is not None:
        rows = [r for r in rows if abs(r["scale"]) <= args.max_scale]
    mels = []
    for r in rows:
        audio, sr = sf.read(run / f"p{pid:02d}_s{seed:05d}_x{r['scale']:+.2f}.flac")
        mel = librosa.feature.melspectrogram(y=audio.mean(1), sr=sr, n_fft=2048, hop_length=1024, n_mels=96, fmax=16000)
        mels.append(librosa.power_to_db(mel, ref=1.0))
    top = max(m.max() for m in mels)
    ends = parts[2].split(",") if len(parts) > 2 else spec.get(name, {}).get("ends", ["-", "+"])
    panels.append(dict(name=name, rows=rows, mels=[np.clip(m - top, -70, 0) for m in mels], ends=ends,
                       prompt=rows[0]["prompt"]))

n = len(panels[0]["rows"])
mid = n // 2
order = list(range(mid, n)) + list(range(n - 2, -1, -1)) + list(range(1, mid + 1))
cols = min(args.cols, len(panels))
nrows = int(np.ceil(len(panels) / cols))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig = plt.figure(figsize=(5.4 * cols, 3.4 * nrows), dpi=80, facecolor=SURFACE)
artists = []
for k, p in enumerate(panels):
    gs = fig.add_gridspec(nrows, cols, left=0.03, right=0.97, top=0.93, bottom=0.03, wspace=0.08, hspace=0.3)[k // cols, k % cols]
    inner = gs.subgridspec(3, 1, height_ratios=[5, 0.9, 0.7], hspace=0.08)
    ax = fig.add_subplot(inner[0])
    im = ax.imshow(p["mels"][mid], origin="lower", aspect="auto", cmap="magma", vmin=-70, vmax=0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{p['name']}", loc="left", fontsize=13, fontweight="bold", color=INK, pad=4)
    ax.text(1.0, 1.03, p["prompt"][:46], transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=MUTED)
    sl = fig.add_subplot(inner[1])
    scales = [r["scale"] for r in p["rows"]]
    sl.set_xlim(scales[0] - 0.9, scales[-1] + 0.9); sl.set_ylim(-1, 1); sl.axis("off")
    sl.plot([scales[0], scales[-1]], [0, 0], color=TRACK, linewidth=6, solid_capstyle="round")
    sl.scatter(scales, [0] * n, s=14, color=MUTED, zorder=2)
    knob = sl.scatter([0], [0], s=260, color=ACCENT, edgecolor=SURFACE, linewidth=2, zorder=3)
    sl.text(scales[0] - 0.25, 0, p["ends"][0], ha="right", va="center", fontsize=10, color=INK)
    sl.text(scales[-1] + 0.25, 0, p["ends"][1], ha="left", va="center", fontsize=10, color=INK)
    tx = fig.add_subplot(inner[2]); tx.axis("off")
    label = tx.text(0.5, 0.5, "", ha="center", va="center", fontsize=10, color=MUTED, family="DejaVu Sans Mono")
    artists.append((im, knob, label))


def draw(frame):
    i = order[frame]
    for p, (im, knob, label) in zip(panels, artists):
        r = p["rows"][i]
        im.set_data(p["mels"][i])
        knob.set_offsets([[r["scale"], 0]])
        label.set_text(f"position {r['scale']:+.1f}   enjoyment {r['ce']:.1f}/10   same piece {r['clap_keep']:.2f}")
    return [a for group in artists for a in group]


out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
FuncAnimation(fig, draw, frames=len(order), blit=False).save(out, writer=PillowWriter(fps=args.fps))
draw(order.index(n - 1))
fig.savefig(out.with_suffix(".png"))
print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB")
