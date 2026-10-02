"""Find axes inside the model: principal directions of its own activations.

    python experiments/discover_internal.py --backbone ace-turbo --split train --seeds 16 --out runs/discovery/internal_ace

Samples every prompt with several seeds, records each clip's mean cross-attention output
in every block, removes each prompt's mean, and decomposes what is left. No embedding
model and no text is involved in choosing the axes. Each axis is saved as one vector per
block, scaled to one standard deviation of natural spread, and can be applied with

    python experiments/evaluate.py --method caa --vectors runs/discovery/internal_ace/axes.pt --axis 0 ...
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment

from audiosliders import steer
from audiosliders.backbone import load_backbone

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--backbone", default="ace-turbo")
ap.add_argument("--split", default="train")
ap.add_argument("--n-prompts", type=int, default=48)
ap.add_argument("--seeds", type=int, default=16)
ap.add_argument("--seconds", type=float, default=10.0)
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--out", required=True)
args = ap.parse_args()

prompts = yaml.safe_load(Path("configs/prompts.yaml").read_text())[args.split][: args.n_prompts]
text = [p for p in prompts for _ in range(args.seeds)]
seeds = [500_000 + 1000 * i + k for i in range(len(prompts)) for k in range(args.seeds)]
groups = np.repeat(np.arange(len(prompts)), args.seeds)

model = load_backbone(args.backbone)
acts, names = steer.record(model, text, seeds, args.seconds, args.batch)
axes, share = steer.internal_axes(acts, groups, args.n)

half = np.random.default_rng(0).permutation(len(prompts))
a = np.isin(groups, half[: len(half) // 2])
first, second = (steer.internal_axes(acts[m], groups[m], args.n)[0].reshape(args.n, -1) for m in (a, ~a))
first /= np.linalg.norm(first, axis=1, keepdims=True)
second /= np.linalg.norm(second, axis=1, keepdims=True)
cos = np.abs(first @ second.T)
r, c = linear_sum_assignment(-cos[:8, :8])
stability = float(cos[:8, :8][r, c].mean())

# Where in depth each axis lives: its squared norm per block, as a share of the whole.
depth = (axes**2).sum(-1)
depth /= depth.sum(1, keepdims=True)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
torch.save(dict(axes=torch.from_numpy(axes), names=names, share=share.tolist(), backbone=args.backbone), out / "axes.pt")
np.save(out / "activations.npy", acts.astype(np.float16))
(out / "axes.json").write_text(json.dumps(dict(
    backbone=args.backbone, clips=len(text), prompts=len(prompts), blocks=len(names), width=int(acts.shape[-1]),
    share=share.tolist(), stability=stability, depth=depth.tolist()), indent=1))
print(f"{len(text)} clips, {len(names)} blocks of width {acts.shape[-1]}")
print("share of within-prompt variance:", " ".join(f"{100 * v:.1f}%" for v in share))
print(f"stability of the first 8 axes across two disjoint halves of the prompts: {stability:.2f}")
