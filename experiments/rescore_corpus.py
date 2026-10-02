"""Recompute descriptors and aesthetics for a stored corpus by decoding its latents.

Used when the measurement code gains new columns after a corpus was generated.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from audiosliders.backbone import load_backbone
from audiosliders.descriptors import describe
from audiosliders.quality import Aesthetics

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--backbone", default="sao")
ap.add_argument("--corpus", required=True)
ap.add_argument("--shard", required=True, help="two-digit shard tag, e.g. 00")
ap.add_argument("--seconds", type=float, default=10.0)
ap.add_argument("--batch", type=int, default=32)
args = ap.parse_args()

path = Path(args.corpus)
rows = [json.loads(line) for line in (path / f"rows_{args.shard}.jsonl").read_text().splitlines() if line]
latents = np.load(path / f"latents_{args.shard}.npy")
model, aesthetics = load_backbone(args.backbone), Aesthetics()
pool = ProcessPoolExecutor(12)
pending = []
for b in range(0, len(rows), args.batch):
    z = torch.from_numpy(latents[b : b + args.batch]).to(model.device, torch.float32)
    with torch.no_grad():
        audio = model.fit_peak(model.decode(z, args.seconds)).cpu().numpy()
    for row, score, clip in zip(rows[b : b + args.batch], aesthetics(audio, model.sample_rate), audio):
        row.update(score)
        pending.append(pool.submit(describe, clip, model.sample_rate))
    print(f"{min(b + args.batch, len(rows))}/{len(rows)}", flush=True)
for row, fut in zip(rows, pending):
    row.update(fut.result())
(path / f"rows_{args.shard}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print("rescored", len(rows))
