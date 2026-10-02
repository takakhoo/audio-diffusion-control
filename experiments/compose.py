"""Two sliders at once: render a grid of positions and check that their effects add.

For sliders A and B the same (prompt, seed) is rendered at every (a, b) in a square grid.
If the sliders compose, descriptor A depends on a and not on b, and the reverse. The
summary reports each descriptor's slope against its own slider and against the other one.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

from audiosliders.backbone import load_backbone
from audiosliders.clap import Clap
from audiosliders.descriptors import describe
from audiosliders.lora import SliderBank
from audiosliders.quality import Aesthetics

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--weights", default="runs/sliders/v1")
ap.add_argument("--backbone", default="sao")
ap.add_argument("--out", required=True)
ap.add_argument("--grid", type=float, nargs="+", default=[-1, 0, 1])
ap.add_argument("--n-prompts", type=int, default=24)
ap.add_argument("--seconds", type=float, default=10.0)
ap.add_argument("--start", type=float, default=1.0, help="sliders are active for t <= start")
args = ap.parse_args()

spec = yaml.safe_load(Path("configs/sliders.yaml").read_text())
prompts = yaml.safe_load(Path("configs/prompts.yaml").read_text())["eval"][: args.n_prompts]
model = load_backbone(args.backbone)
bank = SliderBank(model.dit)
bank.load("a", Path(args.weights) / f"{args.a}.safetensors")
bank.load("b", Path(args.weights) / f"{args.b}.safetensors")
clap, aesthetics = Clap(), Aesthetics()
cells = [(x, y) for x in args.grid for y in args.grid]
sa = torch.tensor([c[0] for c in cells], device=model.device)
sb = torch.tensor([c[1] for c in cells], device=model.device)
pool = ProcessPoolExecutor(12)
rows, pending = [], []
for i, prompt in enumerate(prompts):
    audio = model.generate([prompt] * len(cells), [100 * i] * len(cells), seconds=args.seconds,
                           wrap=lambda p: bank.gated(p, {"a": sa, "b": sb}, start=args.start)).cpu().numpy()
    emb = clap.audio(audio, model.sample_rate)
    zero = cells.index((0.0, 0.0)) if (0.0, 0.0) in cells else 0
    for k, ((x, y), score) in enumerate(zip(cells, aesthetics(audio, model.sample_rate))):
        rows.append(dict(prompt_index=i, a=x, b=y, clap_keep=float(emb[k] @ emb[zero]), **score))
        pending.append(pool.submit(describe, audio[k], model.sample_rate))
    print(f"{i + 1}/{len(prompts)}", flush=True)
for row, fut in zip(rows, pending):
    row.update(fut.result())

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
summary = dict(a=args.a, b=args.b)
for own, other, name in (("a", "b", args.a), ("b", "a", args.b)):
    key = spec[name].get("measure")
    if not key:
        continue
    y = np.array([r[key] for r in rows], dtype=float)
    base = {r["prompt_index"]: r[key] for r in rows if r["a"] == 0 and r["b"] == 0}
    y = y - np.array([base[r["prompt_index"]] for r in rows])
    design = np.array([[r[own], r[other], r[own] * r[other]] for r in rows])
    ok = np.isfinite(y)
    coef, *_ = np.linalg.lstsq(design[ok], y[ok], rcond=None)
    std = np.nanstd(list(base.values()))
    summary[name] = dict(measure=key, own_slope=float(coef[0] / std), other_slope=float(coef[1] / std),
                         interaction=float(coef[2] / std))
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
