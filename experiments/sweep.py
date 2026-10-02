"""Sweep one trained slider over scales on a few prompts and print how the descriptors move."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from audiosliders.backbone import StableAudio
from audiosliders.clap import Clap
from audiosliders.descriptors import content_similarity, describe
from audiosliders.lora import SliderBank

ap = argparse.ArgumentParser()
ap.add_argument("weights")
ap.add_argument("--out", default=None)
ap.add_argument("--scales", type=float, nargs="+", default=[-2, -1, 0, 1, 2])
ap.add_argument("--prompts", type=int, default=8)
ap.add_argument("--start", type=float, default=1.0, help="slider is active for t <= start")
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--split", default="eval")
ap.add_argument("--save-audio", action="store_true")
args = ap.parse_args()

weights = Path(args.weights)
out = Path(args.out or weights.with_suffix(""))
out.mkdir(parents=True, exist_ok=True)
prompts = yaml.safe_load(Path("configs/prompts.yaml").read_text())[args.split][: args.prompts]
model = StableAudio()
bank = SliderBank(model.dit)
meta = bank.load("s", weights)
clap = Clap()
if "direction" in meta:
    direction = torch.tensor(meta["direction"], device=clap.device, dtype=torch.float32)[None]
elif "positive" in meta:
    direction = clap.text([meta["positive"]]) - clap.text([meta["negative"]])
else:
    direction = torch.zeros(1, 512, device=clap.device)
direction = direction / direction.norm().clamp(min=1e-9)

rows = []
audio0 = None
for scale in [0.0] + [s for s in args.scales if s != 0]:
    wrap = (lambda p, s=scale: bank.gated(p, {"s": s}, start=args.start)) if scale else None
    audio = model.generate(prompts, list(range(len(prompts))), wrap=wrap, steps=args.steps).cpu()
    emb = clap.audio(audio, model.sample_rate)
    if scale == 0:
        audio0, emb0 = audio, emb
    for i, p in enumerate(prompts):
        a = audio[i].numpy()
        row = dict(prompt=p, scale=scale, **describe(a, model.sample_rate))
        row["clap_dir"] = float((emb[i] @ direction.T).item())
        row["clap_keep"] = float((emb[i] @ emb0[i]).item())
        row.update(content_similarity(audio0[i].numpy(), a, model.sample_rate))
        row["clipped"] = float((np.abs(a) >= 0.999).mean())
        rows.append(row)
        if args.save_audio:
            sf.write(out / f"p{i:02d}_s{scale:+.1f}.wav", a.T, model.sample_rate)
(out / f"sweep_start{args.start}.json").write_text(json.dumps(rows))

keys = ["centroid_hz", "rolloff_hz", "bass_ratio", "flatness", "flux", "rms_db", "onset_rate", "pulse_bpm",
        "percussive_ratio", "decay_s", "side_ratio", "majorness", "clap_dir", "clap_keep", "chroma_sim", "rhythm_sim",
        "clipped"]
print("scale  " + " ".join(k[:10].rjust(10) for k in keys))
for scale in sorted(set(r["scale"] for r in rows)):
    sel = [r for r in rows if r["scale"] == scale]
    print(f"{scale:+5.1f}  " + " ".join(f"{np.nanmean([r[k] for r in sel]):10.3f}" for k in keys))
