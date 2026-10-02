"""Score saved evaluation clips with MuQ-MuLan, a music-text model that shares nothing with CLAP.

Runs in its own environment (MuQ needs transformers 4.x):

    venv-muq/bin/python experiments/muq_score.py runs/eval/ace/lora_mood --slider mood

Writes muq.jsonl next to rows.jsonl: for every clip, its similarity to the slider's positive
phrase, to its negative phrase, and to the base prompt. audiosliders.metrics merges it in.
"""

import argparse
import json
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml
from muq import MuQMuLan

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("runs", nargs="+")
ap.add_argument("--slider", default=None, help="key in configs/sliders.yaml; default: taken from the directory name")
ap.add_argument("--batch", type=int, default=16)
args = ap.parse_args()

spec = yaml.safe_load(Path("configs/sliders.yaml").read_text())
model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large").cuda().eval()
for run in map(Path, args.runs):
    name = args.slider or run.name.split("_", 1)[1]
    if name not in spec or not (run / "rows.jsonl").exists():
        print("skip", run)
        continue
    rows = [json.loads(line) for line in (run / "rows.jsonl").read_text().splitlines() if line]
    files = [run / f"p{r['prompt_index']:02d}_s{r['seed']:05d}_x{r['scale']:+.2f}.flac" for r in rows]
    if not all(f.exists() for f in files):
        print("no audio", run)
        continue
    prompts = sorted({r["prompt"] for r in rows})
    with torch.no_grad():
        text = model(texts=[spec[name]["positive"], spec[name]["negative"]] + prompts)
    out = []
    for b in range(0, len(files), args.batch):
        wavs = []
        for f in files[b : b + args.batch]:
            audio, sr = sf.read(f)
            wavs.append(librosa.resample(audio.mean(1).astype(np.float32), orig_sr=sr, target_sr=24_000))
        n = min(len(w) for w in wavs)
        with torch.no_grad():
            emb = model(wavs=torch.tensor(np.stack([w[:n] for w in wavs])).cuda())
            sim = model.calc_similarity(emb, text).cpu().numpy()
        for r, f, s in zip(rows[b : b + args.batch], files[b : b + args.batch], sim):
            out.append(dict(file=f.name, muq_pos=float(s[0]), muq_neg=float(s[1]),
                            muq_prompt=float(s[2 + prompts.index(r["prompt"])])))
    (run / "muq.jsonl").write_text("\n".join(json.dumps(o) for o in out) + "\n")
    print("scored", run, len(out))
