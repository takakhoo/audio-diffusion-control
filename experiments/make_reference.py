"""Embed and score a sample of real recordings, as the yardstick for "sounds like music".

Writes CLAP embeddings, aesthetics scores, and descriptors for 10-second excerpts of real
tracks (FMA by default), plus CLAP text embeddings for the tag vocabulary.
"""

import argparse
import json
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from audiosliders.clap import Clap
from audiosliders.descriptors import describe
from audiosliders.quality import Aesthetics
from audiosliders.tags import vocabulary

SR = 44_100


def excerpt(path: str, seconds: float = 10.0, offset: float = 10.0):
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", str(offset), "-t", str(seconds), "-i", path,
           "-ar", str(SR), "-ac", "2", "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    audio = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T
    return audio if audio.shape[1] == int(seconds * SR) else None


def measure(audio):
    return describe(audio, SR)


ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--music", default="/scratch/f004h1v/remaster/data/raw/fma_medium")
ap.add_argument("--out", default="runs/reference/fma")
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
files = sorted(str(p) for p in Path(args.music).rglob("*.mp3"))
random.Random(args.seed).shuffle(files)
clap, aesthetics = Clap(), Aesthetics()
vocab = vocabulary()
np.savez(out.parent / "vocab.npz", text=clap.text([t for _, _, t in vocab]).cpu().numpy(),
         tags=np.array([t for _, t, _ in vocab]), groups=np.array([g for g, _, _ in vocab]))

rows, embeds, pending = [], [], []
pool = ProcessPoolExecutor(12)
with ThreadPoolExecutor(16) as io:
    for b in range(0, len(files), 64):
        if len(rows) >= args.n:
            break
        got = [(f, a) for f, a in zip(files[b : b + 64], io.map(excerpt, files[b : b + 64])) if a is not None]
        got = [(f, a) for f, a in got if np.abs(a).max() > 1e-3]
        audio = np.stack([a for _, a in got])
        embeds.append(clap.audio(audio, SR).cpu().numpy())
        for (f, a), score in zip(got, aesthetics(audio, SR)):
            rows.append(dict(file=Path(f).name, **score))
            pending.append(pool.submit(measure, a))
        print(f"{len(rows)}/{args.n}", flush=True)
for row, fut in zip(rows, pending):
    row.update(fut.result())
np.save(out / "clap.npy", np.concatenate(embeds)[: args.n])
(out / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows[: args.n]) + "\n")
print(f"wrote {min(len(rows), args.n)} reference clips to {out}")
