"""MuQ-MuLan embeddings for a corpus of real recordings, as a second space for axis discovery.

Runs in the MuQ environment:

    venv-muq/bin/python experiments/muq_embed.py --corpus runs/corpus/real_ace --music DIR [DIR ...]

For every rows_XX.jsonl it writes muq_XX.npy in the same order (middle ten seconds of each
track), and it writes the tag vocabulary's MuQ text embeddings next to the CLAP ones.
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from muq import MuQMuLan

sys.path.insert(0, ".")
from audiosliders.tags import vocabulary  # noqa: E402  (pure Python, no heavy imports)

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--corpus", required=True)
ap.add_argument("--music", nargs="+", required=True)
ap.add_argument("--vocab-out", default="runs/reference/vocab_muq.npz")
ap.add_argument("--batch", type=int, default=32)
args = ap.parse_args()


def find(name: str) -> str | None:
    for root in args.music:
        p = Path(root) / name[:3] / name
        if p.exists():
            return str(p)
    return None


def excerpt(path: str) -> np.ndarray:
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", "10", "-t", "10", "-i", path, "-ar", "24000", "-ac", "1", "-f", "f32le", "-"]
    audio = np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, dtype=np.float32)
    out = np.zeros(240_000, dtype=np.float32)
    out[: min(len(audio), 240_000)] = audio[:240_000]
    return out


model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large").cuda().eval()
vocab = vocabulary()
with torch.no_grad():
    text = model(texts=[t for _, _, t in vocab]).cpu().numpy()
np.savez(args.vocab_out, text=text, tags=np.array([t for _, t, _ in vocab]), groups=np.array([g for g, _, _ in vocab]))

corpus = Path(args.corpus)
with ThreadPoolExecutor(12) as io:
    for rows_file in sorted(corpus.glob("rows_*.jsonl")):
        out = corpus / rows_file.name.replace("rows_", "muq_").replace(".jsonl", ".npy")
        if out.exists():
            continue
        names = [json.loads(line)["file"] for line in rows_file.read_text().splitlines() if line]
        embeds = []
        for b in range(0, len(names), args.batch):
            wavs = np.stack(list(io.map(excerpt, [find(n) for n in names[b : b + args.batch]])))
            with torch.no_grad():
                embeds.append(model(wavs=torch.from_numpy(wavs).cuda()).cpu().numpy())
        np.save(out, np.concatenate(embeds))
        print("embedded", rows_file.name, len(names), flush=True)
