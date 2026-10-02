"""Score saved clips with SongEval, a quality predictor that shares nothing with Audiobox Aesthetics.

SongEval (ASLP-lab, arXiv:2505.10793) is a small head on MuQ features trained on human ratings
of generated songs, on a 1 to 5 scale: coherence, musicality, memorability, clarity of structure,
and naturalness. Runs in the MuQ environment and needs a clone of github.com/ASLP-lab/SongEval:

    venv-muq/bin/python experiments/songeval_score.py --songeval ../SongEval runs/eval/ace/lora_*
    venv-muq/bin/python experiments/songeval_score.py --songeval ../SongEval --list files.txt --out scores.jsonl

For an evaluation run it writes songeval.jsonl next to rows.jsonl, in the same order, and
audiosliders.metrics merges the scores in as se_coherence, se_musicality, and so on.
"""

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from muq import MuQ
from safetensors.torch import load_file

NAMES = ["coherence", "musicality", "memorability", "clarity", "naturalness"]

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("runs", nargs="*")
ap.add_argument("--songeval", required=True, help="path to a clone of the SongEval repository")
ap.add_argument("--list", default=None, help="text file of audio paths to score instead of evaluation runs")
ap.add_argument("--out", default=None)
ap.add_argument("--seconds", type=float, default=None, help="with --list: score this many seconds from the middle of each file")
ap.add_argument("--batch", type=int, default=16)
args = ap.parse_args()

sys.path.insert(0, args.songeval)
from model import Generator  # noqa: E402

head = Generator(in_features=1024, ffd_hidden_size=4096, num_classes=5, attn_layer_num=4)
head.load_state_dict(load_file(str(Path(args.songeval) / "ckpt" / "model.safetensors")), strict=False)
head = head.cuda().eval()
muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").cuda().eval()


def load(path, seconds=None):
    audio, sr = sf.read(path, always_2d=True) if str(path).endswith((".flac", ".wav")) else (None, None)
    if audio is None:
        wav, _ = librosa.load(path, sr=24_000, mono=True)
    else:
        wav = librosa.resample(audio.mean(1).astype(np.float32), orig_sr=sr, target_sr=24_000)
    if seconds:
        n = int(seconds * 24_000)
        start = max(0, (len(wav) - n) // 2)
        wav = wav[start : start + n]
    return wav


@torch.no_grad()
def score(files, seconds=None):
    out = []
    for b in range(0, len(files), args.batch):
        wavs = [load(f, seconds) for f in files[b : b + args.batch]]
        n = min(len(w) for w in wavs)
        feats = muq(torch.tensor(np.stack([w[:n] for w in wavs])).cuda(), output_hidden_states=True)["hidden_states"][6]
        for row in head(feats).cpu().numpy():
            out.append({f"se_{k}": float(v) for k, v in zip(NAMES, row)})
    return out


if args.list:
    files = [line.strip() for line in Path(args.list).read_text().splitlines() if line.strip()]
    rows = score(files, args.seconds)
    Path(args.out).write_text("\n".join(json.dumps(dict(file=f, **r)) for f, r in zip(files, rows)) + "\n")
    print("scored", len(rows), "files; mean musicality", np.mean([r["se_musicality"] for r in rows]))
for run in map(Path, args.runs):
    if not (run / "rows.jsonl").exists():
        continue
    rows = [json.loads(line) for line in (run / "rows.jsonl").read_text().splitlines() if line]
    files = [run / f"p{r['prompt_index']:02d}_s{r['seed']:05d}_x{r['scale']:+.2f}.flac" for r in rows]
    if not all(f.exists() for f in files):
        print("no audio", run)
        continue
    (run / "songeval.jsonl").write_text("\n".join(json.dumps(r) for r in score(files)) + "\n")
    print("scored", run, len(files), flush=True)
