"""Encode real recordings into a backbone's latent space and measure them.

    python experiments/make_real_corpus.py --music data/fma/fma_large --metadata data/fma/fma_metadata/tracks.csv \
        --out runs/corpus/real_ace --backbone ace-turbo --shard 0 4

The result has the same layout as a generated corpus (latents, CLAP embeddings, one row of
measurements per clip), so the set trainer can learn sliders between real recordings sorted
by tempo, brightness, harmony, mood or instrument tags, or a principal direction.
"""

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from audiosliders.backbone import load_backbone
from audiosliders.clap import Clap
from audiosliders.descriptors import describe
from audiosliders.quality import Aesthetics
from audiosliders.tags import vocabulary


def excerpt(job):
    path, sr, seconds = job
    cmd = ["ffmpeg", "-loglevel", "error", "-t", str(seconds), "-i", path, "-ar", str(sr), "-ac", "2", "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    audio = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T
    if audio.shape[1] < int(seconds * sr) or np.abs(audio).max() < 1e-3:
        return None
    return audio[:, : int(seconds * sr)].copy()


def measure(job):
    audio, sr = job
    return describe(audio, sr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--music", required=True)
    ap.add_argument("--metadata", default=None, help="FMA tracks.csv, for genre labels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="ace-turbo")
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1])
    ap.add_argument("--seconds", type=float, default=29.5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--tag-offset", type=int, default=0, help="added to the shard index in output file names")
    ap.add_argument("--skip", default=None, help="an existing corpus directory whose tracks are left out")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(str(p) for p in Path(args.music).rglob("*.mp3"))[: args.limit]
    if args.skip:
        seen = {json.loads(line)["file"] for f in Path(args.skip).glob("rows_*.jsonl") for line in f.read_text().splitlines() if line}
        files = [f for f in files if Path(f).name not in seen]
    files = files[args.shard[0] :: args.shard[1]]
    genre = {}
    if args.metadata:
        import pandas as pd

        tracks = pd.read_csv(args.metadata, index_col=0, header=[0, 1])
        genre = tracks[("track", "genre_top")].dropna().to_dict()

    model = load_backbone(args.backbone)
    clap, aesthetics = Clap(), Aesthetics()
    from beat_this.inference import Audio2Beats

    beats = Audio2Beats(checkpoint_path="final0", device="cuda", dbn=False)
    vocab = vocabulary()
    text = clap.text([t for _, _, t in vocab])
    by_group = {g: [i for i, (gg, _, _) in enumerate(vocab) if gg == g] for g in ("genre", "instrument", "mood")}
    vocal = [i for i, (_, t, _) in enumerate(vocab) if t == "vocals"][0]
    sr = model.sample_rate
    mid = slice(int(10 * sr), int(20 * sr))  # descriptors and embeddings use the middle ten seconds

    pool = ProcessPoolExecutor(args.workers)
    rows, latents, embeds, pending = [], [], [], []
    with ThreadPoolExecutor(8) as io:
        for b in range(0, len(files), args.batch):
            chunk = files[b : b + args.batch]
            got = [(f, a) for f, a in zip(chunk, io.map(excerpt, [(f, sr, args.seconds) for f in chunk])) if a is not None]
            if not got:
                continue
            audio = np.stack([a for _, a in got])
            z = torch.cat([model.encode_audio(torch.from_numpy(audio[i : i + 4])) for i in range(0, len(audio), 4)])
            emb = clap.audio(audio[..., mid], sr)
            sims = (emb @ text.T).cpu().numpy()
            scores = aesthetics(audio[..., mid], sr)
            for j, (f, a) in enumerate(got):
                track = int(Path(f).stem)
                top = {g: vocab[idx[int(np.argmax(sims[j, idx]))]][1] for g, idx in by_group.items()}
                label = str(genre.get(track, top["genre"]))
                found, _ = beats(a.mean(0), sr)
                bpm = float(60 / np.median(np.diff(found))) if len(found) > 3 else float("nan")
                rows.append(dict(file=Path(f).name, track=track, genre=label, prompt_index=label,
                                 prompt=f"{label.lower()} music with {top['instrument']}, {top['mood']}",
                                 vocal_score=float(sims[j, vocal]), beat_bpm=bpm, n_beats=int(len(found)), **scores[j]))
                pending.append(pool.submit(measure, (a[..., mid], sr)))
            latents.append(z.cpu().half().numpy())
            embeds.append(emb.cpu().numpy())
            if (b // args.batch) % 20 == 0:
                print(f"{b + len(chunk)}/{len(files)}", flush=True)
    for row, fut in zip(rows, pending):
        row.update(fut.result())
    tag = f"{args.shard[0] + args.tag_offset:02d}"
    np.save(out / f"latents_{tag}.npy", np.concatenate(latents))
    np.save(out / f"clap_{tag}.npy", np.concatenate(embeds))
    (out / f"rows_{tag}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {len(rows)} clips to {out}")


if __name__ == "__main__":
    main()
