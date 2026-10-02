"""Generate a corpus of unsteered clips and store latents, descriptors, and CLAP embeddings.

The corpus is the model's own output distribution for a set of prompts. It is the raw
material for sliders that are not defined by text: sort the clips by a measured
descriptor, or by a principal direction of their CLAP embeddings, and train a slider
between the two ends.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

from audiosliders.backbone import StableAudio, seeded_noise
from audiosliders.clap import Clap
from audiosliders.descriptors import describe


def measure(job):
    audio, sr = job
    return describe(audio, sr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--seeds", type=int, default=64)
    ap.add_argument("--seed-offset", type=int, default=10_000)
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1], metavar=("INDEX", "COUNT"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = yaml.safe_load(Path("configs/prompts.yaml").read_text())[args.split]
    jobs = [(i, args.seed_offset + 1000 * i + k) for i in range(len(prompts)) for k in range(args.seeds)]
    jobs = jobs[args.shard[0] :: args.shard[1]]

    model = StableAudio()
    clap = Clap()
    pool = ProcessPoolExecutor(args.workers)
    latents, embeds, rows, pending = [], [], [], []
    for b in range(0, len(jobs), args.batch):
        chunk = jobs[b : b + args.batch]
        text = [prompts[i] for i, _ in chunk]
        z = model.generate(text, [s for _, s in chunk], seconds=args.seconds, steps=args.steps,
                           guidance=args.guidance, latents=True)
        audio = model.decode(z, args.seconds).cpu().numpy()
        emb = clap.audio(audio, model.sample_rate)
        text_emb = clap.text(text)
        latents.append(z.cpu().half().numpy())
        embeds.append(emb.cpu().numpy())
        for j, (i, seed) in enumerate(chunk):
            rows.append(dict(prompt_index=i, prompt=prompts[i], seed=seed, clap_prompt=float(emb[j] @ text_emb[j])))
            pending.append(pool.submit(measure, (audio[j], model.sample_rate)))
        print(f"{min(b + args.batch, len(jobs))}/{len(jobs)}", flush=True)
    for row, fut in zip(rows, pending):
        row.update(fut.result())
    tag = f"{args.shard[0]:02d}"
    np.save(out / f"latents_{tag}.npy", np.concatenate(latents))
    np.save(out / f"clap_{tag}.npy", np.concatenate(embeds))
    (out / f"rows_{tag}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {len(rows)} clips to {out}")


if __name__ == "__main__":
    main()
