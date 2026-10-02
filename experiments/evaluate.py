"""Run one control method on one slider over the evaluation grid and measure every clip.

For each (prompt, seed) pair the same initial noise is rendered at every slider scale.
Output directory gets rows.jsonl (one line per clip: descriptors, CLAP scores, similarity
to the scale-0 clip), clap.npy (embeddings, same order), and optionally the audio as FLAC.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from audiosliders import methods
from audiosliders.backbone import StableAudio
from audiosliders.clap import Clap
from audiosliders.descriptors import content_similarity, describe
from audiosliders.dsp import EFFECTS
from audiosliders.lora import SliderBank


def measure(job):
    audio, base, sr = job
    row = describe(audio, sr)
    row.update(content_similarity(base, audio, sr))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--method", required=True, choices=["lora", "guidance", "embed", "dsp", "base"])
    ap.add_argument("--slider", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scales", type=float, nargs="+", default=[-3, -2, -1, 0, 1, 2, 3])
    ap.add_argument("--split", default="eval")
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--start", type=float, default=1.0, help="steering is active for t <= start")
    ap.add_argument("--eta", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--pairs-per-batch", type=int, default=4)
    ap.add_argument("--save-audio", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = yaml.safe_load(Path("configs/prompts.yaml").read_text())[args.split][: args.n_prompts]
    spec = yaml.safe_load(Path("configs/sliders.yaml").read_text())[args.slider] if args.slider else None
    scales = [0.0] if args.method == "base" else sorted(set(args.scales) | {0.0})
    zero = scales.index(0.0)

    model = StableAudio()
    clap = Clap()
    sr = model.sample_rate
    bank = None
    if args.method == "lora":
        bank = SliderBank(model.dit)
        meta = bank.load("s", args.weights)
        spec = spec or meta
    direction = None
    if spec:
        direction = clap.text([spec["positive"]]) - clap.text([spec["negative"]])
        direction = (direction / direction.norm())[0]

    pairs = [(i, args.seed_offset + 100 * i + k) for i in range(len(prompts)) for k in range(args.seeds)]
    rows, embeds, pending = [], [], []
    pool = ProcessPoolExecutor(args.workers)
    for b in range(0, len(pairs), args.pairs_per_batch):
        chunk = pairs[b : b + args.pairs_per_batch]
        render = [0.0] if args.method == "dsp" else scales
        text = [prompts[i] for i, _ in chunk for _ in render]
        seeds = [seed for _, seed in chunk for _ in render]
        s = torch.tensor(render * len(chunk), device=model.device)
        kw = dict(seconds=args.seconds, steps=args.steps, guidance=args.guidance)
        if args.method == "lora":
            audio = model.generate(text, seeds, wrap=methods.lora(bank, "s", s, args.start), **kw)
        elif args.method == "guidance":
            wrap = methods.guidance(model, text, spec["positive"], spec["negative"], s, args.seconds, args.eta, args.start)
            audio = model.generate(text, seeds, wrap=wrap, **kw)
        elif args.method == "embed":
            cond = methods.embed(model, text, spec["positive"], spec["negative"], s, args.seconds)
            audio = model.generate(text, seeds, cond=cond, **kw)
        else:
            audio = model.generate(text, seeds, **kw)
        audio = audio.cpu().numpy()
        if args.method == "dsp":
            fx = EFFECTS[spec["dsp"]]
            audio = np.stack([fx(a, sr, sc) for a in audio for sc in scales])
        emb = clap.audio(audio, sr)
        text_emb = clap.text([prompts[i] for i, _ in chunk])
        n = len(scales)
        for j, (i, seed) in enumerate(chunk):
            base = audio[j * n + zero]
            for k, scale in enumerate(scales):
                idx = j * n + k
                row = dict(prompt_index=i, prompt=prompts[i], seed=seed, scale=scale)
                row["clap_prompt"] = float(emb[idx] @ text_emb[j])
                row["clap_keep"] = float(emb[idx] @ emb[j * n + zero])
                if direction is not None:
                    row["clap_dir"] = float(emb[idx] @ direction)
                rows.append(row)
                pending.append(pool.submit(measure, (audio[idx], base, sr)))
                if args.save_audio:
                    sf.write(out / f"p{i:02d}_s{seed:05d}_x{scale:+.2f}.flac", audio[idx].T, sr)
        embeds.append(emb.cpu().numpy())
        print(f"{args.method} {args.slider} {min(b + args.pairs_per_batch, len(pairs))}/{len(pairs)} pairs", flush=True)

    for row, fut in zip(rows, pending):
        row.update(fut.result())
    with (out / "rows.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    np.save(out / "clap.npy", np.concatenate(embeds))
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
