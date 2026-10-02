"""Check the backbone wrapper: speed, short-latent generation, sampler choice, CLAP agreement."""

import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

from audiosliders.backbone import StableAudio
from audiosliders.clap import Clap

out = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/smoke")
out.mkdir(parents=True, exist_ok=True)
prompts = [
    "solo jazz guitar, warm tone, swing feel",
    "128 BPM tech house drum loop",
    "ambient pad with slow evolving texture",
    "upbeat funk bass and drums groove, 110 BPM",
    "solo piano, gentle classical melody",
    "lofi hip hop beat with mellow keys, 85 BPM",
    "acoustic folk guitar fingerpicking",
    "cinematic orchestral strings, dramatic",
]
seeds = list(range(len(prompts)))
model = StableAudio()
clap = Clap()
text = clap.text(prompts)
report = {}
for name, kw in {
    "ode50_10s": dict(seconds=10.0, steps=50),
    "sde50_10s": dict(seconds=10.0, steps=50, sde=True),
    "ode100_10s": dict(seconds=10.0, steps=100),
    "sde100_10s": dict(seconds=10.0, steps=100, sde=True),
    "ode50_47s": dict(seconds=47.0, steps=50),
}.items():
    torch.cuda.synchronize()
    t0 = time.time()
    audio = model.generate(prompts, seeds, guidance=7.0, **kw)
    torch.cuda.synchronize()
    dt = time.time() - t0
    audio = audio[..., : 10 * model.sample_rate].cpu()
    emb = clap.audio(audio, model.sample_rate)
    sim = (emb * text).sum(-1)
    report[name] = dict(
        seconds_per_batch=round(dt, 2),
        clap_text_audio=round(sim.mean().item(), 4),
        per_prompt=[round(x, 3) for x in sim.tolist()],
        peak=round(audio.abs().max().item(), 3),
        rms=round(audio.pow(2).mean().sqrt().item(), 4),
    )
    print(name, report[name], flush=True)
    for i in range(len(prompts)):
        sf.write(out / f"{name}_{i}.wav", audio[i].T.numpy(), model.sample_rate)
(out / "report.json").write_text(json.dumps(report, indent=2))
