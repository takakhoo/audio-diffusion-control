"""Train one slider by distilling prompt-pair guidance into a LoRA (Concept Sliders, v-prediction).

For a base prompt c, attribute phrases a+ and a-, and slider scale s in {+1, -1}:

    target = v(z_t, c) + s * eta * (v(z_t, "c, a+") - v(z_t, "c, a-"))
    loss   = || v_lora(z_t, c; scale = s) - target ||^2

All three target terms come from the frozen model. z_t is drawn by running the sampler
from noise for a random number of steps with the slider active, so the LoRA is trained on
the states it will actually visit at inference.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml

from .backbone import Cond, StableAudio, seeded_noise
from .lora import SliderBank


@dataclass
class TrainConfig:
    name: str
    positive: str
    negative: str
    rank: int = 4
    alpha: float = 1.0
    targets: str = "all"
    lr: float = 2e-4
    iters: int = 1000
    batch: int = 4
    eta: float = 4.0
    guidance: float = 7.0
    seconds: float = 10.0
    sample_steps: int = 30
    max_stop: float = 1.0
    seed: int = 0


def train_slider(model: StableAudio, bank: SliderBank, cfg: TrainConfig, prompts: list[str], log=print) -> list[dict]:
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    params = bank.add(
        cfg.name, rank=cfg.rank, alpha=cfg.alpha, targets=cfg.targets, positive=cfg.positive, negative=cfg.negative
    )
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 50))
    frames = model.frames(cfg.seconds)
    # Half the batch trains the positive direction and half the negative, on the same step.
    scale = torch.tensor([1.0, -1.0], device=model.device).repeat(cfg.batch // 2 + 1)[: cfg.batch]
    history = []
    t0 = time.time()
    for it in range(cfg.iters):
        base = [rng.choice(prompts) for _ in range(cfg.batch)]
        c0 = model.encode(base, cfg.seconds)
        cp = model.encode([f"{p}, {cfg.positive}" for p in base], cfg.seconds)
        cn = model.encode([f"{p}, {cfg.negative}" for p in base], cfg.seconds)
        stop = rng.randrange(0, int(cfg.sample_steps * cfg.max_stop))
        noise = seeded_noise([rng.randrange(2**31) for _ in base], (model.channels, frames), model.device)
        with torch.no_grad():
            with bank.at(**{cfg.name: scale}):
                z, t = model.sample(model.cfg(c0, cfg.guidance), noise, steps=cfg.sample_steps, stop=stop)
            v0, vp, vn = model.v(torch.cat([z, z, z]), t, Cond.cat([c0, cp, cn])).chunk(3)
            target = v0 + scale.view(-1, 1, 1) * cfg.eta * (vp - vn)
        with bank.at(**{cfg.name: scale}):
            pred = model.v(z, t, c0)
        loss = torch.nn.functional.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            # How much of the wanted shift the LoRA has learned: 0 at init, 1 when it matches.
            gap = torch.nn.functional.mse_loss(v0, target)
        row = dict(iter=it, loss=loss.item(), baseline=gap.item(), t=float(t), time=time.time() - t0)
        history.append(row)
        if it % 25 == 0 or it == cfg.iters - 1:
            recent = history[-25:]
            ratio = sum(r["loss"] for r in recent) / max(1e-12, sum(r["baseline"] for r in recent))
            log(f"[{cfg.name}] {it:5d}/{cfg.iters} loss {loss.item():.5f} unexplained {ratio:.3f} {row['time']:.0f}s")
    return history


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("slider", help="key in the sliders config")
    ap.add_argument("--sliders", default="configs/sliders.yaml")
    ap.add_argument("--prompts", default="configs/prompts.yaml")
    ap.add_argument("--out", default="runs/sliders")
    ap.add_argument("--tag", default=None, help="output name, defaults to the slider key")
    for f, typ in [("rank", int), ("alpha", float), ("targets", str), ("lr", float), ("iters", int),
                   ("batch", int), ("eta", float), ("guidance", float), ("seconds", float), ("seed", int)]:
        ap.add_argument(f"--{f}", type=typ, default=None)
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.sliders).read_text())[args.slider]
    prompts = yaml.safe_load(Path(args.prompts).read_text())["train"]
    over = {k: v for k, v in vars(args).items() if k in TrainConfig.__dataclass_fields__ and v is not None}
    cfg = TrainConfig(name=args.slider, positive=spec["positive"], negative=spec["negative"], **over)
    tag = args.tag or args.slider
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = StableAudio()
    bank = SliderBank(model.dit)
    history = train_slider(model, bank, cfg, prompts)
    bank.save(cfg.name, out / f"{tag}.safetensors")
    (out / f"{tag}.json").write_text(json.dumps(dict(config=asdict(cfg), history=history)))
    print(f"saved {out / (tag + '.safetensors')}")


if __name__ == "__main__":
    main()
