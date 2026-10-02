"""Train a slider from two sets of clips instead of two prompts.

At scale +1 the slider is trained with the ordinary denoising loss on a `high` set, and
at scale -1 on a `low` set. A single low-rank update serves both ends with opposite sign,
so whatever the two sets share cancels and only their difference can be learned.

The sets can come from anywhere:
  - the model's own clips sorted by a measured descriptor (no text involved),
  - the same clips sorted along a principal direction of their CLAP embeddings
    (unsupervised discovery, after SliderSpace),
  - before/after renders of a signal-processing effect with a known parameter.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .backbone import StableAudio
from .lora import SliderBank


@dataclass
class ContrastConfig:
    name: str
    rank: int = 4
    alpha: float = 1.0
    targets: str = "all"
    lr: float = 2e-4
    iters: int = 2000
    batch: int = 16
    seconds: float = 10.0
    drop_text: float = 0.1
    paired: bool = False
    seed: int = 0


def load_corpus(path: str | Path) -> dict:
    """Concatenate the shards written by experiments/make_corpus.py."""
    path = Path(path)
    shards = sorted(p.stem.split("_")[1] for p in path.glob("rows_*.jsonl"))
    rows, latents, clap = [], [], []
    for tag in shards:
        rows += [json.loads(line) for line in (path / f"rows_{tag}.jsonl").read_text().splitlines() if line]
        latents.append(np.load(path / f"latents_{tag}.npy"))
        clap.append(np.load(path / f"clap_{tag}.npy"))
    return dict(rows=rows, latents=np.concatenate(latents), clap=np.concatenate(clap))


def split_ends(values: np.ndarray, groups: np.ndarray, fraction: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the top and bottom `fraction` of values inside each group.

    Splitting inside each prompt keeps both sets balanced over prompts, so the slider
    cannot learn "this prompt" as a shortcut for "high value".
    """
    high, low = [], []
    for g in np.unique(groups):
        idx = np.nonzero((groups == g) & np.isfinite(values))[0]
        k = max(1, int(round(len(idx) * fraction)))
        order = idx[np.argsort(values[idx])]
        low += list(order[:k])
        high += list(order[-k:])
    return np.array(high), np.array(low)


def principal_directions(clap: np.ndarray, groups: np.ndarray, n: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """PCA of CLAP embeddings after removing each prompt's mean. Returns (directions, variance share)."""
    centered = clap.astype(np.float64).copy()
    for g in np.unique(groups):
        centered[groups == g] -= centered[groups == g].mean(0)
    _, sing, vt = np.linalg.svd(centered, full_matrices=False)
    share = sing**2 / (sing**2).sum()
    return vt[:n], share[:n]


def train_contrast(
    model: StableAudio,
    bank: SliderBank,
    cfg: ContrastConfig,
    high: tuple[np.ndarray, list[str]],
    low: tuple[np.ndarray, list[str]],
    log=print,
    **meta,
) -> list[dict]:
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    params = bank.add(cfg.name, rank=cfg.rank, alpha=cfg.alpha, targets=cfg.targets, **meta)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 50))
    half = cfg.batch // 2
    scale = torch.cat([torch.ones(half), -torch.ones(half)]).to(model.device)
    history, t0 = [], time.time()
    for it in range(cfg.iters):
        ih = rng.integers(0, len(high[0]), half)
        il = ih if cfg.paired else rng.integers(0, len(low[0]), half)
        x0 = torch.from_numpy(np.concatenate([high[0][ih], low[0][il]])).to(model.device, torch.float32)
        cond = model.encode([high[1][i] for i in ih] + [low[1][i] for i in il], cfg.seconds)
        drop = torch.rand(len(x0), device=model.device) < cfg.drop_text
        cond.cross = torch.where(drop.view(-1, 1, 1), torch.zeros_like(cond.cross), cond.cross)
        t = torch.rand(half, device=model.device) * 0.98 + 0.01
        eps = torch.randn(half, *x0.shape[1:], device=model.device)
        if cfg.paired:
            # Same noise and time on both members of a pair, so the two targets differ only by the edit.
            t, eps = t.repeat(2), eps.repeat(2, 1, 1)
        else:
            t = torch.cat([t, torch.rand(half, device=model.device) * 0.98 + 0.01])
            eps = torch.cat([eps, torch.randn_like(eps)])
        a, s = torch.cos(t * math.pi / 2).view(-1, 1, 1), torch.sin(t * math.pi / 2).view(-1, 1, 1)
        z, target = a * x0 + s * eps, a * eps - s * x0
        with bank.at(**{cfg.name: scale}):
            pred = model.v(z, t, cond)
        loss = torch.nn.functional.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        history.append(dict(iter=it, loss=loss.item(), time=time.time() - t0))
        if it % 100 == 0 or it == cfg.iters - 1:
            recent = np.mean([h["loss"] for h in history[-100:]])
            log(f"[{cfg.name}] {it:5d}/{cfg.iters} loss {recent:.5f} {history[-1]['time']:.0f}s")
    return history


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--by", required=True, help="descriptor:<key>[:-1] or pca:<index>")
    ap.add_argument("--fraction", type=float, default=0.3)
    ap.add_argument("--out", default="runs/sliders")
    for f, typ in [("rank", int), ("alpha", float), ("targets", str), ("lr", float), ("iters", int),
                   ("batch", int), ("seed", int)]:
        ap.add_argument(f"--{f}", type=typ, default=None)
    args = ap.parse_args()

    corpus = load_corpus(args.corpus)
    rows = corpus["rows"]
    groups = np.array([r["prompt_index"] for r in rows])
    prompts = [r["prompt"] for r in rows]
    kind, _, rest = args.by.partition(":")
    meta = dict(by=args.by, fraction=args.fraction)
    if kind == "descriptor":
        key, _, sign = rest.partition(":")
        values = np.array([r[key] for r in rows], dtype=float) * (float(sign) if sign else 1.0)
    elif kind == "pca":
        directions, share = principal_directions(corpus["clap"], groups, int(rest) + 1)
        values = corpus["clap"] @ directions[int(rest)]
        meta.update(variance_share=float(share[int(rest)]), direction=directions[int(rest)].tolist())
    else:
        raise SystemExit(f"unknown --by {args.by!r}")
    ih, il = split_ends(values, groups, args.fraction)
    over = {k: v for k, v in vars(args).items() if k in ContrastConfig.__dataclass_fields__ and v is not None}
    cfg = ContrastConfig(**over)

    model = StableAudio()
    bank = SliderBank(model.dit)
    lat = corpus["latents"]
    history = train_contrast(
        model, bank, cfg, (lat[ih], [prompts[i] for i in ih]), (lat[il], [prompts[i] for i in il]), **meta
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bank.save(cfg.name, out / f"{cfg.name}.safetensors")
    (out / f"{cfg.name}.json").write_text(json.dumps(dict(config=asdict(cfg), meta=meta, history=history)))
    print(f"saved {out / (cfg.name + '.safetensors')}")


if __name__ == "__main__":
    main()
