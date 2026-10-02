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
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .backbone import load_backbone
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
    backbone: str = "sao"


def load_corpus(path: str | Path) -> dict:
    """Concatenate the shards written by experiments/make_corpus.py."""
    path = Path(path)
    shards = sorted(p.stem.split("_")[1] for p in path.glob("rows_*.jsonl"))
    rows, latents, clap, muq = [], [], [], []
    for tag in shards:
        rows += [json.loads(line) for line in (path / f"rows_{tag}.jsonl").read_text().splitlines() if line]
        latents.append(np.load(path / f"latents_{tag}.npy"))
        clap.append(np.load(path / f"clap_{tag}.npy"))
        if (path / f"muq_{tag}.npy").exists():
            muq.append(np.load(path / f"muq_{tag}.npy"))
    out = dict(rows=rows, latents=np.concatenate(latents), clap=np.concatenate(clap))
    if len(muq) == len(shards):
        out["muq"] = np.concatenate(muq)
    return out


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


def independent_directions(emb: np.ndarray, groups: np.ndarray, n: int = 16, subspace: int = 32,
                           seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Independent components of the embeddings, as an alternative to principal components.

    PCA finds orthogonal directions of largest variance, which can each mix several causes.
    ICA, run inside the leading principal subspace, looks instead for directions whose
    projections are statistically independent and non-Gaussian, which tends to isolate one
    cause per direction. Returns (unit directions in embedding space, excess kurtosis of each),
    ordered from most to least heavy-tailed and signed so the heavier tail is positive.
    """
    from sklearn.decomposition import FastICA

    centered = emb.astype(np.float64).copy()
    for g in np.unique(groups):
        centered[groups == g] -= centered[groups == g].mean(0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:subspace]
    ica = FastICA(n_components=n, whiten="unit-variance", random_state=seed, max_iter=2000, tol=1e-4)
    sources = ica.fit_transform(centered @ basis.T)
    directions = ica.components_ @ basis
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    z = (sources - sources.mean(0)) / sources.std(0)
    skew, kurt = (z**3).mean(0), (z**4).mean(0) - 3
    directions *= np.where(skew < 0, -1.0, 1.0)[:, None]
    order = np.argsort(-kurt)
    return directions[order], kurt[order]


def train_contrast(
    model,
    bank: SliderBank,
    cfg: ContrastConfig,
    high: tuple[np.ndarray, list[str]],
    low: tuple[np.ndarray, list[str]],
    log=print,
    **meta,
) -> list[dict]:
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    params = bank.add(cfg.name, rank=cfg.rank, alpha=cfg.alpha, targets=cfg.targets, backbone=model.name, **meta)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 50))
    half = cfg.batch // 2
    shape = model.latent_shape(cfg.seconds)
    window = model.frames(cfg.seconds)
    time_axis = 1 + shape.index(window)
    scale = torch.cat([torch.ones(half), -torch.ones(half)]).to(model.device)
    history, t0 = [], time.time()
    for it in range(cfg.iters):
        ih = rng.integers(0, len(high[0]), half)
        il = ih if cfg.paired else rng.integers(0, len(low[0]), half)
        x0 = np.concatenate([high[0][ih], low[0][il]])
        if x0.shape[time_axis] > window:
            # Longer clips (real recordings are stored at 30 s) give a fresh crop every time they are drawn.
            start = rng.integers(0, x0.shape[time_axis] - window + 1)
            x0 = np.take(x0, np.arange(start, start + window), axis=time_axis)
        x0 = torch.from_numpy(x0).to(model.device, torch.float32)
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
        z, target = model.diffuse(x0, eps, t)
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
    ap.add_argument("--by", required=True,
                    help="descriptor:<key>[:-1], pca:<index>, ica:<index>, or tags:<tag>|<tag> (CLAP similarity to "
                         "the first tag minus the second; needs --vocab)")
    ap.add_argument("--emb", default="clap", choices=["clap", "muq"], help="embedding space for pca and ica")
    ap.add_argument("--vocab", default="runs/reference/vocab.npz")
    ap.add_argument("--max-vocal", type=float, default=None,
                    help="keep only clips whose vocal_score is below this quantile (real recordings)")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--fraction", type=float, default=0.3)
    ap.add_argument("--prompt-index", type=int, default=None, help="use only clips of this prompt")
    ap.add_argument("--out", default="runs/sliders")
    for f, typ in [("rank", int), ("alpha", float), ("targets", str), ("lr", float), ("iters", int),
                   ("batch", int), ("seed", int), ("backbone", str)]:  # --seconds is declared above
        ap.add_argument(f"--{f}", type=typ, default=None)
    args = ap.parse_args()

    corpus = load_corpus(args.corpus)
    if args.prompt_index is not None:
        keep = np.array([r["prompt_index"] == args.prompt_index for r in corpus["rows"]])
        corpus = dict(rows=[r for r, k in zip(corpus["rows"], keep) if k], latents=corpus["latents"][keep],
                      clap=corpus["clap"][keep])
    if args.max_vocal is not None:
        vocal = np.array([r.get("vocal_score", -np.inf) for r in corpus["rows"]])
        keep = vocal <= np.quantile(vocal, args.max_vocal)
        corpus = {k: ([r for r, kk in zip(v, keep) if kk] if k == "rows" else v[keep]) for k, v in corpus.items()}
    rows = corpus["rows"]
    groups = np.array([r["prompt_index"] for r in rows])
    prompts = [r["prompt"] for r in rows]
    kind, _, rest = args.by.partition(":")
    meta = dict(by=args.by, fraction=args.fraction, concept=rows[0]["prompt"] if args.prompt_index is not None else None)
    if kind == "descriptor":
        key, _, sign = rest.partition(":")
        values = np.array([r[key] for r in rows], dtype=float) * (float(sign) if sign else 1.0)
    elif kind == "pca":
        directions, share = principal_directions(corpus[args.emb], groups, int(rest) + 1)
        values = corpus[args.emb] @ directions[int(rest)]
        meta.update(variance_share=float(share[int(rest)]), emb=args.emb)
        if args.emb == "clap":
            meta["direction"] = directions[int(rest)].tolist()
    elif kind == "ica":
        directions, kurt = independent_directions(corpus[args.emb], groups)
        values = corpus[args.emb] @ directions[int(rest)]
        meta.update(kurtosis=float(kurt[int(rest)]), emb=args.emb)
        if args.emb == "clap":
            meta["direction"] = directions[int(rest)].tolist()
    elif kind == "tags":
        vocab = np.load(args.vocab)
        names = list(vocab["tags"])
        a, _, b = rest.partition("|")
        values = corpus["clap"] @ vocab["text"][names.index(a)]
        if b:
            values = values - corpus["clap"] @ vocab["text"][names.index(b)]
    else:
        raise SystemExit(f"unknown --by {args.by!r}")
    ih, il = split_ends(values, groups, args.fraction)
    over = {k: v for k, v in vars(args).items() if k in ContrastConfig.__dataclass_fields__ and v is not None}
    cfg = ContrastConfig(name=args.name, **{k: v for k, v in over.items() if k != "name"})

    model = load_backbone(cfg.backbone)
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
