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

from .backbone import Cond, load_backbone
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
    symmetry: float = 1.0
    weight_decay: float = 0.0
    seed: int = 0
    backbone: str = "sao"


def load_corpus(path: str | Path, latents: bool = True) -> dict:
    """Concatenate the shards written by experiments/make_corpus.py."""
    path = Path(path)
    shards = sorted(p.stem.split("_")[1] for p in path.glob("rows_*.jsonl"))
    rows, lat, clap, muq = [], [], [], []
    for tag in shards:
        rows += [json.loads(line) for line in (path / f"rows_{tag}.jsonl").read_text().splitlines() if line]
        if latents:
            lat.append(np.load(path / f"latents_{tag}.npy"))
        clap.append(np.load(path / f"clap_{tag}.npy"))
        if (path / f"muq_{tag}.npy").exists():
            muq.append(np.load(path / f"muq_{tag}.npy"))
    out = dict(rows=rows, clap=np.concatenate(clap))
    if latents:
        out["latents"] = np.concatenate(lat)
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


def residualize(values: np.ndarray, others: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Remove from `values` what a linear fit on `others` explains, inside each group.

    Sorting clips by the residual gives high and low sets that differ in the target but are
    balanced on the other measurements, so a slider trained between them has no reason to
    move those as well.
    """
    out = values.astype(np.float64).copy()
    for g in np.unique(groups):
        m = (groups == g) & np.isfinite(values)
        if m.sum() < others.shape[1] + 3:
            continue
        x = np.nan_to_num(others[m] - np.nanmean(others[m], 0))
        x = np.c_[x / (x.std(0) + 1e-9), np.ones(m.sum())]
        coef, *_ = np.linalg.lstsq(x, out[m], rcond=None)
        out[m] = out[m] - x @ coef
    return out


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


def axis_stability(emb: np.ndarray, groups: np.ndarray, method: str = "pca", n: int = 8, seed: int = 0) -> float:
    """How reproducible a set of discovered axes is: fit on two random halves of the corpus,
    match axes one to one, and return the mean absolute cosine of the matched pairs."""
    from scipy.optimize import linear_sum_assignment

    order = np.random.default_rng(seed).permutation(len(emb))
    halves = []
    for idx in (order[: len(order) // 2], order[len(order) // 2 :]):
        fn = independent_directions if method == "ica" else principal_directions
        d = fn(emb[idx], groups[idx], n)[0]
        halves.append(d / np.linalg.norm(d, axis=1, keepdims=True))
    cos = np.abs(halves[0] @ halves[1].T)
    rows, cols = linear_sum_assignment(-cos)
    return float(cos[rows, cols].mean())


def _centered(emb: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = emb.astype(np.float64).copy()
    for g in np.unique(groups):
        out[groups == g] -= out[groups == g].mean(0)
    return out


def axis_coverage(real: np.ndarray, real_groups: np.ndarray, generated: np.ndarray, generated_groups: np.ndarray,
                  directions: np.ndarray) -> dict:
    """How much of each axis found in real music a model's output spans.

    `total` is the spread of all generated clips along an axis, as a fraction of the spread
    of real recordings. `within` is the same after removing each prompt's (for real music,
    each genre's) mean, so it measures what changing only the seed can reach. `offset` is
    where the average generated clip sits on the axis, in standard deviations of real music.
    """
    d = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    pr, pg = real.astype(np.float64) @ d.T, generated.astype(np.float64) @ d.T
    wr, wg = _centered(real, real_groups) @ d.T, _centered(generated, generated_groups) @ d.T
    return dict(total=pg.std(0) / pr.std(0), within=wg.std(0) / wr.std(0), offset=(pg.mean(0) - pr.mean(0)) / pr.std(0),
                real_std=pr.std(0), real_within_std=wr.std(0), generated_within_std=wg.std(0))


def subspace_overlap(a: np.ndarray, b: np.ndarray, k: int = 8) -> float:
    """Share of the variance in the top-k principal subspace of `a` that the top-k subspace
    of `b` also contains: the mean squared cosine of the principal angles between them."""
    va = np.linalg.svd(a - a.mean(0), full_matrices=False)[2][:k]
    vb = np.linalg.svd(b - b.mean(0), full_matrices=False)[2][:k]
    return float((np.linalg.svd(va @ vb.T, compute_uv=False) ** 2).mean())


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
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
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
        if cfg.symmetry > 0:
            # Evaluate every sample at +1 and at -1. Each sample is fitted at its own sign, and the
            # two predictions are asked to average to the frozen model's. The wanted effect of a
            # slider is odd in its position; the part that is even is damage, and this removes it.
            both = Cond.cat([cond, cond])
            with bank.at(**{cfg.name: torch.cat([torch.ones_like(scale), -torch.ones_like(scale)])}):
                plus, minus = model.v(torch.cat([z, z]), torch.cat([t, t]), both).chunk(2)
            with torch.no_grad():
                base = model.v(z, t, cond)
            own = torch.where(scale.view(-1, 1, 1) > 0, plus, minus)
            fit = torch.nn.functional.mse_loss(own, target)
            even = torch.nn.functional.mse_loss(0.5 * (plus + minus), base)
            loss = fit + cfg.symmetry * even
        else:
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
                    help="descriptor:<key>[:-1], pca:<index>, ica:<index>, or tags:<tag>,<tag> (CLAP similarity to "
                         "the first tag minus the second; needs --vocab)")
    ap.add_argument("--emb", default="clap", choices=["clap", "muq"], help="embedding space for pca and ica")
    ap.add_argument("--vocab", default="runs/reference/vocab.npz")
    ap.add_argument("--max-vocal", type=float, default=None,
                    help="keep only clips whose vocal_score is below this quantile (real recordings)")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--balance", default=None,
                    help="comma-separated measurements to balance between the two sets, e.g. rms_db,centroid_hz,ce")
    ap.add_argument("--fraction", type=float, default=0.3)
    ap.add_argument("--prompt-index", type=int, default=None, help="use only clips of this prompt")
    ap.add_argument("--out", default="runs/sliders")
    for f, typ in [("rank", int), ("alpha", float), ("targets", str), ("lr", float), ("iters", int),
                   ("batch", int), ("seed", int), ("backbone", str), ("symmetry", float),
                   ("weight_decay", float)]:  # --seconds is declared above
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
    elif kind == "direction":
        # A direction saved by experiments/discover.py, so the axis stays the same as the corpus grows.
        path, _, index = rest.rpartition(":")
        direction = np.load(path)[int(index)]
        values = corpus[args.emb] @ direction
        meta.update(emb=args.emb, source=path, index=int(index))
        if args.emb == "clap":
            meta["direction"] = direction.tolist()
    elif kind == "sae":
        from .sae import activation

        path, _, index = rest.rpartition(":")
        values = activation(path, corpus[args.emb], int(index))
        meta.update(emb=args.emb, source=path, index=int(index))
        if args.emb == "clap":
            meta["direction"] = np.load(path)["atoms"][int(index)].tolist()
    elif kind == "tags":
        vocab = np.load(args.vocab)
        names = list(vocab["tags"])
        a, _, b = rest.replace("|", ",").partition(",")
        values = corpus["clap"] @ vocab["text"][names.index(a)]
        if b:
            values = values - corpus["clap"] @ vocab["text"][names.index(b)]
    else:
        raise SystemExit(f"unknown --by {args.by!r}")
    if args.balance:
        keys = [k for k in args.balance.split(",") if k and k != rest.split(":")[0]]
        others = np.array([[r.get(k, np.nan) for k in keys] for r in rows], dtype=float)
        values = residualize(np.asarray(values, dtype=float), others, groups)
        meta["balance"] = keys
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
