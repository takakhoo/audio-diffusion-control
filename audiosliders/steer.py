"""Activation steering baseline (contrastive activation addition).

For an attribute with a positive and a negative phrase, record the output of every
cross-attention block while the frozen model samples the same prompts with each phrase
appended. The steering vector of a block is the mean output with the positive phrase minus
the mean with the negative one. At inference the vector, times the slider position, is
added to that block's output at every step. No weights are trained.

The same hooks also give axes nobody named: `record` stores each clip's mean activation,
and `internal_axes` returns the principal directions along which clips of the same prompt
differ inside the model, ready to be added back with `steering`.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .backbone import seeded_noise

CROSS_ATTENTION = re.compile(r"(transformer_blocks\.\d+\.attn2|layers\.\d+\.cross_attn)$")


def _blocks(model) -> dict[str, torch.nn.Module]:
    return {name: m for name, m in model.dit.named_modules() if CROSS_ATTENTION.search(name)}


@torch.no_grad()
def collect(model, prompts: Sequence[str], positive: str, negative: str, seconds: float = 10.0,
            layers: Sequence[int] | None = None, seed: int = 0) -> dict[str, Tensor]:
    """Mean cross-attention output difference between the two ends of an attribute, per block."""
    blocks = _blocks(model)
    if layers is not None:
        blocks = {n: m for n, m in blocks.items() if int(re.search(r"\.(\d+)\.", n).group(1)) in set(layers)}
    sums = {}
    for sign, phrase in ((1.0, positive), (-1.0, negative)):
        acc = {n: [0.0, 0] for n in blocks}

        def hook(name):
            def fn(_m, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                half = h[h.shape[0] // 2 :] if h.shape[0] == 2 * len(prompts) else h  # conditional half under guidance
                acc[name][0] = acc[name][0] + half.float().mean(dim=(0, 1))
                acc[name][1] += 1
            return fn

        handles = [m.register_forward_hook(hook(n)) for n, m in blocks.items()]
        try:
            cond = model.encode([f"{p}, {phrase}" for p in prompts], seconds)
            noise = seeded_noise([seed + i for i in range(len(prompts))], model.latent_shape(seconds), model.device)
            model.sample(model.cfg(cond, model.guidance), noise, steps=model.steps)
        finally:
            for h in handles:
                h.remove()
        for n, (total, count) in acc.items():
            sums[n] = sums.get(n, 0.0) + sign * total / max(count, 1)
    return sums


@torch.no_grad()
def record(model, prompts: Sequence[str], seeds: Sequence[int], seconds: float = 10.0,
           batch: int = 8) -> tuple[np.ndarray, list[str]]:
    """Mean cross-attention output of every block for every clip, averaged over tokens and
    sampling steps. Returns an array shaped (clips, blocks, width) and the block names."""
    blocks = _blocks(model)
    names, out = list(blocks), []
    for b in range(0, len(prompts), batch):
        text, chunk = list(prompts[b : b + batch]), list(seeds[b : b + batch])
        acc = {n: [0.0, 0] for n in names}

        def hook(name):
            def fn(_m, _inp, o):
                h = o[0] if isinstance(o, tuple) else o
                half = h[h.shape[0] // 2 :] if h.shape[0] == 2 * len(text) else h
                acc[name][0] = acc[name][0] + half.float().mean(dim=1)
                acc[name][1] += 1
            return fn

        handles = [blocks[n].register_forward_hook(hook(n)) for n in names]
        try:
            cond = model.encode(text, seconds)
            noise = seeded_noise(chunk, model.latent_shape(seconds), model.device)
            model.sample(model.cfg(cond, model.guidance), noise, steps=model.steps)
        finally:
            for h in handles:
                h.remove()
        out.append(torch.stack([acc[n][0] / acc[n][1] for n in names], dim=1).cpu().numpy())
    return np.concatenate(out), names


def internal_axes(acts: np.ndarray, groups: np.ndarray, n: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Principal directions of the recorded activations after removing each prompt's mean.

    All blocks are decomposed jointly, so one axis is a set of vectors, one per block. Each
    axis is returned scaled to one standard deviation of the clips' natural spread along
    it: steering at scale s moves the mean activation by s of those. Returns
    (axes shaped (n, blocks, width), share of within-prompt variance).
    """
    flat = acts.reshape(len(acts), -1).astype(np.float64)
    for g in np.unique(groups):
        flat[groups == g] -= flat[groups == g].mean(0)
    _, sing, vt = np.linalg.svd(flat, full_matrices=False)
    spread = sing[:n] / np.sqrt(len(flat) - 1)
    axes = (vt[:n] * spread[:, None]).reshape(n, *acts.shape[1:])
    return axes.astype(np.float32), (sing**2 / (sing**2).sum())[:n]


@contextmanager
def steering(model, vectors: dict[str, Tensor], scales: Tensor, start: float = 1.0):
    """Add scale * vector to each block's output while the context is open.

    `scales` holds one value per sample. `clock` must be updated by the predictor with the
    current time so that steering can be gated like a slider; see methods.caa.
    """
    state = dict(t=1.0)
    blocks = _blocks(model)

    def hook(name):
        vec = vectors[name]

        def fn(_m, _inp, out):
            if state["t"] > start:
                return out
            h = out[0] if isinstance(out, tuple) else out
            s = scales.repeat(h.shape[0] // scales.shape[0]).view(-1, 1, 1).to(h.dtype)
            h = h + s * vec.to(h.dtype)
            return (h, *out[1:]) if isinstance(out, tuple) else h
        return fn

    handles = [blocks[n].register_forward_hook(hook(n)) for n in vectors]
    try:
        yield state
    finally:
        for h in handles:
            h.remove()
