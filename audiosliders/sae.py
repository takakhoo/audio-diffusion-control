"""A small top-k sparse autoencoder over clip embeddings, as a third way to discover axes.

PCA and ICA return as many axes as the subspace has dimensions. A sparse autoencoder learns
an overcomplete dictionary: several thousand features, each active on few clips, none forced
to be orthogonal to the others. A feature that fires on "solo piano with long reverb" can
coexist with one for "piano" and one for "reverb". Each feature has a decoder atom (a direction
in embedding space, used to label it) and a detector (its encoder row, used to rank clips), so it
plugs into the same set trainer as a principal direction.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TopKSAE(nn.Module):
    def __init__(self, dim: int, features: int = 2048, k: int = 16):
        super().__init__()
        self.k = k
        self.bias = nn.Parameter(torch.zeros(dim))
        self.encoder = nn.Linear(dim, features)
        self.decoder = nn.Linear(features, dim, bias=False)
        with torch.no_grad():
            self.decoder.weight.copy_(nn.functional.normalize(torch.randn(dim, features), dim=0))
            self.encoder.weight.copy_(self.decoder.weight.T)

    def pre(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x - self.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre = self.pre(x)
        top = torch.topk(pre, self.k, dim=-1)
        codes = torch.zeros_like(pre).scatter_(-1, top.indices, torch.relu(top.values))
        return self.decoder(codes) + self.bias, codes


def fit(emb: np.ndarray, features: int = 2048, k: int = 16, steps: int = 20_000, batch: int = 1024, lr: float = 1e-3,
        seed: int = 0, device: str | None = None) -> tuple[TopKSAE, dict]:
    """Train on centred, unit-scaled embeddings. Returns the model and reconstruction statistics."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    x = torch.tensor(emb, dtype=torch.float32, device=device)
    x = (x - x.mean(0)) / x.std()
    model = TopKSAE(x.shape[1], features, k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator(device=device).manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(x), (batch,), device=device, generator=gen)
        recon, _ = model(x[idx])
        loss = (recon - x[idx]).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            model.decoder.weight.copy_(nn.functional.normalize(model.decoder.weight, dim=0))
    with torch.no_grad():
        recon, codes = model(x)
        explained = 1 - ((recon - x).pow(2).sum() / x.pow(2).sum()).item()
        alive = float((codes > 0).any(0).float().mean())
        rate = (codes > 0).float().mean(0).cpu().numpy()
    return model, dict(explained=explained, alive=alive, firing_rate=rate)


def directions(model: TopKSAE) -> np.ndarray:
    """One unit direction in embedding space per feature: its decoder atom, used to label the feature."""
    w = model.decoder.weight.detach().cpu().numpy().T
    return w / np.linalg.norm(w, axis=1, keepdims=True)


def save(model: TopKSAE, emb: np.ndarray, path: str) -> None:
    """Store what is needed to score new clips on every feature."""
    np.savez(path, enc_w=model.encoder.weight.detach().cpu().numpy(), enc_b=model.encoder.bias.detach().cpu().numpy(),
             bias=model.bias.detach().cpu().numpy(), mean=emb.mean(0), std=float(emb.std()), atoms=directions(model))


def activation(path: str, emb: np.ndarray, feature: int) -> np.ndarray:
    """Pre-activation of one feature for every clip: how strongly the feature's detector responds."""
    f = np.load(path)
    x = (emb - f["mean"]) / f["std"] - f["bias"]
    return x @ f["enc_w"][feature] + f["enc_b"][feature]
