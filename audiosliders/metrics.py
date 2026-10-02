"""Summaries of an evaluation run: does the slider move its descriptor, and what else moves.

A run is a list of rows, one per generated clip, each with prompt_index, seed, scale and
measured values (see experiments/evaluate.py). A trajectory is one (prompt, seed) pair
rendered at every scale from the same initial noise.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


def load_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    path = path / "rows.jsonl" if path.is_dir() else path
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def trajectories(rows: Iterable[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (scales, values) with values shaped (n_trajectories, n_scales)."""
    groups: dict[tuple, dict[float, float]] = defaultdict(dict)
    for r in rows:
        groups[(r["prompt_index"], r["seed"])][r["scale"]] = r.get(key, np.nan)
    scales = sorted({s for g in groups.values() for s in g})
    values = np.array([[g.get(s, np.nan) for s in scales] for g in groups.values()], dtype=float)
    return np.array(scales), values


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    ok = np.isfinite(y)
    if ok.sum() < 3 or np.ptp(y[ok]) == 0:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).statistic)


def monotonicity(rows: Iterable[dict], key: str, sign: float = 1.0) -> dict[str, float]:
    """Rank correlation between slider scale and the descriptor, per trajectory.

    rho is the mean Spearman correlation over trajectories. `consistent` is the share of
    trajectories whose two extreme scales are ordered the intended way.
    """
    scales, values = trajectories(rows, key)
    rho = np.array([_spearman(scales, sign * v) for v in values])
    ends = sign * (values[:, -1] - values[:, 0])
    ends = ends[np.isfinite(ends)]
    return dict(
        rho=float(np.nanmean(rho)) if np.isfinite(rho).any() else np.nan,
        consistent=float((ends > 0).mean()) if len(ends) else np.nan,
        n=int(len(values)),
    )


def natural_std(rows: Iterable[dict], key: str) -> float:
    """Spread of the descriptor across unsteered clips: the yardstick for effect sizes."""
    base = np.array([r[key] for r in rows if r["scale"] == 0 and np.isfinite(r.get(key, np.nan))], dtype=float)
    return float(base.std()) if len(base) > 1 else np.nan


def response(rows: Iterable[dict], key: str) -> dict[str, np.ndarray]:
    """Mean change from the scale-0 clip at each scale, with a 95% interval over trajectories."""
    scales, values = trajectories(rows, key)
    zero = int(np.argmin(np.abs(scales)))
    delta = values - values[:, [zero]]
    n = np.maximum(np.isfinite(delta).sum(0), 1)
    return dict(
        scales=scales,
        mean=np.nanmean(delta, 0),
        ci=1.96 * np.nanstd(delta, 0) / np.sqrt(n),
        level=np.nanmean(values, 0),
    )


def slope(rows: Iterable[dict], key: str, unit: float | None = None) -> float:
    """Least-squares change in the descriptor per unit of slider scale, pooled over trajectories.

    With `unit` (usually natural_std) the slope is expressed in those units.
    """
    scales, values = trajectories(rows, key)
    zero = int(np.argmin(np.abs(scales)))
    delta = values - values[:, [zero]]
    x = np.broadcast_to(scales, delta.shape)[np.isfinite(delta)]
    y = delta[np.isfinite(delta)]
    if len(x) < 2 or np.ptp(x) == 0:
        return np.nan
    out = float((x * y).sum() / (x * x).sum())
    return out / unit if unit else out


def leakage(rows: list[dict], keys: list[str]) -> dict[str, float]:
    """Slope of every descriptor per unit scale, in units of its natural spread."""
    return {k: slope(rows, k, natural_std(rows, k) or np.nan) for k in keys}


def frechet(a: np.ndarray, b: np.ndarray) -> float:
    """Fréchet distance between Gaussians fitted to two embedding sets."""
    mu_a, mu_b = a.mean(0), b.mean(0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    # tr sqrt(cov_a cov_b) is the sum of square roots of the product's eigenvalues.
    eig = np.linalg.eigvals(cov_a @ cov_b).real
    tr_sqrt = np.sqrt(np.clip(eig, 0, None)).sum()
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(cov_a) + np.trace(cov_b) - 2 * tr_sqrt)


def kernel_distance(a: np.ndarray, b: np.ndarray, scale: float = 1000.0) -> float:
    """Unbiased squared MMD with a Gaussian kernel (the KAD estimator), times `scale`.

    The bandwidth is the median pairwise distance within the reference set b.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)

    def sq(x, y):
        return np.maximum((x * x).sum(1)[:, None] + (y * y).sum(1)[None] - 2 * x @ y.T, 0)

    dbb = sq(b, b)
    bandwidth = np.median(np.sqrt(dbb[np.triu_indices(len(b), 1)]))
    gamma = 1 / (2 * bandwidth**2)
    kaa, kbb, kab = np.exp(-gamma * sq(a, a)), np.exp(-gamma * dbb), np.exp(-gamma * sq(a, b))
    m, n = len(a), len(b)
    term_a = (kaa.sum() - np.trace(kaa)) / (m * (m - 1))
    term_b = (kbb.sum() - np.trace(kbb)) / (n * (n - 1))
    return float(scale * (term_a + term_b - 2 * kab.mean()))


def summarize(rows: list[dict], key: str | None, sign: float = 1.0) -> dict[str, float]:
    """The headline numbers for one method on one slider."""
    scales = sorted({r["scale"] for r in rows})
    ends = [r for r in rows if r["scale"] in (scales[0], scales[-1])]
    out: dict[str, float] = {}
    if key:
        mono = monotonicity(rows, key, sign)
        resp = response(rows, key)
        std = natural_std(rows, key)
        out.update(
            rho=mono["rho"],
            consistent=mono["consistent"],
            range=float(sign * (resp["level"][-1] - resp["level"][0])),
            range_in_std=float(sign * (resp["level"][-1] - resp["level"][0]) / std) if std else np.nan,
        )
    if any("clap_dir" in r for r in rows):
        mono = monotonicity(rows, "clap_dir")
        resp = response(rows, "clap_dir")
        out.update(clap_rho=mono["rho"], clap_range=float(resp["level"][-1] - resp["level"][0]))
    for k in ("clap_keep", "chroma_sim", "rhythm_sim", "clap_prompt"):
        vals = [r[k] for r in ends if k in r and np.isfinite(r[k])]
        out[f"{k}_at_ends"] = float(np.mean(vals)) if vals else np.nan
    base = [r["clap_prompt"] for r in rows if r["scale"] == 0]
    out["clap_prompt_at_zero"] = float(np.mean(base)) if base else np.nan
    return out
