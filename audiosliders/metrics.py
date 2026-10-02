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
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    beats = path.with_name("beat.jsonl")
    if beats.exists():
        for r, line in zip(rows, beats.read_text().splitlines()):
            r.setdefault("beat_bpm", json.loads(line)["beat_bpm"])
    extra = path.with_name("muq.jsonl")
    if extra.exists():
        # Scores from MuQ-MuLan, written by experiments/muq_score.py in the same order as the rows.
        for r, line in zip(rows, extra.read_text().splitlines()):
            m = json.loads(line)
            r.update(muq_dir=m["muq_pos"] - m["muq_neg"], muq_prompt=m["muq_prompt"])
            if "muq_axis" in m:
                r["muq_axis"] = m["muq_axis"]
    for r in rows:
        for hz, octv in (("centroid_hz", "centroid_oct"), ("rolloff_hz", "rolloff_oct")):
            if hz in r and octv not in r:
                r[octv] = float(np.log2(max(r[hz], 1.0) / 440.0))
    return rows


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
    ok = rho[np.isfinite(rho)]
    return dict(
        rho=float(ok.mean()) if len(ok) else np.nan,
        rho_ci=float(1.96 * ok.std(ddof=1) / np.sqrt(len(ok))) if len(ok) > 1 else np.nan,
        consistent=float((ends > 0).mean()) if len(ends) else np.nan,
        n=int(len(values)),
    )


def end_to_end(rows: Iterable[dict], key: str, sign: float = 1.0, unit: float | None = None) -> dict[str, float]:
    """Mean change in the descriptor from the lowest to the highest position, with a 95% interval over trajectories."""
    _, values = trajectories(rows, key)
    diff = sign * (values[:, -1] - values[:, 0])
    diff = diff[np.isfinite(diff)] / (unit or 1.0)
    if len(diff) < 2:
        return dict(mean=np.nan, ci=np.nan)
    return dict(mean=float(diff.mean()), ci=float(1.96 * diff.std(ddof=1) / np.sqrt(len(diff))))


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


def selectivity(rows: list[dict], key: str, others: list[str], sign: float = 1.0) -> float:
    """How much a slider moves its own descriptor relative to everything else.

    The slope of the target descriptor divided by the mean absolute slope of the other
    descriptors, all in units of their spread across unsteered clips. 1 means the slider moves
    its target no more than it moves an average bystander.
    """
    own = sign * slope(rows, key, natural_std(rows, key) or np.nan)
    drift = [abs(slope(rows, k, natural_std(rows, k) or np.nan)) for k in others if k != key]
    drift = [d for d in drift if np.isfinite(d)]
    return float(own / np.mean(drift)) if drift and np.mean(drift) > 0 else np.nan


def monotone_share(rows: Iterable[dict], key: str, sign: float = 1.0, threshold: float = 0.8) -> float:
    """Share of trajectories whose rank correlation with position exceeds the threshold."""
    scales, values = trajectories(rows, key)
    rho = np.array([_spearman(scales, sign * v) for v in values])
    rho = rho[np.isfinite(rho)]
    return float((rho > threshold).mean()) if len(rho) else np.nan


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


def usable_span(rows: Iterable[dict], quality: str = "ce", tolerance: float = 0.5) -> tuple[float, float]:
    """The slider positions over which mean quality stays within `tolerance` of the unsteered clips.

    Walks outward from zero in each direction and stops at the first position that falls
    below. Returns (lowest usable scale, highest usable scale).
    """
    r = response(rows, quality)
    scales, level = r["scales"], r["level"]
    zero = int(np.argmin(np.abs(scales)))
    floor = level[zero] - tolerance
    lo = hi = zero
    while hi + 1 < len(scales) and level[hi + 1] >= floor:
        hi += 1
    while lo - 1 >= 0 and level[lo - 1] >= floor:
        lo -= 1
    return float(scales[lo]), float(scales[hi])


def within(rows: Iterable[dict], lo: float, hi: float) -> list[dict]:
    return [r for r in rows if lo <= r["scale"] <= hi]


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
            rho_ci=mono["rho_ci"],
            consistent=mono["consistent"],
            range=float(sign * (resp["level"][-1] - resp["level"][0])),
            range_in_std=float(sign * (resp["level"][-1] - resp["level"][0]) / std) if std else np.nan,
        )
    if any("clap_dir" in r for r in rows):
        mono = monotonicity(rows, "clap_dir")
        resp = response(rows, "clap_dir")
        out.update(clap_rho=mono["rho"], clap_range=float(resp["level"][-1] - resp["level"][0]))
    if any("muq_dir" in r for r in rows):
        mono = monotonicity(rows, "muq_dir")
        out.update(muq_rho=mono["rho"], muq_rho_ci=mono["rho_ci"], muq_ordered=mono["consistent"])
    for k in ("clap_keep", "chroma_sim", "rhythm_sim", "clap_prompt", "ce", "pq"):
        vals = [r[k] for r in ends if k in r and np.isfinite(r[k])]
        out[f"{k}_at_ends"] = float(np.mean(vals)) if vals else np.nan
    for k in ("clap_prompt", "ce", "pq"):
        base = [r[k] for r in rows if r["scale"] == 0 and k in r]
        out[f"{k}_at_zero"] = float(np.mean(base)) if base else np.nan
    if any("ce" in r for r in rows):
        # The same questions asked only over the positions where the output still scores as music.
        lo, hi = usable_span(rows)
        out.update(usable_lo=lo, usable_hi=hi)
        inside = within(rows, lo, hi)
        if key and hi > lo:
            std = natural_std(rows, key)
            moved = end_to_end(inside, key, sign, std) if std else dict(mean=np.nan, ci=np.nan)
            out["usable_range_in_std"], out["usable_range_ci"] = moved["mean"], moved["ci"]
        if hi > lo and any("clap_dir" in r for r in rows):
            resp = response(inside, "clap_dir")
            out["usable_clap_range"] = float(resp["level"][-1] - resp["level"][0])
        ends = [r for r in inside if r["scale"] in (lo, hi) and r["scale"] != 0]
        vals = [r["clap_keep"] for r in ends if "clap_keep" in r]
        out["usable_clap_keep"] = float(np.mean(vals)) if vals else np.nan
    return out
