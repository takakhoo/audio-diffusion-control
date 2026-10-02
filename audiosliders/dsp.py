"""Signal-processing edits with a signed strength.

These serve two purposes. They are the baseline a learned slider has to beat ("why not
just apply an EQ?"), and they produce before/after pairs with known parameters for
training sliders from audio pairs. Input and output are float arrays shaped (channels, samples).
"""

from __future__ import annotations

import numpy as np


def _spectral_gain(x: np.ndarray, sr: int, gain_db) -> np.ndarray:
    spec = np.fft.rfft(x, axis=-1)
    freqs = np.fft.rfftfreq(x.shape[-1], 1 / sr)
    return np.fft.irfft(spec * 10 ** (gain_db(freqs) / 20), n=x.shape[-1], axis=-1).astype(np.float32)


def _match_rms(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Keep loudness fixed so an edit is not confounded with a level change."""
    y = y * np.sqrt((x**2).mean() / max((y**2).mean(), 1e-12))
    peak = np.abs(y).max()
    return (y / peak * 0.999 if peak > 1 else y).astype(np.float32)


def tilt_eq(x: np.ndarray, sr: int, amount: float, db_per_octave: float = 1.5) -> np.ndarray:
    """Tilt the spectrum around 1 kHz. amount = +1 adds 1.5 dB per octave upward."""

    def gain(f):
        octaves = np.log2(np.maximum(f, 20.0) / 1000.0)
        return np.clip(amount * db_per_octave * octaves, -24, 24)

    return _match_rms(_spectral_gain(x, sr, gain), x)


def low_shelf(x: np.ndarray, sr: int, amount: float, db: float = 6.0, corner: float = 150.0) -> np.ndarray:
    """Boost or cut everything below the corner. amount = +1 is +6 dB."""

    def gain(f):
        return amount * db / (1 + (np.maximum(f, 1e-3) / corner) ** 4)

    return _match_rms(_spectral_gain(x, sr, gain), x)


def mid_side(x: np.ndarray, sr: int, amount: float, db: float = 6.0) -> np.ndarray:
    """Scale the side channel. amount = +1 is +6 dB of side; large negative values approach mono."""
    if x.shape[0] != 2:
        return x
    mid, side = (x[0] + x[1]) / 2, (x[0] - x[1]) / 2
    side = side * 10 ** (amount * db / 20)
    return _match_rms(np.stack([mid + side, mid - side]), x)


def _impulse_response(sr: int, rt60: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(rt60 * sr * 1.2)
    t = np.arange(n) / sr
    ir = rng.standard_normal((2, n)) * 10 ** (-3 * t / rt60)
    ir[:, : int(0.01 * sr)] *= np.linspace(0, 1, int(0.01 * sr))
    return (ir / np.sqrt((ir**2).sum(-1, keepdims=True))).astype(np.float32)


def reverb(x: np.ndarray, sr: int, amount: float, rt60: float = 2.5) -> np.ndarray:
    """Mix in a synthetic hall tail. amount in [0, 1] sets the wet share at 0.25 per unit.

    There is no signal-processing inverse, so negative amounts return the input unchanged.
    """
    from scipy.signal import fftconvolve

    if amount <= 0:
        return x
    wet_share = min(0.25 * amount, 0.95)
    ir = _impulse_response(sr, rt60)
    wet = np.stack([fftconvolve(x[c], ir[c % 2])[: x.shape[-1]] for c in range(x.shape[0])])
    wet = wet * np.sqrt((x**2).mean() / max((wet**2).mean(), 1e-12))
    return _match_rms((1 - wet_share) * x + wet_share * wet, x)


def drive(x: np.ndarray, sr: int, amount: float) -> np.ndarray:
    """Soft clipping with pre-gain of 9 dB per unit. Negative amounts return the input."""
    if amount <= 0:
        return x
    gain = 10 ** (9 * amount / 20)
    return _match_rms(np.tanh(x * gain), x)


def lofi(x: np.ndarray, sr: int, amount: float) -> np.ndarray:
    """Band-limit and add hiss. The cutoff falls one octave per unit from 16 kHz."""
    if amount <= 0:
        return x
    cutoff = 16_000 / 2**amount

    def gain(f):
        return -24 * np.log2(np.maximum(f / cutoff, 1.0)) - 12 * np.log2(np.maximum(80 / np.maximum(f, 1.0), 1.0))

    y = _spectral_gain(x, sr, lambda f: np.maximum(gain(f), -80))
    rms = np.sqrt((x**2).mean())
    hiss = np.random.default_rng(0).standard_normal(x.shape).astype(np.float32) * rms * 0.02 * amount
    return _match_rms(y + hiss, x)


EFFECTS = dict(tilt_eq=tilt_eq, low_shelf=low_shelf, mid_side=mid_side, reverb=reverb, drive=drive, lofi=lofi)
