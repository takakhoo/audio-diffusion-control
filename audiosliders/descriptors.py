"""Signal descriptors measured from the waveform. No learned model is involved.

Each slider is scored against one of these, so "the brightness slider works" means
"the spectral centroid of the output rises with the slider". All functions take
float audio shaped (channels, samples) or (samples,).
"""

from __future__ import annotations

import numpy as np

N_FFT, HOP = 2048, 512

_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x if x.ndim == 1 else x.mean(0)


def _power(mono: np.ndarray) -> np.ndarray:
    import librosa

    return np.abs(librosa.stft(mono, n_fft=N_FFT, hop_length=HOP)) ** 2


def _db(x: float) -> float:
    return float(10 * np.log10(max(x, 1e-12)))


def centroid_hz(power: np.ndarray, freqs: np.ndarray) -> float:
    """Power-weighted mean frequency of the whole clip."""
    spectrum = power.sum(1)
    return float((freqs * spectrum).sum() / max(spectrum.sum(), 1e-12))


def rolloff_hz(power: np.ndarray, freqs: np.ndarray, fraction: float = 0.95) -> float:
    cum = np.cumsum(power.sum(1))
    return float(freqs[min(np.searchsorted(cum, fraction * cum[-1]), len(freqs) - 1)])


def band_ratio_db(power: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    spectrum = power.sum(1)
    band = spectrum[(freqs >= lo) & (freqs < hi)].sum()
    return _db(band / max(spectrum.sum(), 1e-12))


def flatness(power: np.ndarray, freqs: np.ndarray) -> float:
    """Mean spectral flatness between 100 Hz and 10 kHz: 0 for pure tones, 1 for white noise."""
    p = power[(freqs >= 100) & (freqs < 10_000)] + 1e-10
    return float(np.mean(np.exp(np.log(p).mean(0)) / p.mean(0)))


def flux(power: np.ndarray) -> float:
    """Mean positive change of the log-magnitude spectrum per frame."""
    logmag = np.log1p(1000 * np.sqrt(power))
    return float(np.maximum(np.diff(logmag, axis=1), 0).mean())


def onset_features(mono: np.ndarray, sr: int) -> dict[str, float]:
    import librosa

    env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=HOP)
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=HOP, backtrack=False)
    tempo = librosa.feature.tempo(onset_envelope=env, sr=sr, hop_length=HOP)
    return dict(
        onset_rate=float(len(onsets) / (len(mono) / sr)),
        onset_strength=float(env.mean()),
        pulse_bpm=float(np.atleast_1d(tempo)[0]),
    )


def percussive_ratio(mono: np.ndarray) -> float:
    """Share of spectral energy that median filtering assigns to transients (HPSS)."""
    import librosa

    mag = np.abs(librosa.stft(mono, n_fft=N_FFT, hop_length=HOP))
    harm, perc = librosa.decompose.hpss(mag)
    return float((perc**2).sum() / max((perc**2).sum() + (harm**2).sum(), 1e-12))


def decay_s(mono: np.ndarray, sr: int) -> float:
    """Median time for the energy envelope to fall 10 dB after a local peak.

    Longer in reverberant or sustained material, shorter in dry, damped material. This is a
    proxy for reverberation and is only meaningful when comparing versions of the same clip.
    """
    hop = 256
    frames = np.lib.stride_tricks.sliding_window_view(mono, 1024)[::hop]
    env = 10 * np.log10((frames**2).mean(1) + 1e-10)
    floor = env.max() - 50
    times = []
    for i in range(1, len(env) - 1):
        if env[i] >= env[i - 1] and env[i] > env[i + 1] and env[i] > floor + 20:
            below = np.nonzero(env[i:] < env[i] - 10)[0]
            rising = np.nonzero(env[i + 1 :] > env[i])[0]
            if len(below) and (not len(rising) or below[0] <= rising[0] + 1):
                times.append(below[0] * hop / sr)
    return float(np.median(times)) if times else float("nan")


def stereo_features(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1 or x.shape[0] == 1:
        return dict(side_ratio=float("-inf"), channel_corr=1.0)
    left, right = x[0], x[1]
    mid, side = (left + right) / 2, (left - right) / 2
    corr = float(np.corrcoef(left, right)[0, 1]) if left.std() > 0 and right.std() > 0 else 1.0
    return dict(side_ratio=_db((side**2).mean() / max((mid**2).mean(), 1e-12)), channel_corr=corr)


def majorness(mono: np.ndarray, sr: int) -> float:
    """Best major-key profile correlation minus best minor-key correlation (Krumhansl profiles)."""
    import librosa

    chroma = librosa.feature.chroma_cqt(y=mono, sr=sr, hop_length=HOP).mean(1)
    best = []
    for profile in (_MAJOR, _MINOR):
        best.append(max(np.corrcoef(np.roll(profile, k), chroma)[0, 1] for k in range(12)))
    return float(best[0] - best[1])


def chroma_frames(mono: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    return librosa.feature.chroma_cqt(y=mono, sr=sr, hop_length=HOP)


def describe(x: np.ndarray, sr: int) -> dict[str, float]:
    """All scalar descriptors for one clip."""
    import librosa

    mono = _mono(x)
    power = _power(mono)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    rms = float(np.sqrt((mono**2).mean()))
    out = dict(
        centroid_hz=centroid_hz(power, freqs),
        rolloff_hz=rolloff_hz(power, freqs),
        bass_ratio=band_ratio_db(power, freqs, 0, 150),
        treble_ratio=band_ratio_db(power, freqs, 4000, sr / 2),
        flatness=flatness(power, freqs),
        flux=flux(power),
        rms_db=_db(rms**2),
        crest_db=_db(float(np.abs(mono).max()) ** 2 / max(rms**2, 1e-12)),
        percussive_ratio=percussive_ratio(mono),
        decay_s=decay_s(mono, sr),
        majorness=majorness(mono, sr),
    )
    out.update(onset_features(mono, sr))
    out.update(stereo_features(x))
    return out


def content_similarity(a: np.ndarray, b: np.ndarray, sr: int) -> dict[str, float]:
    """How much of clip a's musical content survives in clip b.

    chroma_sim: mean frame-wise cosine similarity of chroma (same notes at the same time).
    rhythm_sim: correlation of the onset-strength envelopes (same events at the same time).
    """
    import librosa

    ma, mb = _mono(a), _mono(b)
    n = min(len(ma), len(mb))
    ma, mb = ma[:n], mb[:n]
    ca, cb = chroma_frames(ma, sr), chroma_frames(mb, sr)
    cos = (ca * cb).sum(0) / np.maximum(np.linalg.norm(ca, axis=0) * np.linalg.norm(cb, axis=0), 1e-9)
    ea = librosa.onset.onset_strength(y=ma, sr=sr, hop_length=HOP)
    eb = librosa.onset.onset_strength(y=mb, sr=sr, hop_length=HOP)
    rhythm = float(np.corrcoef(ea, eb)[0, 1]) if ea.std() > 0 and eb.std() > 0 else 0.0
    return dict(chroma_sim=float(cos.mean()), rhythm_sim=rhythm)
