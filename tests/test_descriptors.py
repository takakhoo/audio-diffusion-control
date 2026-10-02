import numpy as np
import pytest

from audiosliders import descriptors as D

SR = 22_050


def tone(freq, seconds=2.0, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def clicks(rate, seconds=4.0, sr=SR, decay=0.01):
    x = np.zeros(int(seconds * sr), dtype=np.float32)
    n = int(decay * 8 * sr)
    burst = np.random.default_rng(0).standard_normal(n).astype(np.float32) * np.exp(-np.arange(n) / (decay * sr))
    for start in np.arange(0.25, seconds - 0.5, 1 / rate):
        i = int(start * sr)
        x[i : i + n] += burst[: len(x) - i]
    return x


def test_centroid_tracks_pitch():
    low, high = D.describe(tone(220), SR), D.describe(tone(3520), SR)
    assert abs(low["centroid_hz"] - 220) < 30
    assert abs(high["centroid_hz"] - 3520) < 60
    assert high["rolloff_hz"] > low["rolloff_hz"]


def test_flatness_separates_noise_from_tone():
    noise = np.random.default_rng(1).standard_normal(SR * 2).astype(np.float32) * 0.1
    assert D.describe(noise, SR)["flatness"] > 0.5
    assert D.describe(tone(440), SR)["flatness"] < 0.05


def test_onset_rate_counts_clicks():
    slow, fast = D.describe(clicks(2), SR), D.describe(clicks(6), SR)
    assert abs(slow["onset_rate"] - 2) < 0.6
    assert abs(fast["onset_rate"] - 6) < 1.2
    assert fast["percussive_ratio"] > D.describe(tone(440, 4.0), SR)["percussive_ratio"]


def test_decay_grows_with_tail_length():
    short, long = clicks(1.5, decay=0.01), clicks(1.5, decay=0.08)
    assert D.decay_s(long, SR) > 2 * D.decay_s(short, SR)


def test_bass_ratio():
    assert D.describe(tone(60), SR)["bass_ratio"] > -1
    assert D.describe(tone(2000), SR)["bass_ratio"] < -30


def test_stereo_width():
    x = tone(440)
    mono = np.stack([x, x])
    wide = np.stack([x, -x])
    assert D.stereo_features(mono)["side_ratio"] < -60
    assert D.stereo_features(wide)["side_ratio"] > 60
    assert D.stereo_features(mono)["channel_corr"] == pytest.approx(1.0, abs=1e-4)


def test_majorness_sign():
    def chord(freqs):
        return sum(tone(f, 3.0) for f in freqs) / len(freqs)

    c_major = chord([261.63, 329.63, 392.00, 523.25])
    c_minor = chord([261.63, 311.13, 392.00, 523.25])
    assert D.majorness(c_major, SR) > D.majorness(c_minor, SR)


def test_content_similarity_identity_and_mismatch():
    a = clicks(3) + 0.3 * tone(440, 4.0)
    b = clicks(5) + 0.3 * tone(622, 4.0)
    same = D.content_similarity(a, a, SR)
    diff = D.content_similarity(a, b, SR)
    assert same["chroma_sim"] > 0.99 and same["rhythm_sim"] > 0.99
    assert diff["chroma_sim"] < same["chroma_sim"] and diff["rhythm_sim"] < 0.9


def test_pulse_clarity_separates_steady_beat_from_noise():
    steady = clicks(2, seconds=6.0)
    noise = np.random.default_rng(3).standard_normal(SR * 6).astype(np.float32) * 0.1
    assert D.pulse_clarity(steady, SR) > 0.5 > D.pulse_clarity(noise, SR)


def test_harmonic_change_is_higher_when_chords_move():
    held = tone(261.63, 4.0) + tone(329.63, 4.0) + tone(392.0, 4.0)
    moving = np.concatenate([tone(f, 0.5) + tone(f * 1.26, 0.5) + tone(f * 1.5, 0.5)
                             for f in (261.63, 349.23, 293.66, 392.0, 246.94, 329.63, 220.0, 311.13)])
    assert D.tonal_features(moving, SR)["harmonic_change"] > D.tonal_features(held, SR)["harmonic_change"] + 0.1
    assert D.tonal_features(held, SR)["key_clarity"] > 0.5


def test_dynamics_measures_level_variation():
    flat = tone(440, 4.0) * 0.3
    swell = flat * np.linspace(0.05, 1.0, len(flat)).astype(np.float32)
    assert D.dynamics_db(swell) > D.dynamics_db(flat) + 3
