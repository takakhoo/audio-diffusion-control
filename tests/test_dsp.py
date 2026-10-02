import numpy as np
import pytest

from audiosliders import descriptors as D
from audiosliders import dsp

SR = 22_050


@pytest.fixture(scope="module")
def clip():
    rng = np.random.default_rng(0)
    left = rng.standard_normal(SR * 3).astype(np.float32) * 0.1
    right = 0.8 * left + 0.2 * rng.standard_normal(SR * 3).astype(np.float32) * 0.1
    return np.stack([left, right])


def rms(x):
    return float(np.sqrt((x**2).mean()))


@pytest.mark.parametrize("name", list(dsp.EFFECTS))
def test_zero_amount_changes_nothing_audible(clip, name):
    out = dsp.EFFECTS[name](clip, SR, 0.0)
    assert out.shape == clip.shape
    assert np.allclose(out, clip, atol=1e-4)


def test_tilt_moves_centroid_both_ways_at_constant_loudness(clip):
    lo, mid, hi = (D.describe(dsp.tilt_eq(clip, SR, a), SR) for a in (-2, 0, 2))
    assert lo["centroid_hz"] < mid["centroid_hz"] < hi["centroid_hz"]
    assert abs(lo["rms_db"] - mid["rms_db"]) < 0.5 and abs(hi["rms_db"] - mid["rms_db"]) < 0.5


def test_low_shelf_moves_bass_ratio(clip):
    lo, mid, hi = (D.describe(dsp.low_shelf(clip, SR, a), SR)["bass_ratio"] for a in (-2, 0, 2))
    assert lo < mid < hi and hi - lo > 15


def test_mid_side_moves_width_by_the_stated_amount(clip):
    base = D.stereo_features(clip)["side_ratio"]
    assert D.stereo_features(dsp.mid_side(clip, SR, 1.0))["side_ratio"] == pytest.approx(base + 6, abs=0.2)
    assert D.stereo_features(dsp.mid_side(clip, SR, -1.0))["side_ratio"] == pytest.approx(base - 6, abs=0.2)


def test_one_sided_effects_return_input_for_negative_amounts(clip):
    for name in ("reverb", "drive", "lofi"):
        assert dsp.EFFECTS[name](clip, SR, -1.0) is clip


def test_reverb_lengthens_decay():
    x = np.zeros((2, SR * 3), dtype=np.float32)
    burst = np.random.default_rng(1).standard_normal(400).astype(np.float32) * np.hanning(400)
    for start in (0.3, 1.2, 2.1):
        i = int(start * SR)
        x[:, i : i + 400] = burst
    assert D.decay_s(dsp.reverb(x, SR, 3.0)[0], SR) > 3 * D.decay_s(x[0], SR)


def test_lofi_lowers_rolloff(clip):
    assert D.describe(dsp.lofi(clip, SR, 2.0), SR)["rolloff_hz"] < 0.6 * D.describe(clip, SR)["rolloff_hz"]
