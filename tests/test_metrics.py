import numpy as np
import pytest

from audiosliders import metrics as M


def make_rows(effect, noise=0.0, n_prompts=6, seeds=3, scales=(-2, -1, 0, 1, 2), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_prompts):
        for s in range(seeds):
            base = rng.normal(1000, 200)
            other = rng.normal(5, 1)
            for x in scales:
                rows.append(dict(prompt_index=p, seed=100 * p + s, scale=float(x),
                                 target=base + effect * x + rng.normal(0, noise),
                                 other=other + rng.normal(0, 0.01)))
    return rows


def test_perfect_slider_is_monotone_and_sized_in_natural_units():
    rows = make_rows(effect=100)
    mono = M.monotonicity(rows, "target")
    assert mono["rho"] == pytest.approx(1.0) and mono["consistent"] == 1.0 and mono["n"] == 18
    assert M.slope(rows, "target") == pytest.approx(100)
    std = M.natural_std(rows, "target")
    assert 100 < std < 300
    assert M.slope(rows, "target", std) == pytest.approx(100 / std)


def test_sign_flips_the_verdict():
    rows = make_rows(effect=-50)
    assert M.monotonicity(rows, "target")["rho"] == pytest.approx(-1.0)
    assert M.monotonicity(rows, "target", sign=-1)["rho"] == pytest.approx(1.0)


def test_dead_slider_scores_near_zero():
    rows = make_rows(effect=0, noise=20)
    mono = M.monotonicity(rows, "target")
    assert abs(mono["rho"]) < 0.3 and 0.2 < mono["consistent"] < 0.8


def test_response_is_relative_to_scale_zero():
    resp = M.response(make_rows(effect=10), "target")
    assert np.allclose(resp["mean"], [-20, -10, 0, 10, 20])
    assert np.allclose(resp["ci"], 0)


def test_leakage_separates_target_from_bystander():
    leak = M.leakage(make_rows(effect=100), ["target", "other"])
    assert leak["target"] > 0.3 and abs(leak["other"]) < 0.05


def test_distances_are_zero_for_same_distribution_and_grow_with_shift():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(400, 8)), rng.normal(size=(400, 8))
    shifted = b + 1.0
    assert M.frechet(a, b) < 0.3 < M.frechet(a, shifted)
    assert abs(M.kernel_distance(a, b)) < 5 < M.kernel_distance(a, shifted)


def test_summarize_reports_headline_numbers():
    rows = make_rows(effect=100)
    for r in rows:
        r.update(clap_dir=0.01 * r["scale"], clap_keep=1 - 0.1 * abs(r["scale"]), chroma_sim=0.9,
                 rhythm_sim=0.8, clap_prompt=0.4)
    out = M.summarize(rows, "target")
    assert out["rho"] == pytest.approx(1.0) and out["range"] == pytest.approx(400)
    assert out["clap_range"] == pytest.approx(0.04) and out["clap_keep_at_ends"] == pytest.approx(0.8)


def test_usable_span_stops_where_quality_drops():
    rows = make_rows(effect=100, scales=(-3, -2, -1, 0, 1, 2, 3))
    quality = {-3: 4.0, -2: 6.8, -1: 7.0, 0: 7.1, 1: 6.9, 2: 6.0, 3: 6.9}
    for r in rows:
        r.update(ce=quality[r["scale"]], clap_keep=1 - 0.1 * abs(r["scale"]), clap_dir=0.01 * r["scale"])
    assert M.usable_span(rows) == (-2.0, 1.0)
    out = M.summarize(rows, "target")
    assert (out["usable_lo"], out["usable_hi"]) == (-2.0, 1.0)
    assert out["usable_range_in_std"] == pytest.approx(300 / M.natural_std(rows, "target"))
    assert out["usable_clap_range"] == pytest.approx(0.03)
    assert out["usable_clap_keep"] == pytest.approx(0.85)


def test_intervals_shrink_with_more_trajectories_and_cover_the_truth():
    few, many = make_rows(effect=100, noise=60, n_prompts=4), make_rows(effect=100, noise=60, n_prompts=40)
    a, b = M.end_to_end(few, "target"), M.end_to_end(many, "target")
    assert b["ci"] < a["ci"]
    assert abs(b["mean"] - 400) < b["ci"] * 1.5
    assert M.monotonicity(many, "target")["rho_ci"] < M.monotonicity(few, "target")["rho_ci"]


def test_selectivity_and_monotone_share():
    clean = make_rows(effect=100)
    assert M.selectivity(clean, "target", ["target", "other"]) > 20
    assert M.monotone_share(clean, "target") == 1.0
    leaky = make_rows(effect=100)
    for r in leaky:
        r["other"] += 0.5 * r["scale"]
    assert M.selectivity(leaky, "target", ["target", "other"]) < M.selectivity(clean, "target", ["target", "other"])
    assert M.monotone_share(make_rows(effect=0, noise=20), "target") < 0.3
