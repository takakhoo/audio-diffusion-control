import numpy as np
import pytest

pytest.importorskip("torch")

from audiosliders.contrast import independent_directions, principal_directions, split_ends


def test_split_ends_is_balanced_within_groups():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(5), 20)
    # Each group has a different offset; a global split would pick whole groups.
    values = rng.normal(size=100) + groups * 10
    high, low = split_ends(values, groups, fraction=0.25)
    assert len(high) == len(low) == 25
    for g in range(5):
        assert (groups[high] == g).sum() == (groups[low] == g).sum() == 5
        assert values[high][groups[high] == g].min() > values[low][groups[low] == g].max()
    assert not set(high) & set(low)


def test_split_ends_skips_missing_values():
    values = np.array([1.0, np.nan, 3.0, 2.0, np.nan, 0.0])
    high, low = split_ends(values, np.zeros(6, dtype=int), fraction=0.5)
    assert set(high) == {2, 3} and set(low) == {0, 5}


def test_principal_directions_ignore_between_prompt_offsets():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(4), 50)
    axis = np.zeros(16)
    axis[3] = 1.0
    offsets = rng.normal(size=(4, 16)) * 20
    data = offsets[groups] + rng.normal(size=(200, 1)) * 3 * axis + rng.normal(size=(200, 16)) * 0.1
    directions, share = principal_directions(data, groups, n=2)
    assert abs(directions[0] @ axis) > 0.99
    assert share[0] > 0.9


def test_independent_directions_unmix_what_pca_mixes():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(2)
    n, dim = 4000, 24
    # Two independent heavy-tailed causes of equal strength, each along its own non-orthogonal axis.
    a, b = np.zeros(dim), np.zeros(dim)
    a[0], b[0], b[1] = 1.0, 0.6, 0.8
    sources = rng.laplace(size=(n, 2))
    data = sources[:, :1] * a + sources[:, 1:] * b + rng.normal(size=(n, dim)) * 0.05
    groups = np.zeros(n, dtype=int)
    ica, kurt = independent_directions(data, groups, n=2, subspace=2)
    pca, _ = principal_directions(data, groups, n=2)

    def purity(directions):
        proj = data @ directions.T
        corr = np.abs(np.corrcoef(proj.T, sources.T)[:2, 2:])
        return corr.max(1).min()

    assert purity(ica) > 0.97
    assert purity(pca) < 0.9
    assert (kurt > 1).all()
