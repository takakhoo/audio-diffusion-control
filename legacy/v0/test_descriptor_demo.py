import unittest
import numpy as np
from descriptor_demo import render, descriptors, fit_pca, project


class DescriptorTests(unittest.TestCase):
    def test_render_is_deterministic_and_unclipped(self):
        a = render(0.8, 4)
        np.testing.assert_array_equal(a, render(0.8, 4))
        self.assertLess(np.max(abs(a)), 1)
        self.assertGreater(np.std(a), 0.01)

    def test_brightness_is_monotonic_without_changing_onset_rate(self):
        measurements = np.array([descriptors(render(b, 4)) for b in np.linspace(0, 1, 5)])
        self.assertTrue(np.all(np.diff(measurements[:, 0]) > 0))
        np.testing.assert_allclose(measurements[:, 1], 4)

    def test_note_rate_measured_not_copied_from_settings(self):
        for rate in [2, 3, 4, 6, 8]:
            self.assertAlmostEqual(descriptors(render(0.5, rate))[1], rate)
        np.testing.assert_array_equal(descriptors(np.zeros(16000)), 0)

    def test_pca_held_out_projection_and_full_rank_roundtrip(self):
        rng = np.random.default_rng(7)
        X, held = rng.normal(size=(20, 3)), rng.normal(size=(5, 3))
        pca = fit_pca(X, rank=3)
        restored = project(held, pca) @ pca["components"] * pca["scale"] + pca["mean"]
        np.testing.assert_allclose(restored, held, atol=1e-12)
        self.assertAlmostEqual(pca["variance_ratio"].sum(), 1)

    def test_invalid_and_constant_features(self):
        with self.assertRaises(ValueError):
            render(2, 4)
        with self.assertRaises(ValueError):
            fit_pca([[1, 2, 3]])
        self.assertTrue(np.isfinite(project(np.ones((4, 3)), fit_pca(np.ones((4, 3))))).all())


if __name__ == "__main__":
    unittest.main()
