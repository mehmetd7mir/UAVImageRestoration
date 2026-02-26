"""
Tests for contrast enhancement module.
"""

import pytest
import numpy as np
from skimage import data
from src.enhancement.contrast import (
    histogram_equalization,
    clahe,
    gamma_correction,
    homomorphic_filter,
    simulate_low_light,
    compare_enhancements
)


@pytest.fixture
def test_image():
    return data.camera()


class TestHistogramEqualization:

    def test_output_shape(self, test_image):
        result = histogram_equalization(test_image)
        assert result.shape == test_image.shape

    def test_spreads_histogram(self, test_image):
        # create low contrast image
        dark = simulate_low_light(test_image, 0.3)
        result = histogram_equalization(dark)
        # result should have wider range
        assert result.max() > dark.max()


class TestCLAHE:

    def test_output_shape(self, test_image):
        result = clahe(test_image)
        assert result.shape == test_image.shape

    def test_enhances_contrast(self, test_image):
        dark = simulate_low_light(test_image, 0.3)
        result = clahe(dark)
        assert result.mean() > dark.mean()


class TestGammaCorrection:

    def test_gamma_1_no_change(self, test_image):
        result = gamma_correction(test_image, gamma=1.0)
        # should be very close to original
        diff = np.mean(np.abs(result.astype(float) - test_image.astype(float)))
        assert diff < 1.0

    def test_gamma_less_brightens(self, test_image):
        dark = simulate_low_light(test_image, 0.3)
        result = gamma_correction(dark, gamma=0.4)
        assert result.mean() > dark.mean()

    def test_gamma_more_darkens(self, test_image):
        result = gamma_correction(test_image, gamma=2.0)
        assert result.mean() < test_image.mean()

    def test_output_range(self, test_image):
        result = gamma_correction(test_image, gamma=0.5)
        assert result.min() >= 0
        assert result.max() <= 255


class TestHomomorphicFilter:

    def test_output_shape(self, test_image):
        result = homomorphic_filter(test_image)
        assert result.shape == test_image.shape

    def test_output_range(self, test_image):
        result = homomorphic_filter(test_image)
        assert result.min() >= 0
        assert result.max() <= 255


class TestSimulateLowLight:

    def test_darkens_image(self, test_image):
        dark = simulate_low_light(test_image, 0.3)
        assert dark.mean() < test_image.mean()

    def test_factor_zero(self, test_image):
        dark = simulate_low_light(test_image, 0.0)
        assert dark.max() == 0


class TestCompare:

    def test_returns_dict(self, test_image):
        results = compare_enhancements(test_image)
        assert isinstance(results, dict)
        assert "histogram_eq" in results
        assert "homomorphic" in results
