"""
Tests for spatial domain filters.
"""

import pytest
import numpy as np
from skimage import data
from src.noise.noise_generator import add_salt_pepper, add_gaussian_noise
from src.spatial.spatial_filters import (
    median_filter,
    arithmetic_mean_filter,
    geometric_mean_filter,
    alpha_trimmed_mean_filter,
    compare_spatial_filters
)
from src.metrics.quality import calculate_psnr


@pytest.fixture
def test_image():
    return data.camera()


@pytest.fixture
def sp_noisy(test_image):
    return add_salt_pepper(test_image, amount=0.05)


class TestMedianFilter:

    def test_output_shape(self, sp_noisy):
        result = median_filter(sp_noisy, 3)
        assert result.shape == sp_noisy.shape

    def test_reduces_sp_noise(self, test_image, sp_noisy):
        filtered = median_filter(sp_noisy, 3)
        psnr_noisy = calculate_psnr(test_image, sp_noisy)
        psnr_filtered = calculate_psnr(test_image, filtered)
        # median should improve PSNR for salt-pepper
        assert psnr_filtered > psnr_noisy

    def test_different_kernel_sizes(self, sp_noisy):
        r3 = median_filter(sp_noisy, 3)
        r5 = median_filter(sp_noisy, 5)
        # both should produce valid output
        assert r3.dtype == np.uint8
        assert r5.dtype == np.uint8


class TestArithmeticMean:

    def test_output_shape(self, sp_noisy):
        result = arithmetic_mean_filter(sp_noisy, 3)
        assert result.shape == sp_noisy.shape

    def test_output_dtype(self, sp_noisy):
        result = arithmetic_mean_filter(sp_noisy, 3)
        assert result.dtype == np.uint8

    def test_reduces_gaussian_noise(self, test_image):
        noisy = add_gaussian_noise(test_image, sigma=25)
        filtered = arithmetic_mean_filter(noisy, 3)
        psnr_noisy = calculate_psnr(test_image, noisy)
        psnr_filtered = calculate_psnr(test_image, filtered)
        assert psnr_filtered > psnr_noisy


class TestGeometricMean:

    def test_output_shape(self, sp_noisy):
        result = geometric_mean_filter(sp_noisy, 3)
        assert result.shape == sp_noisy.shape

    def test_output_range(self, sp_noisy):
        result = geometric_mean_filter(sp_noisy, 3)
        assert result.min() >= 0
        assert result.max() <= 255


class TestAlphaTrimmedMean:

    def test_output_shape(self, sp_noisy):
        result = alpha_trimmed_mean_filter(sp_noisy, 5, alpha=2)
        assert result.shape == sp_noisy.shape

    def test_output_dtype(self, sp_noisy):
        result = alpha_trimmed_mean_filter(sp_noisy, 5, alpha=2)
        assert result.dtype == np.uint8


class TestCompare:

    def test_returns_dict(self, test_image, sp_noisy):
        results = compare_spatial_filters(test_image, sp_noisy)
        assert isinstance(results, dict)
        assert "median" in results
        assert "arithmetic_mean" in results
