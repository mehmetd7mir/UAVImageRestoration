"""
Tests for quality metrics module.
"""

import pytest
import numpy as np
from skimage import data
from src.metrics.quality import (
    calculate_psnr,
    calculate_ssim,
    measure_time,
    evaluate_filter
)
from src.noise.noise_generator import add_gaussian_noise
from src.spatial.spatial_filters import median_filter


@pytest.fixture
def test_image():
    return data.camera()


class TestPSNR:

    def test_identical_images(self, test_image):
        psnr = calculate_psnr(test_image, test_image)
        assert psnr == float('inf')

    def test_noisy_lower_psnr(self, test_image):
        noisy = add_gaussian_noise(test_image, sigma=25)
        psnr = calculate_psnr(test_image, noisy)
        assert 15 < psnr < 40  # reasonable range

    def test_more_noise_lower_psnr(self, test_image):
        low = add_gaussian_noise(test_image, sigma=10)
        high = add_gaussian_noise(test_image, sigma=50)
        assert calculate_psnr(test_image, low) > calculate_psnr(test_image, high)

    def test_returns_float(self, test_image):
        noisy = add_gaussian_noise(test_image, sigma=25)
        psnr = calculate_psnr(test_image, noisy)
        assert isinstance(psnr, float)


class TestSSIM:

    def test_identical_images(self, test_image):
        ssim = calculate_ssim(test_image, test_image)
        assert abs(ssim - 1.0) < 0.01

    def test_noisy_lower_ssim(self, test_image):
        noisy = add_gaussian_noise(test_image, sigma=25)
        ssim = calculate_ssim(test_image, noisy)
        assert 0 < ssim < 1

    def test_returns_float(self, test_image):
        noisy = add_gaussian_noise(test_image, sigma=25)
        ssim = calculate_ssim(test_image, noisy)
        assert isinstance(ssim, float)


class TestMeasureTime:

    def test_returns_result_and_time(self):
        result, elapsed = measure_time(np.zeros, (100, 100))
        assert result.shape == (100, 100)
        assert elapsed >= 0

    def test_timing_reasonable(self, test_image):
        noisy = add_gaussian_noise(test_image, sigma=25)
        result, elapsed = measure_time(median_filter, noisy, 3)
        assert elapsed < 5000  # should be less than 5 seconds


class TestEvaluateFilter:

    def test_returns_dict(self, test_image):
        noisy = add_gaussian_noise(test_image, sigma=25)
        filtered = median_filter(noisy, 3)
        result = evaluate_filter(test_image, noisy, filtered, "median")
        assert "psnr_filtered" in result
        assert "ssim_filtered" in result
        assert "psnr_improvement" in result
