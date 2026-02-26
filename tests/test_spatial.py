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
)
from src.metrics.quality import calculate_psnr


@pytest.fixture
def test_image():
    return data.camera()


def test_median_removes_salt_pepper(test_image):
    noisy = add_salt_pepper(test_image, amount=0.05)
    filtered = median_filter(noisy, 3)
    assert calculate_psnr(test_image, filtered) > calculate_psnr(test_image, noisy)


def test_arithmetic_mean_shape(test_image):
    noisy = add_gaussian_noise(test_image, sigma=25)
    result = arithmetic_mean_filter(noisy, 3)
    assert result.shape == test_image.shape
    assert result.dtype == np.uint8


def test_geometric_mean_range(test_image):
    noisy = add_gaussian_noise(test_image, sigma=25)
    result = geometric_mean_filter(noisy, 3)
    assert result.min() >= 0
    assert result.max() <= 255


def test_alpha_trimmed(test_image):
    noisy = add_salt_pepper(test_image, amount=0.05)
    result = alpha_trimmed_mean_filter(noisy, 5, alpha=2)
    assert result.shape == test_image.shape
