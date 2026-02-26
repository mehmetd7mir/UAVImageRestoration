"""
Tests for quality metrics.
"""

import pytest
import numpy as np
from skimage import data
from src.metrics.quality import calculate_psnr, calculate_ssim
from src.noise.noise_generator import add_gaussian_noise


@pytest.fixture
def test_image():
    return data.camera()


def test_psnr_identical(test_image):
    assert calculate_psnr(test_image, test_image) == float('inf')


def test_psnr_noisy(test_image):
    noisy = add_gaussian_noise(test_image, sigma=25)
    psnr = calculate_psnr(test_image, noisy)
    assert 15 < psnr < 40


def test_ssim_identical(test_image):
    ssim = calculate_ssim(test_image, test_image)
    assert abs(ssim - 1.0) < 0.01


def test_ssim_noisy(test_image):
    noisy = add_gaussian_noise(test_image, sigma=25)
    ssim = calculate_ssim(test_image, noisy)
    assert 0 < ssim < 1
