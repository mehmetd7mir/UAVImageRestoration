"""
Tests for frequency domain filters.
"""

import pytest
import numpy as np
from skimage import data
from src.noise.noise_generator import add_motion_blur, get_motion_psf, add_periodic_noise
from src.frequency.freq_filters import (
    inverse_filter,
    wiener_filter,
    notch_filter,
    butterworth_lowpass,
    gaussian_lowpass,
)


@pytest.fixture
def test_image():
    return data.camera()


def test_inverse_filter(test_image):
    psf = get_motion_psf(15, 0)
    blurred = add_motion_blur(test_image, 15, 0)
    result = inverse_filter(blurred, psf)
    assert result.shape == test_image.shape
    assert result.min() >= 0 and result.max() <= 255


def test_wiener_filter(test_image):
    psf = get_motion_psf(15, 0)
    blurred = add_motion_blur(test_image, 15, 0)
    result = wiener_filter(blurred, psf, K=0.01)
    assert result.shape == test_image.shape


def test_notch_removes_periodic(test_image):
    noisy = add_periodic_noise(test_image, frequency=30, amplitude=40)
    result = notch_filter(noisy, frequency=30, radius=10)
    diff_noisy = np.mean(np.abs(noisy.astype(float) - test_image.astype(float)))
    diff_filtered = np.mean(np.abs(result.astype(float) - test_image.astype(float)))
    assert diff_filtered < diff_noisy


def test_butterworth_smooths(test_image):
    result = butterworth_lowpass(test_image, cutoff=20)
    assert np.std(result.astype(float)) < np.std(test_image.astype(float))


def test_gaussian_lowpass(test_image):
    result = gaussian_lowpass(test_image, cutoff=30)
    assert result.shape == test_image.shape
