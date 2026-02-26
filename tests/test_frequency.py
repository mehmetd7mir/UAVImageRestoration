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
    butterworth_highpass,
    compare_freq_filters
)


@pytest.fixture
def test_image():
    return data.camera()


class TestInverseFilter:

    def test_output_shape(self, test_image):
        psf = get_motion_psf(15, 0)
        blurred = add_motion_blur(test_image, 15, 0)
        result = inverse_filter(blurred, psf)
        assert result.shape == test_image.shape

    def test_output_range(self, test_image):
        psf = get_motion_psf(15, 0)
        blurred = add_motion_blur(test_image, 15, 0)
        result = inverse_filter(blurred, psf)
        assert result.min() >= 0
        assert result.max() <= 255


class TestWienerFilter:

    def test_output_shape(self, test_image):
        psf = get_motion_psf(15, 0)
        blurred = add_motion_blur(test_image, 15, 0)
        result = wiener_filter(blurred, psf, K=0.01)
        assert result.shape == test_image.shape

    def test_output_dtype(self, test_image):
        psf = get_motion_psf(15, 0)
        blurred = add_motion_blur(test_image, 15, 0)
        result = wiener_filter(blurred, psf, K=0.01)
        assert result.dtype == np.uint8


class TestNotchFilter:

    def test_output_shape(self, test_image):
        noisy = add_periodic_noise(test_image, frequency=30)
        result = notch_filter(noisy, frequency=30, radius=10)
        assert result.shape == test_image.shape

    def test_removes_pattern(self, test_image):
        noisy = add_periodic_noise(test_image, frequency=30, amplitude=40)
        result = notch_filter(noisy, frequency=30, radius=10)
        # filtered should be closer to original than noisy
        diff_noisy = np.mean(np.abs(noisy.astype(float) - test_image.astype(float)))
        diff_filtered = np.mean(np.abs(result.astype(float) - test_image.astype(float)))
        assert diff_filtered < diff_noisy


class TestButterworthLP:

    def test_output_shape(self, test_image):
        result = butterworth_lowpass(test_image, cutoff=30)
        assert result.shape == test_image.shape

    def test_smooths_image(self, test_image):
        result = butterworth_lowpass(test_image, cutoff=20)
        assert np.std(result.astype(float)) < np.std(test_image.astype(float))


class TestGaussianLP:

    def test_output_shape(self, test_image):
        result = gaussian_lowpass(test_image, cutoff=30)
        assert result.shape == test_image.shape


class TestButterworthHP:

    def test_output_shape(self, test_image):
        result = butterworth_highpass(test_image, cutoff=30)
        assert result.shape == test_image.shape


class TestCompare:

    def test_returns_dict(self, test_image):
        blurred = add_motion_blur(test_image, 15, 0)
        results = compare_freq_filters(test_image, blurred)
        assert isinstance(results, dict)
        assert "wiener" in results
