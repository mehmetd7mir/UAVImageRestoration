"""
Tests for noise generator module.
"""

import pytest
import numpy as np
from skimage import data
from src.noise.noise_generator import (
    add_gaussian_noise,
    add_salt_pepper,
    add_periodic_noise,
    add_motion_blur,
    get_motion_psf,
    generate_degraded_dataset
)


@pytest.fixture
def test_image():
    """standard test image"""
    return data.camera()


class TestGaussianNoise:

    def test_output_shape(self, test_image):
        noisy = add_gaussian_noise(test_image, sigma=25)
        assert noisy.shape == test_image.shape

    def test_output_range(self, test_image):
        noisy = add_gaussian_noise(test_image, sigma=50)
        assert noisy.min() >= 0
        assert noisy.max() <= 255

    def test_noise_changes_image(self, test_image):
        noisy = add_gaussian_noise(test_image, sigma=25)
        assert not np.array_equal(noisy, test_image)

    def test_higher_sigma_more_noise(self, test_image):
        low = add_gaussian_noise(test_image, sigma=5)
        high = add_gaussian_noise(test_image, sigma=50)
        # higher sigma should produce larger differences
        diff_low = np.mean(np.abs(low.astype(float) - test_image.astype(float)))
        diff_high = np.mean(np.abs(high.astype(float) - test_image.astype(float)))
        assert diff_high > diff_low

    def test_output_dtype(self, test_image):
        noisy = add_gaussian_noise(test_image, sigma=25)
        assert noisy.dtype == np.uint8


class TestSaltPepper:

    def test_output_shape(self, test_image):
        noisy = add_salt_pepper(test_image, amount=0.05)
        assert noisy.shape == test_image.shape

    def test_has_salt(self, test_image):
        noisy = add_salt_pepper(test_image, amount=0.1)
        assert np.any(noisy == 255)

    def test_has_pepper(self, test_image):
        noisy = add_salt_pepper(test_image, amount=0.1)
        assert np.any(noisy == 0)

    def test_more_amount_more_noise(self, test_image):
        low = add_salt_pepper(test_image, amount=0.01)
        high = add_salt_pepper(test_image, amount=0.2)
        diff_low = np.sum(low != test_image)
        diff_high = np.sum(high != test_image)
        assert diff_high > diff_low


class TestPeriodicNoise:

    def test_output_shape(self, test_image):
        noisy = add_periodic_noise(test_image, frequency=30)
        assert noisy.shape == test_image.shape

    def test_adds_pattern(self, test_image):
        noisy = add_periodic_noise(test_image, frequency=30, amplitude=40)
        assert not np.array_equal(noisy, test_image)


class TestMotionBlur:

    def test_output_shape(self, test_image):
        blurred = add_motion_blur(test_image, kernel_size=15)
        assert blurred.shape == test_image.shape

    def test_blurs_image(self, test_image):
        blurred = add_motion_blur(test_image, kernel_size=15)
        # blurred image should have less variation
        assert np.std(blurred.astype(float)) < np.std(test_image.astype(float))

    def test_psf_shape(self):
        psf = get_motion_psf(15, 0)
        assert psf.shape == (15, 15)

    def test_psf_normalized(self):
        psf = get_motion_psf(15, 0)
        assert abs(psf.sum() - 1.0) < 0.01


class TestDegradedDataset:

    def test_returns_dict(self, test_image):
        dataset = generate_degraded_dataset(test_image)
        assert isinstance(dataset, dict)

    def test_has_all_types(self, test_image):
        dataset = generate_degraded_dataset(test_image)
        expected = ["original", "gaussian", "salt_pepper",
                    "periodic", "motion_blur", "heavy_noise"]
        for key in expected:
            assert key in dataset

    def test_all_same_shape(self, test_image):
        dataset = generate_degraded_dataset(test_image)
        for name, img in dataset.items():
            assert img.shape == test_image.shape
