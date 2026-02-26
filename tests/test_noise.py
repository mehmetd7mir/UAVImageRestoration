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
    return data.camera()


def test_gaussian_noise_shape(test_image):
    noisy = add_gaussian_noise(test_image, sigma=25)
    assert noisy.shape == test_image.shape
    assert noisy.dtype == np.uint8


def test_salt_pepper_noise(test_image):
    noisy = add_salt_pepper(test_image, amount=0.1)
    assert np.any(noisy == 255)
    assert np.any(noisy == 0)


def test_periodic_noise(test_image):
    noisy = add_periodic_noise(test_image, frequency=30, amplitude=40)
    assert not np.array_equal(noisy, test_image)


def test_motion_blur(test_image):
    blurred = add_motion_blur(test_image, kernel_size=15)
    assert blurred.shape == test_image.shape
    # blur reduces variation
    assert np.std(blurred.astype(float)) < np.std(test_image.astype(float))


def test_psf_normalized():
    psf = get_motion_psf(15, 0)
    assert psf.shape == (15, 15)
    assert abs(psf.sum() - 1.0) < 0.01


def test_degraded_dataset(test_image):
    dataset = generate_degraded_dataset(test_image)
    expected = ["original", "gaussian", "salt_pepper",
                "periodic", "motion_blur", "heavy_noise"]
    for key in expected:
        assert key in dataset
        assert dataset[key].shape == test_image.shape
