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
)


@pytest.fixture
def test_image():
    return data.camera()


def test_histogram_eq_spreads(test_image):
    dark = simulate_low_light(test_image, 0.3)
    result = histogram_equalization(dark)
    assert result.max() > dark.max()


def test_clahe_enhances(test_image):
    dark = simulate_low_light(test_image, 0.3)
    result = clahe(dark)
    assert result.mean() > dark.mean()


def test_gamma_correction(test_image):
    result_bright = gamma_correction(test_image, gamma=0.4)
    result_dark = gamma_correction(test_image, gamma=2.0)
    assert result_bright.mean() > test_image.mean()
    assert result_dark.mean() < test_image.mean()


def test_homomorphic_filter(test_image):
    result = homomorphic_filter(test_image)
    assert result.shape == test_image.shape
    assert result.min() >= 0 and result.max() <= 255
