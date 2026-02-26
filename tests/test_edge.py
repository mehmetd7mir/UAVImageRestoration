"""
Tests for edge detection module.
"""

import pytest
import numpy as np
from skimage import data
from src.edge.edge_detection import (
    sobel_edge,
    laplacian_edge,
    unsharp_mask,
    high_boost_filter,
)


@pytest.fixture
def test_image():
    return data.camera()


def test_sobel(test_image):
    result = sobel_edge(test_image)
    assert result.shape == test_image.shape
    assert result.max() > 0


def test_laplacian(test_image):
    result = laplacian_edge(test_image)
    assert result.shape == test_image.shape
    assert result.max() > 0


def test_unsharp_mask(test_image):
    result = unsharp_mask(test_image, sigma=1.0, strength=1.5)
    assert result.min() >= 0 and result.max() <= 255


def test_high_boost(test_image):
    result = high_boost_filter(test_image, boost_factor=2.0)
    assert result.min() >= 0 and result.max() <= 255
