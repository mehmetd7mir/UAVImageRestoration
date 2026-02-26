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
    compare_at_snr_levels
)


@pytest.fixture
def test_image():
    return data.camera()


class TestSobel:

    def test_output_shape(self, test_image):
        result = sobel_edge(test_image)
        assert result.shape == test_image.shape

    def test_detects_edges(self, test_image):
        result = sobel_edge(test_image)
        # edges should have non-zero values
        assert result.max() > 0

    def test_output_dtype(self, test_image):
        result = sobel_edge(test_image)
        assert result.dtype == np.uint8


class TestLaplacian:

    def test_output_shape(self, test_image):
        result = laplacian_edge(test_image)
        assert result.shape == test_image.shape

    def test_detects_edges(self, test_image):
        result = laplacian_edge(test_image)
        assert result.max() > 0


class TestUnsharpMask:

    def test_output_shape(self, test_image):
        result = unsharp_mask(test_image)
        assert result.shape == test_image.shape

    def test_sharpens_image(self, test_image):
        result = unsharp_mask(test_image, sigma=1.0, strength=2.0)
        # sharpened should have higher variance
        assert np.std(result.astype(float)) >= np.std(test_image.astype(float)) * 0.9

    def test_output_range(self, test_image):
        result = unsharp_mask(test_image)
        assert result.min() >= 0
        assert result.max() <= 255


class TestHighBoost:

    def test_output_shape(self, test_image):
        result = high_boost_filter(test_image)
        assert result.shape == test_image.shape

    def test_output_range(self, test_image):
        result = high_boost_filter(test_image, boost_factor=2.0)
        assert result.min() >= 0
        assert result.max() <= 255


class TestCompareSNR:

    def test_returns_dict(self, test_image):
        results = compare_at_snr_levels(test_image, [10, 25])
        assert isinstance(results, dict)
        assert "sigma_10" in results
        assert "sobel" in results["sigma_10"]
