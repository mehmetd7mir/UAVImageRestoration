"""
Edge Detection and Target Enhancement
-----------------------------------------
Detect and enhance edges in UAV imagery.

Methods:
    - Sobel: first derivative, good for gradual edges
    - Laplacian: second derivative, detects all edges
    - Unsharp masking: sharpen by subtracting blurred version
    - High-boost: amplified unsharp masking

Important for making targets (vehicles, buildings) more
visible in noisy or blurred UAV images.

Covers Chapter 4 and Chapter 7.

Author: Mehmet Demir
"""

import numpy as np
import cv2
from typing import Dict, List


def sobel_edge(image: np.ndarray) -> np.ndarray:
    """
    Sobel edge detection.

    Computes gradient in x and y directions separately,
    then combines them: magnitude = sqrt(Gx^2 + Gy^2)

    Good at detecting edges with direction information.

    Args:
        image: grayscale image

    Returns:
        edge magnitude image
    """
    # compute gradients
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # magnitude
    magnitude = np.sqrt(gx**2 + gy**2)

    # normalize to 0-255
    magnitude = magnitude / magnitude.max() * 255

    return magnitude.astype(np.uint8)


def laplacian_edge(image: np.ndarray) -> np.ndarray:
    """
    Laplacian edge detection.

    Second derivative - detects edges in all directions.
    More sensitive to noise than Sobel.

    Uses the kernel:
        [0  1  0]
        [1 -4  1]
        [0  1  0]

    Args:
        image: grayscale image

    Returns:
        edge image (absolute values)
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # take absolute value (laplacian gives + and - values)
    result = np.abs(laplacian)

    # normalize
    if result.max() > 0:
        result = result / result.max() * 255

    return result.astype(np.uint8)


def unsharp_mask(
    image: np.ndarray,
    sigma: float = 1.0,
    strength: float = 1.5
) -> np.ndarray:
    """
    Unsharp masking for image sharpening.

    Idea: sharp = original + strength * (original - blurred)

    The "mask" is the difference between original and blurred.
    Adding it back enhances edges and details.

    Args:
        image: input image
        sigma: Gaussian blur sigma (controls blur amount)
        strength: enhancement factor (higher = sharper)

    Returns:
        sharpened image
    """
    # create blurred version
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1

    blurred = cv2.GaussianBlur(
        image.astype(np.float64), (ksize, ksize), sigma
    )

    # unsharp mask = original - blurred
    mask = image.astype(np.float64) - blurred

    # sharpened = original + strength * mask
    sharpened = image.astype(np.float64) + strength * mask

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def high_boost_filter(
    image: np.ndarray,
    boost_factor: float = 2.0
) -> np.ndarray:
    """
    High-boost filtering.

    Generalization of unsharp masking.
    result = boost * original - blurred
           = (boost - 1) * original + (original - blurred)
           = (boost - 1) * original + high_pass

    When boost = 1, this is just high-pass filter.
    When boost > 1, original image is preserved with enhanced edges.

    Args:
        image: input image
        boost_factor: amplification factor (>= 1)

    Returns:
        enhanced image
    """
    # low-pass (blur)
    blurred = cv2.GaussianBlur(
        image.astype(np.float64), (5, 5), 1.0
    )

    # high-boost = boost * original - blurred
    result = boost_factor * image.astype(np.float64) - blurred

    return np.clip(result, 0, 255).astype(np.uint8)


def compare_at_snr_levels(
    image: np.ndarray,
    snr_levels: List[float] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compare edge detectors at different noise levels.

    Adds Gaussian noise at various sigma values and shows
    how each edge detector performs.

    Args:
        image: clean image
        snr_levels: list of noise sigma values

    Returns:
        nested dict: {snr_level: {method: edge_image}}
    """
    from src.noise.noise_generator import add_gaussian_noise

    if snr_levels is None:
        snr_levels = [10, 25, 50]

    results = {}

    for sigma in snr_levels:
        noisy = add_gaussian_noise(image, sigma=sigma)

        results[f"sigma_{sigma}"] = {
            "noisy": noisy,
            "sobel": sobel_edge(noisy),
            "laplacian": laplacian_edge(noisy),
            "unsharp": unsharp_mask(noisy, sigma=1.0, strength=1.5),
            "high_boost": high_boost_filter(noisy, boost_factor=2.0),
        }

    return results


# test
if __name__ == "__main__":
    from skimage import data

    image = data.camera()

    print("Edge detection methods:")
    print(f"  Sobel: {sobel_edge(image).shape}")
    print(f"  Laplacian: {laplacian_edge(image).shape}")
    print(f"  Unsharp: {unsharp_mask(image).shape}")
    print(f"  High-boost: {high_boost_filter(image).shape}")
