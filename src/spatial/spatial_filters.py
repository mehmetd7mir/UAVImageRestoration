"""
Spatial Domain Filters
-----------------------
Image restoration filters that work directly on pixel values.

Different filters work best for different noise types:
    - Median filter: best for salt-and-pepper noise
    - Arithmetic mean: good for Gaussian noise
    - Geometric mean: preserves detail better than arithmetic
    - Alpha-trimmed mean: handles mixed/heavy noise

These are all from Chapter 4 and 8 of the textbook.

Author: Mehmet Demir
"""

import numpy as np
import cv2
from typing import Dict, Tuple
from scipy.ndimage import generic_filter


def median_filter(
    image: np.ndarray,
    kernel_size: int = 3
) -> np.ndarray:
    """
    Apply median filter.

    Replaces each pixel with the median of its neighborhood.
    Excellent for salt-and-pepper noise because extreme values
    (0 and 255) are unlikely to be the median.

    Args:
        image: noisy grayscale image
        kernel_size: size of the window (must be odd)

    Returns:
        filtered image
    """
    return cv2.medianBlur(image, kernel_size)


def arithmetic_mean_filter(
    image: np.ndarray,
    kernel_size: int = 3
) -> np.ndarray:
    """
    Apply arithmetic mean (averaging) filter.

    Replaces each pixel with the average of its neighborhood.
    Good for Gaussian noise but tends to blur edges.

    f_hat(x,y) = (1/mn) * sum of g(s,t) over window

    Args:
        image: noisy image
        kernel_size: window size

    Returns:
        filtered image
    """
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64)
    kernel /= kernel.size

    result = cv2.filter2D(image.astype(np.float64), -1, kernel)
    return np.clip(result, 0, 255).astype(np.uint8)


def geometric_mean_filter(
    image: np.ndarray,
    kernel_size: int = 3
) -> np.ndarray:
    """
    Apply geometric mean filter.

    Uses the product of pixel values instead of sum.
    Better at preserving detail compared to arithmetic mean
    while still reducing noise.

    f_hat(x,y) = (product of g(s,t))^(1/mn)

    We compute in log domain to avoid numerical overflow.

    Args:
        image: noisy image
        kernel_size: window size

    Returns:
        filtered image
    """
    # work in log domain to avoid overflow
    # log of product = sum of logs
    img_float = image.astype(np.float64)

    # avoid log(0) by adding small epsilon
    img_log = np.log(img_float + 1e-10)

    # average the log values (this equals log of geometric mean)
    kernel = np.ones((kernel_size, kernel_size))
    kernel /= kernel.size

    mean_log = cv2.filter2D(img_log, -1, kernel)

    # exponentiate to get geometric mean
    result = np.exp(mean_log)

    return np.clip(result, 0, 255).astype(np.uint8)


def alpha_trimmed_mean_filter(
    image: np.ndarray,
    kernel_size: int = 5,
    alpha: int = 2
) -> np.ndarray:
    """
    Apply alpha-trimmed mean filter.

    Sort pixels in the window, remove 'alpha' smallest and
    'alpha' largest values, then take the mean of what's left.

    This is a compromise between mean and median filters.
    Good for images with multiple noise types.

    Args:
        image: noisy image
        kernel_size: window size
        alpha: number of values to trim from each end

    Returns:
        filtered image
    """
    def alpha_trimmed(values):
        # sort, remove extremes, take mean
        sorted_vals = np.sort(values)
        if 2 * alpha >= len(sorted_vals):
            # if alpha too big, just take median
            return np.median(sorted_vals)
        trimmed = sorted_vals[alpha:len(sorted_vals) - alpha]
        return np.mean(trimmed)

    result = generic_filter(
        image.astype(np.float64),
        alpha_trimmed,
        size=kernel_size
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def compare_spatial_filters(
    original: np.ndarray,
    noisy: np.ndarray,
    kernel_size: int = 3
) -> Dict[str, np.ndarray]:
    """
    Apply all spatial filters and return results.

    Useful for side-by-side comparison of filter performance.

    Returns:
        dict with filter name -> filtered image
    """
    results = {
        "median": median_filter(noisy, kernel_size),
        "arithmetic_mean": arithmetic_mean_filter(noisy, kernel_size),
        "geometric_mean": geometric_mean_filter(noisy, kernel_size),
        "alpha_trimmed": alpha_trimmed_mean_filter(
            noisy, kernel_size, alpha=2
        ),
    }

    return results


# test
if __name__ == "__main__":
    from skimage import data
    from src.noise.noise_generator import add_salt_pepper, add_gaussian_noise

    image = data.camera()

    # test with salt-pepper noise
    noisy = add_salt_pepper(image, amount=0.05)
    results = compare_spatial_filters(image, noisy)

    for name, filtered in results.items():
        print(f"{name}: shape={filtered.shape}")
