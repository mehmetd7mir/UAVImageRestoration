"""
Image Quality Metrics
-----------------------
Measure how well our restoration worked.

Metrics:
    - PSNR: Peak Signal-to-Noise Ratio (higher = better)
    - SSIM: Structural Similarity Index (closer to 1 = better)

These are standard metrics in image processing papers.
We also have a timing utility to measure processing speed.

Author: Mehmet Demir
"""

import numpy as np
import time
from typing import Callable, Any, Tuple
from skimage.metrics import structural_similarity


def calculate_psnr(
    original: np.ndarray,
    processed: np.ndarray
) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio.

    PSNR = 10 * log10(MAX^2 / MSE)

    Higher PSNR = processed image is closer to original.
    Typical values:
        30-40 dB: good quality
        20-30 dB: acceptable
        < 20 dB: poor quality

    Args:
        original: reference image
        processed: restored/filtered image

    Returns:
        PSNR value in dB
    """
    original = original.astype(np.float64)
    processed = processed.astype(np.float64)

    mse = np.mean((original - processed) ** 2)

    if mse == 0:
        return float('inf')  # identical images

    max_pixel = 255.0
    psnr = 10 * np.log10(max_pixel**2 / mse)

    return round(psnr, 2)


def calculate_ssim(
    original: np.ndarray,
    processed: np.ndarray
) -> float:
    """
    Calculate Structural Similarity Index.

    SSIM considers luminance, contrast, and structure.
    More perceptually relevant than PSNR.

    Range: -1 to 1 (1 = identical)

    Args:
        original: reference image
        processed: compared image

    Returns:
        SSIM value
    """
    # handle both 2D and 3D images
    if original.ndim == 3:
        channel_axis = 2
    else:
        channel_axis = None

    score = structural_similarity(
        original, processed,
        data_range=255,
        channel_axis=channel_axis
    )

    return round(score, 4)


def measure_time(
    func: Callable,
    *args: Any,
    **kwargs: Any
) -> Tuple[Any, float]:
    """
    Measure execution time of a function.

    Returns:
        (result, elapsed_time_ms)
    """
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = (time.time() - start) * 1000  # to ms

    return result, round(elapsed, 2)


def evaluate_filter(
    original: np.ndarray,
    noisy: np.ndarray,
    filtered: np.ndarray,
    filter_name: str = ""
) -> dict:
    """
    Full evaluation of a filter result.

    Computes PSNR and SSIM for both the noisy image
    and the filtered result, so you can see improvement.

    Returns:
        dict with metrics
    """
    return {
        "filter": filter_name,
        "psnr_noisy": calculate_psnr(original, noisy),
        "psnr_filtered": calculate_psnr(original, filtered),
        "ssim_noisy": calculate_ssim(original, noisy),
        "ssim_filtered": calculate_ssim(original, filtered),
        "psnr_improvement": round(
            calculate_psnr(original, filtered) -
            calculate_psnr(original, noisy), 2
        ),
    }


# test
if __name__ == "__main__":
    from skimage import data
    from src.noise.noise_generator import add_gaussian_noise
    from src.spatial.spatial_filters import median_filter

    image = data.camera()
    noisy = add_gaussian_noise(image, sigma=25)
    filtered = median_filter(noisy, 3)

    psnr = calculate_psnr(image, filtered)
    ssim = calculate_ssim(image, filtered)
    print(f"PSNR: {psnr} dB")
    print(f"SSIM: {ssim}")

    result = evaluate_filter(image, noisy, filtered, "median_3x3")
    for k, v in result.items():
        print(f"  {k}: {v}")
