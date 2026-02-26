"""
Contrast Enhancement Module
------------------------------
Improve image contrast for better visibility.

Techniques:
    - Histogram equalization: spread pixel values evenly
    - CLAHE: adaptive local histogram equalization
    - Gamma correction: power-law intensity transform
    - Homomorphic filtering: separate illumination from reflectance

Especially important for UAV images captured in:
    - Low light / night conditions
    - Hazy / foggy weather
    - High dynamic range scenes

Covers Chapter 3 of the textbook.

Author: Mehmet Demir
"""

import numpy as np
import cv2
from typing import Dict


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Global histogram equalization.

    Spreads the pixel intensity histogram to cover full range.
    Result: improved contrast, especially for low-contrast images.

    Limitation: can over-enhance noise in uniform areas.

    Args:
        image: grayscale image

    Returns:
        equalized image
    """
    return cv2.equalizeHist(image)


def clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: tuple = (8, 8)
) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Like histogram equalization but works on local regions.
    clip_limit prevents over-amplification of noise.

    Much better than global HE for most real-world images
    because it preserves local contrast.

    Args:
        image: grayscale image
        clip_limit: threshold for contrast limiting
        grid_size: size of local regions

    Returns:
        enhanced image
    """
    clahe_obj = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=grid_size
    )
    return clahe_obj.apply(image)


def gamma_correction(
    image: np.ndarray,
    gamma: float = 1.0
) -> np.ndarray:
    """
    Gamma (power-law) correction.

    s = c * r^gamma

    gamma < 1: brighten dark regions (like screen brightness up)
    gamma > 1: darken bright regions
    gamma = 1: no change

    Common use: correct camera sensor response or
    enhance visibility in dark UAV footage.

    Args:
        image: input image
        gamma: gamma value

    Returns:
        corrected image
    """
    # normalize to [0, 1]
    normalized = image.astype(np.float64) / 255.0

    # apply power law
    corrected = np.power(normalized, gamma)

    # scale back to [0, 255]
    return (corrected * 255).astype(np.uint8)


def homomorphic_filter(
    image: np.ndarray,
    gamma_l: float = 0.5,
    gamma_h: float = 2.0,
    cutoff: float = 30.0
) -> np.ndarray:
    """
    Homomorphic filtering.

    Key insight: image = illumination * reflectance
    In log domain: log(image) = log(illumination) + log(reflectance)

    Illumination varies slowly (low frequency)
    Reflectance has details (high frequency)

    By filtering in log-frequency domain we can:
        - Compress illumination range (reduce gamma_l)
        - Enhance reflectance/details (boost gamma_h)

    This is incredibly useful for night vision and
    uneven lighting conditions.

    Args:
        image: grayscale image
        gamma_l: gain for low frequencies (< 1 compresses)
        gamma_h: gain for high frequencies (> 1 enhances)
        cutoff: transition frequency

    Returns:
        enhanced image
    """
    rows, cols = image.shape[:2]
    crow, ccol = rows // 2, cols // 2

    # step 1: take log
    img_log = np.log(image.astype(np.float64) + 1.0)

    # step 2: DFT
    F = np.fft.fft2(img_log)
    F_shift = np.fft.fftshift(F)

    # step 3: create homomorphic filter
    u = np.arange(rows).reshape(-1, 1) - crow
    v = np.arange(cols).reshape(1, -1) - ccol
    D = np.sqrt(u**2 + v**2)

    # filter: goes from gamma_l (low freq) to gamma_h (high freq)
    H = (gamma_h - gamma_l) * (
        1 - np.exp(-(D**2) / (2 * cutoff**2))
    ) + gamma_l

    # step 4: apply filter
    F_filtered = F_shift * H

    # step 5: inverse DFT
    f_ishift = np.fft.ifftshift(F_filtered)
    result_log = np.abs(np.fft.ifft2(f_ishift))

    # step 6: exponentiate (undo log)
    result = np.exp(result_log) - 1.0

    # normalize to 0-255
    result = result - result.min()
    if result.max() > 0:
        result = result / result.max() * 255

    return result.astype(np.uint8)


def simulate_low_light(
    image: np.ndarray,
    factor: float = 0.3
) -> np.ndarray:
    """
    Simulate low light conditions.

    Simply scale down pixel values to mimic underexposure.
    Useful for testing enhancement algorithms.

    Args:
        image: normal image
        factor: darkening factor (0.0 to 1.0)

    Returns:
        darkened image
    """
    dark = image.astype(np.float64) * factor
    return dark.astype(np.uint8)


def compare_enhancements(
    image: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Apply all enhancement methods for comparison.

    Creates a low-light version first, then enhances it
    with each method.

    Returns:
        dict of method name -> enhanced image
    """
    dark = simulate_low_light(image, factor=0.3)

    results = {
        "low_light": dark,
        "histogram_eq": histogram_equalization(dark),
        "clahe": clahe(dark),
        "gamma_0.4": gamma_correction(dark, gamma=0.4),
        "gamma_0.6": gamma_correction(dark, gamma=0.6),
        "homomorphic": homomorphic_filter(dark),
    }

    return results


# test
if __name__ == "__main__":
    from skimage import data

    image = data.camera()

    # test low light enhancement
    dark = simulate_low_light(image, 0.3)
    print(f"Dark image: min={dark.min()}, max={dark.max()}")

    enhanced = homomorphic_filter(dark)
    print(f"Homomorphic: min={enhanced.min()}, max={enhanced.max()}")

    eq = histogram_equalization(dark)
    print(f"Hist eq: min={eq.min()}, max={eq.max()}")
