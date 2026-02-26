"""
Frequency Domain Filters
--------------------------
Restore images using frequency domain techniques.

The key idea: transform image to frequency domain with DFT,
apply filter there, then transform back.

Techniques:
    - Inverse filtering: undo known blur (but amplifies noise)
    - Wiener filtering: smarter inverse that accounts for noise
    - Notch filter: remove specific frequencies (periodic noise)
    - Butterworth LP/HP: smooth or sharpen with tunable roll-off
    - Gaussian LP/HP: smooth or sharpen without ringing

This covers Chapters 6, 7, and 9 from the textbook.

Author: Mehmet Demir
"""

import numpy as np
import cv2
from typing import Tuple, Optional


def _to_frequency_domain(image: np.ndarray) -> np.ndarray:
    """Convert image to centered frequency domain."""
    f = np.fft.fft2(image.astype(np.float64))
    return np.fft.fftshift(f)


def _to_spatial_domain(freq: np.ndarray) -> np.ndarray:
    """Convert frequency domain back to spatial."""
    f = np.fft.ifftshift(freq)
    result = np.fft.ifft2(f)
    return np.abs(result)


def inverse_filter(
    degraded: np.ndarray,
    psf: np.ndarray,
    epsilon: float = 1e-3
) -> np.ndarray:
    """
    Inverse filtering for motion blur removal.

    Theory: if G = H * F (degraded = blur * original),
    then F = G / H (original = degraded / blur transfer function)

    Problem: where H is small, noise gets amplified badly.
    That's why we add epsilon to avoid division by zero.

    Args:
        degraded: blurred image
        psf: point spread function (blur kernel)
        epsilon: small value to prevent division by zero

    Returns:
        restored image (may have artifacts)
    """
    rows, cols = degraded.shape[:2]

    # pad PSF to image size
    psf_padded = np.zeros_like(degraded, dtype=np.float64)
    kh, kw = psf.shape
    psf_padded[:kh, :kw] = psf

    # transform both to frequency domain
    G = np.fft.fft2(degraded.astype(np.float64))
    H = np.fft.fft2(psf_padded)

    # inverse filter: F_hat = G / H
    # add epsilon where H is too small
    H_safe = np.where(np.abs(H) > epsilon, H, epsilon)
    F_hat = G / H_safe

    # back to spatial domain
    result = np.abs(np.fft.ifft2(F_hat))

    return np.clip(result, 0, 255).astype(np.uint8)


def wiener_filter(
    degraded: np.ndarray,
    psf: np.ndarray,
    K: float = 0.01
) -> np.ndarray:
    """
    Wiener filtering for image restoration.

    Smarter than inverse filter because it considers noise.
    The K parameter controls the noise-to-signal ratio estimate.

    Formula: F_hat = (1/H) * (|H|^2 / (|H|^2 + K)) * G

    Larger K = more noise suppression but more blur.
    Smaller K = sharper result but more noise.

    Args:
        degraded: degraded image
        psf: blur kernel
        K: noise-to-signal ratio (try 0.001 to 0.1)

    Returns:
        restored image
    """
    rows, cols = degraded.shape[:2]

    # pad PSF
    psf_padded = np.zeros_like(degraded, dtype=np.float64)
    kh, kw = psf.shape
    psf_padded[:kh, :kw] = psf

    G = np.fft.fft2(degraded.astype(np.float64))
    H = np.fft.fft2(psf_padded)

    # Wiener formula
    H_conj = np.conj(H)
    H_sq = np.abs(H) ** 2

    F_hat = (H_conj / (H_sq + K)) * G

    result = np.abs(np.fft.ifft2(F_hat))

    return np.clip(result, 0, 255).astype(np.uint8)


def notch_filter(
    image: np.ndarray,
    frequency: float = 30.0,
    radius: float = 10.0,
    angle: float = 0.0
) -> np.ndarray:
    """
    Notch filter for periodic noise removal.

    Identifies and removes specific frequency components
    that correspond to the periodic interference pattern.

    Works by creating a filter mask that blocks the
    noise frequencies while leaving everything else.

    Args:
        image: image with periodic noise
        frequency: frequency of the noise pattern
        radius: size of the notch (how much to block)
        angle: orientation of the noise

    Returns:
        filtered image
    """
    rows, cols = image.shape[:2]
    crow, ccol = rows // 2, cols // 2

    # go to frequency domain
    F = _to_frequency_domain(image)

    # create notch filter mask (start with all ones)
    mask = np.ones((rows, cols), dtype=np.float64)

    # calculate notch positions based on noise frequency
    theta = np.radians(angle)
    freq_x = frequency * np.cos(theta) * cols / max(rows, cols)
    freq_y = frequency * np.sin(theta) * rows / max(rows, cols)

    # block the noise frequency and its symmetric pair
    for dx, dy in [(freq_x, freq_y), (-freq_x, -freq_y)]:
        u0 = int(crow + dy)
        v0 = int(ccol + dx)

        # create circular notch
        for u in range(rows):
            for v in range(cols):
                dist = np.sqrt((u - u0)**2 + (v - v0)**2)
                if dist < radius:
                    mask[u, v] = 0.0

    # apply mask
    F_filtered = F * mask

    result = _to_spatial_domain(F_filtered)
    return np.clip(result, 0, 255).astype(np.uint8)


def butterworth_lowpass(
    image: np.ndarray,
    cutoff: float = 30.0,
    order: int = 2
) -> np.ndarray:
    """
    Butterworth low-pass filter.

    Smoother roll-off than ideal LP filter.
    Higher order = sharper cutoff but more ringing.

    H(u,v) = 1 / (1 + (D/D0)^(2n))

    Args:
        image: input image
        cutoff: cutoff frequency D0
        order: filter order n

    Returns:
        filtered image
    """
    rows, cols = image.shape[:2]
    crow, ccol = rows // 2, cols // 2

    F = _to_frequency_domain(image)

    # create filter
    u = np.arange(rows).reshape(-1, 1) - crow
    v = np.arange(cols).reshape(1, -1) - ccol
    D = np.sqrt(u**2 + v**2)

    # Butterworth transfer function
    H = 1.0 / (1.0 + (D / cutoff) ** (2 * order))

    result = _to_spatial_domain(F * H)
    return np.clip(result, 0, 255).astype(np.uint8)


def gaussian_lowpass(
    image: np.ndarray,
    cutoff: float = 30.0
) -> np.ndarray:
    """
    Gaussian low-pass filter.

    No ringing artifacts (unlike Butterworth).
    Smoother but less sharp transition.

    H(u,v) = exp(-D^2 / (2 * D0^2))

    Args:
        image: input image
        cutoff: cutoff frequency D0

    Returns:
        filtered image
    """
    rows, cols = image.shape[:2]
    crow, ccol = rows // 2, cols // 2

    F = _to_frequency_domain(image)

    u = np.arange(rows).reshape(-1, 1) - crow
    v = np.arange(cols).reshape(1, -1) - ccol
    D = np.sqrt(u**2 + v**2)

    H = np.exp(-(D**2) / (2 * cutoff**2))

    result = _to_spatial_domain(F * H)
    return np.clip(result, 0, 255).astype(np.uint8)


def butterworth_highpass(
    image: np.ndarray,
    cutoff: float = 30.0,
    order: int = 2
) -> np.ndarray:
    """
    Butterworth high-pass filter.

    Passes high frequencies (edges, details) and blocks
    low frequencies (smooth areas).

    H_hp = 1 - H_lp

    Args:
        image: input image
        cutoff: cutoff frequency
        order: filter order

    Returns:
        filtered image
    """
    rows, cols = image.shape[:2]
    crow, ccol = rows // 2, cols // 2

    F = _to_frequency_domain(image)

    u = np.arange(rows).reshape(-1, 1) - crow
    v = np.arange(cols).reshape(1, -1) - ccol
    D = np.sqrt(u**2 + v**2)

    # avoid division by zero at center
    D[crow, ccol] = 1e-10

    H_lp = 1.0 / (1.0 + (D / cutoff) ** (2 * order))
    H_hp = 1.0 - H_lp

    result = _to_spatial_domain(F * H_hp)
    return np.clip(result, 0, 255).astype(np.uint8)


def compare_freq_filters(
    original: np.ndarray,
    degraded: np.ndarray
) -> dict:
    """
    Apply all frequency domain filters for comparison.

    Returns:
        dict of filter name -> filtered image
    """
    from src.noise.noise_generator import get_motion_psf

    psf = get_motion_psf(15, 0)

    results = {
        "butterworth_lp": butterworth_lowpass(degraded, cutoff=50),
        "gaussian_lp": gaussian_lowpass(degraded, cutoff=50),
        "butterworth_hp": butterworth_highpass(degraded, cutoff=30),
        "inverse": inverse_filter(degraded, psf),
        "wiener": wiener_filter(degraded, psf, K=0.01),
    }

    return results


# test
if __name__ == "__main__":
    from skimage import data
    from src.noise.noise_generator import add_motion_blur, add_periodic_noise

    image = data.camera()

    # test motion blur restoration
    blurred = add_motion_blur(image, kernel_size=15, angle=0)
    print("Testing frequency domain filters...")

    lp = butterworth_lowpass(image, cutoff=50)
    print(f"  Butterworth LP: {lp.shape}")

    hp = butterworth_highpass(image, cutoff=30)
    print(f"  Butterworth HP: {hp.shape}")

    glp = gaussian_lowpass(image, cutoff=50)
    print(f"  Gaussian LP: {glp.shape}")
