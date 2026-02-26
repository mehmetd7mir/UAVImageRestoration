"""
Noise Generator Module
-----------------------
Add different types of noise and degradation to images.

This simulates what actually happens to UAV camera images:
    - Gaussian noise from sensor electronics
    - Salt-and-pepper noise from transmission errors
    - Periodic noise from electronic interference
    - Motion blur from aircraft movement

We use these to create a synthetic degraded dataset for testing
our restoration algorithms.

Author: Mehmet Demir
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional


def add_gaussian_noise(
    image: np.ndarray,
    mean: float = 0.0,
    sigma: float = 25.0
) -> np.ndarray:
    """
    Add Gaussian noise to image.

    This is the most common type of noise in camera sensors.
    The noise follows a normal distribution.

    Args:
        image: input grayscale image (0-255)
        mean: mean of the noise (usually 0)
        sigma: standard deviation (higher = more noise)

    Returns:
        noisy image clipped to [0, 255]
    """
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = image.astype(np.float64) + noise

    # clip to valid range and convert back
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper(
    image: np.ndarray,
    amount: float = 0.05
) -> np.ndarray:
    """
    Add salt-and-pepper noise.

    Random pixels become either pure white (salt) or
    pure black (pepper). This simulates bit errors in
    data transmission or dead pixels.

    Args:
        image: input image
        amount: fraction of pixels to corrupt (0.0 to 1.0)

    Returns:
        noisy image
    """
    noisy = image.copy()
    total_pixels = image.size

    # salt (white) pixels
    num_salt = int(total_pixels * amount / 2)
    salt_coords = tuple(
        np.random.randint(0, dim, num_salt)
        for dim in image.shape
    )
    noisy[salt_coords] = 255

    # pepper (black) pixels
    num_pepper = int(total_pixels * amount / 2)
    pepper_coords = tuple(
        np.random.randint(0, dim, num_pepper)
        for dim in image.shape
    )
    noisy[pepper_coords] = 0

    return noisy


def add_periodic_noise(
    image: np.ndarray,
    frequency: float = 30.0,
    angle: float = 0.0,
    amplitude: float = 40.0
) -> np.ndarray:
    """
    Add periodic sinusoidal noise.

    Creates a striped pattern on the image, simulating
    electronic interference from nearby equipment.
    This kind of noise shows up as impulses in the
    frequency domain.

    Args:
        image: input image
        frequency: spatial frequency of the pattern
        angle: angle of the stripes in degrees
        amplitude: strength of the noise

    Returns:
        noisy image
    """
    rows, cols = image.shape[:2]

    # create coordinate grids
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)

    # convert angle to radians
    theta = np.radians(angle)

    # sinusoidal pattern along the rotated axis
    pattern = amplitude * np.sin(
        2 * np.pi * frequency * (X * np.cos(theta) + Y * np.sin(theta))
        / max(rows, cols)
    )

    noisy = image.astype(np.float64) + pattern
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_motion_blur(
    image: np.ndarray,
    kernel_size: int = 15,
    angle: float = 0.0
) -> np.ndarray:
    """
    Add motion blur using PSF convolution.

    Simulates camera shake or rapid aircraft movement.
    The blur direction is controlled by the angle parameter.

    The PSF (Point Spread Function) is a line kernel rotated
    to the desired angle. Convolving with this kernel creates
    the blur effect.

    Args:
        image: input image
        kernel_size: length of the motion blur kernel
        angle: direction of blur in degrees

    Returns:
        blurred image
    """
    # create motion blur kernel (a line)
    kernel = np.zeros((kernel_size, kernel_size))

    # draw a line in the center of the kernel
    center = kernel_size // 2
    kernel[center, :] = 1.0

    # rotate the kernel to desired angle
    M = cv2.getRotationMatrix2D(
        (center, center), angle, 1.0
    )
    kernel = cv2.warpAffine(
        kernel, M, (kernel_size, kernel_size)
    )

    # normalize so sum = 1
    kernel = kernel / kernel.sum()

    # apply convolution
    blurred = cv2.filter2D(image, -1, kernel)

    return blurred


def get_motion_psf(
    kernel_size: int = 15,
    angle: float = 0.0
) -> np.ndarray:
    """
    Get the Point Spread Function for motion blur.

    Returns the kernel separately so we can use it later
    for deconvolution/inverse filtering.

    Returns:
        PSF kernel (normalized)
    """
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    kernel[center, :] = 1.0

    M = cv2.getRotationMatrix2D(
        (center, center), angle, 1.0
    )
    kernel = cv2.warpAffine(
        kernel, M, (kernel_size, kernel_size)
    )
    kernel = kernel / kernel.sum()

    return kernel


def generate_degraded_dataset(
    image: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Apply all noise types to create a full degraded dataset.

    Takes a clean image and returns a dictionary with all
    the different degraded versions. Useful for comparing
    restoration methods side by side.

    Args:
        image: clean input image

    Returns:
        dict with keys: 'gaussian', 'salt_pepper', 'periodic',
                        'motion_blur', 'heavy_noise'
    """
    dataset = {
        "original": image.copy(),
        "gaussian": add_gaussian_noise(image, sigma=25),
        "salt_pepper": add_salt_pepper(image, amount=0.05),
        "periodic": add_periodic_noise(image, frequency=30),
        "motion_blur": add_motion_blur(image, kernel_size=15),
        # combination: gaussian + salt-pepper (worst case)
        "heavy_noise": add_salt_pepper(
            add_gaussian_noise(image, sigma=30), amount=0.03
        ),
    }

    return dataset


# quick test
if __name__ == "__main__":
    from skimage import data

    # use built-in test image
    image = data.camera()  # 512x512 grayscale
    print(f"Original image: {image.shape}, dtype: {image.dtype}")

    dataset = generate_degraded_dataset(image)
    for name, img in dataset.items():
        print(f"  {name}: shape={img.shape}, min={img.min()}, max={img.max()}")
