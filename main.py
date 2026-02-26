"""
UAV Image Restoration Pipeline
---------------------------------
Main entry point for the degraded UAV imagery
restoration and target enhancement system.

Usage:
    python main.py --demo
    python main.py --module noise
    python main.py --benchmark

Author: Mehmet Demir
"""

import argparse
import sys
import numpy as np


def get_test_image():
    """Load a test image from scikit-image."""
    from skimage import data
    return data.camera()  # 512x512 grayscale


def demo_noise(image):
    """Demonstrate noise generation."""
    from src.noise.noise_generator import generate_degraded_dataset

    print("\n[Module 1] Noise Modeling")
    print("-" * 40)

    dataset = generate_degraded_dataset(image)
    for name, img in dataset.items():
        print(f"  {name:15s} -> min={img.min():3d}, max={img.max():3d}, "
              f"mean={img.mean():.1f}")


def demo_spatial(image):
    """Demonstrate spatial filters."""
    from src.noise.noise_generator import add_salt_pepper
    from src.spatial.spatial_filters import compare_spatial_filters
    from src.metrics.quality import calculate_psnr, calculate_ssim

    print("\n[Module 2] Spatial Domain Restoration")
    print("-" * 40)

    noisy = add_salt_pepper(image, amount=0.05)
    print(f"  Noisy (S&P):     PSNR={calculate_psnr(image, noisy):.1f} dB")

    results = compare_spatial_filters(image, noisy)
    for name, filtered in results.items():
        psnr = calculate_psnr(image, filtered)
        ssim = calculate_ssim(image, filtered)
        print(f"  {name:18s} PSNR={psnr:.1f} dB, SSIM={ssim:.4f}")


def demo_frequency(image):
    """Demonstrate frequency domain filters."""
    from src.noise.noise_generator import add_motion_blur, get_motion_psf
    from src.frequency.freq_filters import (
        inverse_filter, wiener_filter,
        butterworth_lowpass, gaussian_lowpass
    )
    from src.metrics.quality import calculate_psnr

    print("\n[Module 3] Frequency Domain Restoration")
    print("-" * 40)

    blurred = add_motion_blur(image, kernel_size=15, angle=0)
    psf = get_motion_psf(15, 0)
    print(f"  Motion blurred:  PSNR={calculate_psnr(image, blurred):.1f} dB")

    inv_result = inverse_filter(blurred, psf)
    print(f"  Inverse filter:  PSNR={calculate_psnr(image, inv_result):.1f} dB")

    wiener_result = wiener_filter(blurred, psf, K=0.01)
    print(f"  Wiener filter:   PSNR={calculate_psnr(image, wiener_result):.1f} dB")

    bw_result = butterworth_lowpass(image, cutoff=50)
    print(f"  Butterworth LP:  shape={bw_result.shape}")

    g_result = gaussian_lowpass(image, cutoff=50)
    print(f"  Gaussian LP:     shape={g_result.shape}")


def demo_enhancement(image):
    """Demonstrate contrast enhancement."""
    from src.enhancement.contrast import compare_enhancements
    from src.metrics.quality import calculate_psnr

    print("\n[Module 4] Contrast Enhancement")
    print("-" * 40)

    results = compare_enhancements(image)
    for name, enhanced in results.items():
        if name == "low_light":
            print(f"  {name:18s} mean={enhanced.mean():.1f}")
        else:
            psnr = calculate_psnr(image, enhanced)
            print(f"  {name:18s} PSNR={psnr:.1f} dB, mean={enhanced.mean():.1f}")


def demo_edge(image):
    """Demonstrate edge detection."""
    from src.edge.edge_detection import (
        sobel_edge, laplacian_edge,
        unsharp_mask, high_boost_filter
    )

    print("\n[Module 5] Edge & Target Enhancement")
    print("-" * 40)

    methods = {
        "Sobel": sobel_edge(image),
        "Laplacian": laplacian_edge(image),
        "Unsharp Mask": unsharp_mask(image),
        "High-Boost": high_boost_filter(image),
    }

    for name, result in methods.items():
        print(f"  {name:18s} range=[{result.min()}, {result.max()}], "
              f"mean={result.mean():.1f}")


def demo_benchmark(image):
    """Run full benchmark."""
    from src.benchmark.dashboard import (
        run_full_benchmark,
        generate_metrics_table,
        save_results
    )

    print("\n[Module 6] Benchmark Dashboard")
    print("-" * 40)

    results = run_full_benchmark(image)
    save_results(results)

    print("\nMetrics Table:")
    print(generate_metrics_table(results))


def main():
    parser = argparse.ArgumentParser(
        description="UAV Image Restoration Pipeline"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run full demonstration"
    )
    parser.add_argument(
        "--module",
        choices=["noise", "spatial", "frequency",
                 "enhancement", "edge", "benchmark"],
        help="Run specific module demo"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run full benchmark with plots"
    )

    args = parser.parse_args()

    print("=" * 50)
    print("  UAV Image Restoration Pipeline")
    print("  Degraded Imagery Restoration & Enhancement")
    print("=" * 50)

    image = get_test_image()
    print(f"\nTest image: {image.shape}, dtype: {image.dtype}")

    if args.module:
        module_map = {
            "noise": demo_noise,
            "spatial": demo_spatial,
            "frequency": demo_frequency,
            "enhancement": demo_enhancement,
            "edge": demo_edge,
            "benchmark": demo_benchmark,
        }
        module_map[args.module](image)

    elif args.benchmark:
        demo_benchmark(image)

    elif args.demo:
        demo_noise(image)
        demo_spatial(image)
        demo_frequency(image)
        demo_enhancement(image)
        demo_edge(image)
        print("\n" + "=" * 50)
        print("  All modules completed successfully!")
        print("=" * 50)

    else:
        # no args = run demo
        demo_noise(image)
        demo_spatial(image)
        demo_frequency(image)
        demo_enhancement(image)
        demo_edge(image)
        print("\n" + "=" * 50)
        print("  Pipeline complete! Use --benchmark for full results.")
        print("=" * 50)


if __name__ == "__main__":
    main()
