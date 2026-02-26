"""
Benchmark Dashboard
---------------------
Run all modules and generate comparison results.

Creates visual comparisons and metrics tables for
every restoration and enhancement method.

Author: Mehmet Demir
"""

import numpy as np
import os
from typing import Dict, List

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from src.noise.noise_generator import generate_degraded_dataset
from src.spatial.spatial_filters import compare_spatial_filters
from src.frequency.freq_filters import (
    butterworth_lowpass, gaussian_lowpass, wiener_filter
)
from src.enhancement.contrast import compare_enhancements
from src.edge.edge_detection import sobel_edge, laplacian_edge, unsharp_mask
from src.metrics.quality import (
    calculate_psnr, calculate_ssim, measure_time
)


def run_full_benchmark(image: np.ndarray) -> Dict:
    """
    Run the complete benchmark pipeline.

    1. Generate degraded dataset
    2. Apply spatial filters to each noise type
    3. Apply frequency filters
    4. Apply enhancement methods
    5. Apply edge detectors
    6. Calculate metrics for everything

    Args:
        image: clean test image

    Returns:
        nested dict with all results and metrics
    """
    results = {
        "degraded": {},
        "spatial": {},
        "frequency": {},
        "enhancement": {},
        "edge": {},
        "metrics": [],
    }

    # step 1: generate noisy versions
    dataset = generate_degraded_dataset(image)
    results["degraded"] = dataset

    # step 2: spatial filters on salt-pepper noise
    sp_noisy = dataset["salt_pepper"]
    spatial_results = compare_spatial_filters(image, sp_noisy)
    results["spatial"] = spatial_results

    for name, filtered in spatial_results.items():
        psnr = calculate_psnr(image, filtered)
        ssim = calculate_ssim(image, filtered)
        results["metrics"].append({
            "module": "spatial",
            "method": name,
            "noise": "salt_pepper",
            "psnr": psnr,
            "ssim": ssim,
        })

    # step 3: frequency filters on gaussian noise
    gauss_noisy = dataset["gaussian"]
    freq_results = {
        "butterworth_lp": butterworth_lowpass(gauss_noisy, cutoff=50),
        "gaussian_lp": gaussian_lowpass(gauss_noisy, cutoff=50),
    }
    results["frequency"] = freq_results

    for name, filtered in freq_results.items():
        psnr = calculate_psnr(image, filtered)
        ssim = calculate_ssim(image, filtered)
        results["metrics"].append({
            "module": "frequency",
            "method": name,
            "noise": "gaussian",
            "psnr": psnr,
            "ssim": ssim,
        })

    # step 4: enhancement
    enhancement_results = compare_enhancements(image)
    results["enhancement"] = enhancement_results

    # step 5: edge detection
    edge_results = {
        "sobel": sobel_edge(image),
        "laplacian": laplacian_edge(image),
        "unsharp": unsharp_mask(image),
    }
    results["edge"] = edge_results

    return results


def generate_comparison_plots(
    results: Dict,
    output_dir: str = "results"
) -> List[str]:
    """
    Generate matplotlib comparison plots.

    Creates side-by-side visual comparisons for each module.

    Returns:
        list of saved file paths
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping plots")
        return []

    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    # plot 1: noise types
    degraded = results["degraded"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Noise Types Comparison", fontsize=16)

    for idx, (name, img) in enumerate(degraded.items()):
        ax = axes[idx // 3, idx % 3]
        ax.imshow(img, cmap='gray')
        ax.set_title(name.replace("_", " ").title())
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(output_dir, "noise_comparison.png")
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    saved_files.append(path)

    # plot 2: spatial filter results
    spatial = results["spatial"]
    fig, axes = plt.subplots(1, len(spatial) + 1, figsize=(20, 5))
    fig.suptitle("Spatial Filters (Salt-Pepper Noise)", fontsize=14)

    axes[0].imshow(degraded["salt_pepper"], cmap='gray')
    axes[0].set_title("Noisy")
    axes[0].axis('off')

    for idx, (name, img) in enumerate(spatial.items()):
        axes[idx + 1].imshow(img, cmap='gray')
        axes[idx + 1].set_title(name.replace("_", " ").title())
        axes[idx + 1].axis('off')

    plt.tight_layout()
    path = os.path.join(output_dir, "spatial_comparison.png")
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    saved_files.append(path)

    # plot 3: enhancement results
    enhancement = results["enhancement"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Contrast Enhancement Methods", fontsize=16)

    for idx, (name, img) in enumerate(enhancement.items()):
        ax = axes[idx // 3, idx % 3]
        ax.imshow(img, cmap='gray')
        ax.set_title(name.replace("_", " ").title())
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(output_dir, "enhancement_comparison.png")
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    saved_files.append(path)

    # plot 4: edge detection
    edge = results["edge"]
    fig, axes = plt.subplots(1, len(edge) + 1, figsize=(20, 5))
    fig.suptitle("Edge Detection Methods", fontsize=14)

    axes[0].imshow(degraded["original"], cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')

    for idx, (name, img) in enumerate(edge.items()):
        axes[idx + 1].imshow(img, cmap='gray')
        axes[idx + 1].set_title(name.replace("_", " ").title())
        axes[idx + 1].axis('off')

    plt.tight_layout()
    path = os.path.join(output_dir, "edge_comparison.png")
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    saved_files.append(path)

    return saved_files


def generate_metrics_table(results: Dict) -> str:
    """
    Generate a formatted metrics table.

    Returns:
        string table (or pandas DataFrame if available)
    """
    metrics = results["metrics"]

    if PANDAS_AVAILABLE:
        df = pd.DataFrame(metrics)
        return df.to_string(index=False)

    # fallback: manual formatting
    header = f"{'Module':<12} {'Method':<20} {'Noise':<15} {'PSNR':>8} {'SSIM':>8}"
    lines = [header, "-" * len(header)]

    for m in metrics:
        line = f"{m['module']:<12} {m['method']:<20} {m['noise']:<15} {m['psnr']:>8.2f} {m['ssim']:>8.4f}"
        lines.append(line)

    return "\n".join(lines)


def save_results(
    results: Dict,
    output_dir: str = "results"
) -> None:
    """Save benchmark results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # save metrics table
    table = generate_metrics_table(results)
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(table)

    # save comparison plots
    saved = generate_comparison_plots(results, output_dir)

    print(f"Saved {len(saved)} comparison plots to {output_dir}/")
    print(f"Saved metrics table to {output_dir}/metrics.txt")


# test
if __name__ == "__main__":
    from skimage import data

    image = data.camera()
    print("Running full benchmark...")

    results = run_full_benchmark(image)
    save_results(results)

    print("\nMetrics:")
    print(generate_metrics_table(results))
