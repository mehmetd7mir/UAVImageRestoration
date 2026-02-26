# Architecture

## Overview

Image processing pipeline that takes degraded UAV imagery and applies multiple restoration and enhancement techniques.

## Data Flow

```
Input Image
    │
    ▼
[Noise Generator] ──→ Synthetic degraded versions
    │                    ├── Gaussian noise
    │                    ├── Salt-and-pepper
    │                    ├── Periodic interference
    │                    └── Motion blur (PSF)
    ▼
[Spatial Filters] ──→ Pixel-domain restoration
    │                    ├── Median (best for S&P)
    │                    ├── Arithmetic mean (Gaussian)
    │                    ├── Geometric mean (detail-preserving)
    │                    └── Alpha-trimmed (mixed noise)
    ▼
[Frequency Filters] ──→ DFT-based restoration
    │                     ├── Inverse filter (deconvolution)
    │                     ├── Wiener filter (noise-aware)
    │                     ├── Notch filter (periodic noise)
    │                     └── Butterworth/Gaussian LP/HP
    ▼
[Enhancement] ──→ Contrast improvement
    │               ├── Histogram equalization
    │               ├── CLAHE (local adaptive)
    │               ├── Gamma correction
    │               └── Homomorphic filtering
    ▼
[Edge Detection] ──→ Target enhancement
    │                  ├── Sobel gradient
    │                  ├── Laplacian
    │                  ├── Unsharp masking
    │                  └── High-boost filtering
    ▼
[Benchmark] ──→ PSNR + SSIM metrics + comparison plots
```

## Module Dependencies

- `noise` → standalone (no internal deps)
- `spatial` → uses `noise` for comparison function
- `frequency` → uses `noise` for PSF generation
- `enhancement` → standalone
- `edge` → uses `noise` for SNR comparison
- `metrics` → standalone
- `benchmark` → uses all modules

## Key Design Decisions

1. **Grayscale first**: All algorithms work on grayscale. UAV target detection doesn't need color.
2. **NumPy-based**: Custom implementations use NumPy for transparency. OpenCV used only where it's clearly better (median filter, Gaussian blur).
3. **scikit-image for metrics**: SSIM from scikit-image is the standard reference implementation.
4. **Built-in test images**: Using `skimage.data.camera()` avoids dataset download requirements.
