"""
Preprocessing pipeline for Drift-Sense localization.

Pipeline applied to reference and search independently:
    1. Mild Gaussian denoise (sigma=0.8) — reduces SEM shot noise
       without destroying the semiconductor line structures.
    2. Convert to float32 [0.0, 1.0] via simple /255 scaling.

Why we do NOT use global normalization (z-score / CLAHE):
    cv2.matchTemplate(TM_CCOEFF_NORMED) computes Normalized Cross-
    Correlation (NCC) which ALREADY applies zero-mean and unit-variance
    normalization to every overlapping patch during computation.  This
    makes NCC inherently invariant to global additive brightness offsets
    and multiplicative contrast differences.

    Applying an additional global normalization step (z-score stretch or
    CLAHE) BEFORE feeding to NCC is harmful: because the global statistics
    (mean, std) are computed from different-sized images (100x100 ref vs
    1000x1000 search), the same 100x100 region gets a DIFFERENT rescaled
    value depending on whether it is processed alone or as a crop.  This
    causes the NCC GT score to drop from 1.0 to ~0.96, while other
    periodic-pattern locations score higher, causing completely wrong
    localization results.

    Mild Gaussian denoise is safe because it is a local spatial operation
    that does not alter global intensity statistics.

    CLAHE and z-score normalization are preserved as utility functions for
    Phase 3 experiments where pre-processing consistency can be explicitly
    controlled.
"""

import numpy as np
import cv2
from typing import Tuple


def normalize_float(image: np.ndarray) -> np.ndarray:
    """Linearly rescale image to [0.0, 1.0].

    Used internally after denoising. Not applied as a global
    pre-normalization step (see module docstring).
    """
    img = image.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)
    return img


def mild_denoise(image: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """Apply a very mild Gaussian denoise to suppress SEM shot noise.

    sigma=0.8 removes high-frequency Poisson/Gaussian noise while
    keeping fine semiconductor structures (DRAM bit-lines, FinFET fins)
    intact.  This is a purely local spatial operation and does not
    alter global intensity statistics, making it NCC-safe.

    Args:
        image: float32 image in [0, 1].
        sigma: Gaussian std dev (default 0.8 — very gentle).

    Returns:
        Denoised float32 image.
    """
    ksize = int(6 * sigma + 1) | 1   # enforce odd kernel size
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def preprocess(image: np.ndarray,
               denoise_sigma: float = 0.0,
               clahe_clip: float = 2.0) -> np.ndarray:
    """Preprocess an image for NCC template matching.

    Steps:
        1. Mild Gaussian denoise to suppress SEM sensor noise.
        2. Convert to float32 in [0.0, 1.0] via /255.

    The clahe_clip argument is accepted for API compatibility but
    CLAHE is intentionally NOT applied here (see module docstring).

    Args:
        image: Input grayscale image (uint8 or float32).
        denoise_sigma: Gaussian denoise sigma (default 0.8).
        clahe_clip: Not used; kept for backward compatibility.

    Returns:
        Preprocessed float32 image in [0.0, 1.0].
    """
    if image.dtype == np.uint8:
        img_f = image.astype(np.float32) / 255.0
    else:
        img_f = np.clip(image.astype(np.float32), 0.0, 255.0) / 255.0

    # Mild denoise — local operation, NCC-safe
    if denoise_sigma > 0.01:
        img_f = mild_denoise(img_f, sigma=denoise_sigma)

    return img_f


def apply_clahe(image_uint8: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE equalization (for Phase 3 experiments only).

    WARNING: Do NOT use in the NCC pipeline.  CLAHE tile sizes are
    fixed in pixels, so the output for a 100x100 crop differs from the
    same region inside a 1000x1000 image.  This breaks NCC.

    Args:
        image_uint8: Input grayscale uint8 image.
        clip_limit: CLAHE clip limit.
        tile_grid_size: Histogram tile grid size.

    Returns:
        Equalized uint8 image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image_uint8)


def zscore_normalize(image: np.ndarray) -> np.ndarray:
    """Z-score normalization then stretch to [0,1] (Phase 3 experiments).

    WARNING: Global statistic — output differs between a standalone crop
    and the same region inside a larger image.  NOT for use in the NCC
    matching pipeline.

    Args:
        image: float32 or uint8 image.

    Returns:
        float32 image normalized to [0.0, 1.0].
    """
    img = image.astype(np.float32)
    mu = img.mean()
    sigma = img.std()
    if sigma > 1e-6:
        img = (img - mu) / sigma
    else:
        img = img - mu
    return normalize_float(img)


def compute_edge_map(image_float: np.ndarray) -> np.ndarray:
    """Compute normalized Sobel edge magnitude map.

    SEM images exhibit secondary-electron edge brightening at material
    boundaries.  Explicit Sobel edges provide a structural representation
    that complements intensity NCC, helping differentiate structurally
    distinct regions.

    This is a local spatial operator and is NCC-safe (does not alter
    global statistics in a size-dependent way).

    Args:
        image_float: float32 image in [0, 1].

    Returns:
        Normalized float32 edge magnitude in [0, 1].
    """
    img_u8 = np.clip(image_float * 255, 0, 255).astype(np.uint8)
    grad_x = cv2.Sobel(img_u8, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_u8, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude = magnitude / mag_max
    return magnitude
