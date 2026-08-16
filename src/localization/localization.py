"""
Top-level localization function for Drift-Sense.

This is the single public entry point for running inference on a
reference-search image pair.

Algorithm summary:
    1. Validate inputs (shape, non-empty, readable).
    2. Run multi-scale matching to collect all candidates.
    3. Deduplicate candidates that cluster near the same location.
    4. If no candidate passes the threshold, fall back to the global
       best match at the nominal scale (scale=1.0).
    5. Apply center-priority selection: if multiple candidates remain,
       return the one closest to the search image center (500, 500).
    6. Return a LocalizationResult with the predicted center.

Coordinate convention:
    center_x = horizontal (column direction)
    center_y = vertical   (row direction)
    Both are integer pixel offsets measured from the top-left of the
    search image, representing the CENTER of the matched region.
"""

import time
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .preprocessing import preprocess, compute_edge_map
from .matching import intensity_match, structural_match, combined_score_map
from .candidates import (
    Candidate,
    extract_candidates,
    cluster_and_deduplicate,
    select_by_center_priority,
)
from .multiscale import multiscale_match


@dataclass
class LocalizationResult:
    """Result of a single localization inference call.

    Attributes:
        center_x: Predicted horizontal center in search image (pixels).
        center_y: Predicted vertical center in search image (pixels).
        score: Confidence score of the best candidate (0-1 scale).
        scale: Template scale factor that produced the best match.
        num_candidates: Number of candidates before selection.
        runtime_ms: Wall-clock inference time in milliseconds.
        fallback_used: True if no candidate passed threshold and a
                       fallback to global maximum was used.
    """
    center_x: int
    center_y: int
    score: float
    scale: float
    num_candidates: int
    runtime_ms: float
    fallback_used: bool = False


def _fallback_match(
    reference: np.ndarray,
    search: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> Candidate:
    """Last-resort: global maximum NCC at scale=1.0.

    Used when no candidate passes the score threshold at any scale.
    The global max always exists; we just return it as our best guess.

    Args:
        reference: Preprocessed float32 reference.
        search: Preprocessed float32 search.
        alpha: Intensity weight.
        beta: Structural weight.

    Returns:
        A single Candidate at the global best match position.
    """
    ref_proc = preprocess(reference)
    search_proc = preprocess(search)

    th, tw = ref_proc.shape[:2]

    # Clamp template to be smaller than search
    if th >= search_proc.shape[0] or tw >= search_proc.shape[1]:
        # Extreme edge case: return search center
        sh, sw = search.shape[:2]
        return Candidate(
            center_x=sw // 2, center_y=sh // 2,
            score=0.0, scale=1.0,
            width=tw, height=th,
        )

    i_map = intensity_match(ref_proc, search_proc)
    search_edges = compute_edge_map(search_proc)
    ref_edges = compute_edge_map(ref_proc)
    s_map = structural_match(ref_edges, search_edges)
    c_map = combined_score_map(i_map, s_map, alpha=alpha, beta=beta)

    idx = np.argmax(c_map)
    row, col = np.unravel_index(idx, c_map.shape)
    cx = col + tw // 2
    cy = row + th // 2

    return Candidate(
        center_x=int(cx), center_y=int(cy),
        score=float(c_map[row, col]),
        scale=1.0,
        width=tw, height=th,
    )


def localize(
    reference: np.ndarray,
    search: np.ndarray,
    scales: Optional[List[float]] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    score_threshold: float = 0.3,
    nms_radius: int = 50,
    cluster_radius: int = 30,
    max_candidates_per_scale: int = 10,
    search_center: Optional[Tuple[int, int]] = None,
) -> LocalizationResult:
    """Localize the reference pattern inside the search image.

    This is the main inference function. It does NOT require knowledge
    of the semiconductor architecture — it works equally for DRAM and
    FinFET patterns.

    Args:
        reference: Grayscale image of the reference patch (uint8).
                   Expected shape: approximately (100, 100).
        search: Grayscale search image (uint8).
                Expected shape: (1000, 1000).
        scales: List of scale factors to test. If None, uses default
                [0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4].
        alpha: Weight for intensity NCC in combined score.
        beta: Weight for structural NCC in combined score.
        score_threshold: Minimum combined score to be a valid candidate.
        nms_radius: Non-maximum suppression radius (pixels).
        cluster_radius: Deduplication radius across scales (pixels).
        max_candidates_per_scale: Candidates extracted per scale.
        search_center: (x, y) center of search image. If None, uses
                       (search_w // 2, search_h // 2).

    Returns:
        LocalizationResult with predicted (center_x, center_y).

    Raises:
        ValueError: If images are invalid, empty, or incompatible.
    """
    t0 = time.perf_counter()

    # --- Input validation ---
    if reference is None or not isinstance(reference, np.ndarray):
        raise ValueError("reference must be a numpy array.")
    if search is None or not isinstance(search, np.ndarray):
        raise ValueError("search must be a numpy array.")

    if reference.size == 0:
        raise ValueError("reference image is empty.")
    if search.size == 0:
        raise ValueError("search image is empty.")

    if reference.ndim != 2:
        raise ValueError(
            f"reference must be a 2D grayscale image, got shape {reference.shape}."
        )
    if search.ndim != 2:
        raise ValueError(
            f"search must be a 2D grayscale image, got shape {search.shape}."
        )

    rh, rw = reference.shape
    sh, sw = search.shape

    if rh > sh or rw > sw:
        raise ValueError(
            f"Reference ({rw}x{rh}) must not be larger than search ({sw}x{sh})."
        )

    if reference.std() < 1.0:
        raise ValueError(
            "Reference image appears to be blank (std < 1). "
            "Cannot perform localization on an empty image."
        )

    # Determine search image center
    if search_center is None:
        search_center = (sw // 2, sh // 2)

    # --- Multi-scale matching ---
    candidates = multiscale_match(
        reference=reference,
        search=search,
        scales=scales,
        alpha=alpha,
        beta=beta,
        score_threshold=score_threshold,
        nms_radius=nms_radius,
        max_candidates_per_scale=max_candidates_per_scale,
    )

    fallback_used = False

    # --- Deduplicate across scales ---
    candidates = cluster_and_deduplicate(candidates, cluster_radius=cluster_radius)

    if not candidates:
        # No candidate passed threshold — use fallback (global max at scale=1)
        best = _fallback_match(reference, search, alpha=alpha, beta=beta)
        fallback_used = True
        candidates = [best]

    # --- Center-priority selection ---
    selected = select_by_center_priority(candidates, search_center=search_center)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return LocalizationResult(
        center_x=selected.center_x,
        center_y=selected.center_y,
        score=selected.score,
        scale=selected.scale,
        num_candidates=len(candidates),
        runtime_ms=elapsed_ms,
        fallback_used=fallback_used,
    )


def load_grayscale(path: str) -> np.ndarray:
    """Load an image file as a grayscale uint8 array.

    Args:
        path: Path to image file.

    Returns:
        Grayscale uint8 numpy array.

    Raises:
        FileNotFoundError: If file does not exist.
        IOError: If file cannot be read as an image.
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(
            f"Could not read image (unsupported format or corrupt file): {path}"
        )
    return img
