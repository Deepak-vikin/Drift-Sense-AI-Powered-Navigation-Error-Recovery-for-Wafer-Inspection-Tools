"""
Template matching core for Drift-Sense localization.

Two complementary similarity signals are computed:

1. Intensity NCC (Normalized Cross-Correlation):
   Classic template matching using cv2.TM_CCOEFF_NORMED.
   Robust to global brightness offsets because it works on
   zero-mean, unit-variance patches.

2. Structural (Edge) NCC:
   Same NCC but applied to Sobel edge magnitude maps.
   More invariant to lighting differences between the two
   captures; focuses on the line structure of DRAM / FinFET patterns.

Combined score:
    combined = alpha * intensity_ncc + beta * structural_ncc

The weights alpha=0.6, beta=0.4 are initial defaults.
Phase 3 should tune these using the validation set.

All output maps are in the coordinate frame of the SEARCH IMAGE,
where value at (row, col) represents the similarity when the
template top-left corner is at (col, row).
"""

import numpy as np
import cv2


def ncc_match(template: np.ndarray, search: np.ndarray) -> np.ndarray:
    """Normalized Cross-Correlation template match.

    Uses cv2.TM_CCOEFF_NORMED which produces values in [-1, 1].
    Returns an unnormalized raw similarity map.

    Args:
        template: float32 template (smaller image, e.g. resized ref).
        search: float32 search image (1000x1000).

    Returns:
        float32 similarity map. Shape: (search_h - tmpl_h + 1,
                                        search_w - tmpl_w + 1).
        Values in [-1, 1]. Higher = better match.
    """
    if template.shape[0] > search.shape[0] or template.shape[1] > search.shape[1]:
        # Template larger than search — return empty map (no valid position)
        return np.full((1, 1), -1.0, dtype=np.float32)

    # cv2.matchTemplate requires uint8 or float32
    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    return result.astype(np.float32)


def intensity_match(template_float: np.ndarray,
                    search_float: np.ndarray) -> np.ndarray:
    """Intensity NCC similarity map.

    Args:
        template_float: Preprocessed float32 template in [0, 1].
        search_float: Preprocessed float32 search image in [0, 1].

    Returns:
        float32 similarity map in [-1, 1].
    """
    return ncc_match(template_float, search_float)


def structural_match(template_edges: np.ndarray,
                     search_edges: np.ndarray) -> np.ndarray:
    """Structural (edge) NCC similarity map.

    Args:
        template_edges: Sobel edge magnitude map of template, in [0, 1].
        search_edges: Sobel edge magnitude map of search image, in [0, 1].

    Returns:
        float32 similarity map in [-1, 1].
    """
    return ncc_match(template_edges, search_edges)


def combined_score_map(
    intensity_map: np.ndarray,
    structural_map: np.ndarray,
    alpha: float = 0.6,
    beta: float = 0.4,
) -> np.ndarray:
    """Combine intensity and structural similarity maps.

    If the maps have different shapes (can happen at edge cases), we
    take the intersection and pad the rest to -1.

    Args:
        intensity_map: float32 NCC map from intensity matching.
        structural_map: float32 NCC map from structural matching.
        alpha: Weight for intensity score (default 0.6).
        beta: Weight for structural score (default 0.4).

    Returns:
        float32 combined map, same shape as intensity_map.
    """
    h1, w1 = intensity_map.shape
    h2, w2 = structural_map.shape

    if (h1, w1) != (h2, w2):
        # Shapes differ (rare): align by taking minimum valid region
        h = min(h1, h2)
        w = min(w1, w2)
        combined = (alpha * intensity_map[:h, :w] +
                    beta * structural_map[:h, :w])
        # Pad remainder with -1 (worst score)
        out = np.full((h1, w1), -1.0, dtype=np.float32)
        out[:h, :w] = combined
    else:
        out = alpha * intensity_map + beta * structural_map

    return out.astype(np.float32)
