"""
Candidate extraction from similarity maps.

After computing a combined similarity map for one scale, we need to
extract plausible match locations rather than only the global maximum.
This is important because:
  - The search image contains a periodic semiconductor pattern, so
    multiple regions may score highly.
  - The official specification requires returning the candidate
    closest to the search image center when multiple matches exist.

Algorithm:
    1. Apply a score threshold to discard weak candidates.
    2. Extract local maxima using non-maximum suppression (NMS):
       for each peak, suppress all locations within `nms_radius` pixels.
    3. Convert template-corner coordinates to center coordinates.
    4. Store score, scale, center_x, center_y, width, height.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List


@dataclass
class Candidate:
    """A single match candidate.

    Attributes:
        center_x: Horizontal center in search image coordinates.
        center_y: Vertical center in search image coordinates.
        score: Combined similarity score (higher = better).
        scale: Scale factor of template that produced this candidate.
        width: Template width at this scale (pixels).
        height: Template height at this scale (pixels).
    """
    center_x: int
    center_y: int
    score: float
    scale: float
    width: int
    height: int


def extract_local_maxima(
    score_map: np.ndarray,
    threshold: float = 0.3,
    nms_radius: int = 50,
    max_candidates: int = 20,
) -> List[dict]:
    """Extract local maxima from a similarity map with NMS.

    Args:
        score_map: float32 similarity map (values in [-1, 1]).
        threshold: Minimum score to be considered a valid candidate.
        nms_radius: Suppress all maxima within this pixel radius.
        max_candidates: Maximum number of candidates to return.

    Returns:
        List of dicts with keys: 'row', 'col', 'score'.
        Coordinates are in the score_map frame (template top-left corner).
    """
    peaks = []
    # Work on a copy to mark suppressed regions
    working_map = score_map.copy()

    for _ in range(max_candidates):
        idx = np.argmax(working_map)
        row, col = np.unravel_index(idx, working_map.shape)
        val = float(working_map[row, col])

        if val < threshold:
            break

        peaks.append({"row": int(row), "col": int(col), "score": val})

        # Suppress the neighbourhood
        r0 = max(0, row - nms_radius)
        r1 = min(working_map.shape[0], row + nms_radius + 1)
        c0 = max(0, col - nms_radius)
        c1 = min(working_map.shape[1], col + nms_radius + 1)
        working_map[r0:r1, c0:c1] = -1.0

    return peaks


def corners_to_center(
    top_left_col: int,
    top_left_row: int,
    tmpl_w: int,
    tmpl_h: int,
) -> tuple:
    """Convert template top-left corner to center coordinate.

    The score_map value at (row, col) means the template's top-left
    corner is at (col, row) in the search image. The center is offset
    by half the template size.

    Args:
        top_left_col: Column of template top-left corner (= x).
        top_left_row: Row of template top-left corner (= y).
        tmpl_w: Template width in pixels.
        tmpl_h: Template height in pixels.

    Returns:
        Tuple (center_x, center_y) in search image pixels.
    """
    cx = top_left_col + tmpl_w // 2
    cy = top_left_row + tmpl_h // 2
    return int(cx), int(cy)


def extract_candidates(
    score_map: np.ndarray,
    template_w: int,
    template_h: int,
    scale: float,
    threshold: float = 0.3,
    nms_radius: int = 50,
    max_candidates: int = 20,
) -> List[Candidate]:
    """Extract Candidate objects from a score map.

    Args:
        score_map: Combined similarity map for this scale.
        template_w: Width of the resized template used.
        template_h: Height of the resized template used.
        scale: Scale factor that produced this template.
        threshold: Minimum score threshold.
        nms_radius: NMS suppression radius in pixels.
        max_candidates: Maximum candidates to extract.

    Returns:
        List of Candidate objects.
    """
    peaks = extract_local_maxima(
        score_map,
        threshold=threshold,
        nms_radius=nms_radius,
        max_candidates=max_candidates,
    )

    candidates = []
    for p in peaks:
        cx, cy = corners_to_center(p["col"], p["row"], template_w, template_h)
        candidates.append(Candidate(
            center_x=cx,
            center_y=cy,
            score=p["score"],
            scale=scale,
            width=template_w,
            height=template_h,
        ))

    return candidates


def select_by_center_priority(
    candidates: List[Candidate],
    search_center: tuple = (500, 500),
) -> Candidate:
    """Apply the official center-priority selection rule.

    Per specification:
        "If more than one matching region is found, return the one
         closest to the center of the Search Image."

    Implementation:
        Only candidates whose scores are within a small margin of the
        global maximum are considered "matching regions" that trigger
        the center-priority tiebreaker.
    """
    if not candidates:
        raise ValueError("No candidates to select from.")

    if len(candidates) == 1:
        return candidates[0]

    best_score = max(c.score for c in candidates)
    
    # 0.05 tie margin handles small score variations due to noise/blur
    # while preventing clearly inferior matches from winning just because
    # they are closer to the center.
    TIE_MARGIN = 0.05
    tied = [c for c in candidates if c.score >= best_score - TIE_MARGIN]

    scx, scy = search_center
    best = min(
        tied,
        key=lambda c: (c.center_x - scx) ** 2 + (c.center_y - scy) ** 2
    )
    return best


def cluster_and_deduplicate(
    candidates: List[Candidate],
    cluster_radius: int = 30,
) -> List[Candidate]:
    """Merge nearby candidates, keeping the highest-scoring one per cluster.

    Candidates from different scales that agree on nearly the same
    location are redundant. We keep only the best-scoring per cluster
    to avoid a single strong location dominating the candidate pool
    with many duplicates.

    Args:
        candidates: Candidates from ALL scales.
        cluster_radius: Merge candidates within this pixel distance.

    Returns:
        Deduplicated list of best-per-cluster candidates.
    """
    if not candidates:
        return []

    # Sort by score descending so greedy selection picks best first
    sorted_cands = sorted(candidates, key=lambda c: c.score, reverse=True)
    kept = []

    for cand in sorted_cands:
        is_dup = False
        for k in kept:
            dist = ((cand.center_x - k.center_x) ** 2 +
                    (cand.center_y - k.center_y) ** 2) ** 0.5
            if dist <= cluster_radius:
                is_dup = True
                break
        if not is_dup:
            kept.append(cand)

    return kept
