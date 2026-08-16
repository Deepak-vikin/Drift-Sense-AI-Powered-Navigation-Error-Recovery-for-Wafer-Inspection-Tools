"""
Multi-scale template matching for Drift-Sense localization.

Two-signal strategy:
    1. INTENSITY NCC (primary): Finds candidate locations using
       cv2.TM_CCOEFF_NORMED on the intensity-preprocessed image pair.
       NCC is exact-match optimal — on a clean embed the GT scores 1.0.
       Candidates are extracted from the intensity map alone.

    2. STRUCTURAL NCC (refinement): Sobel edge maps are used to compute
       an additional structural similarity.  Rather than combining the
       two maps before candidate extraction (which reduces the GT score
       when periodic edge patterns dominate), structural scores are used
       only AFTER candidates are found, to provide a secondary ranking
       signal when multiple candidates are within a small intensity-
       score window (score_tie_margin).

Why NOT combine before extraction:
    On perfectly-periodic semiconductor patterns (DRAM, FinFET), the edge
    NCC map has many near-equal peaks (every bit-line row looks the same).
    Averaging intensity NCC (GT rank 0) with structural NCC (GT rank 316)
    pushes GT to combined rank 74, making it invisible to candidate
    extraction.  Structural must NOT override the primary intensity signal.

Scale range:
    Default [0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4]
    covers the ±5% default augmentation and wider field-of-view drift.
"""

import numpy as np
import cv2
from typing import List, Optional

from .preprocessing import preprocess, compute_edge_map
from .matching import intensity_match, structural_match, combined_score_map
from .candidates import Candidate, extract_candidates


def _resize_template(template: np.ndarray, scale: float) -> Optional[np.ndarray]:
    """Resize template by scale factor using the best interpolation method.

    Args:
        template: float32 template image.
        scale: Scale factor (1.0 = no change).

    Returns:
        Resized float32 image, or None if result would be < 10x10 px.
    """
    h, w = template.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    if new_w < 10 or new_h < 10:
        return None

    interp = cv2.INTER_LINEAR if scale >= 1.0 else cv2.INTER_AREA
    return cv2.resize(template, (new_w, new_h), interpolation=interp)


def _refine_with_structural(
    candidates: List[Candidate],
    tmpl_edges: np.ndarray,
    search_edges: np.ndarray,
    search_shape: tuple,
    structural_weight: float = 0.25,
    score_tie_margin: float = 0.02,
) -> List[Candidate]:
    """Re-score candidates using structural information as a soft signal.

    Only candidates within `score_tie_margin` of the top intensity score
    are re-scored.  This avoids over-riding a clear intensity winner with
    structural noise.

    For each candidate, we extract the corresponding search_edges patch
    and compute the NCC against tmpl_edges to obtain a per-candidate
    structural score.  The combined score updates the candidate's score.

    Args:
        candidates: Candidates from intensity-NCC extraction (sorted or not).
        tmpl_edges: Sobel edge map of the resized template.
        search_edges: Sobel edge map of the full search image.
        search_shape: (height, width) of the full search image.
        structural_weight: How much weight to give structural in re-scoring.
        score_tie_margin: Only re-score candidates within this delta of top.

    Returns:
        Updated candidate list with refined scores.
    """
    if not candidates:
        return candidates

    top_score = max(c.score for c in candidates)
    tie_threshold = top_score - score_tie_margin

    sh, sw = search_shape
    th, tw = tmpl_edges.shape[:2]

    refined = []
    for c in candidates:
        if c.score < tie_threshold:
            # Clear loser — keep intensity score, no structural overhead
            refined.append(c)
            continue

        # Extract matching patch from search_edges
        r0 = c.center_y - c.height // 2
        c0 = c.center_x - c.width // 2
        r1 = r0 + c.height
        c1 = c0 + c.width

        # Skip if out of bounds
        if r0 < 0 or c0 < 0 or r1 > sh or c1 > sw:
            refined.append(c)
            continue

        patch_edges = search_edges[r0:r1, c0:c1]
        if patch_edges.shape != tmpl_edges.shape:
            refined.append(c)
            continue

        # Local structural NCC between template edges and search patch edges
        s_score = float(intensity_match(tmpl_edges, patch_edges).max())
        # Weighted combination: intensity dominates
        new_score = (1.0 - structural_weight) * c.score + structural_weight * s_score

        from dataclasses import replace
        refined.append(Candidate(
            center_x=c.center_x,
            center_y=c.center_y,
            score=new_score,
            scale=c.scale,
            width=c.width,
            height=c.height,
        ))

    return refined


def multiscale_match(
    reference: np.ndarray,
    search: np.ndarray,
    scales: List[float] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    score_threshold: float = 0.3,
    nms_radius: int = 50,
    max_candidates_per_scale: int = 10,
) -> List[Candidate]:
    """Run multi-scale template matching and collect candidates.

    Primary signal: intensity NCC (finds exact/near-exact locations).
    Secondary signal: structural NCC (used to refine tied candidates).

    The alpha/beta parameters control the structural refinement weight
    (beta is mapped to structural_weight = beta / (alpha + beta)).

    Args:
        reference: Raw grayscale reference image (uint8 or float32).
        search: Raw grayscale search image (uint8 or float32).
        scales: List of scale factors to test.
                Default: [0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4]
        alpha: Intensity NCC weight (used to compute structural_weight).
        beta: Structural NCC weight (used to compute structural_weight).
        score_threshold: Minimum intensity NCC score to be a candidate.
        nms_radius: NMS suppression radius in pixels.
        max_candidates_per_scale: Max candidates extracted per scale.

    Returns:
        List of all Candidate objects across all scales, with scores
        refined by structural information for tied candidates.
    """
    if scales is None:
        scales = [0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4]

    structural_weight = beta / (alpha + beta) if (alpha + beta) > 0 else 0.0

    # Preprocess images once
    ref_proc = preprocess(reference)
    search_proc = preprocess(search)

    # Edge maps for structural refinement (computed once)
    search_edges = compute_edge_map(search_proc)

    all_candidates: List[Candidate] = []

    for scale in scales:
        # Resize preprocessed reference
        tmpl = _resize_template(ref_proc, scale)
        if tmpl is None:
            continue

        th, tw = tmpl.shape[:2]

        # Skip if template >= search size
        if th >= search_proc.shape[0] or tw >= search_proc.shape[1]:
            continue

        # ── Primary: Intensity NCC ──────────────────────────────────────
        i_map = intensity_match(tmpl, search_proc)

        # Extract candidates from intensity map ONLY
        cands = extract_candidates(
            score_map=i_map,
            template_w=tw,
            template_h=th,
            scale=scale,
            threshold=score_threshold,
            nms_radius=nms_radius,
            max_candidates=max_candidates_per_scale,
        )

        # ── Secondary: Structural refinement for tied candidates ────────
        if cands and structural_weight > 0.0:
            tmpl_edges = compute_edge_map(tmpl)
            cands = _refine_with_structural(
                cands,
                tmpl_edges=tmpl_edges,
                search_edges=search_edges,
                search_shape=search_proc.shape,
                structural_weight=structural_weight,
                score_tie_margin=0.0005,
            )

        all_candidates.extend(cands)

    return all_candidates
