"""
Tests for the Drift-Sense localization inference engine (Phase 2).

Test coverage:
    1.  Exact synthetic target (clean embed — no augmentation)
    2.  Noisy target
    3.  Scaled target
    4.  Rotated target
    5.  Blurred target
    6.  Multiple candidate regions (center-priority rule)
    7.  Center-priority selection logic
    8.  Coordinate correctness (center not top-left)
    9.  Invalid input handling
    10. CLI invocation (DRAM + FinFET)

Design note on test patterns:
    Classical NCC template matching is inherently ambiguous on perfectly
    periodic patterns — every period-aligned crop scores identically.
    Real SEM images have process variation and thermal drift that breaks
    this symmetry.  For unit tests we embed the reference DIRECTLY into
    the search image (clean embed) so the GT location has a uniquely
    high NCC score.  Augmentation-robustness tests (noise, blur, rotation)
    use generous tolerances consistent with the centre-priority spec.
"""

import pytest
import numpy as np
import json
import subprocess
import sys
import re
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.localization.preprocessing import preprocess, compute_edge_map, normalize_float
from src.localization.matching import (
    intensity_match, structural_match, combined_score_map
)
from src.localization.candidates import (
    Candidate, extract_candidates, select_by_center_priority,
    cluster_and_deduplicate, corners_to_center,
)
from src.localization.localization import localize, load_grayscale


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_unique_pattern(
    size: int = 1000, seed: int = 7, target_cx: int = None, target_cy: int = None
) -> np.ndarray:
    """Build a search image with sufficient uniqueness for NCC.

    Uses the real DRAMGenerator. If target_cx/cy are provided, places a
    guaranteed unique, high-contrast marker at that location so the 100x100
    reference crop is unambiguously unique across the image.
    """
    rng = np.random.RandomState(seed)
    from src.architectures.dram import DRAMGenerator
    gen = DRAMGenerator(seed=seed)
    base = gen.generate_base_pattern(size + 200, size + 200)
    img = base[:size, :size].copy().astype(np.int32)

    # Add background blobs
    for _ in range(40):
        bx = rng.randint(10, size - 10)
        by = rng.randint(10, size - 10)
        cv2.circle(img, (bx, by), rng.randint(4, 10), int(rng.randint(180, 255)), -1)

    # Add guaranteed unique marker at GT location if provided
    if target_cx is not None and target_cy is not None:
        cv2.circle(img, (target_cx, target_cy), 15, 255, -1)
        cv2.circle(img, (target_cx - 10, target_cy - 10), 5, 0, -1)

    return np.clip(img, 0, 255).astype(np.uint8)


def _make_clean_pair(
    target_cx: int = 400,
    target_cy: int = 600,
    ref_size: int = 100,
    search_size: int = 1000,
    seed: int = 7,
) -> tuple:
    """Build (reference, search) by embedding ref DIRECTLY into search.

    The reference is a crop from the search pattern. With the unique marker,
    this guarantees a unique, unambiguous NCC peak at the GT location.
    """
    search = _make_unique_pattern(
        search_size, seed=seed, target_cx=target_cx, target_cy=target_cy
    )
    half = ref_size // 2
    ref = search[target_cy - half: target_cy + half,
                 target_cx - half: target_cx + half].copy()
    return ref, search, target_cx, target_cy


def _make_augmented_pair(
    target_cx: int = 400,
    target_cy: int = 600,
    ref_size: int = 100,
    search_size: int = 1000,
    noise_sigma: float = 0.0,
    blur_sigma: float = 0.0,
    rotation_deg: float = 0.0,
    scale: float = 1.0,
    seed: int = 7,
) -> tuple:
    """Build a pair where ref is augmented (noise/blur/rotation/scale)."""
    search = _make_unique_pattern(
        search_size, seed=seed, target_cx=target_cx, target_cy=target_cy
    )
    half = ref_size // 2
    ref = search[target_cy - half: target_cy + half,
                 target_cx - half: target_cx + half].copy()

    rng = np.random.RandomState(seed + 1)

    if abs(scale - 1.0) > 1e-4:
        new_h = max(10, int(ref_size * scale))
        new_w = max(10, int(ref_size * scale))
        ref = cv2.resize(ref, (new_w, new_h),
                         interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA)
        # Center-crop/pad back to ref_size
        pad = np.full((ref_size, ref_size), 30, dtype=np.uint8)
        h, w = ref.shape[:2]
        ph = min(h, ref_size)
        pw = min(w, ref_size)
        y0 = (h - ph) // 2 if h > ref_size else 0
        x0 = (w - pw) // 2 if w > ref_size else 0
        oy = (ref_size - ph) // 2
        ox = (ref_size - pw) // 2
        pad[oy:oy+ph, ox:ox+pw] = ref[y0:y0+ph, x0:x0+pw]
        ref = pad

    if abs(rotation_deg) > 0.01:
        h, w = ref.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rotation_deg, 1.0)
        ref = cv2.warpAffine(ref, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    if blur_sigma > 0.1:
        ksize = int(6 * blur_sigma + 1) | 1
        ref = cv2.GaussianBlur(ref, (ksize, ksize), blur_sigma)

    if noise_sigma > 0:
        noise = rng.normal(0, noise_sigma, ref.shape)
        ref = np.clip(ref.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        noise_s = rng.normal(0, noise_sigma * 1.5, search.shape)
        search = np.clip(search.astype(np.float32) + noise_s, 0, 255).astype(np.uint8)

    return ref, search, target_cx, target_cy


# ─────────────────────────────────────────────────────────────────────────────
# 1. Preprocessing tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_normalize_float_range(self):
        img = np.array([[0, 128, 255]], dtype=np.uint8)
        out = normalize_float(img.astype(np.float32))
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-6

    def test_preprocess_output_shape(self):
        img = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        out = preprocess(img)
        assert out.shape == (100, 100)
        assert out.dtype == np.float32

    def test_preprocess_output_range(self):
        img = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        out = preprocess(img)
        assert out.min() >= -0.01
        assert out.max() <= 1.01

    def test_edge_map_shape(self):
        img = np.random.rand(100, 100).astype(np.float32)
        edges = compute_edge_map(img)
        assert edges.shape == (100, 100)
        assert edges.dtype == np.float32

    def test_edge_map_detects_edges(self):
        flat = np.full((100, 100), 0.5, dtype=np.float32)
        edges_flat = compute_edge_map(flat)
        step = np.zeros((100, 100), dtype=np.float32)
        step[:, 50:] = 1.0
        edges_step = compute_edge_map(step)
        assert edges_step.max() > edges_flat.max()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Matching tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMatching:
    def test_ncc_map_shape(self):
        tmpl = np.random.rand(50, 50).astype(np.float32)
        search = np.random.rand(200, 200).astype(np.float32)
        result = intensity_match(tmpl, search)
        assert result.shape == (200 - 50 + 1, 200 - 50 + 1)

    def test_ncc_perfect_match(self):
        """Embedded pattern must be found at the correct location."""
        rng = np.random.RandomState(0)
        base = rng.randint(0, 255, (300, 300), dtype=np.uint8).astype(np.float32) / 255.0
        tmpl = base[100:150, 100:150].copy()
        result = intensity_match(tmpl, base)
        # Allow ±1 pixel rounding
        idx = np.unravel_index(np.argmax(result), result.shape)
        assert abs(idx[0] - 100) <= 1
        assert abs(idx[1] - 100) <= 1

    def test_combined_map_shape(self):
        i_map = np.random.rand(100, 100).astype(np.float32)
        s_map = np.random.rand(100, 100).astype(np.float32)
        c = combined_score_map(i_map, s_map, alpha=0.6, beta=0.4)
        assert c.shape == (100, 100)

    def test_combined_map_values(self):
        i_map = np.ones((10, 10), dtype=np.float32) * 0.8
        s_map = np.ones((10, 10), dtype=np.float32) * 0.6
        c = combined_score_map(i_map, s_map, alpha=0.6, beta=0.4)
        expected = 0.6 * 0.8 + 0.4 * 0.6
        np.testing.assert_allclose(c, expected, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Candidates tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCandidates:
    def test_corners_to_center(self):
        cx, cy = corners_to_center(100, 200, 50, 60)
        assert cx == 125
        assert cy == 230

    def test_extract_candidates_basic(self):
        score_map = np.zeros((900, 900), dtype=np.float32)
        score_map[100, 200] = 0.9
        score_map[700, 400] = 0.8
        cands = extract_candidates(score_map, 100, 100, 1.0, threshold=0.3)
        assert len(cands) >= 2
        assert all(isinstance(c, Candidate) for c in cands)

    def test_center_priority_selection(self):
        # Tie margin is 0.05, so scores must be within 0.05 of each other
        c1 = Candidate(center_x=490, center_y=510, score=0.88,
                       scale=1.0, width=100, height=100)
        c2 = Candidate(center_x=100, center_y=100, score=0.90,
                       scale=1.0, width=100, height=100)
        selected = select_by_center_priority([c1, c2], search_center=(500, 500))
        # c1 is closer to center → wins despite slightly lower score
        assert selected.center_x == 490
        assert selected.center_y == 510

    def test_cluster_deduplication(self):
        c1 = Candidate(center_x=400, center_y=400, score=0.9,
                       scale=1.0, width=100, height=100)
        c2 = Candidate(center_x=405, center_y=395, score=0.8,
                       scale=1.1, width=110, height=110)
        c3 = Candidate(center_x=700, center_y=700, score=0.7,
                       scale=1.0, width=100, height=100)
        deduped = cluster_and_deduplicate([c1, c2, c3], cluster_radius=30)
        assert len(deduped) == 2
        centers = [(c.center_x, c.center_y) for c in deduped]
        assert (400, 400) in centers

    def test_select_by_center_priority_single(self):
        c = Candidate(center_x=300, center_y=300, score=0.8,
                      scale=1.0, width=100, height=100)
        result = select_by_center_priority([c])
        assert result.center_x == 300

    def test_select_by_center_priority_empty_raises(self):
        with pytest.raises(ValueError):
            select_by_center_priority([])


# ─────────────────────────────────────────────────────────────────────────────
# 4. End-to-end localization tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLocalization:
    """End-to-end localize() tests.

    Clean-embed tests use ref directly cropped from search (no augmentation)
    to guarantee a uniquely-high NCC peak.  Augmentation tests use
    wider tolerances.
    """

    CLEAN_TOL = 20    # pixels; clean embed should be nearly exact
    AUG_TOL   = 120   # pixels; augmented patterns allow more drift

    def test_exact_synthetic_target(self):
        """Test 1: Clean-embed, no augmentation — exact match expected."""
        ref, search, gt_cx, gt_cy = _make_clean_pair(
            target_cx=400, target_cy=600, seed=7
        )
        result = localize(ref, search)
        err = ((result.center_x - gt_cx) ** 2 +
               (result.center_y - gt_cy) ** 2) ** 0.5
        assert err <= self.CLEAN_TOL, (
            f"Exact test: pred=({result.center_x},{result.center_y}) "
            f"gt=({gt_cx},{gt_cy}) err={err:.1f}px"
        )

    def test_noisy_target(self):
        """Test 2: Reference + search receive independent noise."""
        ref, search, gt_cx, gt_cy = _make_augmented_pair(
            target_cx=500, target_cy=300, noise_sigma=12, seed=11
        )
        result = localize(ref, search)
        err = ((result.center_x - gt_cx) ** 2 +
               (result.center_y - gt_cy) ** 2) ** 0.5
        assert err <= self.AUG_TOL, (
            f"Noisy test: pred=({result.center_x},{result.center_y}) "
            f"gt=({gt_cx},{gt_cy}) err={err:.1f}px"
        )

    def test_scaled_target(self):
        """Test 3: Reference scaled to 0.95x."""
        ref, search, gt_cx, gt_cy = _make_augmented_pair(
            target_cx=300, target_cy=700, scale=0.95, seed=13
        )
        result = localize(ref, search,
                          scales=[0.85, 0.9, 0.95, 1.0, 1.05])
        err = ((result.center_x - gt_cx) ** 2 +
               (result.center_y - gt_cy) ** 2) ** 0.5
        assert err <= self.AUG_TOL * 2, (
            f"Scale test: pred=({result.center_x},{result.center_y}) "
            f"gt=({gt_cx},{gt_cy}) err={err:.1f}px"
        )

    def test_rotated_target(self):
        """Test 4: Reference rotated 1.5 degrees."""
        ref, search, gt_cx, gt_cy = _make_augmented_pair(
            target_cx=600, target_cy=200, rotation_deg=1.5, seed=17
        )
        result = localize(ref, search)
        err = ((result.center_x - gt_cx) ** 2 +
               (result.center_y - gt_cy) ** 2) ** 0.5
        assert err <= self.AUG_TOL * 2, (
            f"Rotation test: pred=({result.center_x},{result.center_y}) "
            f"gt=({gt_cx},{gt_cy}) err={err:.1f}px"
        )

    def test_blurred_target(self):
        """Test 5: Reference blurred sigma=1.5."""
        ref, search, gt_cx, gt_cy = _make_augmented_pair(
            target_cx=700, target_cy=400, blur_sigma=1.5, seed=19
        )
        result = localize(ref, search)
        err = ((result.center_x - gt_cx) ** 2 +
               (result.center_y - gt_cy) ** 2) ** 0.5
        assert err <= self.AUG_TOL * 2, (
            f"Blur test: pred=({result.center_x},{result.center_y}) "
            f"gt=({gt_cx},{gt_cy}) err={err:.1f}px"
        )

    def test_result_is_center_not_topleft(self):
        """Test 8: Output coordinate must be CENTER, not top-left."""
        ref, search, gt_cx, gt_cy = _make_clean_pair(
            target_cx=400, target_cy=400, seed=23
        )
        result = localize(ref, search)
        dist_to_center = ((result.center_x - gt_cx) ** 2 +
                          (result.center_y - gt_cy) ** 2) ** 0.5
        # The top-left corner is (gt_cx - 50, gt_cy - 50)
        dist_to_topleft = ((result.center_x - (gt_cx - 50)) ** 2 +
                           (result.center_y - (gt_cy - 50)) ** 2) ** 0.5
        assert dist_to_center < dist_to_topleft, (
            f"Output looks like top-left corner! "
            f"pred=({result.center_x},{result.center_y}) "
            f"center=({gt_cx},{gt_cy}) topleft=({gt_cx-50},{gt_cy-50})"
        )

    def test_center_priority_rule(self):
        """Test 7: Center-priority selects closest to (500,500)."""
        # Tie margin is 0.05
        c_near = Candidate(center_x=480, center_y=520, score=0.92,
                           scale=1.0, width=100, height=100)
        c_far  = Candidate(center_x=100, center_y=900, score=0.95,
                           scale=1.0, width=100, height=100)
        selected = select_by_center_priority(
            [c_near, c_far], search_center=(500, 500)
        )
        assert selected.center_x == 480
        assert selected.center_y == 520

    def test_multiple_candidate_regions(self):
        """Test 6: Multiple candidates — closest to center wins."""
        # Insert TWO bright patches; the one closer to (500,500) should win.
        rng = np.random.RandomState(42)
        search = np.full((1000, 1000), 30, dtype=np.uint8)
        # Patch A: far from center
        patch_a = rng.randint(100, 200, (100, 100), dtype=np.uint8)
        search[100:200, 100:200] = patch_a        # center ≈ (150,150)
        # Patch B: near center
        patch_b = patch_a.copy()                   # same pattern
        search[450:550, 450:550] = patch_b        # center ≈ (500,500)

        # Reference = patch_b (near-center one)
        ref = patch_b.copy()
        result = localize(ref, search)
        # Both patches match equally; center-priority should pick ~(500,500)
        # Allow ±80px given possible NMS / scale effects
        assert abs(result.center_x - 500) <= 80, \
            f"Expected center near 500, got x={result.center_x}"
        assert abs(result.center_y - 500) <= 80, \
            f"Expected center near 500, got y={result.center_y}"

    def test_invalid_input_none(self):
        """Test 9: None inputs raise ValueError."""
        with pytest.raises((ValueError, AttributeError)):
            localize(None, np.zeros((1000, 1000), dtype=np.uint8))

    def test_invalid_input_empty(self):
        """Test 9: Empty array raises ValueError."""
        with pytest.raises(ValueError):
            localize(
                np.array([], dtype=np.uint8).reshape(0, 0),
                np.zeros((1000, 1000), dtype=np.uint8)
            )

    def test_invalid_input_color_image(self):
        """Test 9: 3-channel image raises ValueError."""
        with pytest.raises(ValueError, match="2D"):
            localize(
                np.zeros((100, 100, 3), dtype=np.uint8),
                np.zeros((1000, 1000), dtype=np.uint8)
            )

    def test_invalid_ref_larger_than_search(self):
        """Test 9: Reference larger than search raises ValueError."""
        with pytest.raises(ValueError):
            localize(
                np.zeros((500, 500), dtype=np.uint8),
                np.zeros((100, 100), dtype=np.uint8)
            )

    def test_runtime_reasonable(self):
        """Inference must complete in < 10 seconds."""
        import time
        ref, search, _, _ = _make_clean_pair()
        t0 = time.perf_counter()
        result = localize(ref, search)
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, f"Inference took {elapsed:.2f}s — too slow."
        assert result.runtime_ms < 10000.0

    def test_result_fields(self):
        """LocalizationResult has all expected typed fields."""
        ref, search, _, _ = _make_clean_pair()
        result = localize(ref, search)
        assert isinstance(result.center_x, int)
        assert isinstance(result.center_y, int)
        assert isinstance(result.score, float)
        assert isinstance(result.scale, float)
        assert isinstance(result.num_candidates, int)
        assert isinstance(result.runtime_ms, float)
        assert isinstance(result.fallback_used, bool)

    def test_coordinates_within_search_image(self):
        """Predicted coords must be inside 1000×1000 search."""
        ref, search, _, _ = _make_clean_pair(target_cx=750, target_cy=250)
        result = localize(ref, search)
        assert 0 <= result.center_x < 1000
        assert 0 <= result.center_y < 1000

    def test_second_clean_location(self):
        """Test clean-embed at a different location."""
        ref, search, gt_cx, gt_cy = _make_clean_pair(
            target_cx=750, target_cy=250, seed=31
        )
        result = localize(ref, search)
        err = ((result.center_x - gt_cx) ** 2 +
               (result.center_y - gt_cy) ** 2) ** 0.5
        assert err <= self.CLEAN_TOL, (
            f"Second clean test: pred=({result.center_x},{result.center_y}) "
            f"gt=({gt_cx},{gt_cy}) err={err:.1f}px"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Load grayscale tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadGrayscale:
    def test_load_valid_image(self, tmp_path):
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        p = str(tmp_path / "test.png")
        cv2.imwrite(p, img)
        loaded = load_grayscale(p)
        assert loaded.shape == (100, 100)

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_grayscale("/nonexistent/path/image.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLI tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCLI:
    """Test CLI invocation of inference.py."""

    def _run_cli(self, ref_path, search_path, extra_args=None):
        cmd = [sys.executable, "inference.py",
               str(ref_path), str(search_path)]
        if extra_args:
            cmd.extend(extra_args)
        return subprocess.run(
            cmd,
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
        )

    def test_cli_output_format(self, tmp_path):
        """Test 10: stdout is exactly '(x,y)'."""
        ref, search, _, _ = _make_clean_pair(target_cx=400, target_cy=500)
        ref_p  = str(tmp_path / "ref.png")
        srch_p = str(tmp_path / "search.png")
        cv2.imwrite(ref_p, ref)
        cv2.imwrite(srch_p, search)

        result = self._run_cli(ref_p, srch_p)
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        m = re.match(r"^\((\d+),(\d+)\)$", result.stdout.strip())
        assert m is not None, \
            f"Output does not match (x,y): '{result.stdout.strip()}'"

    def test_cli_missing_reference(self, tmp_path):
        """CLI exits != 0 when reference file is missing."""
        search_img = np.zeros((1000, 1000), dtype=np.uint8)
        srch_p = str(tmp_path / "search.png")
        cv2.imwrite(srch_p, search_img)
        result = self._run_cli("/no/such/file.png", srch_p)
        assert result.returncode != 0

    def test_cli_missing_search(self, tmp_path):
        """CLI exits != 0 when search file is missing."""
        ref_img = np.zeros((100, 100), dtype=np.uint8)
        ref_p = str(tmp_path / "ref.png")
        cv2.imwrite(ref_p, ref_img)
        result = self._run_cli(ref_p, "/no/such/search.png")
        assert result.returncode != 0

    def test_cli_verbose_flag(self, tmp_path):
        """--verbose: diagnostics go to stderr; stdout stays clean."""
        ref, search, _, _ = _make_clean_pair(target_cx=300, target_cy=300)
        ref_p  = str(tmp_path / "ref.png")
        srch_p = str(tmp_path / "search.png")
        cv2.imwrite(ref_p, ref)
        cv2.imwrite(srch_p, search)

        result = self._run_cli(ref_p, srch_p, ["--verbose"])
        assert result.returncode == 0
        m = re.match(r"^\((\d+),(\d+)\)$", result.stdout.strip())
        assert m is not None, \
            f"Stdout not clean with --verbose: '{result.stdout.strip()}'"
        assert "[INFO]" in result.stderr

    def test_cli_dram_fresh(self, tmp_path):
        """Test 10: CLI on a real DRAM pair — clean embed."""
        ref, search, gt_cx, gt_cy = _make_clean_pair(
            target_cx=450, target_cy=350, seed=99
        )
        ref_p  = str(tmp_path / "dram_ref.png")
        srch_p = str(tmp_path / "dram_search.png")
        cv2.imwrite(ref_p, ref)
        cv2.imwrite(srch_p, search)

        result = self._run_cli(ref_p, srch_p)
        assert result.returncode == 0, f"DRAM CLI failed:\n{result.stderr}"
        m = re.match(r"^\((\d+),(\d+)\)$", result.stdout.strip())
        assert m is not None
        pred_x, pred_y = int(m.group(1)), int(m.group(2))
        err = ((pred_x - gt_cx) ** 2 + (pred_y - gt_cy) ** 2) ** 0.5
        assert err < 40, (
            f"DRAM CLI: pred=({pred_x},{pred_y}) "
            f"gt=({gt_cx},{gt_cy}) err={err:.1f}px"
        )

    def test_cli_finfet_fresh(self, tmp_path):
        """Test 10: CLI on a fresh FinFET pair — clean embed."""
        from src.architectures.finfet import FinFETGenerator
        gen = FinFETGenerator(seed=77)
        base = gen.generate_base_pattern(1200, 1200)

        # Add sparse blobs for uniqueness
        rng = np.random.RandomState(77)
        img = base[:1000, :1000].astype(np.int32)
        for _ in range(40):
            bx = rng.randint(10, 990)
            by = rng.randint(10, 990)
            cv2.circle(img, (bx, by), rng.randint(4, 10),
                       int(rng.randint(180, 255)), -1)

        gt_cx, gt_cy = 600, 200
        
        # Add guaranteed unique marker at GT location
        cv2.circle(img, (gt_cx, gt_cy), 15, 255, -1)
        cv2.circle(img, (gt_cx - 10, gt_cy - 10), 5, 0, -1)
        
        search = np.clip(img, 0, 255).astype(np.uint8)

        half = 50
        ref = search[gt_cy-half:gt_cy+half,
                     gt_cx-half:gt_cx+half].copy()

        ref_p  = str(tmp_path / "finfet_ref.png")
        srch_p = str(tmp_path / "finfet_search.png")
        cv2.imwrite(ref_p, ref)
        cv2.imwrite(srch_p, search)

        result = self._run_cli(ref_p, srch_p)
        assert result.returncode == 0, f"FinFET CLI failed:\n{result.stderr}"
        m = re.match(r"^\((\d+),(\d+)\)$", result.stdout.strip())
        assert m is not None
        pred_x, pred_y = int(m.group(1)), int(m.group(2))
        err = ((pred_x - gt_cx) ** 2 + (pred_y - gt_cy) ** 2) ** 0.5
        assert err < 40, (
            f"FinFET CLI: pred=({pred_x},{pred_y}) "
            f"gt=({gt_cx},{gt_cy}) err={err:.1f}px"
        )

    def test_cli_save_debug(self, tmp_path):
        """--save-debug: debug image saved, stdout still clean."""
        ref, search, _, _ = _make_clean_pair(target_cx=400, target_cy=400)
        ref_p   = str(tmp_path / "ref.png")
        srch_p  = str(tmp_path / "search.png")
        debug_d = str(tmp_path / "dbg")
        cv2.imwrite(ref_p, ref)
        cv2.imwrite(srch_p, search)

        result = self._run_cli(ref_p, srch_p, ["--save-debug", debug_d])
        assert result.returncode == 0
        m = re.match(r"^\((\d+),(\d+)\)$", result.stdout.strip())
        assert m is not None
        # Debug dir should exist with at least one PNG
        debug_files = list(Path(debug_d).glob("*.png"))
        assert len(debug_files) >= 1
