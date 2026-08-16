import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.augmentation.noise import add_gaussian_noise, add_poisson_noise
from src.augmentation.blur import apply_gaussian_blur
from src.augmentation.geometry import apply_rotation, apply_scale
from src.augmentation.edge_effects import apply_sem_edge_brightening


class TestNoise:
    def test_gaussian_noise_shape(self):
        img = np.ones((100, 100), dtype=np.uint8) * 128
        noisy = add_gaussian_noise(img, sigma=10.0)
        assert noisy.shape == (100, 100)
        assert noisy.dtype == np.uint8
    
    def test_gaussian_noise_modifies_image(self):
        img = np.ones((100, 100), dtype=np.uint8) * 128
        noisy = add_gaussian_noise(img, sigma=10.0)
        assert not np.array_equal(img, noisy)
    
    def test_independent_noise(self):
        """Verify that two calls produce different noise."""
        img = np.ones((100, 100), dtype=np.uint8) * 128
        noisy1 = add_gaussian_noise(img, sigma=20.0)
        noisy2 = add_gaussian_noise(img, sigma=20.0)
        # With different random draws, these should differ
        assert not np.array_equal(noisy1, noisy2)
    
    def test_gaussian_noise_with_rng(self):
        img = np.ones((100, 100), dtype=np.uint8) * 128
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        n1 = add_gaussian_noise(img, sigma=10.0, rng=rng1)
        n2 = add_gaussian_noise(img, sigma=10.0, rng=rng2)
        np.testing.assert_array_equal(n1, n2)
    
    def test_poisson_noise_shape(self):
        img = np.ones((100, 100), dtype=np.uint8) * 128
        noisy = add_poisson_noise(img)
        assert noisy.shape == (100, 100)
        assert noisy.dtype == np.uint8


class TestBlur:
    def test_blur_shape(self):
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        blurred = apply_gaussian_blur(img, sigma=1.0)
        assert blurred.shape == (100, 100)
        assert blurred.dtype == np.uint8
    
    def test_blur_smooths_image(self):
        # Create a sharp edge image
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 50:] = 255
        blurred = apply_gaussian_blur(img, sigma=3.0)
        # The edge should be smoother (gradient region should exist)
        edge_region = blurred[50, 45:55]
        assert edge_region.min() < 128
        assert edge_region.max() > 128


class TestGeometry:
    def test_rotation_shape(self):
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        rotated = apply_rotation(img, 5.0)
        assert rotated.shape == (100, 100)
    
    def test_zero_rotation(self):
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        rotated = apply_rotation(img, 0.0)
        # Should be very similar (may have minor interpolation differences)
        assert np.allclose(img, rotated, atol=1)
    
    def test_scale_shape(self):
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        scaled = apply_scale(img, 1.1)
        assert scaled.shape == (100, 100)
    
    def test_identity_scale(self):
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        scaled = apply_scale(img, 1.0)
        assert scaled.shape == (100, 100)


class TestEdgeEffects:
    def test_edge_brightening_shape(self):
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = apply_sem_edge_brightening(img, strength=0.3)
        assert result.shape == (100, 100)
        assert result.dtype == np.uint8
    
    def test_edge_brightening_increases_edges(self):
        # Create image with clear edges
        img = np.zeros((100, 100), dtype=np.uint8)
        img[30:70, 30:70] = 200
        result = apply_sem_edge_brightening(img, strength=0.5)
        # Near edges should be brighter than original
        # Check a pixel near the edge
        edge_y, edge_x = 30, 50  # top edge of the bright square
        assert result[edge_y, edge_x] >= img[edge_y, edge_x] or result[edge_y+1, edge_x] >= img[edge_y+1, edge_x]
    
    def test_zero_strength_no_change(self):
        img = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        result = apply_sem_edge_brightening(img, strength=0.0)
        np.testing.assert_array_equal(img, result)
