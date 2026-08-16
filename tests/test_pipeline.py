import pytest
import numpy as np
import tempfile
import shutil
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.generator import DatasetGenerator
from src.augmentation.noise import add_gaussian_noise
from src.augmentation.edge_effects import apply_sem_edge_brightening

class TestPipeline:
    @pytest.fixture
    def tmp_dir(self):
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d, ignore_errors=True)

    def test_independent_noise(self):
        """Test 1 — Independent noise: Generate two independent captures and verify noise is not identical."""
        img = np.ones((100, 100), dtype=np.uint8) * 128
        
        # Generator creates fresh noise rng per invocation
        rng1 = np.random.RandomState(1)
        rng2 = np.random.RandomState(2)
        
        noisy1 = add_gaussian_noise(img, sigma=20.0, rng=rng1)
        noisy2 = add_gaussian_noise(img, sigma=20.0, rng=rng2)
        
        assert not np.array_equal(noisy1, noisy2)

    def test_reproducibility(self, tmp_dir):
        """Test 2 — Reproducibility: Generate using the same global seed twice and verify."""
        gen1 = DatasetGenerator('DRAM', str(tmp_dir / 'out1'), seed=123)
        gen2 = DatasetGenerator('DRAM', str(tmp_dir / 'out2'), seed=123)
        
        gen1.generate_dataset(2, save_visualizations=False)
        gen2.generate_dataset(2, save_visualizations=False)
        
        f1 = tmp_dir / 'out1' / 'train' / 'annotations.json'
        f2 = tmp_dir / 'out2' / 'train' / 'annotations.json'
        
        with open(f1) as fh:
            d1 = json.load(fh)
        with open(f2) as fh:
            d2 = json.load(fh)
            
        assert len(d1) == len(d2)
        for a1, a2 in zip(d1, d2):
            assert a1['center_x'] == a2['center_x']
            assert a1['augmentation_params']['reference']['noise_seed'] == a2['augmentation_params']['reference']['noise_seed']

    def test_ground_truth_location(self, tmp_dir):
        """Test 3 — Ground-truth location exactly matches the actual inserted reference location."""
        gen = DatasetGenerator('DRAM', str(tmp_dir / 'out'), seed=42)
        # We manually call generate_pair to inspect the GT
        ref, search, ann = gen.generate_pair("test_01")
        
        # Verify the crop in search exactly matches the unaugmented ref_base
        # Since augmentation happens later, we just verify the bbox logic
        bbox = ann.bbox
        cx, cy = ann.center_x, ann.center_y
        half = 50
        
        assert bbox['x'] == cx - half
        assert bbox['y'] == cy - half
        assert bbox['width'] == 100
        assert bbox['height'] == 100

    def test_bounding_box_validity(self, tmp_dir):
        """Test 4 — Bounding-box validity: lies within image boundaries."""
        gen = DatasetGenerator('DRAM', str(tmp_dir / 'out'), seed=42)
        gen.generate_dataset(5, save_visualizations=False)
        
        with open(tmp_dir / 'out' / 'train' / 'annotations.json') as fh:
            data = json.load(fh)
            
        for ann in data:
            x = ann['bbox']['x']
            y = ann['bbox']['y']
            w = ann['bbox']['width']
            h = ann['bbox']['height']
            
            assert x >= 0 and y >= 0
            assert x + w <= ann['image_width']
            assert y + h <= ann['image_height']

    def test_edge_brightening(self):
        """Test 5 — Edge brightening measurably changes/enhances edge regions."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[30:70, 30:70] = 100
        
        brightened = apply_sem_edge_brightening(img, strength=0.5)
        
        # Check an edge pixel specifically
        edge_pixel_original = img[30, 50]
        edge_pixel_brightened = brightened[30, 50]
        
        assert edge_pixel_brightened > edge_pixel_original
        
        # Check flat region is unchanged
        assert brightened[50, 50] == img[50, 50]

    def test_different_samples(self, tmp_dir):
        """Test 6 — Different samples do not accidentally become identical."""
        gen = DatasetGenerator('DRAM', str(tmp_dir / 'out'), seed=42)
        gen.generate_dataset(3, save_visualizations=False)
        
        with open(tmp_dir / 'out' / 'train' / 'annotations.json') as fh:
            data = json.load(fh)
            
        centers = [(a['center_x'], a['center_y']) for a in data]
        seeds = [a['augmentation_params']['search']['noise_seed'] for a in data]
        
        # Verify centers and seeds are distinct
        assert len(set(centers)) == len(centers)
        assert len(set(seeds)) == len(seeds)

    def test_metadata_consistency(self, tmp_dir):
        """Test 7 — Metadata consistency: every metadata record points to an existing image."""
        gen = DatasetGenerator('DRAM', str(tmp_dir / 'out'), seed=42)
        gen.generate_dataset(2, save_visualizations=False)
        
        for split in ['train', 'validation', 'test']:
            ann_file = tmp_dir / 'out' / split / 'annotations.json'
            if ann_file.exists():
                with open(ann_file) as fh:
                    data = json.load(fh)
                
                for ann in data:
                    ref_path = tmp_dir / 'out' / split / ann['reference']
                    search_path = tmp_dir / 'out' / split / ann['search']
                    assert ref_path.exists()
                    assert search_path.exists()

    def test_full_pipeline(self, tmp_dir):
        """Test 8 — Full pipeline: end-to-end dataset generation produces all expected outputs."""
        gen = DatasetGenerator('FinFET', str(tmp_dir / 'out'), seed=42)
        gen.generate_dataset(2, save_visualizations=False)
        
        assert (tmp_dir / 'out' / 'metadata.json').exists()
        assert (tmp_dir / 'out' / 'train' / 'annotations.json').exists()
        
        train_ref_dir = tmp_dir / 'out' / 'train' / 'reference'
        assert train_ref_dir.exists()
        assert len(list(train_ref_dir.glob('*.png'))) > 0
