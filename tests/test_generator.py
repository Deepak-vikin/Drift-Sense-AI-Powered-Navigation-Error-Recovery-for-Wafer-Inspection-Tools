import pytest
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.architectures import DRAMGenerator, FinFETGenerator, ARCHITECTURE_MAP
from src.dataset.generator import DatasetGenerator


class TestDRAMGenerator:
    def test_dram_pattern_shape(self):
        gen = DRAMGenerator(seed=42)
        pattern = gen.generate_base_pattern(200, 200)
        assert pattern.shape == (200, 200)
        assert pattern.dtype == np.uint8
    
    def test_dram_pattern_not_empty(self):
        gen = DRAMGenerator(seed=42)
        pattern = gen.generate_base_pattern(200, 200)
        assert pattern.max() > 100  # has bright features
        assert pattern.min() < 100  # has dark background
        assert pattern.std() > 10   # has meaningful contrast
    
    def test_dram_pattern_large(self):
        gen = DRAMGenerator(seed=42)
        pattern = gen.generate_base_pattern(1200, 1200)
        assert pattern.shape == (1200, 1200)
    
    def test_dram_reproducibility(self):
        gen1 = DRAMGenerator(seed=42)
        gen2 = DRAMGenerator(seed=42)
        p1 = gen1.generate_base_pattern(200, 200)
        p2 = gen2.generate_base_pattern(200, 200)
        np.testing.assert_array_equal(p1, p2)
    
    def test_dram_architecture_name(self):
        gen = DRAMGenerator()
        assert gen.get_architecture_name() == 'DRAM'


class TestFinFETGenerator:
    def test_finfet_pattern_shape(self):
        gen = FinFETGenerator(seed=42)
        pattern = gen.generate_base_pattern(200, 200)
        assert pattern.shape == (200, 200)
        assert pattern.dtype == np.uint8
    
    def test_finfet_pattern_not_empty(self):
        gen = FinFETGenerator(seed=42)
        pattern = gen.generate_base_pattern(200, 200)
        assert pattern.max() > 100
        assert pattern.min() < 100
        assert pattern.std() > 10
    
    def test_finfet_pattern_large(self):
        gen = FinFETGenerator(seed=42)
        pattern = gen.generate_base_pattern(1200, 1200)
        assert pattern.shape == (1200, 1200)
    
    def test_finfet_reproducibility(self):
        gen1 = FinFETGenerator(seed=42)
        gen2 = FinFETGenerator(seed=42)
        p1 = gen1.generate_base_pattern(200, 200)
        p2 = gen2.generate_base_pattern(200, 200)
        np.testing.assert_array_equal(p1, p2)
    
    def test_finfet_architecture_name(self):
        gen = FinFETGenerator()
        assert gen.get_architecture_name() == 'FinFET'


class TestArchitectureMap:
    def test_dram_in_map(self):
        assert 'DRAM' in ARCHITECTURE_MAP
    
    def test_finfet_in_map(self):
        assert 'FinFET' in ARCHITECTURE_MAP


class TestDatasetGenerator:
    @pytest.fixture
    def tmp_dir(self):
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d, ignore_errors=True)
    
    def test_dram_generation(self, tmp_dir):
        gen = DatasetGenerator('DRAM', str(tmp_dir / 'out'), seed=42)
        gen.generate_dataset(5)
        # Check output structure
        assert (tmp_dir / 'out' / 'train' / 'annotations.json').exists()
    
    def test_finfet_generation(self, tmp_dir):
        gen = DatasetGenerator('FinFET', str(tmp_dir / 'out'), seed=42)
        gen.generate_dataset(5)
        assert (tmp_dir / 'out' / 'train' / 'annotations.json').exists()
    
    def test_reference_dimensions(self, tmp_dir):
        gen = DatasetGenerator('DRAM', str(tmp_dir / 'out'), seed=42)
        gen.generate_dataset(3)
        # Load a reference image and check dimensions
        import cv2
        ref_dir = tmp_dir / 'out' / 'train' / 'reference'
        if ref_dir.exists():
            refs = list(ref_dir.glob('*.png'))
            if refs:
                img = cv2.imread(str(refs[0]), cv2.IMREAD_GRAYSCALE)
                assert img is not None
                h, w = img.shape
                assert 80 <= h <= 120  # approximately 100
                assert 80 <= w <= 120
    
    def test_search_dimensions(self, tmp_dir):
        gen = DatasetGenerator('DRAM', str(tmp_dir / 'out'), seed=42)
        gen.generate_dataset(3)
        import cv2
        for split in ['train', 'validation', 'test']:
            search_dir = tmp_dir / 'out' / split / 'search'
            if search_dir.exists():
                for img_path in search_dir.glob('*.png'):
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    assert img is not None
                    assert img.shape == (1000, 1000), f"Search image {img_path} has wrong shape: {img.shape}"
    
    def test_ground_truth_inside_image(self, tmp_dir):
        gen = DatasetGenerator('DRAM', str(tmp_dir / 'out'), seed=42)
        gen.generate_dataset(5)
        for split in ['train', 'validation', 'test']:
            ann_file = tmp_dir / 'out' / split / 'annotations.json'
            if ann_file.exists():
                with open(ann_file) as f:
                    data = json.load(f)
                for ann in data:
                    assert 0 <= ann['center_x'] < 1000
                    assert 0 <= ann['center_y'] < 1000
    
    def test_bbox_valid(self, tmp_dir):
        gen = DatasetGenerator('DRAM', str(tmp_dir / 'out'), seed=42)
        gen.generate_dataset(5)
        for split in ['train', 'validation', 'test']:
            ann_file = tmp_dir / 'out' / split / 'annotations.json'
            if ann_file.exists():
                with open(ann_file) as f:
                    data = json.load(f)
                for ann in data:
                    bbox = ann['bbox']
                    assert bbox['x'] >= 0
                    assert bbox['y'] >= 0
                    assert bbox['x'] + bbox['width'] <= 1000
                    assert bbox['y'] + bbox['height'] <= 1000
    
    def test_annotations_valid_json(self, tmp_dir):
        gen = DatasetGenerator('DRAM', str(tmp_dir / 'out'), seed=42)
        gen.generate_dataset(5)
        for split in ['train', 'validation', 'test']:
            ann_file = tmp_dir / 'out' / split / 'annotations.json'
            if ann_file.exists():
                with open(ann_file) as f:
                    data = json.load(f)  # should not raise
                assert isinstance(data, list)
    
    def test_no_empty_images(self, tmp_dir):
        gen = DatasetGenerator('DRAM', str(tmp_dir / 'out'), seed=42)
        gen.generate_dataset(5)
        import cv2
        for split in ['train', 'validation', 'test']:
            for subdir in ['reference', 'search']:
                img_dir = tmp_dir / 'out' / split / subdir
                if img_dir.exists():
                    for img_path in img_dir.glob('*.png'):
                        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                        assert img is not None, f"Image is None: {img_path}"
                        assert img.std() > 1, f"Image appears empty: {img_path}"
    
    def test_30_pairs(self, tmp_dir):
        gen = DatasetGenerator('DRAM', str(tmp_dir / 'out'), seed=42)
        gen.generate_dataset(30)
        total = 0
        for split in ['train', 'validation', 'test']:
            ann_file = tmp_dir / 'out' / split / 'annotations.json'
            if ann_file.exists():
                with open(ann_file) as f:
                    data = json.load(f)
                total += len(data)
        assert total == 30
    
    def test_seed_reproducibility(self, tmp_dir):
        gen1 = DatasetGenerator('DRAM', str(tmp_dir / 'out1'), seed=42)
        gen2 = DatasetGenerator('DRAM', str(tmp_dir / 'out2'), seed=42)
        gen1.generate_dataset(3)
        gen2.generate_dataset(3)
        # Check annotations match
        for split in ['train', 'validation', 'test']:
            f1 = tmp_dir / 'out1' / split / 'annotations.json'
            f2 = tmp_dir / 'out2' / split / 'annotations.json'
            if f1.exists() and f2.exists():
                with open(f1) as fh:
                    d1 = json.load(fh)
                with open(f2) as fh:
                    d2 = json.load(fh)
                assert len(d1) == len(d2)
                for a1, a2 in zip(d1, d2):
                    assert a1['center_x'] == a2['center_x']
                    assert a1['center_y'] == a2['center_y']
    
    def test_metadata_json(self, tmp_dir):
        gen = DatasetGenerator('DRAM', str(tmp_dir / 'out'), seed=42)
        gen.generate_dataset(5)
        meta = tmp_dir / 'out' / 'metadata.json'
        assert meta.exists()
        with open(meta) as f:
            data = json.load(f)
        assert data['architecture'] == 'DRAM'
        assert data['seed'] == 42
    
    def test_search_noisier_than_reference(self, tmp_dir):
        """Verify search images get higher noise sigma than reference images on average."""
        gen = DatasetGenerator('DRAM', str(tmp_dir / 'out'), seed=42)
        gen.generate_dataset(10)
        ref_noises = []
        search_noises = []
        for split in ['train', 'validation', 'test']:
            ann_file = tmp_dir / 'out' / split / 'annotations.json'
            if ann_file.exists():
                with open(ann_file) as f:
                    data = json.load(f)
                for ann in data:
                    ref_noises.append(ann['augmentation_params']['reference']['noise_sigma'])
                    search_noises.append(ann['augmentation_params']['search']['noise_sigma'])
        # On average, search should have higher noise sigma
        avg_ref = sum(ref_noises) / len(ref_noises)
        avg_search = sum(search_noises) / len(search_noises)
        assert avg_search > avg_ref, (
            f"Search noise ({avg_search:.2f}) should be higher than reference noise ({avg_ref:.2f})"
        )


class TestCLI:
    """Test the command-line interface."""

    def test_cli_dram(self, tmp_path):
        """Test that the CLI runs successfully for DRAM architecture."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, 'dataset_generator.py',
             '--architecture', 'DRAM',
             '--num_pairs', '5',
             '--output_dir', str(tmp_path / 'cli_test'),
             '--seed', '0',
             '--no-visualizations'],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        assert (tmp_path / 'cli_test' / 'train' / 'annotations.json').exists()

    def test_cli_finfet(self, tmp_path):
        """Test that the CLI runs successfully for FinFET architecture."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, 'dataset_generator.py',
             '--architecture', 'FinFET',
             '--num_pairs', '5',
             '--output_dir', str(tmp_path / 'cli_finfet'),
             '--seed', '1',
             '--no-visualizations'],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        assert (tmp_path / 'cli_finfet' / 'train' / 'annotations.json').exists()

    def test_cli_invalid_architecture(self, tmp_path):
        """Test that invalid architecture is rejected."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, 'dataset_generator.py',
             '--architecture', 'INVALID',
             '--num_pairs', '5',
             '--output_dir', str(tmp_path / 'bad')],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        assert result.returncode != 0
