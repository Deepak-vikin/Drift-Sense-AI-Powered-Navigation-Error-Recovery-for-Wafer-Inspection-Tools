import pytest
import json
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.annotations import Annotation, AnnotationManager


class TestAnnotation:
    def test_create_annotation(self):
        ann = Annotation(
            id='000001',
            architecture='DRAM',
            reference='reference/000001.png',
            search='search/000001.png',
            reference_width_px=1000,
            reference_height_px=1000,
            reference_pixel_size_nm=1.0,
            search_width_px=1000,
            search_height_px=1000,
            search_pixel_size_nm=10.0,
            reference_physical_width_nm=1000.0,
            reference_physical_height_nm=1000.0,
            expected_footprint_width_px=100.0,
            expected_footprint_height_px=100.0,
            ground_truth_x=450,
            ground_truth_y=350,
            ground_truth_width=100,
            ground_truth_height=100,
            ground_truth_center_x=500.0,
            ground_truth_center_y=400.0,
            image_width=1000,
            image_height=1000,
            center_x=500,
            center_y=400,
            bbox={'x': 450, 'y': 350, 'width': 100, 'height': 100},
            bbox_x_min=450,
            bbox_y_min=350,
            bbox_x_max=550,
            bbox_y_max=450,
            augmentation_params={'noise_sigma': 10.0}
        )
        assert ann.center_x == 500
        assert ann.architecture == 'DRAM'


class TestAnnotationManager:
    def _create_dummy_ann(self, **kwargs):
        defaults = dict(
            id='000001', architecture='DRAM',
            reference='reference/000001.png',
            search='search/000001.png',
            reference_width_px=1000, reference_height_px=1000, reference_pixel_size_nm=1.0,
            search_width_px=1000, search_height_px=1000, search_pixel_size_nm=10.0,
            reference_physical_width_nm=1000.0, reference_physical_height_nm=1000.0,
            expected_footprint_width_px=100.0, expected_footprint_height_px=100.0,
            ground_truth_x=450, ground_truth_y=350, ground_truth_width=100, ground_truth_height=100,
            ground_truth_center_x=500.0, ground_truth_center_y=400.0,
            image_width=1000, image_height=1000, center_x=500, center_y=400,
            bbox={'x': 450, 'y': 350, 'width': 100, 'height': 100},
            bbox_x_min=450, bbox_y_min=350, bbox_x_max=550, bbox_y_max=450,
            augmentation_params={}
        )
        defaults.update(kwargs)
        return Annotation(**defaults)

    def test_add_and_save(self):
        manager = AnnotationManager()
        ann = self._create_dummy_ann()
        manager.add(ann)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            tmp_path = Path(f.name)
        
        manager.save(tmp_path)
        assert tmp_path.exists()
        
        with open(tmp_path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]['center_x'] == 500
        tmp_path.unlink()
    
    def test_load(self):
        manager = AnnotationManager()
        ann = self._create_dummy_ann(
            center_x=634, center_y=421,
            ground_truth_center_x=634.0, ground_truth_center_y=421.0,
            bbox={'x': 584, 'y': 371, 'width': 100, 'height': 100}
        )
        manager.add(ann)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            tmp_path = Path(f.name)
        manager.save(tmp_path)
        
        loaded = AnnotationManager.load(tmp_path)
        assert len(loaded.annotations) == 1
        assert loaded.annotations[0].center_x == 634
        tmp_path.unlink()
    
    def test_validate_valid(self):
        manager = AnnotationManager()
        ann = self._create_dummy_ann()
        manager.add(ann)
        errors = manager.validate()
        assert len(errors) == 0
    
    def test_validate_invalid_center(self):
        manager = AnnotationManager()
        ann = self._create_dummy_ann(
            center_x=1500, center_y=500,  # outside image
            bbox={'x': 1450, 'y': 450, 'width': 100, 'height': 100}
        )
        manager.add(ann)
        errors = manager.validate()
        assert len(errors) > 0
