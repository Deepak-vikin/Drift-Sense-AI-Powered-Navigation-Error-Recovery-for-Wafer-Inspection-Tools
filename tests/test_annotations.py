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
            center_x=500,
            center_y=400,
            image_width=1000,
            image_height=1000,
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
    def test_add_and_save(self):
        manager = AnnotationManager()
        ann = Annotation(
            id='000001', architecture='DRAM',
            reference='reference/000001.png',
            search='search/000001.png',
            center_x=500, center_y=400,
            image_width=1000, image_height=1000,
            bbox={'x': 450, 'y': 350, 'width': 100, 'height': 100},
            bbox_x_min=450, bbox_y_min=350, bbox_x_max=550, bbox_y_max=450,
            augmentation_params={}
        )
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
        ann = Annotation(
            id='000001', architecture='DRAM',
            reference='ref/000001.png', search='search/000001.png',
            center_x=634, center_y=421,
            image_width=1000, image_height=1000,
            bbox={'x': 584, 'y': 371, 'width': 100, 'height': 100},
            bbox_x_min=584, bbox_y_min=371, bbox_x_max=684, bbox_y_max=471,
            augmentation_params={'noise_sigma': 10.0}
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
        ann = Annotation(
            id='000001', architecture='DRAM',
            reference='ref/000001.png', search='search/000001.png',
            center_x=500, center_y=500,
            image_width=1000, image_height=1000,
            bbox={'x': 450, 'y': 450, 'width': 100, 'height': 100},
            bbox_x_min=450, bbox_y_min=450, bbox_x_max=550, bbox_y_max=550,
            augmentation_params={}
        )
        manager.add(ann)
        errors = manager.validate()
        assert len(errors) == 0
    
    def test_validate_invalid_center(self):
        manager = AnnotationManager()
        ann = Annotation(
            id='000001', architecture='DRAM',
            reference='ref/000001.png', search='search/000001.png',
            center_x=1500, center_y=500,  # outside image
            image_width=1000, image_height=1000,
            bbox={'x': 1450, 'y': 450, 'width': 100, 'height': 100},
            bbox_x_min=1450, bbox_y_min=450, bbox_x_max=1550, bbox_y_max=550,
            augmentation_params={}
        )
        manager.add(ann)
        errors = manager.validate()
        assert len(errors) > 0
