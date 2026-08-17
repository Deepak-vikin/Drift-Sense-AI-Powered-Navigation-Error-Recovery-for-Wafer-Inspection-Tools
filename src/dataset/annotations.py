from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional
import json
from pathlib import Path

@dataclass
class Annotation:
    """Ground-truth annotation for a single reference-search image pair.

    Coordinate convention:
        x = horizontal axis (column index), 0 = left edge
        y = vertical axis   (row index),    0 = top edge
        All coordinates are zero-based pixel indices in the search image.

    Bounding box:
        bbox dict contains {x, y, width, height} where (x, y) is the
        top-left corner of the reference region inside the search image.
        bbox_x_min/y_min/x_max/y_max provide explicit corner coordinates.
    """
    id: str
    architecture: str
    reference: str          # relative path to reference image
    search: str             # relative path to search image
    
    reference_width_px: int
    reference_height_px: int
    reference_pixel_size_nm: float
    search_width_px: int
    search_height_px: int
    search_pixel_size_nm: float
    reference_physical_width_nm: float
    reference_physical_height_nm: float
    expected_footprint_width_px: float
    expected_footprint_height_px: float
    
    ground_truth_x: int
    ground_truth_y: int
    ground_truth_width: int
    ground_truth_height: int
    ground_truth_center_x: float
    ground_truth_center_y: float

    # Legacy fields maintained for backward compatibility in parts of code, though replaced by ground_truth_*
    image_width: int        
    image_height: int       
    center_x: int           
    center_y: int           
    bbox: dict              
    bbox_x_min: int         
    bbox_y_min: int         
    bbox_x_max: int         
    bbox_y_max: int         
    
    augmentation_params: dict
    seed: Optional[int] = None

class AnnotationManager:
    """Manages collection, saving, and loading of dataset annotations."""

    def __init__(self):
        self.annotations: List[Annotation] = []

    def add(self, annotation: Annotation) -> None:
        """Add an annotation record."""
        self.annotations.append(annotation)

    def save(self, filepath: Path) -> None:
        """Save annotations to JSON file."""
        data = [asdict(a) for a in self.annotations]
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> 'AnnotationManager':
        """Load annotations from JSON file."""
        manager = cls()
        with open(filepath) as f:
            data = json.load(f)
        for item in data:
            manager.annotations.append(Annotation(**item))
        return manager

    def validate(self) -> List[str]:
        """Validate all annotations. Return list of error messages."""
        errors = []
        seen_ids = set()
        for ann in self.annotations:
            # Duplicate ID check
            if ann.id in seen_ids:
                errors.append(f"Annotation {ann.id}: duplicate sample ID")
            seen_ids.add(ann.id)

            w = ann.image_width
            h = ann.image_height

            # Center inside image (strict < for zero-based)
            if not (0 <= ann.center_x < w and 0 <= ann.center_y < h):
                errors.append(
                    f"Annotation {ann.id}: center ({ann.center_x}, {ann.center_y}) "
                    f"out of bounds for {w}x{h} image"
                )

            # Bbox dict structure
            if not isinstance(ann.bbox, dict):
                errors.append(f"Annotation {ann.id}: bbox is not a dict")
            else:
                for k in ["x", "y", "width", "height"]:
                    if k not in ann.bbox:
                        errors.append(f"Annotation {ann.id}: bbox missing '{k}'")

                # Positive dimensions
                if ann.bbox.get("width", 0) <= 0 or ann.bbox.get("height", 0) <= 0:
                    errors.append(f"Annotation {ann.id}: bbox has non-positive dimensions")

                # Bbox inside image
                bx = ann.bbox.get("x", 0)
                by = ann.bbox.get("y", 0)
                bw = ann.bbox.get("width", 0)
                bh = ann.bbox.get("height", 0)
                if bx < 0 or by < 0 or bx + bw > w or by + bh > h:
                    errors.append(
                        f"Annotation {ann.id}: bbox ({bx},{by},{bw},{bh}) "
                        f"outside image bounds {w}x{h}"
                    )

            # Explicit corner fields match bbox dict
            if ann.bbox_x_min != ann.bbox.get("x"):
                errors.append(f"Annotation {ann.id}: bbox_x_min != bbox['x']")
            if ann.bbox_y_min != ann.bbox.get("y"):
                errors.append(f"Annotation {ann.id}: bbox_y_min != bbox['y']")
            expected_x_max = ann.bbox.get("x", 0) + ann.bbox.get("width", 0)
            expected_y_max = ann.bbox.get("y", 0) + ann.bbox.get("height", 0)
            if ann.bbox_x_max != expected_x_max:
                errors.append(f"Annotation {ann.id}: bbox_x_max mismatch")
            if ann.bbox_y_max != expected_y_max:
                errors.append(f"Annotation {ann.id}: bbox_y_max mismatch")

        return errors
