#!/usr/bin/env python3
"""
Drift-Sense Dataset Validation Utility

Inspects a generated dataset and reports validity statistics.
Usage:
    python validate_dataset.py --dataset path/to/dataset
"""

import argparse
import json
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Validate a Drift-Sense generated dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory '{dataset_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"ERROR: Missing metadata.json in '{dataset_dir}'", file=sys.stderr)
        sys.exit(1)

    total_samples = 0
    valid_samples = 0
    invalid_samples = 0
    missing_images = 0
    missing_metadata = 0
    duplicate_ids = 0
    bbox_errors = 0
    noise_seed_dupes = 0
    
    seen_ids = set()
    seen_noise_seeds = set()

    print(f"Validating dataset at: {dataset_dir}")
    
    splits = ["train", "validation", "test"]
    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
            
        ann_path = split_dir / "annotations.json"
        if not ann_path.exists():
            print(f"WARNING: Missing annotations.json in {split_dir}")
            missing_metadata += 1
            continue

        try:
            with open(ann_path, 'r') as f:
                annotations = json.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load {ann_path}: {e}")
            invalid_samples += 1
            continue
            
        for ann in annotations:
            total_samples += 1
            is_valid = True
            ann_id = ann.get('id')
            
            if not ann_id:
                is_valid = False
            elif ann_id in seen_ids:
                duplicate_ids += 1
                is_valid = False
            else:
                seen_ids.add(ann_id)
                
            # Check images
            ref_path = split_dir / ann.get('reference', '')
            search_path = split_dir / ann.get('search', '')
            
            if not ref_path.exists() or not search_path.exists():
                missing_images += 1
                is_valid = False
            
            # Check bbox and geometry
            bbox = ann.get('bbox', {})
            x = bbox.get('x', -1)
            y = bbox.get('y', -1)
            w = bbox.get('width', -1)
            h = bbox.get('height', -1)
            
            img_w = ann.get('search_width_px', 1000)
            img_h = ann.get('search_height_px', 1000)
            
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > img_w or y + h > img_h:
                bbox_errors += 1
                is_valid = False
                
            # Verify Physical Scale Geometry
            ref_w = ann.get('reference_width_px')
            ref_px_nm = ann.get('reference_pixel_size_nm')
            search_px_nm = ann.get('search_pixel_size_nm')
            
            if ref_w is not None and ref_px_nm is not None and search_px_nm is not None:
                expected_footprint_w = (ref_w * ref_px_nm) / search_px_nm
                actual_footprint_w = ann.get('ground_truth_width', w)
                
                if abs(expected_footprint_w - actual_footprint_w) > 1.0:
                    bbox_errors += 1
                    is_valid = False
                    print(f"Scale Geometry ERROR on {ann_id}: expected footprint {expected_footprint_w}, actual {actual_footprint_w}")
                    
            # Check noise independence
            aug_params = ann.get('augmentation_params', {})
            ref_params = aug_params.get('reference', {})
            search_params = aug_params.get('search', {})
            
            ref_seed = ref_params.get('noise_seed')
            search_seed = search_params.get('noise_seed')
            
            if ref_seed is not None:
                if ref_seed in seen_noise_seeds:
                    noise_seed_dupes += 1
                    is_valid = False
                seen_noise_seeds.add(ref_seed)
                
            if search_seed is not None:
                if search_seed in seen_noise_seeds:
                    noise_seed_dupes += 1
                    is_valid = False
                seen_noise_seeds.add(search_seed)

            if is_valid:
                valid_samples += 1
            else:
                invalid_samples += 1

    print("\n--- Validation Report ---")
    print(f"Total samples:           {total_samples}")
    print(f"Valid samples:           {valid_samples}")
    print(f"Invalid samples:         {invalid_samples}")
    print(f"Missing images:          {missing_images}")
    print(f"Missing metadata files:  {missing_metadata}")
    print(f"Duplicate sample IDs:    {duplicate_ids}")
    print(f"Invalid bounding boxes:  {bbox_errors}")
    print(f"Duplicate noise seeds:   {noise_seed_dupes}")
    
    if invalid_samples == 0 and missing_images == 0 and missing_metadata == 0 and duplicate_ids == 0 and bbox_errors == 0 and noise_seed_dupes == 0:
        print("\n[PASS] Dataset is completely valid.")
        sys.exit(0)
    else:
        print("\n[FAIL] Dataset has validation errors.")
        sys.exit(1)

if __name__ == "__main__":
    main()
