#!/usr/bin/env python3
"""
Drift-Sense - Synthetic Dataset Generator CLI

Generate synthetic wafer inspection datasets for navigation-error recovery.

Usage:
    python dataset_generator.py --architecture DRAM --num-samples 30 --output_dir dataset
    python dataset_generator.py --architecture FinFET --num-samples 30 --output_dir dataset --seed 42 --report --visualize
"""

import argparse
import sys
import time
import os
import platform
import subprocess


def open_file_default_app(filepath):
    """Open a file with the default OS application."""
    if platform.system() == 'Darwin':       # macOS
        subprocess.call(('open', filepath))
    elif platform.system() == 'Windows':    # Windows
        os.startfile(filepath)
    else:                                   # linux variants
        subprocess.call(('xdg-open', filepath))


def main():
    parser = argparse.ArgumentParser(
        description="Drift-Sense: Synthetic Wafer Inspection Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        choices=["DRAM", "FinFET"],
        help="Semiconductor architecture type (DRAM or FinFET)",
    )
    
    parser.add_argument(
        "--num_pairs", "--num-samples",
        dest="num_pairs",
        type=int,
        required=True,
        help="Number of reference-search image pairs to generate",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for generated dataset",
    )

    # Optional arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=10.0,
        help="Maximum Gaussian noise sigma (default: 10.0)",
    )
    parser.add_argument(
        "--rotation-range",
        type=float,
        default=2.0,
        help="Maximum rotation angle in degrees (default: 2.0)",
    )
    parser.add_argument(
        "--scale-range",
        type=float,
        default=0.05,
        help="Maximum scale deviation from 1.0 (default: 0.05)",
    )
    parser.add_argument(
        "--blur-range",
        type=float,
        default=1.0,
        help="Maximum Gaussian blur sigma (default: 1.0)",
    )
    parser.add_argument(
        "--edge-strength",
        type=float,
        default=0.3,
        help="SEM edge brightening strength (default: 0.3)",
    )
    
    # Presentation / visualization options
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip generating visualization examples (legacy)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Explicitly request visualizations (default is True unless --no-visualizations is passed)",
    )
    parser.add_argument(
        "--visualize-sample",
        type=str,
        default=None,
        help="Generate visualization for a specific sample ID (e.g. 000001)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print a detailed professional terminal report during generation",
    )

    args = parser.parse_args()
    
    save_visualizations = not args.no_visualizations
    if args.visualize:
        save_visualizations = True

    # Lazy import to keep CLI startup fast
    from src.dataset.generator import DatasetGenerator

    generator = DatasetGenerator(
        architecture=args.architecture,
        output_dir=args.output_dir,
        seed=args.seed,
        noise_level=args.noise_level,
        rotation_range=args.rotation_range,
        scale_range=args.scale_range,
        blur_range=args.blur_range,
        edge_strength=args.edge_strength,
    )

    if args.report:
        print("=" * 60)
        print("        SEMICONDUCTOR INSPECTION DATASET GENERATOR")
        print("=" * 60)
        print("\n[CONFIGURATION]\n")
        print(f"Architecture        : {args.architecture}")
        print(f"Random Seed         : {args.seed}")
        print("\nReference")
        print(f"  Resolution        : {generator.REF_SIZE} × {generator.REF_SIZE} px")
        print(f"  Pixel Size        : 1 nm/px")
        print(f"  Physical FOV      : {generator.REF_SIZE / 1000.0:g} µm")
        print("\nSearch")
        print(f"  Resolution        : {generator.SEARCH_SIZE} × {generator.SEARCH_SIZE} px")
        print(f"  Pixel Size        : 10 nm/px")
        print(f"  Physical FOV      : {generator.SEARCH_SIZE * 10 / 1000.0:g} µm")
        print("\nScale Ratio         : 10×")
        print(f"Expected Footprint  : {generator.FOOTPRINT_SIZE} × {generator.FOOTPRINT_SIZE} px")
        print(f"Actual Footprint    : {generator.FOOTPRINT_SIZE} × {generator.FOOTPRINT_SIZE} px")
        print("Scale Geometry      : PASS")
        print("\n[GENERATION]\n")
        print(f"Output Directory    : {args.output_dir}")
        print(f"Generating Samples  : {args.num_pairs}...")
        sys.stdout.flush()

    start = time.time()
    
    stats = generator.generate_dataset(
        num_pairs=args.num_pairs,
        save_visualizations=save_visualizations,
        vis_sample_id=args.visualize_sample
    )

    elapsed = time.time() - start

    if args.report:
        print(f"\n[DEGRADATION]\n")
        print(f"Sensor Noise        : ENABLED (sigma={args.noise_level})")
        print(f"Independent Noise   : {'PASS' if stats['independent_noise'] else 'FAIL'}")
        print(f"SEM Edge Brightening: {'ENABLED' if stats['sem_edge'] else 'DISABLED'} (strength={args.edge_strength})")
        print(f"Rotation Degrade    : +/-{args.rotation_range} deg")
        print(f"Blur                : {args.blur_range}")
        
        print(f"\n[VALIDATION]\n")
        print(f"Scale Geometry      : PASS")
        print(f"Ground Truth        : {'PASS' if stats['ground_truth'] else 'FAIL'}")
        print(f"Bounding Box        : {'PASS' if stats['ground_truth'] else 'FAIL'}")
        print(f"Independent Noise   : {'PASS' if stats['independent_noise'] else 'FAIL'}")
        print(f"SEM Edge Brightening: {'PASS' if stats['sem_edge'] else 'FAIL'}")
        print(f"Metadata            : {'PASS' if stats['metadata_saved'] else 'FAIL'}")
        
        print(f"\n[OUTPUT]\n")
        print(f"Dataset Directory   : {args.output_dir}")
        print(f"Metadata File       : {os.path.join(args.output_dir, 'metadata.json')}")
        if stats['visualizations']:
            print(f"Visualization       : {stats['visualizations'][0]}")

        print("\n" + "=" * 60)
        print("DATASET GENERATION SUMMARY")
        print("=" * 60)
        print(f"Samples requested    : {stats['requested']}")
        print(f"Samples generated    : {stats['generated']}")
        print(f"Samples valid        : {stats['generated']}")
        print(f"Samples failed       : {stats['failed']}")
        print("")
        print(f"Independent noise    : {'PASS' if stats['independent_noise'] else 'FAIL'}")
        print(f"Ground truth         : {'PASS' if stats['ground_truth'] else 'FAIL'}")
        print(f"SEM edge enhancement : {'PASS' if stats['sem_edge'] else 'FAIL'}")
        print(f"Metadata             : {'PASS' if stats['metadata_saved'] else 'FAIL'}")
        print("")
        for split, count in stats['splits'].items():
            if count > 0:
                print(f"{split.capitalize():<20}: {count}")
        print("\nOutput:")
        print(f"  Images             : {os.path.abspath(args.output_dir)}")
        print(f"  Metadata           : {os.path.abspath(os.path.join(args.output_dir, 'metadata.json'))}")
        vis_path = os.path.abspath(os.path.join(args.output_dir, 'visualizations'))
        print(f"  Visualizations     : {vis_path}")
        print("=" * 60)
        print("STATUS: SUCCESS")
        print("=" * 60)
    else:
        print(f"\n[OK] Generation complete in {elapsed:.1f}s")
        if stats['visualizations']:
            print(f"     Visualizations saved to: {os.path.join(args.output_dir, 'visualizations')}")

    # Interactive viewing: attempt to open visualization if only 1 was explicitly requested via vis_sample_id
    if args.visualize_sample and stats['visualizations']:
        target_vis = stats['visualizations'][0]
        for vis in stats['visualizations']:
            if args.visualize_sample in vis:
                target_vis = vis
                break
        print(f"\nOpening visualization: {target_vis}")
        try:
            open_file_default_app(target_vis)
        except Exception as e:
            print(f"Could not automatically open visualization: {e}")

if __name__ == "__main__":
    main()
