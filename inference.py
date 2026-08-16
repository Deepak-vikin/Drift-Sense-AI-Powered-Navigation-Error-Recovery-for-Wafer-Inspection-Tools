#!/usr/bin/env python3
"""
Drift-Sense - Inference Engine

Localize a reference patch inside a search image.

Official usage:
    python inference.py path/to/reference.png path/to/search.png

Output (single line, machine-readable):
    (634,421)

The output is the predicted CENTER coordinate (x, y) of the reference
pattern within the search image.

Coordinate convention:
    x = horizontal axis (column), 0 = left edge
    y = vertical axis (row),      0 = top edge

No architecture argument is required.
No ground-truth file is required.
No training is performed at inference time.

Optional flags:
    --verbose           Print detailed diagnostics to stderr.
    --save-debug DIR    Save similarity map and candidate visualization
                        to DIR/ (does not affect stdout output).
    --scales FLOAT...   Override default scale range, e.g. --scales 0.9 1.0 1.1
    --alpha FLOAT       Intensity NCC weight (default 0.6).
    --beta FLOAT        Structural NCC weight (default 0.4).
    --threshold FLOAT   Score threshold for valid candidates (default 0.3).
"""

import sys
import os
import argparse


def _parse_args():
    p = argparse.ArgumentParser(
        description="Drift-Sense: Localize reference patch in search image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("reference", help="Path to reference image (~100x100 px, grayscale).")
    p.add_argument("search", help="Path to search image (1000x1000 px, grayscale).")
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print diagnostics to stderr (stdout stays clean).",
    )
    p.add_argument(
        "--save-debug",
        metavar="DIR",
        default=None,
        help="Save debug visualizations to this directory.",
    )
    p.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=None,
        metavar="SCALE",
        help="Scale factors to test (default: 0.7 to 1.4 in steps).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Weight for intensity NCC in combined score (default 1.0).",
    )
    p.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="Weight for structural NCC in combined score (default 0.0).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Minimum score for a valid candidate (default 0.3).",
    )
    return p.parse_args()


def _save_debug_outputs(
    reference, search, result, debug_dir: str
):
    """Save debug visualizations to debug_dir."""
    import cv2
    import numpy as np

    os.makedirs(debug_dir, exist_ok=True)

    # Draw candidate bounding box on search image
    vis = cv2.cvtColor(search, cv2.COLOR_GRAY2BGR)
    cx, cy = result.center_x, result.center_y
    hw = result.scale * 50  # half-width estimate
    x0 = int(cx - hw)
    y0 = int(cy - hw)
    x1 = int(cx + hw)
    y1 = int(cy + hw)
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.drawMarker(vis, (cx, cy), (0, 0, 255),
                   cv2.MARKER_CROSS, 20, 2)

    # Label
    label = f"({cx},{cy}) score={result.score:.3f} scale={result.scale:.2f}"
    cv2.putText(vis, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    out_path = os.path.join(debug_dir, "debug_localization.png")
    cv2.imwrite(out_path, vis)

    # Save reference alongside
    ref_path = os.path.join(debug_dir, "debug_reference.png")
    cv2.imwrite(ref_path, reference)

    return out_path


def main():
    args = _parse_args()

    # --- Lazy import (keeps CLI startup snappy) ---
    try:
        import cv2
        import numpy as np
        from src.localization.localization import localize, load_grayscale
    except ImportError as e:
        print(
            f"ERROR: Missing dependency: {e}\n"
            "Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(2)

    # --- Load images ---
    try:
        reference = load_grayscale(args.reference)
    except FileNotFoundError:
        print(f"ERROR: Reference image not found: {args.reference}", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"ERROR: Cannot read reference image: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        search = load_grayscale(args.search)
    except FileNotFoundError:
        print(f"ERROR: Search image not found: {args.search}", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"ERROR: Cannot read search image: {e}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(
            f"[INFO] reference: {args.reference} shape={reference.shape}",
            file=sys.stderr,
        )
        print(
            f"[INFO] search:    {args.search} shape={search.shape}",
            file=sys.stderr,
        )
        print(
            f"[INFO] alpha={args.alpha} beta={args.beta} "
            f"threshold={args.threshold}",
            file=sys.stderr,
        )

    # --- Run localization ---
    try:
        result = localize(
            reference=reference,
            search=search,
            scales=args.scales,
            alpha=args.alpha,
            beta=args.beta,
            score_threshold=args.threshold,
        )
    except ValueError as e:
        print(f"ERROR: Localization failed: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Debug outputs (optional) ---
    if args.save_debug:
        try:
            out_path = _save_debug_outputs(
                reference, search, result, args.save_debug
            )
            if args.verbose:
                print(f"[INFO] Debug saved to: {out_path}", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Could not save debug output: {e}", file=sys.stderr)

    # --- Verbose diagnostics to stderr ---
    if args.verbose:
        fb = " [FALLBACK]" if result.fallback_used else ""
        print(
            f"[INFO] candidates={result.num_candidates} "
            f"score={result.score:.4f} "
            f"scale={result.scale:.3f} "
            f"runtime={result.runtime_ms:.1f}ms"
            f"{fb}",
            file=sys.stderr,
        )

    # --- OFFICIAL OUTPUT to stdout — exactly one coordinate ---
    print(f"({result.center_x},{result.center_y})")


if __name__ == "__main__":
    main()
