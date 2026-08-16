# Drift-Sense: AI-Powered Navigation-Error Recovery for Wafer Inspection Tools

## 1. Project Purpose
Drift-Sense addresses navigation-error recovery for wafer inspection tools. Thermal expansion, vibration, and mechanical drift can cause inspection tools to land several pixels away from the intended location. Drift-Sense generates training datasets containing clean reference images and degraded, noisy search images mimicking Scanning Electron Microscope (SEM) captures, complete with strict ground truth, to train AI models for recovery.

## 2. Requirements
- Python 3.10+
- Dependencies listed in `requirements.txt`

## 3. Installation
```bash
git clone https://github.com/TeamDriftSense/drift-sense
cd drift-sense
pip install -r requirements.txt
```

## 4. Dataset Generation Command
Generate a dataset of DRAM or FinFET paired images:
```bash
python dataset_generator.py --architecture DRAM --num_pairs 30 --output_dir dataset/sample --seed 42
```

## 5. Configuration
The dataset generator exposes the following configurations:
- `--architecture`: Structure to synthesize (`DRAM` or `FinFET`)
- `--num_pairs`: Number of total generated image pairs
- `--output_dir`: Output directory
- `--seed`: Deterministic global random seed
- `--noise-level`: Standard deviation of the Gaussian sensor noise
- `--rotation-range`: Maximum random rotation angle applied
- `--scale-range`: Maximum scale variation
- `--blur-range`: Maximum Gaussian blur sigma
- `--edge-strength`: Intensity of the SEM edge-brightening

## 6. Dataset Structure
```text
dataset/sample/
├── train/
│   ├── reference/
│   ├── search/
│   └── annotations.json
├── validation/
│   ├── reference/
│   ├── search/
│   └── annotations.json
├── test/
│   ├── reference/
│   ├── search/
│   └── annotations.json
└── metadata.json
```

## 7. Metadata Format
The `annotations.json` follows this strict schema:
```json
{
  "id": "000000",
  "architecture": "DRAM",
  "reference": "reference/000000.png",
  "search": "search/000000.png",
  "image_width": 1000,
  "image_height": 1000,
  "center_x": 500,
  "center_y": 500,
  "bbox": {
    "x": 450,
    "y": 450,
    "width": 100,
    "height": 100
  },
  "bbox_x_min": 450,
  "bbox_y_min": 450,
  "bbox_x_max": 550,
  "bbox_y_max": 550,
  "augmentation_params": {
    "reference": { "noise_sigma": 10.0, "noise_seed": 12345 },
    "search": { "noise_sigma": 15.0, "noise_seed": 67890 }
  }
}
```

## 8. Ground-Truth Coordinate Convention
- Coordinates are zero-based `(x, y)` pixel indices in the search image.
- `x` is the horizontal axis (column index, 0 = left edge).
- `y` is the vertical axis (row index, 0 = top edge).
- The bounding box `(x, y)` represents the exact top-left corner where the reference pattern was placed before augmentations.

## 9. Noise-Generation Strategy
**Independent Sensor Noise**: Every single physical image capture receives completely independent sensor noise. To preserve statistical independence while maintaining deterministic reproducibility, a global seed initiates a root RNG. For every generated image (both reference and search separately), a deterministic sub-seed is drawn from the RNG and stored directly in the output metadata (`noise_seed`). This ensures the same `noise_seed` is never reused twice across splits or captures, effectively eliminating data leakage through identical noise structures.

## 10. SEM Edge-Brightening Approach
We simulate SEM secondary-electron emission characteristics where structural edges appear significantly brighter. 
- A Sobel gradient filter isolates structural edges.
- The gradient map is gently blurred to prevent harsh unphysical artifacts.
- The map is normalized and scaled by a configurable `edge_strength` factor.
- This contrast overlay is added physically to the image before sensor noise generation.

## 11. Reproducibility
The pipeline is strictly reproducible via the `--seed` argument. Providing a global seed guarantees exact generation outputs without sacrificing independent random noise draws for the individual reference and search physical captures.

## 12. Testing & Validation Commands
Run the complete regression and validation suite:
```bash
pytest tests/
```
Validate an existing dataset for correctness, bounding boxes, and missing data:
```bash
python validate_dataset.py --dataset dataset/sample
```

## 13. Example Generated Sample
To review a visual overlay of the bounding box and exact center coordinates:
```bash
# Generated if --no-visualizations is omitted
open results/vis_DRAM_000000.png 
```

## 14. Known Limitations
- The generator currently synthesizes purely 2D layouts and ignores z-height perspective variation.
- Perfect periodic ambiguity cannot be localized successfully without a unique contextual marker, representing a physical limitation of structural scanning.
