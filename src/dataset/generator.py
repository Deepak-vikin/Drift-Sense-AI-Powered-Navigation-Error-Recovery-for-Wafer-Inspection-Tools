"""
Core dataset generator for Drift-Sense synthetic wafer inspection datasets.

Augmentation Pipeline Order:
    base semiconductor structure
        ->
    geometric variation (rotation)
        ->
    scale variation
        ->
    blur
        ->
    SEM edge brightening
        ->
    independent sensor noise
        ->
    final image

The reference and search images share the same underlying semiconductor
pattern but receive independently sampled augmentations.

Scale Relationship:
    The search image (1000x1000) covers 10x the field-of-view of the
    reference image (~100x100). Both are at the same pixel resolution.
    The reference pattern occupies a 100x100 pixel region within the
    search image.
"""

import numpy as np
import cv2
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

from src.architectures import ARCHITECTURE_MAP
from src.augmentation.noise import add_gaussian_noise
from src.augmentation.blur import apply_gaussian_blur
from src.augmentation.geometry import apply_rotation, apply_scale
from src.augmentation.edge_effects import apply_sem_edge_brightening
from src.utils.visualization import visualize_pair_advanced, visualize_dataset_summary, save_visualization
from .annotations import Annotation, AnnotationManager


class DatasetGenerator:
    """Core orchestrator for generating synthetic wafer inspection datasets.

    Generates reference-search image pairs with independent augmentations
    and ground-truth annotations.

    Attributes:
        architecture: Name of the semiconductor architecture ('DRAM' or 'FinFET').
        output_dir: Output directory path.
        seed: Random seed for reproducibility.
        noise_level: Maximum Gaussian noise sigma.
        rotation_range: Maximum rotation in degrees (±).
        scale_range: Maximum scale deviation from 1.0.
        blur_range: Maximum Gaussian blur sigma.
        edge_strength: SEM edge brightening strength.
    """

    # Reference image size (exact)
    REF_SIZE = 1000
    # Search image size (exact)
    SEARCH_SIZE = 1000
    # Footprint size
    FOOTPRINT_SIZE = 100
    
    # Reference pixel size (nm)
    REF_PIXEL_NM = 1.0
    # Search pixel size (nm)
    SEARCH_PIXEL_NM = 10.0

    def __init__(
        self,
        architecture: str,
        output_dir: str,
        seed: Optional[int] = None,
        noise_level: float = 10.0,
        rotation_range: float = 2.0,
        scale_range: float = 0.05,
        blur_range: float = 1.0,
        edge_strength: float = 0.3,
    ):
        if architecture not in ARCHITECTURE_MAP:
            raise ValueError(
                f"Unknown architecture '{architecture}'. "
                f"Supported: {list(ARCHITECTURE_MAP.keys())}"
            )

        self.architecture = architecture
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.noise_level = noise_level
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.blur_range = blur_range
        self.edge_strength = edge_strength

        # Instantiate the architecture generator
        self.arch_gen = ARCHITECTURE_MAP[architecture](seed=seed)

    # ------------------------------------------------------------------ #
    #  Augmentation pipeline                                              #
    # ------------------------------------------------------------------ #

    def _sample_aug_params(self, is_search: bool) -> Dict[str, float]:
        """Sample random augmentation parameters for one image.

        Search images get higher noise than reference images to model
        the lower SNR of the wider-field capture.
        """
        noise_sigma = float(self.rng.uniform(
            self.noise_level * 0.2, self.noise_level
        ))
        # Search images are noisier
        if is_search:
            noise_sigma *= 1.5

        return {
            "rotation_deg": float(
                self.rng.uniform(-self.rotation_range, self.rotation_range)
            ),
            "scale_factor": float(
                1.0 + self.rng.uniform(-self.scale_range, self.scale_range)
            ),
            "blur_sigma": float(
                self.rng.uniform(0.3, self.blur_range)
            ),
            "edge_strength": float(self.edge_strength),
            "noise_sigma": noise_sigma,
        }

    def _apply_pre_noise_augmentation(
        self, image: np.ndarray, params: Dict[str, float]
    ) -> np.ndarray:
        """Apply geometric, blur, and edge augmentations (NO NOISE)."""
        img = image.copy()

        # 1. Rotation
        if abs(params["rotation_deg"]) > 1e-6:
            img = apply_rotation(img, params["rotation_deg"])

        # 2. Scale
        if abs(params["scale_factor"] - 1.0) > 1e-6:
            img = apply_scale(img, params["scale_factor"])

        # 3. Blur
        if params["blur_sigma"] > 0.1:
            img = apply_gaussian_blur(img, params["blur_sigma"])

        # 4. SEM edge brightening
        if params["edge_strength"] > 0.0:
            img = apply_sem_edge_brightening(img, params["edge_strength"])
            
        return img
        
    def _apply_noise(self, image: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Apply independent sensor noise."""
        noise_seed = int(self.rng.randint(0, 2**31))
        noise_rng = np.random.RandomState(noise_seed)
        img = add_gaussian_noise(
            image, mean=0.0, sigma=params["noise_sigma"], rng=noise_rng
        )
        params["noise_seed"] = noise_seed
        return img

    def _apply_augmentation_pipeline(
        self, image: np.ndarray, params: Dict[str, float]
    ) -> np.ndarray:
        """Apply the full augmentation pipeline (Legacy backward compat)."""
        img = self._apply_pre_noise_augmentation(image, params)
        img = self._apply_noise(img, params)
        return img

    # ------------------------------------------------------------------ #
    #  Pair generation                                                    #
    # ------------------------------------------------------------------ #

    def generate_pair(
        self, pair_id: str
    ) -> Tuple[np.ndarray, np.ndarray, Annotation]:
        """Generate a single reference-search image pair with annotation.
        
        1. Generate 1000x1000 reference base at 1 nm/px.
        2. Generate 1000x1000 search base at 10 nm/px.
        3. Apply geometric, blur, edge augmentations to both.
        4. Downsample augmented reference by 10x to 100x100 footprint.
        5. Insert footprint into random location in augmented search image.
        6. Apply independent noise to reference and search images.
        """
        # 1. Generate independent base patterns
        ref_base = self.arch_gen.generate_base_pattern(self.REF_SIZE, self.REF_SIZE)
        search_base = self.arch_gen.generate_base_pattern(self.SEARCH_SIZE, self.SEARCH_SIZE)

        # 2. Sample independent parameters
        ref_params = self._sample_aug_params(is_search=False)
        search_params = self._sample_aug_params(is_search=True)

        # 3. Apply PRE-NOISE augmentations (geometric, blur, edge)
        ref_img_pre = self._apply_pre_noise_augmentation(ref_base, ref_params)
        search_img_pre = self._apply_pre_noise_augmentation(search_base, search_params)

        # 4. Downsample reference to footprint size (1000 -> 100)
        footprint = cv2.resize(
            ref_img_pre, 
            (self.FOOTPRINT_SIZE, self.FOOTPRINT_SIZE), 
            interpolation=cv2.INTER_AREA
        )

        # 5. Pick a random target center fully inside the search area
        half_footprint = self.FOOTPRINT_SIZE // 2
        margin = half_footprint + 10
        cx = int(self.rng.randint(margin, self.SEARCH_SIZE - margin))
        cy = int(self.rng.randint(margin, self.SEARCH_SIZE - margin))

        # Insert footprint into search image
        search_img_pre[
            cy - half_footprint : cy + half_footprint,
            cx - half_footprint : cx + half_footprint
        ] = footprint

        # 6. Apply independent noise AFTER insertion
        ref_img = self._apply_noise(ref_img_pre, ref_params)
        search_img = self._apply_noise(search_img_pre, search_params)

        # Validate dimensions
        assert ref_img.shape == (self.REF_SIZE, self.REF_SIZE)
        assert search_img.shape == (self.SEARCH_SIZE, self.SEARCH_SIZE)

        bbox_x = cx - half_footprint
        bbox_y = cy - half_footprint
        
        ref_phys_w = self.REF_SIZE * self.REF_PIXEL_NM
        ref_phys_h = self.REF_SIZE * self.REF_PIXEL_NM
        
        expected_w = ref_phys_w / self.SEARCH_PIXEL_NM
        expected_h = ref_phys_h / self.SEARCH_PIXEL_NM

        annotation = Annotation(
            id=pair_id,
            architecture=self.architecture,
            reference=f"reference/{pair_id}.png",
            search=f"search/{pair_id}.png",
            seed=self.seed,
            
            # New exhaustive metadata
            reference_width_px=self.REF_SIZE,
            reference_height_px=self.REF_SIZE,
            reference_pixel_size_nm=self.REF_PIXEL_NM,
            search_width_px=self.SEARCH_SIZE,
            search_height_px=self.SEARCH_SIZE,
            search_pixel_size_nm=self.SEARCH_PIXEL_NM,
            reference_physical_width_nm=ref_phys_w,
            reference_physical_height_nm=ref_phys_h,
            expected_footprint_width_px=expected_w,
            expected_footprint_height_px=expected_h,
            
            ground_truth_x=bbox_x,
            ground_truth_y=bbox_y,
            ground_truth_width=self.FOOTPRINT_SIZE,
            ground_truth_height=self.FOOTPRINT_SIZE,
            ground_truth_center_x=float(cx),
            ground_truth_center_y=float(cy),
            
            # Legacy backward compat metadata
            image_width=self.SEARCH_SIZE,
            image_height=self.SEARCH_SIZE,
            center_x=cx,
            center_y=cy,
            bbox={
                "x": bbox_x,
                "y": bbox_y,
                "width": self.FOOTPRINT_SIZE,
                "height": self.FOOTPRINT_SIZE,
            },
            bbox_x_min=bbox_x,
            bbox_y_min=bbox_y,
            bbox_x_max=bbox_x + self.FOOTPRINT_SIZE,
            bbox_y_max=bbox_y + self.FOOTPRINT_SIZE,
            
            augmentation_params={
                "reference": ref_params,
                "search": search_params,
            },
        )

        return ref_img, search_img, annotation

    # ------------------------------------------------------------------ #
    #  Full dataset generation                                            #
    # ------------------------------------------------------------------ #

    def generate_dataset(
        self,
        num_pairs: int,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        save_visualizations: bool = True,
        vis_sample_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate the full dataset with train/validation/test splits.

        Args:
            num_pairs: Total number of image pairs to generate.
            split_ratios: Fraction of pairs for (train, validation, test).
            save_visualizations: Whether to save example visualizations.
            vis_sample_id: Generate visualization for specific sample ID.
            
        Returns:
            Dictionary with generation statistics.
        """
        # Compute split counts
        num_train = int(num_pairs * split_ratios[0])
        num_val = int(num_pairs * split_ratios[1])
        num_test = num_pairs - num_train - num_val

        split_counts = {
            "train": num_train,
            "validation": num_val,
            "test": num_test,
        }

        # Save metadata
        metadata = {
            "architecture": self.architecture,
            "num_pairs": num_pairs,
            "seed": self.seed,
            "noise_level": self.noise_level,
            "rotation_range": self.rotation_range,
            "scale_range": self.scale_range,
            "blur_range": self.blur_range,
            "edge_strength": self.edge_strength,
            "split_ratios": list(split_ratios),
            "split_counts": split_counts,
            "reference_size": self.REF_SIZE,
            "search_size": self.SEARCH_SIZE,
            "scale_relationship": "10x field-of-view (search covers 10x reference area per dimension)",
            "augmentation_pipeline": [
                "geometric_rotation",
                "scale_variation",
                "gaussian_blur",
                "sem_edge_brightening",
                "independent_sensor_noise",
            ],
            "generated_at": datetime.now().isoformat(),
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Create dir for visualizations
        vis_dir = self.output_dir / "visualizations"
        if save_visualizations or vis_sample_id:
            vis_dir.mkdir(parents=True, exist_ok=True)

        current_idx = 0
        vis_count = 0
        max_vis = min(5, num_pairs)  # save up to 5 example visualizations
        
        # Collect for dataset summary
        summary_anns = []
        summary_refs = []
        summary_searches = []
        
        # Stats tracking
        stats = {
            "requested": num_pairs,
            "generated": 0,
            "failed": 0,
            "independent_noise": True,
            "ground_truth": True,
            "sem_edge": True,
            "metadata_saved": True,
            "splits": split_counts,
            "visualizations": [],
            "summary_vis": None
        }

        for split_name, count in split_counts.items():
            if count == 0:
                continue

            split_dir = self.output_dir / split_name
            ref_dir = split_dir / "reference"
            search_dir = split_dir / "search"

            ref_dir.mkdir(parents=True, exist_ok=True)
            search_dir.mkdir(parents=True, exist_ok=True)

            manager = AnnotationManager()

            for i in range(count):
                pair_id = f"{current_idx:06d}"
                ref_img, search_img, ann = self.generate_pair(pair_id)

                # Save images
                cv2.imwrite(str(ref_dir / f"{pair_id}.png"), ref_img)
                cv2.imwrite(str(search_dir / f"{pair_id}.png"), search_img)

                manager.add(ann)

                # Validation checks for stats
                ref_noise = ann.augmentation_params.get('reference', {}).get('noise_seed')
                search_noise = ann.augmentation_params.get('search', {}).get('noise_seed')
                if ref_noise is None or search_noise is None or ref_noise == search_noise:
                    stats["independent_noise"] = False
                    
                edge_str = ann.augmentation_params.get('reference', {}).get('edge_strength', 0.0)
                if edge_str <= 0.0:
                    stats["sem_edge"] = False

                # Save visualizations
                is_vis_target = (vis_sample_id and vis_sample_id == pair_id)
                should_vis = is_vis_target or (save_visualizations and vis_count < max_vis)
                
                if should_vis:
                    fig = visualize_pair_advanced(
                        ref_img,
                        search_img,
                        ann,
                        ref_scale_nm=1.0,
                        search_scale_nm=10.0,
                        seed=self.seed
                    )
                    vis_path = vis_dir / f"{pair_id}_comparison.png"
                    save_visualization(fig, vis_path)
                    stats["visualizations"].append(str(vis_path))
                    vis_count += 1
                    
                if len(summary_anns) < 3:
                    summary_anns.append(ann)
                    summary_refs.append(ref_img)
                    summary_searches.append(search_img)

                current_idx += 1
                stats["generated"] += 1

            manager.save(split_dir / "annotations.json")

        if save_visualizations and len(summary_anns) > 1:
            fig_summary = visualize_dataset_summary(summary_anns, summary_refs, summary_searches)
            if fig_summary:
                summary_path = vis_dir / "dataset_summary.png"
                save_visualization(fig_summary, summary_path)
                stats["summary_vis"] = str(summary_path)

        return stats
