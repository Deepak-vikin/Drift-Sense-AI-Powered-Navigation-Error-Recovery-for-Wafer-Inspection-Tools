import time
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple
from src.architectures.dram import DRAMGenerator
from src.architectures.finfet import FinFETGenerator
from src.localization.localization import localize
from src.evaluation.metrics import calculate_error

def _add_guaranteed_marker(img: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """Add a unique morphological marker at the center to break perfect periodicity."""
    cv2.circle(img, (cx, cy), 15, 255, -1)
    cv2.circle(img, (cx - 10, cy - 10), 5, 0, -1)
    return np.clip(img, 0, 255).astype(np.uint8)

def generate_benchmark_set(
    condition: str, architecture: str, num_samples: int, seed_offset: int = 0
) -> List[Dict[str, Any]]:
    """Generate a benchmark dataset with controlled degradations.
    
    Conditions: 'clean', 'low_noise', 'medium_noise', 'high_noise',
                'blur', 'rotation', 'scale', 'combined', 'high_periodicity',
                'search_noisier'
    """
    dataset = []
    
    # We use a base generator. For high_periodicity we omit the 40 random sparse blobs.
    # We still add the guaranteed marker for all tests EXCEPT high_periodicity 
    # to evaluate how the algorithm handles pure periodic ambiguity vs unique features.
    
    for i in range(num_samples):
        seed = 1000 * seed_offset + i
        rng = np.random.RandomState(seed)
        
        if architecture.lower() == "dram":
            gen = DRAMGenerator(seed=seed)
        else:
            gen = FinFETGenerator(seed=seed)
            
        base = gen.generate_base_pattern(1200, 1200)
        img = base[:1000, :1000].astype(np.int32)
        
        # Default target center
        cx = int(rng.randint(200, 800))
        cy = int(rng.randint(200, 800))
        ref_size = 100
        
        is_highly_periodic = (condition == 'high_periodicity')
        
        if not is_highly_periodic:
            # Add sparse blobs to break symmetry
            for _ in range(40):
                bx = rng.randint(10, 990)
                by = rng.randint(10, 990)
                cv2.circle(img, (bx, by), rng.randint(4, 10), int(rng.randint(180, 255)), -1)
            # Add unique marker at GT
            img = _add_guaranteed_marker(img, cx, cy)
            
        search = np.clip(img, 0, 255).astype(np.uint8)
        
        # Crop reference
        half = ref_size // 2
        ref = search[cy - half:cy + half, cx - half:cx + half].copy()
        
        # Apply degradations based on condition
        ref_noise, search_noise = 0.0, 0.0
        ref_blur, search_blur = 0.0, 0.0
        scale_factor = 1.0
        rotation = 0.0
        
        if condition == 'low_noise':
            ref_noise, search_noise = 5.0, 5.0
        elif condition == 'medium_noise':
            ref_noise, search_noise = 15.0, 15.0
        elif condition == 'high_noise':
            ref_noise, search_noise = 30.0, 30.0
        elif condition == 'search_noisier':
            ref_noise, search_noise = 5.0, 25.0
        elif condition == 'blur':
            search_blur = 1.5
            ref_blur = 1.5
        elif condition == 'rotation':
            rotation = rng.uniform(-1.5, 1.5)
        elif condition == 'scale':
            scale_factor = rng.uniform(0.95, 1.05)
        elif condition == 'combined':
            ref_noise, search_noise = 10.0, 20.0
            search_blur = 1.0
            rotation = rng.uniform(-1.0, 1.0)
            scale_factor = rng.uniform(0.97, 1.03)
            
        # Apply transforms to ref
        if scale_factor != 1.0:
            new_s = int(ref_size * scale_factor)
            ref = cv2.resize(ref, (new_s, new_s))
            # pad or crop back to ref_size
            pad = np.full((ref_size, ref_size), 30, dtype=np.uint8)
            h, w = ref.shape[:2]
            dy = (ref_size - h) // 2
            dx = (ref_size - w) // 2
            if scale_factor < 1.0:
                pad[dy:dy+h, dx:dx+w] = ref
                ref = pad
            else:
                ref = ref[-dy:-dy+ref_size, -dx:-dx+ref_size]
                
        if rotation != 0.0:
            M = cv2.getRotationMatrix2D((ref_size/2, ref_size/2), rotation, 1.0)
            ref = cv2.warpAffine(ref, M, (ref_size, ref_size), borderMode=cv2.BORDER_REPLICATE)
            
        if ref_blur > 0:
            ref = cv2.GaussianBlur(ref, (5, 5), ref_blur)
        if search_blur > 0:
            search = cv2.GaussianBlur(search, (5, 5), search_blur)
            
        if ref_noise > 0:
            ref_noise_rng = np.random.RandomState(seed + 9000 + i)
            noise = ref_noise_rng.normal(0, ref_noise, ref.shape)
            ref = np.clip(ref.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if search_noise > 0:
            search_noise_rng = np.random.RandomState(seed + 18000 + i)
            noise = search_noise_rng.normal(0, search_noise, search.shape)
            search = np.clip(search.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        dataset.append({
            "sample_id": f"{architecture}_{condition}_{i}",
            "architecture": architecture,
            "condition": condition,
            "true_x": cx,
            "true_y": cy,
            "reference": ref,
            "search": search
        })
        
    return dataset


def baseline_match(reference: np.ndarray, search: np.ndarray) -> Tuple[int, int, float, float]:
    """Simple single-scale normalized template matching."""
    t0 = time.perf_counter()
    res = cv2.matchTemplate(search, reference, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    th, tw = reference.shape[:2]
    cx = max_loc[0] + tw // 2
    cy = max_loc[1] + th // 2
    runtime = (time.perf_counter() - t0) * 1000.0
    return cx, cy, float(max_val), runtime


def run_benchmark(dataset: List[Dict[str, Any]], method: str = 'proposed', **kwargs) -> List[Dict[str, Any]]:
    """Run evaluation on a dataset."""
    results = []
    
    for item in dataset:
        ref = item["reference"]
        search = item["search"]
        true_x = item["true_x"]
        true_y = item["true_y"]
        
        success = True
        error = float('inf')
        
        if method == 'baseline':
            pred_x, pred_y, score, runtime = baseline_match(ref, search)
        else:
            try:
                res = localize(ref, search, **kwargs)
                pred_x, pred_y = res.center_x, res.center_y
                score = res.score
                runtime = res.runtime_ms
            except Exception as e:
                success = False
                pred_x, pred_y, score, runtime = -1, -1, 0.0, 0.0
                print(f"Error on {item['sample_id']}: {e}")
                
        if success:
            error = calculate_error(true_x, true_y, pred_x, pred_y)
            
        results.append({
            "sample_id": item["sample_id"],
            "architecture": item["architecture"],
            "condition": item["condition"],
            "true_x": true_x,
            "true_y": true_y,
            "pred_x": pred_x,
            "pred_y": pred_y,
            "error_pixels": error,
            "score": score,
            "inference_time_ms": runtime,
            "success": success,
            "method": method
        })
        
    return results
