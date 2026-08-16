import math
from typing import List, Dict, Any

def calculate_error(true_x: float, true_y: float, pred_x: float, pred_y: float) -> float:
    """Calculate L2 distance between predicted and true coordinates."""
    return math.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)

def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate individual sample results into summary metrics."""
    if not results:
        return {}

    errors = [r["error_pixels"] for r in results if r["success"]]
    times = [r["inference_time_ms"] for r in results if r["success"]]
    num_samples = len(results)
    num_success = len(errors)

    if num_success == 0:
        return {
            "num_samples": num_samples,
            "valid_samples": 0,
            "mean_error": float('inf'),
            "median_error": float('inf'),
            "max_error": float('inf'),
            "acc_2px": 0.0,
            "acc_5px": 0.0,
            "acc_10px": 0.0,
            "mean_time_ms": 0.0,
            "median_time_ms": 0.0,
        }

    errors.sort()
    times.sort()

    mean_error = sum(errors) / num_success
    median_error = errors[num_success // 2] if num_success % 2 != 0 else (errors[num_success // 2 - 1] + errors[num_success // 2]) / 2.0
    max_error = max(errors)
    
    acc_2px = sum(1 for e in errors if e <= 2.0) / num_samples
    acc_5px = sum(1 for e in errors if e <= 5.0) / num_samples
    acc_10px = sum(1 for e in errors if e <= 10.0) / num_samples

    mean_time = sum(times) / num_success
    median_time = times[num_success // 2] if num_success % 2 != 0 else (times[num_success // 2 - 1] + times[num_success // 2]) / 2.0

    return {
        "num_samples": num_samples,
        "valid_samples": num_success,
        "mean_error": mean_error,
        "median_error": median_error,
        "max_error": max_error,
        "acc_2px": acc_2px,
        "acc_5px": acc_5px,
        "acc_10px": acc_10px,
        "mean_time_ms": mean_time,
        "median_time_ms": median_time,
    }
