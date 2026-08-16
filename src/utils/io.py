import cv2
import numpy as np
from pathlib import Path

def save_image(image: np.ndarray, filepath: Path) -> None:
    """Save grayscale image to file. Creates parent directories."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(filepath), image)

def load_image(filepath: Path) -> np.ndarray:
    """Load grayscale image from file."""
    img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {filepath}")
    return img

def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist. Returns the path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
