import cv2
import numpy as np

def apply_gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur with given sigma.
    
    Args:
        image: Input grayscale uint8 image
        sigma: Gaussian blur standard deviation. Kernel size computed automatically.
    
    Returns:
        Blurred uint8 image
    """
    ksize = int(6 * sigma + 1) | 1
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    return blurred
