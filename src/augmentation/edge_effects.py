import cv2
import numpy as np

def apply_sem_edge_brightening(image: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """Apply controlled SEM-like edge brightening.
    
    Mimics the edge effect in scanning electron microscopy where edges
    appear brighter due to increased secondary electron emission.
    
    Implementation:
    1. Compute Sobel gradients (dx, dy)
    2. Compute gradient magnitude as edge map
    3. Normalize edge map to [0, 1]
    4. Multiply by strength factor
    5. Add scaled edge map to original image
    6. Clip to [0, 255] uint8
    
    Args:
        image: Input grayscale uint8 image
        strength: Edge brightening strength (0.0 = none, 1.0 = maximum). 
                  Typical realistic values: 0.15-0.4
    
    Returns:
        Image with edge brightening applied, uint8
    """
    if strength <= 0.0:
        return image.copy()
        
    # 1. Compute Sobel gradients (dx, dy)
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    
    # 2. Compute gradient magnitude as edge map
    magnitude = cv2.magnitude(grad_x, grad_y)
    
    # Apply a slight Gaussian blur to the edge map before adding to avoid harsh artifacts
    magnitude = cv2.GaussianBlur(magnitude, (3, 3), 0.5)
    
    # 3. Normalize edge map to [0, 1]
    max_val = np.max(magnitude)
    if max_val > 0:
        magnitude = magnitude / max_val
        
    # 4. Multiply by strength factor and scale for image addition
    edge_boost = magnitude * strength * 255.0
    
    # 5. Add scaled edge map to original image
    brightened = image.astype(np.float32) + edge_boost
    
    # 6. Clip to [0, 255] uint8
    brightened = np.clip(brightened, 0, 255).astype(np.uint8)
    
    return brightened
