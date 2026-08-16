import cv2
import numpy as np

def apply_rotation(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Apply rotation around image center. Fills borders with nearest pixel.
    
    Args:
        image: Input image
        angle_deg: Rotation angle in degrees (positive = counter-clockwise)
    
    Returns:
        Rotated image of same size
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated

def apply_scale(image: np.ndarray, scale_factor: float) -> np.ndarray:
    """Apply uniform scaling to image, then center-crop/pad to original size.
    
    Args:
        image: Input image
        scale_factor: Scale factor (1.0 = no change, >1 = zoom in, <1 = zoom out)
    
    Returns:
        Scaled image of same size as input
    """
    h, w = image.shape[:2]
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    
    if new_w == 0 or new_h == 0:
        return np.zeros_like(image)
        
    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    out_image = np.zeros_like(image)
    
    # Calculate crop/pad coordinates
    start_y = max(0, (new_h - h) // 2)
    start_x = max(0, (new_w - w) // 2)
    
    out_start_y = max(0, (h - new_h) // 2)
    out_start_x = max(0, (w - new_w) // 2)
    
    copy_h = min(h, new_h)
    copy_w = min(w, new_w)
    
    out_image[out_start_y:out_start_y+copy_h, out_start_x:out_start_x+copy_w] = \
        scaled[start_y:start_y+copy_h, start_x:start_x+copy_w]
        
    return out_image
