import numpy as np
import cv2

class DRAMGenerator:
    """Generates realistic DRAM-style semiconductor patterns."""
    
    def __init__(self, seed: int | None = None):
        """
        Initialize the DRAM generator.
        
        Args:
            seed (int | None, optional): Random seed for reproducibility.
        """
        self.rng = np.random.RandomState(seed)
        
    def get_architecture_name(self) -> str:
        """
        Returns the architecture name.
        
        Returns:
            str: The architecture name.
        """
        return 'DRAM'
        
    def generate_base_pattern(self, width: int, height: int) -> np.ndarray:
        """
        Generate a base DRAM pattern with horizontal word-lines, vertical bit-lines,
        and contact dots at intersections.
        
        Args:
            width (int): Pattern width in pixels.
            height (int): Pattern height in pixels.
            
        Returns:
            np.ndarray: Grayscale uint8 image.
        """
        # Start with a dark background (~30)
        bg_val = self.rng.randint(25, 40)
        pattern = np.full((height, width), bg_val, dtype=np.uint8)
        
        # Add some background noise
        noise = self.rng.normal(0, 3, (height, width))
        pattern = np.clip(pattern + noise, 0, 255).astype(np.uint8)
        
        # Word-lines (horizontal)
        wl_pitch = 14
        wl_width = 3
        wl_brightness = self.rng.randint(180, 210)
        
        # Bit-lines (vertical)
        bl_pitch = 12
        bl_width = 2
        bl_brightness = self.rng.randint(170, 200)
        
        # Contact dots
        dot_radius = 2
        dot_brightness = self.rng.randint(220, 240)
        
        # Draw word-lines
        for y in range(wl_pitch // 2, height, wl_pitch):
            val = np.clip(wl_brightness + self.rng.normal(0, 5), 0, 255)
            pattern[y:y+wl_width, :] = val
                
        # Draw bit-lines
        for x in range(bl_pitch // 2, width, bl_pitch):
            val = np.clip(bl_brightness + self.rng.normal(0, 5), 0, 255)
            pattern[:, x:x+bl_width] = val
            
        # Draw contact dots at intersections
        for y in range(wl_pitch // 2, height, wl_pitch):
            for x in range(bl_pitch // 2, width, bl_pitch):
                cv2.circle(
                    pattern, 
                    (x + bl_width // 2, y + wl_width // 2), 
                    dot_radius, 
                    int(np.clip(dot_brightness + self.rng.normal(0, 5), 0, 255)), 
                    -1
                )
                
        # Optional blur to make it look optical/SEM-like
        pattern = cv2.GaussianBlur(pattern, (3, 3), 0)
        
        return pattern
