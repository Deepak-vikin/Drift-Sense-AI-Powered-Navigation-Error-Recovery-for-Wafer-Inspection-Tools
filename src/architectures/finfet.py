import numpy as np
import cv2

class FinFETGenerator:
    """Generates realistic FinFET-style semiconductor patterns."""
    
    def __init__(self, seed: int | None = None):
        """
        Initialize the FinFET generator.
        
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
        return 'FinFET'
        
    def generate_base_pattern(self, width: int, height: int) -> np.ndarray:
        """
        Generate a base FinFET pattern with dense vertical fins and horizontal gates.
        
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
        noise = self.rng.normal(0, 2, (height, width))
        pattern = np.clip(pattern + noise, 0, 255).astype(np.uint8)
        
        # Fins (dense vertical lines)
        fin_pitch = 8
        fin_width = 2
        fin_brightness = self.rng.randint(170, 200)
        
        # Gates (horizontal bars)
        # Let's place 1-2 horizontal gate bars
        num_gates = self.rng.randint(1, 3)
        gate_spacing = height // (num_gates + 1)
        gate_width = 5
        gate_brightness = self.rng.randint(200, 230)
        
        # Draw fins
        for x in range(fin_pitch // 2, width, fin_pitch):
            val = np.clip(fin_brightness + self.rng.normal(0, 5), 0, 255)
            pattern[:, x:x+fin_width] = val
            
        # Draw gates
        for i in range(1, num_gates + 1):
            y = i * gate_spacing
            val = np.clip(gate_brightness + self.rng.normal(0, 5), 0, 255)
            pattern[y:y+gate_width, :] = val
            
            # Brighten intersections
            for x in range(fin_pitch // 2, width, fin_pitch):
                intersect_val = np.clip(val + 20 + self.rng.normal(0, 5), 0, 255)
                pattern[y:y+gate_width, x:x+fin_width] = intersect_val
                
        # Optional blur to make it look optical/SEM-like
        pattern = cv2.GaussianBlur(pattern, (3, 3), 0)
        
        return pattern
