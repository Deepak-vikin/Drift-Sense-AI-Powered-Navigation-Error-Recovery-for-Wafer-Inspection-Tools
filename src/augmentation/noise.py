import numpy as np

def add_gaussian_noise(image: np.ndarray, mean: float = 0.0, sigma: float = 10.0, rng: np.random.RandomState = None) -> np.ndarray:
    """Add Gaussian noise to image. Returns uint8 clipped image.
    
    Each call generates INDEPENDENT noise (new random draw).
    """
    if rng is None:
        rng = np.random.RandomState()
    noise = rng.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_image

def add_poisson_noise(image: np.ndarray, scale: float = 1.0, rng: np.random.RandomState = None) -> np.ndarray:
    """Add Poisson-distributed noise to simulate photon counting noise."""
    if rng is None:
        rng = np.random.RandomState()
    
    # Scale image to control the intensity of Poisson noise
    scaled_image = image.astype(np.float32) * scale
    
    # Add Poisson noise (poisson takes lambda parameters, typically expected as intensities)
    noisy_image = rng.poisson(scaled_image) / scale
    
    # Clip and convert back to uint8
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image
