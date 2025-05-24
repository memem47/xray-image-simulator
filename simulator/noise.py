import numpy as np

def add_quantum_noise(image: np.ndarray, photons_per_pixel: float,
                      rng=np.random.default_rng()) -> np.ndarray:
    """
    Poisson noise proportional to photon count.
    Assumes 'image' is signal in arbitary units propotional to photons.
    """
    # Convert to expected photon counts
    expected = image * photons_per_pixel / image.max()
    noisy = rng.poisson(expected)
    # Rescale back to 0-1
    return noisy / noisy.max()

def add_system_noise(image: np.ndarray, sigma: float = 0.01,
                     rng=np.random.default_rng()) -> np.ndarray:
    
    return np.clip(image + rng.normal(0, sigma, image.shape), 0, 1)
