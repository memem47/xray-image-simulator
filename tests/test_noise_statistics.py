import numpy as np
from simulator.noise import add_quantum_noise, add_system_noise

def test_noise_reproducibility():
    rng = np.random.default_rng(42)
    image = np.ones((32, 32))
    noisy1 = add_quantum_noise(image, photons_per_pixel=1e5, rng=rng)
    rng = np.random.default_rng(42)
    noisy2 = add_quantum_noise(image, photons_per_pixel=1e5, rng=rng)
    assert np.array_equal(noisy1, noisy2), "Poisson noise not reproducible with fixed seed"

    rng = np.random.default_rng(123)
    sys1 = add_system_noise(image, sigma=0.02, rng=rng)
    rng = np.random.default_rng(123)
    sys2 = add_system_noise(image, sigma=0.02, rng=rng)
    assert np.allclose(sys1, sys2), "Gaussian noise not reproducible with fixed seed"