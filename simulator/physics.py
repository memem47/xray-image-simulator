import numpy as np

def air_kerma(kvp: float, mas: float) -> float:
    """
    Very corse empirical model:
        Air kerma = k *(kVp)^2 *mAs
        
    k is chosen so that output is on the order of 1e6 photons/pixel for 
    typical parameters. Turn later with measured data.
    """
    k = 1.2e2
    return k * (kvp ** 2) * mas