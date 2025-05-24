import argparse
import numpy as np
import matplotlib.pyplot as plt
from simulator.physics import air_kerma
from simulator.phantom import cone_thickness_map
from simulator.noise import add_quantum_noise, add_system_noise

MU_MM_INV = 0.005 # linear attenuation coefficient [1/mm], toy value

def simulate(kvp: float, mas: float,
             height_pix=512, width_pix=512) -> np.ndarray:
    # --- 1. Photon fluence ---
    photons = air_kerma(kvp, mas)

    # --- 2. Cone phantom thickness map ---
    thickness = cone_thickness_map(height_pix, width_pix)

    # --- 3. Bear-Lamber attenuation ---
    transmission = np.exp(-MU_MM_INV * thickness)

    # --- 4. Scale to photon counts ---
    primary_signal = photons * transmission
    primary_norm = primary_signal / primary_signal.max()

    # --- 5. Quantum noise ---
    quantum_noisy = add_quantum_noise(primary_norm, photons_per_pixel=photons)

    # --- 6. System noise ---
    final_img = add_system_noise(quantum_noisy, sigma=0.02)

    return final_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cone X-ray image simulator")
    parser.add_argument("--kvp", type=float, required=True, help="Tube voltage (kVp)")
    parser.add_argument("--mas", type=float, required=True, help="Tube current-time product (mAs)")
    parser.add_argument("--out", type=str, default="output.png", help="Output PNG filename")
    args = parser.parse_args()

    img = simulate(args.kvp, args.mas)

    plt.imsave(args.out, img, cmap="gray")
    print(f"Saved image to {args.out}")