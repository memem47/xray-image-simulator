import argparse
import numpy as np
import matplotlib.pyplot as plt
from simulator.physics import air_kerma
from simulator.phantom import cone_thickness_map
from simulator.noise import add_quantum_noise, add_system_noise
from PIL import Image

MU_MM_INV = 0.005 # linear attenuation coefficient [1/mm], toy value

def simulate(kvp: float, mas: float,
             height_pix=512, width_pix=512,
             cone_scale: float = 1.0,
             cone_offset: tuple[int, int] = (0,0),
             photons: float | None = None,
             sigma: float = 0.02) -> np.ndarray:
    """
    Core simulation wrapper used by both CLI and GUI.
    """
    # --- 1. Photon fluence ---
    if photons is None:
        photons = air_kerma(kvp, mas)

    # --- 2. Cone phantom thickness map ---
    thickness = cone_thickness_map(height_pix, width_pix,scale=cone_scale,offset_xy=cone_offset)

    # --- 3. Bear-Lamber attenuation ---
    transmission = np.exp(-MU_MM_INV * thickness)

    # --- 4. Scale to photon counts ---
    primary_signal = photons * transmission
    primary_norm = primary_signal / primary_signal.max()

    # --- 5. Quantum noise ---
    quantum_noisy = add_quantum_noise(primary_norm, photons_per_pixel=photons)

    # --- 6. System noise ---
    final_img = add_system_noise(quantum_noisy, sigma)

    return final_img


def main():
    parser = argparse.ArgumentParser(description="Cone X-ray image simulator")
    parser.add_argument("--kvp", type=float, required=True, help="Tube voltage (kVp)")
    parser.add_argument("--mas", type=float, required=True, help="Tube current-time product (mAs)")
    parser.add_argument("--out", type=str, default="output.png", help="Output PNG filename")
    
    # 追加オプション
    parser.add_argument("--cone-scale", type=float, default=1.0, 
                        help="Cone height/width fraction of full frame (0-1)")
    parser.add_argument("--cone-offset", type=int, nargs=2, metavar=("DX","DY"), 
                        default=(0,0), help="Cone top-left offset in pixels")
    parser.add_argument("--photons", type=float, default=None,
                        help="Override photons per pixel (skip air-kerma)")
    parser.add_argument("--sigma", type=float, default=0.02, 
                        help="Additive Gaussian system-noise sigma")
    
    args = parser.parse_args()

    img = simulate(
        args.kvp,
        args.mas,
        cone_scale=args.cone_scale,
        cone_offset=tuple(args.cone_offset),
        photons=args.photons,
        sigma=args.sigma,
    )

    img_u8 = (img * 255).astype("uint8")
    Image.fromarray(img_u8, mode="L").save(args.out)
    print(f"Saved image to {args.out}")

if __name__ == "__main__":
    main()
