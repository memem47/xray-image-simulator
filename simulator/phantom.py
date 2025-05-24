import numpy as np

def cone_thickness_map(height_px=512, width_px=512,
                       base_mm=40.0, height_mm=80.0) -> np.ndarray:
        """
        Returns thickness [mm] map for a right circular core observed from the side.
        x-axis: width direction, y-axis: projection height.

        Linear thickness profile: at each y, the chord length through the cone.
        """
        y = np.linspace(0, height_mm, height_px)
        radius_at_y = (1 - y / height_mm) * (base_mm / 2.0)
        thickness_mm = radius_at_y * 2.0
        # Broadcast thickness to every column
        return np.repeat(thickness_mm[:, None], width_px, axis=1)
