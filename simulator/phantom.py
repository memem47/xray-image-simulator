import numpy as np

def cone_thickness_map(height_px=512, width_px=512,
                       base_mm=40.0, height_mm=80.0,
                       scale: float = 1.0,
                       offset_xy: tuple[int, int] = (0,0)) -> np.ndarray:
    """
    Returns thickness [mm] map for a right circular core observed from the side.
    x-axis: width direction, y-axis: projection height.

    Linear thickness profile: at each y, the chord length through the cone.

    scale: 1.0 = full height of frame, 0.5 = half
    offset_xy: (dx, dy) shift of the cone apex downward/rightward
    """
    h_px = int(height_px * scale)
    w_px = int(width_px * scale)

    thickness_small = _cone_cone(h_px, w_px, base_mm * scale, height_mm * scale)
    
    # blank canvas
    canvas = np.zeros((height_px, width_px), dtype=np.float32)
    dx, dy = offset_xy
    canvas[dy:dy+h_px, dx:dx+w_px] = thickness_small
    return canvas
    
def _cone_cone(h_px, w_px, base_mm, height_mm):
    y = np.linspace(0, height_mm, h_px)
    radius = (1 - y / height_mm) * (base_mm / 2.0)
    thickness = 2.0 * radius
    # Broadcast thickness to every column
    return np.repeat(thickness[:, None], w_px, axis=1)
