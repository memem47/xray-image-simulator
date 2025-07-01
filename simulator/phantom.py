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
    y = np.linspace(0, height_mm, h_px, endpoint=False)
    radius = (1 - y / height_mm) * (base_mm / 2.0)
    thickness = 2.0 * radius
    # Broadcast thickness to every column
    return np.repeat(thickness[:, None], w_px, axis=1)

def sphere_thickness_map(height_px: int = 512, width_px: int = 512,
                         diameter_mm: float = 40.0,
                         scale: float = 1.0,
                         offset_xy: tuple[int, int] = (0, 0)) -> np.ndarray:
    """
    Returns a thickness [mm] map for a solid sphere observed along
    the projection (Z) axis.

    Each pixel value = chord length through the sphere at that (x,y) position.
    * scale  : 1.0 = full-frame diameter, 0.5 = half-size sphere, …
    * offset : (dx, dy) in pixels moves the sphere centre right/down.
    """

    # --- スフィア パラメータ ---
    # 最終キャンバス上での直径（px）と半径（px）
    D_px = int(min(height_px, width_px) * scale)
    R_px = D_px / 2.0
    # 実寸→画素変換係数（mm / px）
    mm_per_px = diameter_mm / D_px

    # --- キャンバス作成 ---
    canvas = np.zeros((height_px, width_px), dtype=np.float32)

    # 中心座標 (cx, cy)
    dx, dy = offset_xy
    cx = width_px  / 2 + dx
    cy = height_px / 2 + dy

    # --- 距離マップを利用して厚さを計算 ---
    # 画素座標格子
    y, x = np.ogrid[:height_px, :width_px]
    r2 = (x - cx) ** 2 + (y - cy) ** 2        # 中心からの距離^2 [px^2]

    mask = r2 <= R_px ** 2                    # 球の内部か判定
    # chord length [px] = 2 * sqrt(R^2 - r^2)
    thickness_px = 2.0 * np.sqrt(R_px ** 2 - r2[mask])
    # px → mm 変換
    canvas[mask] = thickness_px * mm_per_px

    return canvas