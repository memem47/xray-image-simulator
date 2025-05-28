import numpy as np
from simulator.phantom import cone_thickness_map

def test_cone_position_and_scale():
    h, w = 200, 300
    scale = 0.4
    dx, dy = 50, 30
    thick = cone_thickness_map(h, w, scale=scale, offset_xy=(dx, dy))
    # Non-zero region bounds
    ys, xs = np.nonzero(thick)
    assert ys.min() == dy
    assert xs.min() == dx
    # Size roughly matches scale
    assert ys.max() - ys.min() + 1 == int(h * scale)
    assert xs.max() - xs.min() + 1 == int(w * scale)
