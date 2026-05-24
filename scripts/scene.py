import numpy as np
from scripts.objects import BlackHole, ObjectData

def build_scene(black_hole):
    return [
        # yellow star – directly in front, well outside the black hole
        ObjectData(
            np.array([-1.5e11, 0.0, 0.0, 6e9], dtype=np.float32),  # 15×10¹⁰ m
            np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
            1.98892e30,
        ),
        # red star – slightly to the left and a bit closer
        ObjectData(
            np.array([-1.2e11, 0.0, -0.8e11, 6e9], dtype=np.float32),
            np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float32),
            1.98892e30,
        ),
        # accretion disk (black hole silhouette)
        ObjectData(
            np.array([0.0, 0.0, 0.0, black_hole.r_s], dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            black_hole.mass * 0.3,
        ),
    ]