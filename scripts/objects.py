from dataclasses import dataclass, field
import numpy as np
from settings import *

@dataclass
class BlackHole:
    position: np.ndarray
    mass: float

    def __post_init__(self):
        self.r_s = 2.0 * G * self.mass / (C * C)

@dataclass
class ObjectData:
    pos_radius: np.ndarray
    color: np.ndarray
    mass: float
    velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )