import math
import numpy as np
import pygame
from dataclasses import dataclass, field

@dataclass
class Camera:
    target: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32)
    )
    radius: float = 6.34194e10
    min_radius: float = 1e10
    max_radius: float = 1e12
    azimuth: float = 0.0
    elevation: float = math.pi / 2.0
    orbit_speed: float = 0.01
    zoom_speed: float = 25e9
    dragging: bool = False
    panning: bool = False
    moving: bool = False
    last_x: float = 0.0
    last_y: float = 0.0

    def position(self):
        elevation = max(0.01, min(math.pi - 0.01, self.elevation))
        return np.array(
            [
                self.radius * math.sin(elevation) * math.cos(self.azimuth),
                self.radius * math.cos(elevation),
                self.radius * math.sin(elevation) * math.sin(self.azimuth),
            ],
            dtype=np.float32,
        )

    def update(self):
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.moving = self.dragging or self.panning

    def process_mouse_move(self, dx, dy):
        if self.dragging and not self.panning:
            self.azimuth += dx * self.orbit_speed
            self.elevation -= dy * self.orbit_speed
            self.elevation = max(0.01, min(math.pi - 0.01, self.elevation))
        self.update()

    def process_mouse_button(self, button, pressed):
        # Gravity toggling is handled in the app, not here.
        if button == 1 or button == 2:  # left / middle
            self.dragging = pressed
            self.panning = False
            if pressed:
                self.last_x, self.last_y = pygame.mouse.get_pos()
        self.update()

    def process_scroll(self, yoffset):
        self.radius -= yoffset * self.zoom_speed
        self.radius = max(self.min_radius, min(self.max_radius, self.radius))
        self.update()