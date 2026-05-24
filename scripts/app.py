import math
import time
import numpy as np
import glm
import pygame
from scripts.camera import Camera
from scripts.objects import BlackHole
from scripts.scene import build_scene
from scripts.renderer import Renderer
from settings import *

# Global gravity toggle – kept at module level for simplicity.
GRAVITY_ENABLED = False

class BlackHoleApp:
    def __init__(self):
        self.camera = Camera()
        self.black_hole = BlackHole(
            np.array([0.0, 0.0, 0.0], dtype=np.float32), 8.54e36
        )
        self.objects = build_scene(self.black_hole)
        self.engine = Renderer()
        self.grid_dirty = True
        self.last_time = time.time()
        self.world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN or event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE and event.type == pygame.KEYDOWN:
                    return False
                self._process_key(event.key, event.type == pygame.KEYDOWN)
            elif event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.MOUSEBUTTONUP:
                if event.button <= 3:
                    self._process_mouse_button(
                        event.button, event.type == pygame.MOUSEBUTTONDOWN
                    )
            elif event.type == pygame.MOUSEMOTION:
                if self.camera.dragging and not self.camera.panning:
                    dx, dy = event.rel
                    self.camera.process_mouse_move(dx, dy)
                    self.camera.last_x, self.camera.last_y = event.pos
            elif event.type == pygame.MOUSEWHEEL:
                self.camera.process_scroll(event.y)
        return True

    def _process_key(self, key, pressed):
        global GRAVITY_ENABLED
        if pressed and key == pygame.K_g:
            GRAVITY_ENABLED = not GRAVITY_ENABLED
            print(f"[INFO] Gravity turned {'ON' if GRAVITY_ENABLED else 'OFF'}")

    def _process_mouse_button(self, button, pressed):
        global GRAVITY_ENABLED
        if button == 1 or button == 2:  # left / middle
            self.camera.process_mouse_button(button, pressed)
        elif button == 3:  # right button toggles gravity
            GRAVITY_ENABLED = pressed

    def integrate_gravity(self, dt):
        if not GRAVITY_ENABLED:
            return
        self.grid_dirty = True
        for obj in self.objects:
            acc = np.zeros(3, dtype=np.float32)
            for other in self.objects:
                if obj is other:
                    continue
                delta = other.pos_radius[:3] - obj.pos_radius[:3]
                dist = np.linalg.norm(delta)
                if dist <= 0.0:
                    continue
                direction = delta / dist
                force = (G * obj.mass * other.mass) / (dist * dist)
                acc += direction * (force / obj.mass)
            obj.velocity += acc * dt
        for obj in self.objects:
            obj.pos_radius[:3] += obj.velocity * dt

    def render_frame(self):
        self.engine.ctx.clear(0.0, 0.0, 0.0, 1.0)

        now = time.time()
        dt = min(max(now - self.last_time, 0.0), 1.0 / 30.0)
        self.last_time = now

        self.integrate_gravity(dt)

        if self.grid_dirty:
            self.engine.generate_grid(self.objects)
            self.grid_dirty = False

        eye = self.camera.position()
        view = glm.lookAt(glm.vec3(*eye), glm.vec3(*self.camera.target), glm.vec3(*self.world_up))
        fov = glm.radians(60.0)
        proj = glm.perspective(
            fov,
            float(self.engine.width) / float(self.engine.height),
            1e9,
            1e14,
        )
        view_proj = proj * view

        self.engine.dispatch_compute(self.camera, self.black_hole, self.objects)
        self.engine.draw_fullscreen_quad()
        self.engine.draw_grid(view_proj)

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.render_frame()
        self.engine.shutdown()