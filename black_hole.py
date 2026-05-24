"""Black hole visualiser – modernGL + pygame + GLM"""

import math
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pygame
import moderngl
from moderngl.glm import perspective, lookAt

C = 299792458.0
G = 6.67430e-11
GRID_SIZE = 25
GRID_SPACING = 1e10
MAX_OBJECTS = 16
CAMERA_UBO_SIZE = 80
DISK_UBO_SIZE = 16
OBJECTS_UBO_SIZE = 16 + MAX_OBJECTS * 16 * 2
GRAVITY_ENABLED = False


# ---------- Helper (still needed for camera) ----------
def normalize(vec):
    length = np.linalg.norm(vec)
    if length == 0.0:
        return vec
    return vec / length


# ---------- Camera ----------
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
        global GRAVITY_ENABLED
        if button == 1 or button == 2:  # left / middle
            self.dragging = pressed
            self.panning = False
            if pressed:
                self.last_x, self.last_y = pygame.mouse.get_pos()
        elif button == 3:  # right
            GRAVITY_ENABLED = pressed
        self.update()

    def process_scroll(self, yoffset):
        self.radius -= yoffset * self.zoom_speed
        self.radius = max(self.min_radius, min(self.max_radius, self.radius))
        self.update()

    def process_key(self, key, pressed):
        global GRAVITY_ENABLED
        if pressed and key == pygame.K_g:
            GRAVITY_ENABLED = not GRAVITY_ENABLED
            print(f"[INFO] Gravity turned {'ON' if GRAVITY_ENABLED else 'OFF'}")


# ---------- Black hole & objects ----------
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


# ---------- Renderer (modernGL) ----------
class Renderer:
    def __init__(self):
        # Pygame window with OpenGL context
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
        )
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 1)

        self.width = 800
        self.height = 600
        self.low_compute_width = 200
        self.low_compute_height = 150
        self.high_compute_width = 400
        self.high_compute_height = 300
        self.texture_width = 0
        self.texture_height = 0

        self.window = pygame.display.set_mode(
            (self.width, self.height), pygame.OPENGL | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("Black Hole (modernGL)")

        self.ctx = moderngl.create_context(require=430)
        print("OpenGL", self.ctx.version_code)

        root = Path(__file__).resolve().parent
        self.screen_prog = self._make_screen_program()
        self.grid_prog = self.ctx.program(
            vertex_shader=(root / "grid.vert").read_text(),
            fragment_shader=(root / "grid.frag").read_text(),
        )
        self.compute_shader = self.ctx.compute_shader(
            (root / "geodesic.comp").read_text()
        )

        # Uniform buffers – use ctx.binding for compatibility
        self.camera_ubo = self.ctx.buffer(reserve=CAMERA_UBO_SIZE)
        self.ctx.binding(1, self.camera_ubo)

        self.disk_ubo = self.ctx.buffer(reserve=DISK_UBO_SIZE)
        self.ctx.binding(2, self.disk_ubo)

        self.objects_ubo = self.ctx.buffer(reserve=OBJECTS_UBO_SIZE)
        self.ctx.binding(3, self.objects_ubo)

        # Fullscreen quad & texture
        self.quad_vao, self.render_texture = self._create_quad()

        # Grid
        self.grid_vao = None
        self.grid_index_count = 0

    def _make_screen_program(self):
        vs = """
            #version 330 core
            layout (location = 0) in vec2 aPos;
            layout (location = 1) in vec2 aTexCoord;
            out vec2 TexCoord;
            void main() {
                gl_Position = vec4(aPos, 0.0, 1.0);
                TexCoord = aTexCoord;
            }
        """
        fs = """
            #version 330 core
            in vec2 TexCoord;
            out vec4 FragColor;
            uniform sampler2D screenTexture;
            void main() {
                FragColor = texture(screenTexture, TexCoord);
            }
        """
        return self.ctx.program(vertex_shader=vs, fragment_shader=fs)

    def _create_quad(self):
        vertices = np.array(
            [
                -1.0, 1.0, 0.0, 1.0,
                -1.0, -1.0, 0.0, 0.0,
                1.0, -1.0, 1.0, 0.0,
                -1.0, 1.0, 0.0, 1.0,
                1.0, -1.0, 1.0, 0.0,
                1.0, 1.0, 1.0, 1.0,
            ],
            dtype=np.float32,
        )
        vbo = self.ctx.buffer(vertices)
        vao = self.ctx.simple_vertex_array(self.screen_prog, vbo, "aPos", "aTexCoord")
        tex = self.ctx.texture(
            (self.low_compute_width, self.low_compute_height), 4, dtype="f1"
        )
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.texture_width = self.low_compute_width
        self.texture_height = self.low_compute_height
        return vao, tex

    # ---------- grid generation ----------
    def generate_grid(self, objects):
        verts = []
        indices = []

        for z_idx in range(GRID_SIZE + 1):
            for x_idx in range(GRID_SIZE + 1):
                wx = (x_idx - GRID_SIZE / 2) * GRID_SPACING
                wz = (z_idx - GRID_SIZE / 2) * GRID_SPACING
                y = 0.0
                for obj in objects:
                    obj_pos = obj.pos_radius[:3]
                    r_s = 2.0 * G * obj.mass / (C * C)
                    dx = wx - obj_pos[0]
                    dz = wz - obj_pos[2]
                    dist = math.sqrt(dx * dx + dz * dz)
                    if dist > r_s:
                        y += 2.0 * math.sqrt(r_s * (dist - r_s)) - 3e10
                    else:
                        y += 2.0 * math.sqrt(r_s * r_s) - 3e10
                verts.extend([wx, y, wz])

        for z_idx in range(GRID_SIZE):
            for x_idx in range(GRID_SIZE):
                base = z_idx * (GRID_SIZE + 1) + x_idx
                indices.extend([base, base + 1, base, base + GRID_SIZE + 1])

        vbo = self.ctx.buffer(np.array(verts, dtype=np.float32))
        ibo = self.ctx.buffer(np.array(indices, dtype=np.uint32))
        self.grid_vao = self.ctx.simple_vertex_array(
            self.grid_prog, vbo, "in_position", index_buffer=ibo
        )
        self.grid_index_count = len(indices)

    # ---------- UBO uploads ----------
    def upload_camera_ubo(self, camera):
        pos = camera.position()
        forward = normalize(camera.target - pos)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = normalize(np.cross(forward, up))
        up = np.cross(right, forward)
        payload = struct.pack(
            "<16f2f2i",
            pos[0], pos[1], pos[2], 0.0,
            right[0], right[1], right[2], 0.0,
            up[0], up[1], up[2], 0.0,
            forward[0], forward[1], forward[2], 0.0,
            math.tan(math.radians(60.0 * 0.5)),
            float(self.width) / float(self.height),
            int(camera.moving),
            0,
        )
        self.camera_ubo.write(payload)

    def upload_disk_ubo(self, black_hole):
        payload = struct.pack(
            "<4f",
            black_hole.r_s * 2.2,
            black_hole.r_s * 5.2,
            2.0,
            1e9,
        )
        self.disk_ubo.write(payload)

    def upload_objects_ubo(self, objects):
        count = min(len(objects), MAX_OBJECTS)
        payload = bytearray(OBJECTS_UBO_SIZE)
        struct.pack_into("<i", payload, 0, count)

        pos_offset = 16
        color_offset = pos_offset + MAX_OBJECTS * 16

        for idx, obj in enumerate(objects[:count]):
            struct.pack_into(
                "<4f", payload, pos_offset + idx * 16, *obj.pos_radius
            )
            struct.pack_into(
                "<4f", payload, color_offset + idx * 16, *obj.color
            )

        self.objects_ubo.write(payload)

    # ---------- compute dispatch ----------
    def dispatch_compute(self, camera, black_hole, objects):
        w = self.low_compute_width if camera.moving else self.high_compute_width
        h = self.low_compute_height if camera.moving else self.high_compute_height

        if w != self.texture_width or h != self.texture_height:
            self.render_texture = self.ctx.texture((w, h), 4, dtype="f1")
            self.render_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.texture_width = w
            self.texture_height = h

        self.upload_camera_ubo(camera)
        self.upload_disk_ubo(black_hole)
        self.upload_objects_ubo(objects)

        self.render_texture.bind_to_image(0, read=False, write=True)
        groups_x = math.ceil(w / 16.0)
        groups_y = math.ceil(h / 16.0)
        self.compute_shader.run(groups_x, groups_y, 1)

    # ---------- drawing ----------
    def draw_fullscreen_quad(self):
        self.render_texture.use(location=0)
        self.screen_prog["screenTexture"].value = 0
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.quad_vao.render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def draw_grid(self, view_proj):
        self.grid_prog["viewProj"].write(view_proj.astype("f4").tobytes())
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.grid_vao.render(moderngl.LINES)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def shutdown(self):
        pygame.quit()


# ---------- Application ----------
class BlackHoleApp:
    def __init__(self):
        self.camera = Camera()
        self.black_hole = BlackHole(
            np.array([0.0, 0.0, 0.0], dtype=np.float32), 8.54e36
        )
        self.objects = self._build_scene()
        self.engine = Renderer()
        self.grid_dirty = True
        self.last_time = time.time()
        self.world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def _build_scene(self):
        return [
            ObjectData(
                np.array([4e11, 0.0, 0.0, 4e10], dtype=np.float32),
                np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
                1.98892e30,
            ),
            ObjectData(
                np.array([0.0, 0.0, 4e11, 4e10], dtype=np.float32),
                np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
                1.98892e30,
            ),
            ObjectData(
                np.array([0.0, 0.0, 0.0, self.black_hole.r_s], dtype=np.float32),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                self.black_hole.mass * 0.3,
            ),
        ]

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN or event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE and event.type == pygame.KEYDOWN:
                    return False
                self.camera.process_key(event.key, event.type == pygame.KEYDOWN)
            elif event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.MOUSEBUTTONUP:
                if event.button <= 3:
                    self.camera.process_mouse_button(
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
        # Use glm functions for view/projection matrices
        view = lookAt(eye, self.camera.target, self.world_up)
        proj = perspective(
            math.radians(60.0),
            float(self.engine.width) / float(self.engine.height),
            1e9,
            1e14,
        )
        view_proj = proj @ view

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


def main():
    try:
        app = BlackHoleApp()
        app.run()
    except Exception as exc:
        print(f"Failed to start: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()