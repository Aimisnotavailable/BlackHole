import math
import struct
from pathlib import Path
import numpy as np
import pygame
import moderngl
from settings import *

def normalize(vec):
    length = np.linalg.norm(vec)
    if length == 0.0:
        return vec
    return vec / length

class Renderer:
    def __init__(self):
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

        # Shader directory (relative to project root)
        shader_root = Path.cwd() / "shaders"

        # Load screen program from external files
        self.screen_prog = self._load_program(
            shader_root / "screen.vert",
            shader_root / "screen.frag"
        )

        # Load grid program from external files
        self.grid_prog = self._load_program(
            shader_root / "grid.vert",
            shader_root / "grid.frag"
        )

        # Load compute shader
        self.compute_shader = self.ctx.compute_shader(
            (shader_root / "geodesic.comp").read_text()
        )

        # Uniform buffers
        self.camera_ubo = self.ctx.buffer(reserve=CAMERA_UBO_SIZE)
        self.camera_ubo.bind_to_uniform_block(binding=1)

        self.disk_ubo = self.ctx.buffer(reserve=DISK_UBO_SIZE)
        self.disk_ubo.bind_to_uniform_block(binding=2)

        self.objects_ubo = self.ctx.buffer(reserve=OBJECTS_UBO_SIZE)
        self.objects_ubo.bind_to_uniform_block(binding=3)

        # Fullscreen quad & texture
        self.quad_vao, self.render_texture = self._create_quad()

        # Grid (will be generated later)
        self.grid_vao = None
        self.grid_index_count = 0

    def _load_program(self, vertex_path, fragment_path):
        """Load a shader program from two files, print errors on failure."""
        try:
            return self.ctx.program(
                vertex_shader=vertex_path.read_text(),
                fragment_shader=fragment_path.read_text(),
            )
        except moderngl.error.Error as e:
            print(f"Shader compilation failed for {vertex_path} or {fragment_path}:")
            print(e)
            raise

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
        vao = self.ctx.simple_vertex_array(
            self.screen_prog, vbo, "aPos", "aTexCoord"
        )
        tex = self.ctx.texture(
            (self.low_compute_width, self.low_compute_height), 4, dtype="f1"
        )
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.texture_width = self.low_compute_width
        self.texture_height = self.low_compute_height
        return vao, tex

    def generate_grid(self, objects):
        # Create a meshgrid of wx, wz coordinates
        xs = (np.arange(GRID_SIZE + 1) - GRID_SIZE / 2) * GRID_SPACING
        zs = (np.arange(GRID_SIZE + 1) - GRID_SIZE / 2) * GRID_SPACING
        wx, wz = np.meshgrid(xs, zs)               # both shape (26,26)

        # Start with zero displacement
        y = np.zeros_like(wx, dtype=np.float32)

        # Add the deformation for each object using vectorised operations
        for obj in objects:
            obj_pos = obj.pos_radius[:3]            # (3,)
            r_s = 2.0 * G * obj.mass / (C * C)
            # Distance from every grid point to the object's (x, z) projection
            dx = wx - obj_pos[0]
            dz = wz - obj_pos[2]
            dist = np.sqrt(dx * dx + dz * dz)       # shape (26,26)

            # Deformation formula
            mask = dist > r_s
            # For dist > r_s
            y = y + np.where(mask,
                            2.0 * np.sqrt(r_s * (dist - r_s)) - 3e10,
                            2.0 * np.sqrt(r_s * r_s) - 3e10)

        # Flatten to vertex list
        verts = np.column_stack((wx.ravel(), y.ravel(), wz.ravel())).astype(np.float32)

        indices = np.empty(((GRID_SIZE) * (GRID_SIZE) * 4,), dtype=np.uint32)
        idx = 0
        stride = GRID_SIZE + 1
        for z_idx in range(GRID_SIZE):
            for x_idx in range(GRID_SIZE):
                base = z_idx * stride + x_idx
                indices[idx]   = base
                indices[idx+1] = base + 1
                indices[idx+2] = base
                indices[idx+3] = base + stride
                idx += 4

        # Upload to GPU
        vbo = self.ctx.buffer(verts)
        ibo = self.ctx.buffer(indices)
        self.grid_vao = self.ctx.simple_vertex_array(
            self.grid_prog, vbo, "in_position", index_buffer=ibo
        )
        self.grid_index_count = len(indices)

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
            struct.pack_into("<4f", payload, pos_offset + idx * 16, *obj.pos_radius)
            struct.pack_into("<4f", payload, color_offset + idx * 16, *obj.color)

        self.objects_ubo.write(payload)

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

    def draw_fullscreen_quad(self):
        self.render_texture.use(location=0)
        self.screen_prog["screenTexture"].value = 0
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.quad_vao.render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def draw_grid(self, view_proj):
        self.grid_prog["viewProj"].write(view_proj.to_bytes())
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.grid_vao.render(moderngl.LINES)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def shutdown(self):
        pygame.quit()