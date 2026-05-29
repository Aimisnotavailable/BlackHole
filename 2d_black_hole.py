import pygame
import sys
import math
from collections import deque
import pygame.gfxdraw

# --- Constants ---
WIDTH, HEIGHT = 500, 300
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 600
BH_POS = (WIDTH // 2, HEIGHT // 2)    # (250, 150)
MASS = 15.0
RS = 2 * MASS                         # 30 px
PHOTON_SPHERE = 3 * MASS              # 45 px
SPEED_SCALE =  400                    # pixels/second mapping
TRAIL_LENGTH = 500                    # show many turns

# Critical impact parameter for the photon sphere
B_CRIT = 3 * math.sqrt(3) * MASS     # ≈ 77.9423
DELTA_B = 0.0001                     # tiny offset → more orbits before decay

# --- Ray (null geodesic) ---
class Ray:
    def __init__(self, x0, y0, bh_pos, mass, color=(255, 255, 0)):
        self.bh_pos = bh_pos
        self.mass = mass
        self.rs = 2 * mass
        self.color = color

        rel_x = x0 - bh_pos[0]
        rel_y = y0 - bh_pos[1]
        self.r = math.hypot(rel_x, rel_y)
        self.angle = math.atan2(rel_y, rel_x)

        # Angular momentum from impact parameter b = y0 - bh_center_y
        b = y0 - bh_pos[1]
        self.L = -b

        self.dr_sign = -1
        V = self._potential(self.r, self.L)
        self.dr_dlambda = self.dr_sign * math.sqrt(max(V, 0))

        self.alive = True
        self.trail = deque(maxlen=TRAIL_LENGTH)

    def _potential(self, r, L):
        if r <= self.rs:
            return 0
        return 1.0 - (1.0 - self.rs / r) * L * L / (r * r)

    def update(self, dlambda):
        if not self.alive:
            return

        r = self.r
        angle = self.angle
        L = self.L

        dphi_dlambda = L / (r * r) if r > self.rs else 0

        r_new = r + self.dr_dlambda * dlambda
        angle_new = angle + dphi_dlambda * dlambda

        # Capture inside horizon
        if r_new <= self.rs + 0.5:
            self.alive = False
            self.r = self.rs
            sx = self.bh_pos[0] + self.r * math.cos(angle_new)
            sy = self.bh_pos[1] + self.r * math.sin(angle_new)
            self.trail.append((sx, sy))
            return

        V_new = self._potential(r_new, L)
        if self.dr_sign == -1 and V_new < 0:
            # Turning point → switch to outgoing
            if self._potential(r, L) > 0:
                r_new = r
                V_new = self._potential(r_new, L)
                self.dr_sign = 1
                self.dr_dlambda = math.sqrt(max(V_new, 0))
            else:
                self.dr_sign = 1
                self.dr_dlambda = 0
                r_new = r
        else:
            self.dr_dlambda = self.dr_sign * math.sqrt(max(V_new, 0))

        self.r = r_new
        self.angle = angle_new

        sx = self.bh_pos[0] + self.r * math.cos(self.angle)
        sy = self.bh_pos[1] + self.r * math.sin(self.angle)
        self.trail.append((sx, sy))

        if self.r > max(WIDTH, HEIGHT) * 2:
            self.alive = False

    def render(self, surf):
        # Draw anti‑aliased trail as connected line segments
        pts = list(self.trail)
        if len(pts) > 1:
            for i in range(len(pts) - 1):
                pygame.gfxdraw.line(surf,
                                    int(pts[i][0]), int(pts[i][1]),
                                    int(pts[i+1][0]), int(pts[i+1][1]),
                                    self.color)
        # Draw a brighter tip for the head
        # if self.alive and pts:
        #     px, py = pts[-1]
        #     pygame.gfxdraw.aacircle(surf, int(px), int(py), 3, self.color)
        #     pygame.gfxdraw.filled_circle(surf, int(px), int(py), 3, self.color)


# --- Game ---
class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.display = pygame.Surface((WIDTH, HEIGHT))
        pygame.init()
        self.clock = pygame.time.Clock()
        self.running = True
        self.reset()

    def reset(self):
        """Create all initial rays (yellow reference + two special orbital rays)."""
        self.rays = []

        # Normal yellow rays (sparse)
        for y in range(10, HEIGHT - 10, 5):
            self.rays.append(Ray(0, y, BH_POS, MASS, color=(255, 255, 50)))

        # Orbital ray that will eventually plunge (cyan)
        y_plunge = BH_POS[1] + (B_CRIT - DELTA_B)
        self.rays.append(Ray(0, y_plunge, BH_POS, MASS, color=(0, 255, 255)))

        # Orbital ray that will eventually escape (white)
        y_escape = BH_POS[1] + (B_CRIT + DELTA_B)
        self.rays.append(Ray(0, y_escape, BH_POS, MASS, color=(255, 255, 255)))

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            dlambda = SPEED_SCALE * dt

            self.display.fill((10, 10, 20))   # deep dark background

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset()

            # Update and render all rays
            for ray in self.rays:
                ray.update(dlambda)
                if ray.alive:
                    ray.render(self.display)

            # --- Black hole and photon sphere ---
            # Photon sphere (glowing ring)
            pygame.gfxdraw.aacircle(self.display, BH_POS[0], BH_POS[1],
                                    int(PHOTON_SPHERE), (80, 80, 100))
            pygame.gfxdraw.aacircle(self.display, BH_POS[0], BH_POS[1],
                                    int(PHOTON_SPHERE)+1, (80, 80, 100))
            # Event horizon (black core)
            pygame.gfxdraw.filled_circle(self.display, BH_POS[0], BH_POS[1],
                                         int(RS), (0, 0, 0))
            pygame.gfxdraw.aacircle(self.display, BH_POS[0], BH_POS[1],
                                    int(RS), (100, 20, 20))

            # Blit scaled surface
            self.screen.blit(pygame.transform.scale(self.display, self.screen.get_size()), (0, 0))
            pygame.display.update()


if __name__ == "__main__":
    Game().run()