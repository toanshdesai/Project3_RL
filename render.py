"""
render.py - drawing code shared by train.py and viewer.py: the 3D cube and a few plain UI helpers.

A tiny software renderer: 27 cubies -> 162 square faces -> rotate for the camera -> perspective
projection -> hide faces pointing away -> draw far-to-near (painter's algorithm).
A face turn is animated by rotating the 9 cubies of that layer.
"""

import math
from itertools import product

import numpy as np
import pygame

import cube as C

BODY = (18, 18, 20)
# colour index -> RGB     0=U white  1=D yellow  2=F green  3=B blue  4=L orange  5=R red
STICKER_RGB = [(244, 244, 244), (252, 214, 0), (0, 158, 80), (0, 80, 190), (255, 110, 10), (205, 25, 35)]
CAM_DIST = 9.5


def axis_rotation(axis, theta):
    """3x3 rotation matrix: rotate by `theta` radians about unit vector `axis` (Rodrigues)."""
    x, y, z = axis
    c, s = math.cos(theta), math.sin(theta)
    k = 1 - c
    return np.array([[c + x * x * k, x * y * k - z * s, x * z * k + y * s],
                     [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
                     [z * x * k - y * s, z * y * k + x * s, c + z * z * k]])


def build_faces():
    """162 face quads = 27 cubies x 6 sides.  Sides that carry a sticker remember their state slot."""
    slot_of = {(tuple(p), tuple(n)): i for i, (p, n) in enumerate(zip(C.SLOT_POS, C.SLOT_NORMAL))}
    cub, full, inner, normals, centres, slots = [], [], [], [], [], []
    for p in product((-1, 0, 1), repeat=3):
        for n in C.FACE_NORMALS:
            axis = int(np.argmax(np.abs(n)))
            t1, t2 = [np.eye(3)[a] for a in range(3) if a != axis]
            c = np.array(p, float) + 0.5 * n
            corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
            full.append([c + a * 0.5 * t1 + b * 0.5 * t2 for a, b in corners])
            inner.append([c + a * 0.43 * t1 + b * 0.43 * t2 for a, b in corners])
            cub.append(p)
            normals.append(n.astype(float))
            centres.append(c)
            slots.append(slot_of.get((p, tuple(n)), -1))
    return (np.array(cub), np.array(full), np.array(inner), np.array(normals),
            np.array(centres), np.array(slots))


CUBIE, FULL, INNER, NORMAL, CENTRE, SLOT = build_faces()


def render_cube(surface, state, cx, cy, scale, yaw, pitch, anim=None):
    """Draw `state` centred at (cx, cy).  anim = (action, t) with t in 0..1 draws that turn part-way."""
    corners_full, corners_inner = FULL.copy(), INNER.copy()
    normals, centres = NORMAL.copy(), CENTRE.copy()
    if anim is not None:
        a, t = anim
        n = C.FACE_NORMALS[a % 6].astype(float)
        axis = int(np.argmax(np.abs(n)))
        layer = CUBIE[:, axis] == int(n[axis])
        t = min(1.0, max(0.0, t))
        t = t * t * (3 - 2 * t)                                   # ease in/out
        theta = (-1 if a < 6 else 1) * (math.pi / 2) * t          # clockwise = -90 deg about outward normal
        R = axis_rotation(n, theta)
        corners_full[layer] = corners_full[layer] @ R.T
        corners_inner[layer] = corners_inner[layer] @ R.T
        normals[layer] = normals[layer] @ R.T
        centres[layer] = centres[layer] @ R.T
    Rv = axis_rotation((1, 0, 0), pitch) @ axis_rotation((0, 1, 0), yaw)
    cf, ci = corners_full @ Rv.T, corners_inner @ Rv.T
    nv, cv = normals @ Rv.T, centres @ Rv.T
    d = CAM_DIST
    visible = np.einsum("ij,ij->i", nv, np.array([0, 0, d]) - cv) > 1e-6

    def proj(p):
        k = d / (d - p[..., 2])
        return np.stack([cx + scale * p[..., 0] * k, cy - scale * p[..., 1] * k], axis=-1)

    pf, pi = proj(cf), proj(ci)
    for i in sorted(np.flatnonzero(visible), key=lambda j: cv[j, 2]):      # painter's algorithm: far -> near
        pygame.draw.polygon(surface, BODY, pf[i])
        if SLOT[i] >= 0:
            pygame.draw.polygon(surface, STICKER_RGB[int(state[SLOT[i]])], pi[i])


# ---------------------------------------------------------------------------- plain UI helpers
BG = (247, 247, 244)
INK = (26, 26, 28)
GRAY = (128, 128, 126)
LIGHT = (212, 212, 208)
RED = (200, 55, 45)


def fmt_time(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


class UI:
    """One window, three font sizes, text and outlined buttons. Buttons return True when clicked."""

    def __init__(self, size, caption):
        pygame.display.init()
        pygame.font.init()
        pygame.display.set_caption(caption)
        self.screen = pygame.display.set_mode(size)
        self.clock = pygame.time.Clock()
        family = "helveticaneue,helvetica,arial,dejavusans"
        self.fonts = {"big": pygame.font.SysFont(family, 24, bold=True), "n": pygame.font.SysFont(family, 16),
                      "bold": pygame.font.SysFont(family, 16, bold=True), "small": pygame.font.SysFont(family, 13)}
        self.click = None

    def begin(self, events):
        """Call once per frame with this frame's events; remembers a left click for the buttons."""
        self.click = None
        for ev in events:
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                self.click = ev.pos
        self.screen.fill(BG)

    def text(self, s, pos, size="n", col=INK, anchor="topleft"):
        img = self.fonts[size].render(s, True, col)
        rect = img.get_rect(**{anchor: pos})
        self.screen.blit(img, rect)
        return rect

    def para(self, s, x, y, width, size="n", col=INK):
        """Word-wrapped text; returns the y below it."""
        font, line = self.fonts[size], ""
        for word in s.split():
            trial = (line + " " + word).strip()
            if font.size(trial)[0] > width and line:
                self.text(line, (x, y), size, col)
                y += font.get_linesize()
                line = word
            else:
                line = trial
        if line:
            self.text(line, (x, y), size, col)
            y += font.get_linesize()
        return y

    def button(self, label, rect, enabled=True):
        rect = pygame.Rect(rect)
        hover = enabled and rect.collidepoint(pygame.mouse.get_pos())
        if hover:
            pygame.draw.rect(self.screen, LIGHT, rect)
        pygame.draw.rect(self.screen, INK if enabled else LIGHT, rect, 1)
        self.text(label, rect.center, "n", INK if enabled else LIGHT, "center")
        return bool(enabled and self.click and rect.collidepoint(self.click))
