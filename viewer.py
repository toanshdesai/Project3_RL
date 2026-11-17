"""
viewer.py - scramble the cube and let your trained agent solve it.

    python viewer.py                    # uses your newest training run in models/
    python viewer.py models/quick_...   # use a specific run

Solving = beam search guided by the network's Q-values (agent.beam_solve).
Keys:  SPACE scramble   ENTER solve   BACKSPACE reset   U D F B L R turn a face (SHIFT = reverse)
       LEFT/RIGHT earlier/later snapshot of the agent   UP/DOWN scramble length   TAB search width
Drag the cube to rotate it. Only whole-face turns exist, so the cube can never reach an illegal state.
"""

import glob
import math
import os
import sys
import threading
import time
from collections import deque

import numpy as np
import pygame

import cube as C
from agent import QNet, beam_solve, find_latest_run
from render import GRAY, INK, UI, render_cube

W, H = 1000, 640
WIDTHS = [1, 100, 1000, 10000]
FACE_KEYS = {pygame.K_u: 0, pygame.K_d: 1, pygame.K_f: 2, pygame.K_b: 3, pygame.K_l: 4, pygame.K_r: 5}


class Viewer:
    def __init__(self, run_dir):
        self.ui = UI((W, H), "Rubik's Cube: solver")
        self.yaw, self.pitch = math.radians(-32), math.radians(26)
        self.state = C.solved_state()
        self.queue, self.turn = deque(), None            # pending (action, seconds) turns; turn being animated
        self.scramble, self.solution = [], []
        self.length, self.width_i = 6, 2
        self.status = "Scramble the cube, then solve it."
        self.searching, self.result, self.dragging = False, None, False
        self.rng = np.random.default_rng()
        # snapshots of the agent saved during training, oldest first
        self.models = sorted(glob.glob(os.path.join(run_dir, "ckpt_*.npz"))) if run_dir else []
        if run_dir and os.path.exists(os.path.join(run_dir, "final.npz")):
            self.models.append(os.path.join(run_dir, "final.npz"))
        self.model_i = len(self.models) - 1
        self.load_model()
        if not self.models:
            self.status = "No trained agent found. Run train.py first. This one is untrained."

    def load_model(self):
        if not self.models:
            self.net, self.model_name = QNet(), "untrained"
            return
        path = self.models[self.model_i]
        self.net = QNet.load(path)
        iterations = int(self.net.meta.get("iteration", 0))
        self.model_name = "untrained" if iterations == 0 else f"final ({iterations:,})" if path.endswith("final.npz") else f"{iterations:,} iterations"

    def busy(self):
        return bool(self.queue) or self.turn is not None or self.searching

    # ---- actions
    def do_scramble(self):
        if not self.busy():
            self.state = C.solved_state()
            _, self.scramble = C.random_scramble(self.length, self.rng)
            self.solution = []
            self.queue.extend((a, 0.18) for a in self.scramble)
            self.status = "Scrambled. Press Solve."

    def do_solve(self):
        if self.busy() or C.is_solved(self.state):
            return
        start, net, width = self.state.copy(), self.net, WIDTHS[self.width_i]
        self.searching, self.status = True, "Searching..."

        def search():                                     # runs in a thread so the window does not freeze
            began = time.time()
            self.result = (beam_solve(net, start, width=width, max_depth=40), time.time() - began)

        threading.Thread(target=search, daemon=True).start()

    def do_reset(self):
        if not self.searching:
            self.queue.clear()
            self.turn, self.state, self.scramble, self.solution = None, C.solved_state(), [], []
            self.status = "Scramble the cube, then solve it."

    def change_model(self, step):
        if self.models and not self.busy():
            self.model_i = (self.model_i + step) % len(self.models)
            self.load_model()

    # ---- per-frame update
    def update(self, dt):
        if self.result is not None:
            (moves, seconds), self.result, self.searching = self.result, None, False
            if moves is None:
                self.status = f"No solution found ({seconds:.1f} s). Try a wider search or a later snapshot."
            else:
                self.solution = moves
                self.status = f"Found a {len(moves)}-move solution in {seconds:.1f} s."
                self.queue.extend((a, 0.32) for a in moves)
        if self.turn is None and self.queue:
            action, duration = self.queue.popleft()
            self.turn = [action, 0.0, duration]
        if self.turn is not None:
            self.turn[1] += dt
            if self.turn[1] >= self.turn[2]:
                self.state = C.apply_move(self.state, self.turn[0])
                self.turn = None
                if not self.queue and self.solution and C.is_solved(self.state):
                    self.status = f"Solved in {len(self.solution)} moves."

    def handle(self, ev):
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_SPACE:
                self.do_scramble()
            elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.do_solve()
            elif ev.key == pygame.K_BACKSPACE:
                self.do_reset()
            elif ev.key in (pygame.K_LEFT, pygame.K_RIGHT):
                self.change_model(1 if ev.key == pygame.K_RIGHT else -1)
            elif ev.key in (pygame.K_UP, pygame.K_DOWN):
                self.length = min(20, max(1, self.length + (1 if ev.key == pygame.K_UP else -1)))
            elif ev.key == pygame.K_TAB:
                self.width_i = (self.width_i + 1) % len(WIDTHS)
            elif ev.key in FACE_KEYS and not self.searching:
                self.queue.append((FACE_KEYS[ev.key] + (6 if ev.mod & pygame.KMOD_SHIFT else 0), 0.25))
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and ev.pos[0] < 660:
            self.dragging = True
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self.dragging = False
        elif ev.type == pygame.MOUSEMOTION and self.dragging:
            self.yaw += ev.rel[0] * 0.01
            self.pitch = max(-1.45, min(1.45, self.pitch + ev.rel[1] * 0.01))

    # ---- drawing
    def stepper(self, label, value, y, minus="-", plus="+"):
        """A 'label  [-] value [+]' row. Returns -1, 0 or +1."""
        ui = self.ui
        ui.text(label, (680, y + 6), "n", GRAY)
        change = -1 if ui.button(minus, (810, y, 26, 30)) else 0
        ui.text(value, (890, y + 15), "n", INK, "center")
        return 1 if ui.button(plus, (944, y, 26, 30)) else change

    def draw(self):
        ui = self.ui
        turn = None if self.turn is None else (self.turn[0], self.turn[1] / self.turn[2])
        render_cube(ui.screen, self.state, 340, 310, 105, self.yaw, self.pitch, turn)
        ui.text("Solver", (680, 28), "big")
        step = self.stepper("Agent", self.model_name, 90, "<", ">")
        if step:
            self.change_model(step)
        self.length = min(20, max(1, self.length + self.stepper("Scramble length", str(self.length), 132)))
        self.width_i = (self.width_i + self.stepper("Search width", f"{WIDTHS[self.width_i]:,}", 174)) % len(WIDTHS)
        free = not self.busy()
        if ui.button("Scramble", (680, 232, 100, 34), free):
            self.do_scramble()
        if ui.button("Solve", (790, 232, 90, 34), free):
            self.do_solve()
        if ui.button("Reset", (890, 232, 80, 34), not self.searching):
            self.do_reset()
        y = ui.para(self.status, 680, 290, 290, "bold") + 16
        for title, moves in (("Scramble", self.scramble), ("Agent's solution", self.solution)):
            if moves:
                ui.text(f"{title} ({len(moves)} moves)", (680, y), "small", GRAY)
                y = ui.para(C.moves_to_str(moves), 680, y + 18, 290) + 12
        ui.text("Drag to rotate.  U D F B L R turn a face, Shift reverses.", (40, H - 34), "small", GRAY)
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            dt = self.ui.clock.tick(60) / 1000.0
            events = pygame.event.get()
            self.ui.begin(events)
            for ev in events:
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    running = False
                self.handle(ev)
            self.update(dt)
            self.draw()
        pygame.quit()


if __name__ == "__main__":
    models_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    Viewer(sys.argv[1] if len(sys.argv) > 1 else find_latest_run(models_root)).run()
