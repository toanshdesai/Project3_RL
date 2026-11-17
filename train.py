"""
train.py - train the agent yourself and watch it learn.

    python train.py          # choose 5, 20 or 60 minutes in the window

ONE TRAINING ITERATION (Trainer.step, repeated until the time is up)
  1. Make practice cubes: scramble a solved cube with 1..max_depth random turns, 2048 times.
  2. Bellman targets: for each cube and each of its 12 moves, ask the frozen TARGET network how far
     the resulting cube s' is from solved:   target(s, a) = 1 + min_a' Q_target(s', a')
  3. Gradient steps: move the network's Q(s, a) toward the targets (shrink the TD error).
  4. Every 10 iterations: copy the network into the target network.

Keys:  SPACE pause   T test the agent on a fresh scramble   UP/DOWN test length   S stop and save
Each run is saved in models/<preset>_<date_time>/ ; viewer.py opens the newest run.
"""

import csv
import math
import os
import threading
import time
from collections import deque

import numpy as np
import pygame

import cube as C
from agent import QNet, greedy_rollouts
from render import GRAY, INK, LIGHT, RED, STICKER_RGB, UI, fmt_time, render_cube

# name: (minutes, hidden layer sizes, longest practice scramble)
PRESETS = {"quick": (5, (512, 256), 14), "standard": (20, (1024, 512), 20), "full": (60, (1024, 512), 20)}
BATCH = 2048          # practice cubes per iteration
MINIBATCH = 512       # cubes per gradient step
EPOCHS = 4            # passes over each batch -> 16 gradient steps per iteration
SYNC = 10             # iterations between target-network refreshes
LR_START, LR_END = 1e-3, 1e-4
TEST_DEPTHS = [1, 2, 3, 4, 5, 6, 8, 10]
SNAPSHOT_AT = [0.10, 0.25, 0.50, 0.75]      # fractions of the run at which a snapshot is saved


class Trainer:
    """Holds the network and does the learning. The window below only displays it."""

    def __init__(self, preset, models_root, minutes=None):
        default_minutes, hidden, self.max_depth = PRESETS[preset]
        self.preset = preset
        self.budget = 60.0 * (minutes or default_minutes)        # seconds of computing; pauses do not count
        self.rng = np.random.default_rng()
        seed = int(self.rng.integers(1 << 30))
        self.net = QNet((324, *hidden, 12), seed=seed)            # the network being trained
        self.target = QNet((324, *hidden, 12), seed=seed)         # frozen copy used to build targets
        test_rng = np.random.default_rng(12345)                   # fixed test cubes, never trained on
        self.test_set = {d: np.array([C.random_scramble(d, test_rng)[0] for _ in range(200)]) for d in TEST_DEPTHS}

        self.run_dir = os.path.join(models_root, f"{preset}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.log_path = os.path.join(self.run_dir, "training_log.csv")
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(["iteration", "seconds", "td_error"] + [f"solved_{d}_moves" for d in TEST_DEPTHS])

        self.iteration, self.seconds, self.step_name = 0, 0.0, ""
        self.lr, self.td_error = LR_START, float("nan")
        self.sample = None                    # practice cube #1 with its Q-values and targets (for display)
        self.history = []                     # [(seconds, {depth: share solved})]
        self.done = False
        self._next_test, self._snapshots_left = 0.0, list(SNAPSHOT_AT)
        self.save("ckpt_00000.npz")           # the untrained network
        self.test()

    def save(self, name):
        self.net.save(os.path.join(self.run_dir, name), iteration=self.iteration)

    def test(self):
        """Greedy policy (always the lowest Q) on the fixed test cubes. No learning happens here."""
        result = {d: float(greedy_rollouts(self.net, cubes, 30).mean()) for d, cubes in self.test_set.items()}
        self.history.append((self.seconds, result))
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([self.iteration, round(self.seconds, 1), round(self.td_error, 4)] +
                                   [result[d] for d in TEST_DEPTHS])
        self._next_test = self.seconds + min(60.0, max(8.0, self.budget / 40))

    def step(self):
        start = time.perf_counter()
        progress = min(1.0, self.seconds / self.budget)
        self.lr = LR_START + (LR_END - LR_START) * progress               # learning rate shrinks over the run

        # 1. practice cubes, made backwards from the solved cube
        self.step_name = "cubes"
        states, depths = C.random_walk_batch(BATCH, self.max_depth, self.rng)
        states[0], moves = C.random_scramble(int(self.rng.integers(1, 7)), self.rng)   # cube #1: known scramble, shown on screen
        depths[0] = len(moves)

        # 2. Bellman targets from the frozen target network
        self.step_name = "targets"
        children = C.expand(states).reshape(-1, C.N_STICKERS)             # every cube after each of its 12 moves
        targets = (1.0 + self.target.value(children).reshape(BATCH, 12)).astype(np.float32)
        self.td_error = float(np.mean(np.abs(targets - self.net.predict(states))))
        weights = (1.0 / depths).astype(np.float32)                       # short scrambles have more reliable targets
        weights[C.is_solved(states)] = 0.0
        weights *= BATCH / weights.sum()
        inputs = C.encode(states)

        # 3. gradient steps toward the targets
        self.step_name = "gradient"
        for _ in range(EPOCHS):
            order = self.rng.permutation(BATCH)
            for i in range(0, BATCH, MINIBATCH):
                idx = order[i:i + MINIBATCH]
                self.net.train_step(inputs[idx], targets[idx], weights[idx], self.lr)
        self.iteration += 1
        self.sample = dict(state=states[0].copy(), moves=moves, q=self.net.predict(states[0])[0], target=targets[0].copy())

        # 4. refresh the target network
        if self.iteration % SYNC == 0:
            self.step_name = "sync"
            self.target.copy_weights_from(self.net)

        self.seconds += time.perf_counter() - start
        if self.seconds >= self._next_test:
            self.test()
        if self._snapshots_left and self.seconds / self.budget >= self._snapshots_left[0]:
            self._snapshots_left.pop(0)
            self.save(f"ckpt_{self.iteration:05d}.npz")
        if self.seconds >= self.budget:
            self.finish()

    def finish(self):
        if not self.done:
            self.test()
            self.save("final.npz")
            self.step_name, self.done = "", True

    def copy_of_network(self):
        copy = QNet(self.net.sizes)
        copy.copy_weights_from(self.net)
        return copy


# ------------------------------------------------------------------------------------ the window
W, H = 1000, 640
LOOP_LINES = [("cubes", f"1  Make {BATCH:,} practice cubes from the solved cube"),
              ("targets", "2  Score every move with the frozen target network"),
              ("gradient", "3  Adjust the network toward those scores"),
              ("sync", f"4  Every {SYNC} iterations, refresh the target network")]
CURVES = {2: STICKER_RGB[3], 4: STICKER_RGB[2], 6: STICKER_RGB[4], 8: STICKER_RGB[5]}     # scramble length -> line colour


class Window:
    def __init__(self):
        self.ui = UI((W, H), "Rubik's Cube: training")
        self.models_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.trainer, self.thread = None, None
        self.paused = self.stop = self.want_copy = False
        self.copy = None
        self.test, self.test_len, self.last_test = None, 4, ""
        self.open_solver = False

    # ---- training runs in a second thread so the window stays responsive
    def start(self, preset, minutes=None):
        self.ui.begin([])
        self.ui.text("Starting...", (W // 2, H // 2), "n", GRAY, "center")
        pygame.display.flip()
        self.trainer = Trainer(preset, self.models_root, minutes)
        self.thread = threading.Thread(target=self.work, daemon=True)
        self.thread.start()

    def work(self):
        t = self.trainer
        while not t.done and not self.stop:
            if self.paused:
                time.sleep(0.05)
            else:
                t.step()
            if self.want_copy:
                self.copy, self.want_copy = t.copy_of_network(), False
        t.finish()                                    # also saves when stopped early

    # ---- "test the agent": a copy of the current network plays a fresh scramble greedily
    def start_test(self):
        if self.trainer and not self.trainer.done and self.test is None:
            self.copy, self.want_copy, self.test = None, True, {"stage": "wait"}

    def plan_test(self):
        state, scramble = C.random_scramble(self.test_len, np.random.default_rng())
        cube, moves, q_values, seen, outcome = state.copy(), [], [], {state.tobytes()}, "not solved"
        for _ in range(2 * self.test_len + 6):
            q = self.copy.predict(cube)[0]
            q_values.append(q)
            moves.append(int(q.argmin()))                       # greedy: the move with the lowest Q
            cube = C.apply_move(cube, moves[-1])
            if C.is_solved(cube):
                outcome = "solved"
                break
            if cube.tobytes() in seen:
                outcome = "stuck in a loop"
                break
            seen.add(cube.tobytes())
        self.test = dict(stage="scramble", state=C.solved_state(), scramble=scramble, moves=moves, q=q_values,
                         outcome=outcome, queue=deque((a, 0.14) for a in scramble), turn=None, wait=0.0, played=0)

    def update_test(self, dt):
        t = self.test
        if t is None:
            return
        if t["stage"] == "wait":
            if self.copy is not None:
                self.plan_test()
            return
        if t["turn"] is None and t["queue"]:
            action, duration = t["queue"].popleft()
            t["turn"] = [action, 0.0, duration]
        if t["turn"] is not None:
            t["turn"][1] += dt
            if t["turn"][1] >= t["turn"][2]:
                t["state"] = C.apply_move(t["state"], t["turn"][0])
                t["played"] += t["stage"] == "play"
                t["turn"] = None
            return
        t["wait"] += dt
        if t["stage"] == "scramble" and t["wait"] > 0.7:
            t.update(stage="play", wait=0.0, queue=deque((a, 0.4) for a in t["moves"]))
        elif t["stage"] == "play":
            t.update(stage="result", wait=0.0)
            self.last_test = f"Last test: {len(t['scramble'])}-move scramble, {t['outcome']}"
        elif t["stage"] == "result" and t["wait"] > 3.0:
            self.test = None

    # ---- drawing
    def draw_menu(self):
        ui = self.ui
        ui.text("Train the Rubik's Cube agent", (80, 90), "big")
        y = ui.para("The agent starts with random Q-values and knows nothing about the cube. It learns from one rule: "
                    "every move costs 1, and the episode ends when the cube is solved.", 80, 135, 700, "n", GRAY)
        rows = [("quick", "Quick", "5 minutes, small network. A first look, or a live demo."),
                ("standard", "Standard", "20 minutes, full-size network."),
                ("full", "Full", "60 minutes, full-size network. Best on long scrambles.")]
        for i, (key, name, note) in enumerate(rows):
            top = y + 40 + i * 70
            if ui.button(f"{i + 1}   {name}", (80, top, 180, 44)):
                self.start(key)
                return
            ui.text(note, (285, top + 13), "n", GRAY)
        ui.text("Press 1, 2 or 3, or click.", (80, y + 270), "small", GRAY)

    def draw_bars(self, q, target, chosen):
        ui, x0, y0, width, height = self.ui, 100, 440, 340, 110
        top = max(4.0, math.ceil(max(float(q.max()), float(target.max()) if target is not None else 0) / 2) * 2)
        pygame.draw.line(ui.screen, LIGHT, (x0, y0 + height), (x0 + width, y0 + height))
        ui.text(f"{top:.0f} moves", (x0 - 6, y0 - 6), "small", GRAY, "topright")
        ui.text("0", (x0 - 6, y0 + height - 8), "small", GRAY, "topright")
        for a in range(12):
            cx = x0 + width * (a + 0.5) / 12
            bar = height * min(top, max(0.0, float(q[a]))) / top
            pygame.draw.rect(ui.screen, INK if a == chosen else GRAY, (cx - 7, y0 + height - bar, 14, max(1, bar)))
            if target is not None:
                ty = y0 + height - height * min(top, max(0.0, float(target[a]))) / top
                pygame.draw.line(ui.screen, RED, (cx - 11, ty), (cx + 11, ty), 2)
            ui.text(C.ACTION_NAMES[a], (cx, y0 + height + 6), "small", INK if a == chosen else GRAY, "midtop")

    def draw_chart(self):
        ui, t = self.ui, self.trainer
        x0, y0, width, height = 560, 330, 384, 190
        ui.text("Test scrambles solved by the greedy policy (no search)", (x0, y0 - 28), "small", GRAY)
        pygame.draw.line(ui.screen, LIGHT, (x0, y0), (x0, y0 + height))
        pygame.draw.line(ui.screen, LIGHT, (x0, y0 + height), (x0 + width, y0 + height))
        ui.text("100%", (x0 - 6, y0 - 7), "small", GRAY, "topright")
        ui.text("0", (x0 - 6, y0 + height - 8), "small", GRAY, "topright")
        ui.text("0:00", (x0, y0 + height + 6), "small", GRAY)
        ui.text(fmt_time(t.budget), (x0 + width, y0 + height + 6), "small", GRAY, "topright")
        history = list(t.history)
        for i, (length, colour) in enumerate(CURVES.items()):
            points = [(float(x0 + width * min(1.0, s / t.budget)), float(y0 + height - height * r[length])) for s, r in history]
            if len(points) > 1:
                pygame.draw.lines(ui.screen, colour, False, points, 2)
            lx = x0 - 20 + i * 104
            pygame.draw.rect(ui.screen, colour, (lx, y0 + height + 34, 10, 10))
            ui.text(f"{length} moves  {100 * history[-1][1][length]:.0f}%", (lx + 15, y0 + height + 31), "small")

    def draw_training(self):
        ui, t = self.ui, self.trainer
        pygame.draw.rect(ui.screen, LIGHT, (0, 0, W, 4))
        pygame.draw.rect(ui.screen, INK, (0, 0, int(W * min(1.0, t.seconds / t.budget)), 4))
        ui.text(f"Training: {t.preset}", (60, 28), "big")
        state = "finished" if t.done else "paused" if self.paused else ""
        ui.text(f"{state}     iteration {t.iteration:,}     {fmt_time(t.seconds)} of {fmt_time(t.budget)}", (W - 60, 36), "n", GRAY, "topright")

        # left: the cube and its Q-values
        test = self.test if self.test and self.test["stage"] != "wait" else None
        if test:
            turn = None if test["turn"] is None else (test["turn"][0], test["turn"][1] / test["turn"][2])
            render_cube(ui.screen, test["state"], 250, 235, 48, math.radians(-32), math.radians(26), turn)
            caption = {"scramble": f"Test: scrambling with {len(test['scramble'])} random turns",
                       "play": "Test: the agent plays the move with the lowest bar",
                       "result": f"Test: {test['outcome']}" + (f" in {len(test['moves'])} moves" if test["outcome"] == "solved" else "")}[test["stage"]]
            ui.text(caption, (250, 385), "bold", INK, "midtop")
            if test["stage"] == "play" and test["played"] < len(test["q"]):
                self.draw_bars(test["q"][test["played"]], None, test["moves"][test["played"]])
        elif t.sample:
            s = t.sample
            render_cube(ui.screen, s["state"], 250, 235, 48, math.radians(-32), math.radians(26))
            ui.text(f"Practice cube: solved + {C.moves_to_str(s['moves'])}", (250, 385), "n", INK, "midtop")
            self.draw_bars(s["q"], s["target"], int(s["q"].argmin()))
        ui.text("Bar: moves the network expects to need after that move. Dark: its choice.", (60, 582), "small", GRAY)
        if not test:
            ui.text("Red line: the Bellman target it is being pulled toward.", (60, 599), "small", GRAY)

        # right: the loop, two numbers, the learning curve
        ui.text("The training loop", (560, 90), "small", GRAY)
        for i, (name, line) in enumerate(LOOP_LINES):
            active = t.step_name == name and not self.paused
            ui.text(line, (560, 114 + i * 26), "bold" if active else "n", INK if active else GRAY)
        ui.text(f"TD error {t.td_error:.2f} moves", (560, 236), "n")
        ui.text(f"learning rate {t.lr:.5f}", (760, 236), "n")
        ui.text("TD error = average gap between the network's Q-values and the targets.", (560, 260), "small", GRAY)
        self.draw_chart()

        # bottom right: controls
        if t.done:
            ui.text(f"Saved in models/{os.path.basename(t.run_dir)}", (560, 575), "small", GRAY)
            if ui.button("Open the solver", (700, 598, 150, 30)) or self.key == pygame.K_RETURN:
                self.open_solver = True
            if ui.button("Close", (860, 598, 80, 30)):
                self.running = False
        else:
            if ui.button("Resume" if self.paused else "Pause", (560, 598, 80, 30)) or self.key == pygame.K_SPACE:
                self.paused = not self.paused
            if ui.button("Test agent", (650, 598, 100, 30), self.test is None) or self.key == pygame.K_t:
                self.start_test()
            if ui.button("-", (760, 598, 26, 30)) or self.key == pygame.K_DOWN:
                self.test_len = max(1, self.test_len - 1)
            ui.text(str(self.test_len), (800, 613), "n", INK, "center")
            if ui.button("+", (814, 598, 26, 30)) or self.key == pygame.K_UP:
                self.test_len = min(20, self.test_len + 1)
            if ui.button("Stop & save", (850, 598, 90, 30)) or self.key == pygame.K_s:
                self.stop = True
            ui.text(self.last_test or "Test length is the number of scramble turns.", (560, 575), "small", GRAY)

    def frame(self, events, dt):
        """Handle one frame. Returns False when the window should close."""
        self.running, self.key = True, None
        for ev in events:
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                self.running = False
            elif ev.type == pygame.KEYDOWN:
                self.key = ev.key
        self.ui.begin(events)
        if self.trainer is None:
            if self.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                self.start(list(PRESETS)[self.key - pygame.K_1])
            else:
                self.draw_menu()
        else:
            self.update_test(dt)
            self.draw_training()
        pygame.display.flip()
        return self.running and not self.open_solver

    def run(self):
        while self.frame(pygame.event.get(), self.ui.clock.tick(30) / 1000.0):
            pass
        self.stop = True                               # closing the window also saves the network
        if self.thread:
            self.thread.join()
        if self.open_solver:
            from viewer import Viewer
            Viewer(self.trainer.run_dir).run()
        else:
            pygame.quit()
            if self.trainer:
                print(f"Saved in {self.trainer.run_dir}. Open the solver with: python viewer.py")


if __name__ == "__main__":
    Window().run()
