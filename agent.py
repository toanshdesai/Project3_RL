"""
agent.py - the learning agent: a Q-network (pure NumPy), greedy policy, and beam-search solver.

Reinforcement-learning framing
------------------------------
  environment : cube.py  (a scrambled cube; the agent never sees a "rulebook",
                          only the 54 sticker colours and the result of a move)
  state  s    : 54 sticker colours, one-hot encoded -> 324 inputs
  action a    : one of 12 quarter turns
  reward      : -1 for every move, 0 once the cube is solved (episode ends)
  Q(s, a)     : the network's estimate of "how many more moves until solved if I
                play a now and then play well".  We store it as a COST
                (positive number of moves) rather than a negative return;
                cost = -return, so "maximise return" == "minimise Q" below.
  policy      : greedy  a* = argmin_a Q(s, a)
"""

import glob
import os

import numpy as np
import cube as C


class QNet:
    """Fully-connected network  324 -> h1 -> h2 -> 12  with ReLU, trained by Adam.
    Written from scratch in NumPy so every step (forward, backprop, update) is visible."""

    def __init__(self, sizes=(324, 1024, 512, 12), seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = tuple(sizes)
        self.W = [(rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)
                  for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, dtype=np.float32) for b in sizes[1:]]
        self.W[-1] *= 0.1                      # small output layer -> start near 0
        # Adam state
        self._m = [np.zeros_like(p) for p in self.W + self.b]
        self._v = [np.zeros_like(p) for p in self.W + self.b]
        self._t = 0

    # ---- forward -----------------------------------------------------------
    def forward(self, x, keep=False):
        acts = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = np.maximum(x, 0)
            acts.append(x)
        return (x, acts) if keep else x

    def predict(self, states, chunk=8192):
        """Q-values for a batch of raw cube states (N,54) -> (N,12)."""
        states = np.atleast_2d(states)
        out = []
        for i in range(0, len(states), chunk):
            out.append(self.forward(C.encode(states[i:i + chunk])))
        return np.concatenate(out)

    def value(self, states):
        """V(s) = min_a Q(s,a): estimated moves-to-go.  Solved cubes are exactly 0."""
        v = self.predict(states).min(axis=1)
        return np.where(C.is_solved(states), 0.0, v)

    # ---- one weighted-MSE gradient step with Adam --------------------------
    def train_step(self, x, y, w, lr=1e-3, clip=5.0):
        out, acts = self.forward(x, keep=True)
        n = x.shape[0]
        err = out - y
        loss = float(np.mean(w[:, None] * err ** 2))
        g = (2.0 * w[:, None] * err / (n * out.shape[1])).astype(np.float32)
        grads_W, grads_b = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            grads_W[i] = acts[i].T @ g
            grads_b[i] = g.sum(axis=0)
            if i > 0:
                g = (g @ self.W[i].T) * (acts[i] > 0)
        grads = grads_W + grads_b
        # global-norm gradient clipping (keeps early, huge-error steps from blowing up)
        norm = np.sqrt(sum(float(np.sum(gr * gr)) for gr in grads))
        if norm > clip:
            grads = [gr * (clip / norm) for gr in grads]
        # Adam
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        params = self.W + self.b
        for k, (p, gr) in enumerate(zip(params, grads)):
            self._m[k] = b1 * self._m[k] + (1 - b1) * gr
            self._v[k] = b2 * self._v[k] + (1 - b2) * gr * gr
            mhat = self._m[k] / (1 - b1 ** self._t)
            vhat = self._v[k] / (1 - b2 ** self._t)
            p -= (lr * mhat / (np.sqrt(vhat) + eps)).astype(np.float32)
        return loss

    # ---- copy / save / load ------------------------------------------------
    def copy_weights_from(self, other):
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]

    def save(self, path, **meta):
        data = {f"W{i}": w for i, w in enumerate(self.W)}
        data.update({f"b{i}": b for i, b in enumerate(self.b)})
        data["sizes"] = np.array(self.sizes)
        for k, v in meta.items():
            data[f"meta_{k}"] = np.array(v)
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        net = cls(tuple(int(s) for s in d["sizes"]))
        net.W = [d[f"W{i}"] for i in range(len(net.W))]
        net.b = [d[f"b{i}"] for i in range(len(net.b))]
        net.meta = {k[5:]: d[k].item() if d[k].ndim == 0 else d[k] for k in d.files if k.startswith("meta_")}
        return net


# ----------------------------------------------------------------------------
# Using the trained network
# ----------------------------------------------------------------------------
def greedy_rollouts(net, states, max_steps=30):
    """Play the greedy policy on a whole batch at once.  Returns a boolean array: solved within max_steps?"""
    states = states.copy()
    solved = C.is_solved(states)
    for _ in range(max_steps):
        if solved.all():
            break
        a = net.predict(states).argmin(axis=1)
        states = np.where(solved[:, None], states, C.apply_move_per_row(states, a))
        solved |= C.is_solved(states)
    return solved


def beam_solve(net, state, width=1000, max_depth=30):
    """Beam search guided by the Q-network.

    At every level we expand each of the `width` best states into its 12 children and keep the
    `width` children the network likes best (smallest predicted moves-to-go).  We never revisit
    a state and never immediately undo a move.  Returns a list of action ids, or None.
    """
    state = np.asarray(state, dtype=np.int8)
    if C.is_solved(state):
        return []
    beam = state[None, :]
    last_action = np.array([-1])
    seen = {state.tobytes()}
    history = []                                   # per level: (parent_index, action) arrays
    for _ in range(max_depth):
        q = net.predict(beam)                      # (B,12) estimated cost of each move
        children = C.expand(beam)                  # (B,12,54)
        B = beam.shape[0]
        par = np.repeat(np.arange(B), C.N_ACTIONS)
        act = np.tile(np.arange(C.N_ACTIONS), B)
        flat = children.reshape(-1, C.N_STICKERS)
        score = q.reshape(-1)
        keep = ~((last_action[par] >= 0) & (act == C.INVERSE_ACTION[np.maximum(last_action[par], 0)]))
        hit = np.flatnonzero(C.is_solved(flat) & keep)
        if hit.size:                               # solved! rebuild the move list
            i = hit[0]
            moves = [int(act[i])]
            p = int(par[i])
            for lvl in range(len(history) - 1, -1, -1):
                pa, ac = history[lvl]
                moves.append(int(ac[p]))
                p = int(pa[p])
            return moves[::-1]
        idx = np.flatnonzero(keep)
        idx = idx[np.argsort(score[idx], kind="stable")]
        chosen = []
        for i in idx:                              # best-first, skipping states already seen
            kb = flat[i].tobytes()
            if kb not in seen:
                seen.add(kb)
                chosen.append(i)
                if len(chosen) >= width:
                    break
        if not chosen:
            return None
        chosen = np.array(chosen)
        history.append((par[chosen], act[chosen]))
        beam = flat[chosen]
        last_action = act[chosen]
    return None


def find_latest_run(models_root):
    """Newest folder inside models/ that holds a saved network (or None)."""
    runs = [d for d in glob.glob(os.path.join(models_root, "*")) if glob.glob(os.path.join(d, "*.npz"))]
    return max(runs, key=os.path.getmtime) if runs else None
