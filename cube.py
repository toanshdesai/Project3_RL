"""
cube.py - The Rubik's Cube environment (the "game world" the RL agent lives in).

Design goals
------------
1. LEGAL BY CONSTRUCTION. The only actions are quarter turns of one of the six
   outer faces (clockwise or counter-clockwise) -> 12 actions.  There is no
   action that twists a single corner or swaps two stickers, so every state the
   environment can reach is a state a real cube can reach.
2. FAST. Every move is a precomputed permutation of the 54 stickers, so a
   whole batch of cubes can be moved at once with one NumPy fancy-index.
3. NO HAND-TYPED TABLES. The permutations are derived from 3D geometry (rotate
   the stickers of a layer by 90 degrees), which avoids transcription errors.

State representation
--------------------
A state is an int8 vector of length 54.  Entry i is the COLOUR (0..5) of
sticker slot i.  The solved cube has colour c on all 9 slots of face c.

Face order / colours:   0=U(white) 1=D(yellow) 2=F(green) 3=B(blue) 4=L(orange) 5=R(red)
Moves (action ids):     0..5  = U D F B L R clockwise  (as seen looking AT that face)
                        6..11 = the same faces counter-clockwise (U' D' F' B' L' R')
"""

import numpy as np

# ----------------------------------------------------------------------------
# Geometry: build the 54 sticker slots
# ----------------------------------------------------------------------------
FACE_NAMES = ["U", "D", "F", "B", "L", "R"]
# Outward unit normal of each face (x = right, y = up, z = toward the viewer)
FACE_NORMALS = np.array([
    [0, 1, 0],    # U
    [0, -1, 0],   # D
    [0, 0, 1],    # F
    [0, 0, -1],   # B
    [-1, 0, 0],   # L
    [1, 0, 0],    # R
], dtype=np.int64)

N_STICKERS = 54
N_ACTIONS = 12
ACTION_NAMES = [f + "" for f in FACE_NAMES] + [f + "'" for f in FACE_NAMES]


def _build_slots():
    """Return (positions, normals), each (54, 3).  Position = cubie coordinates in {-1,0,1}."""
    pos, nrm = [], []
    for n in FACE_NORMALS:
        axis = int(np.argmax(np.abs(n)))
        other = [a for a in range(3) if a != axis]
        for u in (-1, 0, 1):
            for v in (-1, 0, 1):
                p = np.zeros(3, dtype=np.int64)
                p[axis] = n[axis]          # the cubie sits on that face's layer
                p[other[0]] = u
                p[other[1]] = v
                pos.append(p)
                nrm.append(n.copy())
    return np.array(pos), np.array(nrm)


SLOT_POS, SLOT_NORMAL = _build_slots()
_SLOT_INDEX = {(tuple(p), tuple(n)): i for i, (p, n) in enumerate(zip(SLOT_POS, SLOT_NORMAL))}


def _rotate_cw(vec, axis_normal):
    """Rotate integer vector `vec` by 90 degrees CLOCKWISE when looking at the face
    whose outward normal is `axis_normal`.  (Clockwise from outside = -90 degrees
    about the outward normal by the right-hand rule.)  Rodrigues' formula for
    theta=-90: v' = n(n.v) - (n x v)."""
    n = axis_normal
    return n * np.dot(n, vec) - np.cross(n, vec)


def _build_permutations():
    """perm[a][j] = i means: after move a, the sticker now in slot j came from slot i.
    Applying a move to a state s is therefore  s[perm[a]]."""
    perms = np.zeros((N_ACTIONS, N_STICKERS), dtype=np.int64)
    for f in range(6):
        n = FACE_NORMALS[f]
        axis = int(np.argmax(np.abs(n)))
        layer_val = n[axis]
        cw = np.arange(N_STICKERS)
        for i in range(N_STICKERS):
            if SLOT_POS[i][axis] == layer_val:                 # sticker is in the turning layer
                new_p = _rotate_cw(SLOT_POS[i], n)
                new_n = _rotate_cw(SLOT_NORMAL[i], n)
                j = _SLOT_INDEX[(tuple(new_p), tuple(new_n))]
                cw[j] = i                                       # content of slot i lands in slot j
        perms[f] = cw
        perms[f + 6] = np.argsort(cw)                           # inverse permutation = counter-clockwise
    return perms


PERMS = _build_permutations()
INVERSE_ACTION = np.array([a + 6 if a < 6 else a - 6 for a in range(N_ACTIONS)])

SOLVED = np.repeat(np.arange(6, dtype=np.int8), 9)              # slots 0-8 = colour 0, 9-17 = colour 1, ...


# ----------------------------------------------------------------------------
# Environment functions (all work on a single state (54,) or a batch (N, 54))
# ----------------------------------------------------------------------------
def apply_move(state, action):
    """Apply one action to a single state or a whole batch with the same action."""
    return state[..., PERMS[action]]


def apply_moves(state, actions):
    """Apply a list of actions in order."""
    for a in actions:
        state = apply_move(state, a)
    return state


def apply_move_per_row(states, actions):
    """Batch of states, one (possibly different) action per row.  states: (N,54), actions: (N,)"""
    return np.take_along_axis(states, PERMS[actions], axis=1)


def expand(states):
    """All 12 children of every state.  (N,54) -> (N,12,54)."""
    return states[:, PERMS]            # PERMS is (12,54) -> result (N,12,54)


def is_solved(states):
    """True where the cube is solved.  Works on (54,) or (N,54).  Uses face-uniformity, so a
    cube that is solved but whole-cube-rotated cannot occur (centre stickers never move)."""
    return np.all(states == SOLVED, axis=-1)


def solved_state():
    return SOLVED.copy()


def random_scramble(n_moves, rng=None):
    """Return (scrambled_state, move_list).  Moves never undo the move just before them
    (so a 'scramble of 5' really is 5 moves away in most cases)."""
    rng = rng or np.random.default_rng()
    moves, last = [], -1
    for _ in range(n_moves):
        while True:
            a = int(rng.integers(N_ACTIONS))
            if last < 0 or a != INVERSE_ACTION[last]:
                break
        moves.append(a)
        last = a
    return apply_moves(SOLVED.copy(), moves), moves


def random_walk_batch(batch, max_depth, rng):
    """Generate `batch` states by random walks FROM the solved cube, each of length
    uniform in 1..max_depth.  Returns (states (batch,54), depths (batch,)).
    This is the training distribution for Autodidactic Iteration."""
    depths = rng.integers(1, max_depth + 1, size=batch)
    states = np.tile(SOLVED, (batch, 1))
    last = np.full(batch, -1)
    for step in range(max_depth):
        active = depths > step
        if not active.any():
            break
        acts = rng.integers(0, N_ACTIONS, size=batch)
        # forbid immediately undoing the previous move: resample those rows by shifting
        bad = (last >= 0) & (acts == INVERSE_ACTION[np.maximum(last, 0)])
        acts[bad] = (acts[bad] + 1 + rng.integers(0, N_ACTIONS - 1, size=bad.sum())) % N_ACTIONS
        # the shift above can (rarely) land on the forbidden action again -> do a second fix
        bad = (last >= 0) & (acts == INVERSE_ACTION[np.maximum(last, 0)])
        acts[bad] = (acts[bad] + 1) % N_ACTIONS
        moved = apply_move_per_row(states, acts)
        states = np.where(active[:, None], moved, states)
        last = np.where(active, acts, last)
    return states, depths


def moves_to_str(moves):
    return " ".join(ACTION_NAMES[a] for a in moves)


def str_to_moves(text):
    lookup = {name: i for i, name in enumerate(ACTION_NAMES)}
    return [lookup[tok] for tok in text.split()]


# ----------------------------------------------------------------------------
# Neural-network input encoding
# ----------------------------------------------------------------------------
def encode(states):
    """One-hot encode colours: (N,54) int8 -> (N,324) float32  (54 slots x 6 colours)."""
    states = np.atleast_2d(states)
    out = np.zeros((states.shape[0], N_STICKERS * 6), dtype=np.float32)
    rows = np.repeat(np.arange(states.shape[0]), N_STICKERS)
    cols = (np.tile(np.arange(N_STICKERS), states.shape[0]) * 6) + states.reshape(-1)
    out[rows, cols] = 1.0
    return out
