"""
test_cube.py - proves the environment only ever produces LEGAL Rubik's Cube states.

Run:  python test_cube.py

Checks (independent of how the move tables were built):
  1. Every move is a true permutation, never moves a centre, moves exactly 20 stickers.
  2. Each quarter turn has order 4; a move followed by its inverse is the identity.
  3. Known cube facts: (R U R' U') has order 6, (R U) has order 105.
  4. BFS state counts match the published quarter-turn-metric numbers
     1, 12, 114, 1068, 10011, 93840 (this fails for any wrong/buggy move table).
  5. Group invariants of the real cube hold after thousands of random scrambles:
       - sum of corner twists = 0 (mod 3)
       - sum of edge flips    = 0 (mod 2)
       - corner permutation parity == edge permutation parity
     and a negative control shows these checks DO catch an illegal state
     (e.g. a single twisted corner, which is what "turning a corner" would do).
"""

import numpy as np
import cube as C

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- helpers
def key(s):
    return s.tobytes()


def order_of(moves, limit=200):
    s = C.SOLVED.copy()
    for k in range(1, limit + 1):
        s = C.apply_moves(s, moves)
        if C.is_solved(s):
            return k
    return None


def cubie_slots():
    """Group sticker slots by the cubie they belong to -> {position: [slot indices]}."""
    groups = {}
    for i, p in enumerate(C.SLOT_POS):
        groups.setdefault(tuple(p), []).append(i)
    return groups


GROUPS = cubie_slots()
CORNERS = [p for p, sl in GROUPS.items() if len(sl) == 3]
EDGES = [p for p, sl in GROUPS.items() if len(sl) == 2]


def piece_id(state, pos):
    """Identify which physical cubie sits at `pos` by the SET of colours on it."""
    return frozenset(int(state[i]) for i in GROUPS[pos])


def perm_parity(perm):
    seen, parity = set(), 0
    for i in range(len(perm)):
        if i in seen:
            continue
        j, length = i, 0
        while j not in seen:
            seen.add(j)
            j = perm[j]
            length += 1
        parity += length - 1
    return parity % 2


def corner_twist_sum(state):
    """Standard corner orientation: order the 3 stickers of a corner so that
    (n0, n1, n2) is right-handed and n0 is the vertical (U/D) one.  Twist = index of the
    sticker showing a white/yellow colour."""
    total = 0
    for pos in CORNERS:
        slots = GROUPS[pos]
        ny = [i for i in slots if abs(C.SLOT_NORMAL[i][1]) == 1][0]
        others = [i for i in slots if i != ny]
        a, b = others
        if np.linalg.det(np.array([C.SLOT_NORMAL[ny], C.SLOT_NORMAL[a], C.SLOT_NORMAL[b]])) < 0:
            a, b = b, a
        order = [ny, a, b]
        twist = [k for k, i in enumerate(order) if state[i] in (0, 1)][0]
        total += twist
    return total % 3


def edge_flip_sum(state):
    """Edge orientation with the usual convention: the piece's 'primary' colour is its U/D
    colour, or its F/B colour if it has none.  The reference sticker of a POSITION is its
    vertical (y) sticker, or its z sticker if it has no y sticker.  Flipped = primary colour
    is not on the reference sticker."""
    total = 0
    for pos in EDGES:
        slots = GROUPS[pos]
        colours = [int(state[i]) for i in slots]
        primary = [c for c in colours if c in (0, 1)] or [c for c in colours if c in (2, 3)]
        ref = [i for i in slots if abs(C.SLOT_NORMAL[i][1]) == 1] or \
              [i for i in slots if abs(C.SLOT_NORMAL[i][2]) == 1]
        total += 0 if int(state[ref[0]]) == primary[0] else 1
    return total % 2


def parities(state):
    solved_corner_ids = [piece_id(C.SOLVED, p) for p in CORNERS]
    solved_edge_ids = [piece_id(C.SOLVED, p) for p in EDGES]
    cp = [solved_corner_ids.index(piece_id(state, p)) for p in CORNERS]
    ep = [solved_edge_ids.index(piece_id(state, p)) for p in EDGES]
    return perm_parity(cp), perm_parity(ep)


def is_legal(state):
    cp, ep = parities(state)
    return corner_twist_sum(state) == 0 and edge_flip_sum(state) == 0 and cp == ep


# ---------------------------------------------------------------- tests
def test_permutations():
    for a in range(C.N_ACTIONS):
        p = C.PERMS[a]
        assert sorted(p) == list(range(54)), f"move {a} is not a permutation"
        moved = np.flatnonzero(p != np.arange(54))
        assert len(moved) == 20, f"move {C.ACTION_NAMES[a]} moves {len(moved)} stickers (expected 20)"
        for centre in (4, 13, 22, 31, 40, 49):
            assert p[centre] == centre, "a centre sticker moved"
    print("ok  permutations: 12 valid moves, 20 stickers each, centres fixed")


def test_orders_and_inverses():
    for a in range(6):
        assert order_of([a]) == 4, f"{C.ACTION_NAMES[a]} should have order 4"
        s = C.apply_moves(C.SOLVED.copy(), [a, C.INVERSE_ACTION[a]])
        assert C.is_solved(s)
    print("ok  every quarter turn has order 4 and inverse cancels it")


def test_known_orders():
    R, U = 5, 0
    Ri, Ui = C.INVERSE_ACTION[R], C.INVERSE_ACTION[U]
    assert order_of([R, U, Ri, Ui]) == 6
    assert order_of([R, U]) == 105
    print("ok  (R U R' U') has order 6, (R U) has order 105 (known cube facts)")


def test_bfs_counts():
    expected = [1, 12, 114, 1068, 10011, 93840]
    seen = {key(C.SOLVED)}
    frontier = C.SOLVED[None, :]
    counts = [1]
    for depth in range(1, 6):
        children = C.expand(frontier).reshape(-1, 54)
        new, new_rows = [], []
        for row in children:
            k = key(row)
            if k not in seen:
                seen.add(k)
                new_rows.append(row)
        frontier = np.array(new_rows)
        counts.append(len(frontier))
    assert counts == expected, f"BFS counts {counts} != {expected}"
    print(f"ok  BFS distinct states by depth: {counts} (matches published values)")


def test_batch_functions():
    states, _ = C.random_walk_batch(50, 10, RNG)
    kids = C.expand(states)
    for a in range(12):
        assert np.array_equal(kids[:, a], C.apply_move(states, a))
    acts = RNG.integers(0, 12, 50)
    per_row = C.apply_move_per_row(states, acts)
    for i in range(50):
        assert np.array_equal(per_row[i], C.apply_move(states[i], acts[i]))
    print("ok  batched move functions agree with single-state moves")


def test_scramble_and_undo():
    for n in (1, 5, 20, 50):
        s, moves = C.random_scramble(n, RNG)
        back = C.apply_moves(s, [C.INVERSE_ACTION[a] for a in reversed(moves)])
        assert C.is_solved(back)
    print("ok  scramble followed by reversed inverse moves returns to solved")


def test_invariants():
    for _ in range(3000):
        s, _ = C.random_scramble(int(RNG.integers(1, 60)), RNG)
        assert is_legal(s)
    states, _ = C.random_walk_batch(2000, 30, RNG)
    for s in states:
        assert is_legal(s)
    print("ok  5000 random scrambles all satisfy corner-twist, edge-flip and parity invariants")


def test_negative_controls():
    """The invariant checker must REJECT illegal cubes, otherwise the test above proves nothing."""
    s, _ = C.random_scramble(25, RNG)
    # 1. twist one corner in place (what 'turning a corner' would do)
    bad = s.copy()
    slots = GROUPS[CORNERS[0]]
    bad[slots[0]], bad[slots[1]], bad[slots[2]] = s[slots[1]], s[slots[2]], s[slots[0]]
    assert not is_legal(bad), "twisted corner was not detected"
    # 2. flip one edge in place
    bad = s.copy()
    a, b = GROUPS[EDGES[0]]
    bad[a], bad[b] = s[b], s[a]
    assert not is_legal(bad), "flipped edge was not detected"
    print("ok  negative controls: twisted corner and flipped edge are detected as illegal")


def test_viewer_animation_matches_engine():
    """The 3D window animates a turn by rotating geometry; at the end of the animation every
    sticker must sit in exactly the slot the engine says it moved to (so what you SEE is the state)."""
    import math
    import render as V
    slot_of = {(tuple(p), tuple(n)): i for i, (p, n) in enumerate(zip(C.SLOT_POS, C.SLOT_NORMAL))}
    for a in range(C.N_ACTIONS):
        n = C.FACE_NORMALS[a % 6].astype(float)
        axis = int(np.argmax(np.abs(n)))
        R = V.axis_rotation(n, (-1 if a < 6 else 1) * math.pi / 2)
        in_layer = (V.CUBIE[:, axis] == int(n[axis])) & (V.SLOT >= 0)
        assert in_layer.sum() == 21
        for idx in np.flatnonzero(in_layer):
            new_pos = tuple(np.rint(R @ V.CUBIE[idx]).astype(int))
            new_nrm = tuple(np.rint(R @ V.NORMAL[idx]).astype(int))
            j = slot_of[(new_pos, new_nrm)]
            assert C.PERMS[a][j] == V.SLOT[idx], f"animation of {C.ACTION_NAMES[a]} disagrees with engine"
    print("ok  3D animation geometry matches the engine for all 12 moves")


if __name__ == "__main__":
    test_viewer_animation_matches_engine()
    test_permutations()
    test_orders_and_inverses()
    test_known_orders()
    test_bfs_counts()
    test_batch_functions()
    test_scramble_and_undo()
    test_invariants()
    test_negative_controls()
    print("\nALL TESTS PASSED")
