"""
evaluate.py - measure how good your trained agent is.

    python evaluate.py                           # newest training run
    python evaluate.py models/quick_.../final.npz

For each scramble length it makes 30 fresh scrambles, lets the agent search with several beam widths,
re-plays every solution on the cube to confirm it, and saves the table as results.md in the run's folder.
"""

import os
import sys
import time

import numpy as np

import cube as C
from agent import QNet, beam_solve, find_latest_run

LENGTHS = [4, 6, 8, 10, 12, 14, 20]
WIDTHS = [1, 100, 1000]
N = 30

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        run = find_latest_run(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
        if run is None or not os.path.exists(os.path.join(run, "final.npz")):
            raise SystemExit("No finished training run found. Train first: python train.py")
        path = os.path.join(run, "final.npz")
    net = QNet.load(path)
    print("Benchmarking", path)
    rng = np.random.default_rng(777)
    lines = [f"# Benchmark of {path} ({int(net.meta.get('iteration', 0))} training iterations)", "",
             f"{N} fresh scrambles per row. Width 1 is close to the greedy policy with no search.", "",
             "| Scramble length | Search width | Solved | Average solution length | Seconds per cube |", "|---|---|---|---|---|"]
    for length in LENGTHS:
        scrambles = [C.random_scramble(length, rng)[0] for _ in range(N)]
        for width in WIDTHS:
            start, solved = time.time(), []
            for cube_state in scrambles:
                moves = beam_solve(net, cube_state, width=width, max_depth=40)
                if moves is not None and C.is_solved(C.apply_moves(cube_state, moves)):     # verify the solution
                    solved.append(len(moves))
            average = f"{np.mean(solved):.1f}" if solved else "-"
            lines.append(f"| {length} | {width} | {100 * len(solved) / N:.0f}% | {average} | {(time.time() - start) / N:.2f} |")
            print(lines[-1], flush=True)
    out = os.path.join(os.path.dirname(path), "results.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out)
