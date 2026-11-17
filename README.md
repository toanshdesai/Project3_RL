# Project03_RL: a reinforcement-learning agent that learns to solve the Rubik's Cube

ADV CS III: Machine Learning and AI, Project 03 (Pygame RL project)

A 3×3×3 Rubik's Cube with a scramble button, a 3D Pygame view, and an agent that learns to solve it from one rule:
every move costs 1, and the episode ends when the cube is solved. No trained agent is included. You train it yourself
and watch it learn.

## Run it

```bash
pip install -r requirements.txt
python test_cube.py     # proves the cube only makes legal moves
python train.py         # train: choose 5, 20 or 60 minutes
python viewer.py        # scramble and solve with the agent you trained
python evaluate.py      # benchmark it; writes results.md into your run's folder
```

Each training run is saved in `models/<preset>_<date>_<time>/` with six snapshots, from untrained to final.

## What the training window shows

- **The training loop** (top right): the four steps, with the one running now in bold.
- **A practice cube and its 12 Q-values** (left): each bar is how many moves the network expects to need after that move.
  The dark bar is its choice. The red lines are the Bellman targets the bars are being pulled toward.
- **The learning curve** (bottom right): the share of fixed test scrambles solved by the greedy policy, with no search.
- **Test agent**: a copy of the current network tries a fresh scramble. Early in training it fails; a minute later it solves it.

Keys: `Space` pause, `T` test, `↑` `↓` test length, `S` stop and save.

## How it learns

| RL concept | In this project                                                         |
|------------|-------------------------------------------------------------------------|
| State      | Colours of the 54 stickers, one-hot encoded (324 inputs)                |
| Actions    | 12 quarter turns: `U D F B L R` and their reverses                      |
| Reward     | −1 per move; the episode ends when solved                               |
| Q(s, a)    | Moves still needed if the agent plays `a` now and plays well afterwards |
| Network    | 324 → 1024 → 512 → 12, written from scratch in NumPy                    |
| Policy     | Pick the move with the lowest Q                                         |

The cube has about 4.3 × 10¹⁹ states and one goal, so random play never reaches a reward. The agent therefore learns
backwards from the goal (Autodidactic Iteration). Each iteration:

1. Scramble a solved cube with 1 to 20 random turns, 2,048 times. Every practice cube is near the goal.
2. For each cube and each move, build the Bellman target: `1 + moves-to-go of the resulting cube`, scored by a frozen
   copy of the network (the target network).
3. Take gradient steps that move the network's Q-values toward those targets.
4. Every 10 iterations, refresh the target network.

Correct values start at the solved cube and spread outward one step per refresh. The viewer then solves with a beam
search that uses the learned Q-values to decide which move sequences to keep.

`PROJECT_EXPLAINED.md` maps this onto the class Bellman equation, with a worked example and likely defense questions.

## Files

| File           | What it does                                                                                              |
|----------------|-----------------------------------------------------------------------------------------------------------|
| `cube.py`      | The cube: 54 stickers, 12 legal moves computed from 3D geometry                                           |
| `agent.py`     | The Q-network (NumPy, Adam), greedy play, beam search                                                     |
| `train.py`     | The training loop and the training window                                                                 |
| `viewer.py`    | The solver window                                                                                         |
| `render.py`    | 3D cube drawing and the shared plain UI                                                                   |
| `evaluate.py`  | Benchmark by scramble length                                                                              |
| `test_cube.py` | Tests: move tables match published state counts (1, 12, 114, 1068, 10011, 93840) and real cube invariants |

## Results

Your own numbers come from `python evaluate.py` (`results.md`) and from `training_log.csv` in your run's folder.

For comparison, runs made during development on a 2-core cloud CPU:

| Run                         | Greedy, no search                           | With beam search (30 scrambles per length)                                              |
|-----------------------------|---------------------------------------------|-----------------------------------------------------------------------------------------|
| Quick, 5 min                | 99% of 4-move, 77% of 6-move, 33% of 8-move | 100% of 6-, 8- and 10-move at width 1,000                                               |
| Full settings, about 1 hour | 99% of 4-move, 89% of 6-move, 59% of 8-move | 93% of 12-move and 53% of 20-move at width 1,000; 100% of 8- to 20-move at width 10,000 |

Tuning done during development (one run each, so small differences are noise): a bigger network and four passes per
batch instead of two both helped; refreshing the target every 10 vs. 20 iterations and practising up to 14 vs. 20 turns
made little difference; Double DQN did not help and was dropped. To run your own experiment, change one constant at the
top of `train.py` (`SYNC`, `EPOCHS`, or a preset's layer sizes), train the quick preset again and compare.

## Limits

- The agent can ask the simulator what any move does, during training and search. It learns values, not the cube's
  rules, so it is model-based, unlike the trial-and-error Q-learning in Snake tutorials.
- Deep scrambles need the search; the greedy policy alone solves few 10-move scrambles.
- Solutions to deep scrambles are longer than optimal.
- 30 test cubes per row: 30 of 30 means "probably at least 90%", not "always".

## Resources

- McAleer et al., *Solving the Rubik's Cube Without Human Knowledge* (2018): Autodidactic Iteration.
- Agostinelli et al., *Solving the Rubik's Cube with Deep Reinforcement Learning and Search* (2019): DeepCubeA.
- Mnih et al., *Human-level control through deep reinforcement learning* (2015): DQN and target networks.
- Class resources on Q-learning, the Bellman equation and DQN, listed in `PROJECT_EXPLAINED.md`.

## AI assistance disclosure (Honor Code)

Claude code was used
