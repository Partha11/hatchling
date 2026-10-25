# Diagnostics

`rl/diagnose.py` plays many games with a trained model and reports how they ended. It is the most reliable way to compare two models, because single games vary a lot.

## Run it

```bash
python rl/diagnose.py --weights snake_ppo_child_best.pt --episodes 200
```

The script plays 50 games at a time, so 200 games take about four minutes on a GPU. It reads the model's settings from the weights file.

| Option | Default | Meaning |
| --- | --- | --- |
| `--weights` | | The model to evaluate |
| `--episodes` | 100 | Number of games |
| `--parallel` | 50 | Games played at the same time |
| `--mask-reverse` | off | Blocks the reverse key, fixed view only |
| `--merge-reverse` | off | Treats the reverse key and the forward key as one move, fixed view only |

## Reading the output

This is the diagnostic of the final agent:

```text
Weights: snake_ppo_child_best.pt | View: fixed | Order: relative | Episodes: 200 | Time: 4.3 min | Mean score: 267.11 | Max: 297
Score distribution: min 1 | 10% 232 | 25% 282 | median 289 | 75% 294 | 90% 297 | max 297
Games ending below: <50: 4% | <100: 6% | <150: 8%

Cause       Count    Share   Mean score   Forced   Unforced
wall            1      0%         1.00        0          1
body          157     78%       260.80       13        144
win            42     21%       297.00

Reverse key: pressed 813116 times (30.19% of moves), caused 37 of 158 crashes (23%)

Body crashes by steps until the hit segment would have cleared (157 crashes):
  1 (tail) steps:    8 (5%)
         2 steps:    2 (1%)
       3-5 steps:    4 (3%)
      6-20 steps:    6 (4%)
       21+ steps:  137 (87%)
```

=== "Scores"

    The first lines show the average, the best game and the score distribution. The percentiles tell you more than the average alone. Here, 10 percent of games scored 232 or less, and half scored 289 or more.

=== "Causes"

    Every game ends with a crash into a wall, a crash into the body, a timeout or a win. A crash is forced when all three moves were blocked, so there was nothing the agent could do. Unforced crashes had at least one free move left.

=== "Reverse key"

    Only shown for the fixed view. It counts how often the agent pressed the direction it came from, and how many crashes followed such a press. Compare it with `--merge-reverse` before drawing conclusions, as described in [Raising the Stakes](../story/08-raising-the-stakes.md).

=== "Crash distance"

    For crashes into the body, how many steps the hit cell would have needed to clear. Crashes at 1 to 5 steps are usually timing mistakes near the tail. Crashes at 21 steps or more hit solid body, which on a nearly full board often means the snake had already trapped itself.

## Comparing models

Run the same number of games for every model you compare. 200 games is a good default. With 50 games, differences of 10 to 20 points can still be chance.
