# Engine

The engine is the heart of the project. It holds the complete state of a game and applies the rules, and it knows nothing about screens or neural networks. The terminal game and the Python environment are both thin layers on top of it.

## Moves and steps

The engine thinks in relative moves: forward, left and right. Each call to `step` moves the snake once and returns a result with four fields.

| Field | Meaning |
| --- | --- |
| `reward` | The reward for this move, see [Rules and rewards](rules.md) |
| `done` | The game ended with a crash or a win |
| `truncated` | The game was cut off by the step limit |
| `score` | The number of food items eaten so far |

## The board buffer

The engine keeps the board as a stack of five grids of size 15 by 20. Each cell holds a value from 0 to 255.

| Channel | Content |
| --- | --- |
| 0 | Head position |
| 1 | Neck position, the body cell right behind the head |
| 2 | All body cells |
| 3 | Body order, how soon each body cell will clear |
| 4 | Food position |

The body order channel can be encoded in two ways. In the default relative encoding, the value of a cell is proportional to the share of the snake that still has to pass before it clears. In the inverse encoding, cells near the tail get the highest values. The final agent does not use this channel at all.

The buffer is exposed to Python without copying, so reading it on every step costs almost nothing.

## Useful extras

Besides playing normally, the engine supports a few things that training relies on.

| Feature | What it does |
| --- | --- |
| Reset with a length | Starts a game with a snake of any length, laid out randomly on the board |
| Danger check | Reports which of the three moves would end the game right away |
| Tail segments | Returns the last cells of the snake, starting from the tail |
| Clone and restore | Copies the complete game state and loads it back later |

Clone and restore are what make [failure replay](../story/06-learning-from-mistakes.md) possible. A training game can save its state every few moves, and any saved state can later start a new game.

## The Python environment

`rl/env.py` wraps the engine in a standard Gymnasium environment called `SnakeEnv`. It turns the board buffer into what the network sees, and it supports two views.

=== "Fixed view"

    The view the final agent uses. The network sees the whole board with a one cell wall around it, so the input is 17 by 22 cells. It has six channels: head, neck, body, tail, food and wall. The agent chooses between four absolute moves: up, right, down and left. Pressing the direction the snake came from keeps it going straight.

=== "Snake centered view"

    The view of the earlier agents. The board is rotated and shifted so that the head is always in the center, facing up, in a 39 by 39 window. It has four channels: body, body order, food and wall. Optional extra channels mark the head or the last 64 cells of the tail. The agent chooses between three relative moves: forward, left and right.

The environment also exposes the engine's extras to the training code. `reset` accepts a starting length, `danger` returns the danger check, and `snapshot` and `restore` save and load complete game states.
