# Rules and rewards

The game follows classic Snake rules with a few details that matter for training. All of them are defined as constants in `include/snake/engine.hpp`.

## The rules

The snake starts with 3 cells in the middle of a 20 by 15 board, moving right. On every step, it moves one cell forward. It can keep going straight or turn left or right, but never reverse into itself.

When the head reaches the food, the snake grows by one cell and new food appears on a random empty cell. When the head hits a wall or any part of the body, the game ends. That includes the tail. The tail would move away on the same step, but the engine still counts it as body, so every piece of body is fatal.

If the snake fills all 300 cells, the game is won.

## Rewards

Every step returns a reward. These are the values the final agent was trained with:

| Event | Reward | Game ends |
| --- | --- | --- |
| Step into an empty cell | -0.01 | No |
| Eat food | +10 | No |
| Hit a wall or the body | -50 | Yes |
| Fill the whole board | +100 | Yes |

The death penalty was -10 for most of the project. It was raised to -50 for the last part of training, which is described in [Raising the Stakes](../story/08-raising-the-stakes.md).

## The step limit

A snake that keeps moving without eating would never end the game, so there is a limit. If the snake makes 100 moves for every cell of its length without eating, the game is cut off. A snake of length 3 gets 300 moves, a snake of length 100 gets 10,000.

A game that hits this limit is reported as truncated, not as a loss. It gets no penalty, and training treats it as an interrupted game rather than a failure.

## The constants

| Constant | Final value | Meaning |
| --- | --- | --- |
| `STEP_REWARD` | -0.01 | Reward for every move |
| `FOOD_REWARD` | 10.0 | Reward for eating |
| `DEATH_REWARD` | -50.0 | Reward for crashing |
| `DEATH_PER_LENGTH` | 0.0 | Extra crash penalty per cell of length |
| `WIN_REWARD` | 100.0 | Reward for filling the board |
| `MAX_STEPS_PER_LENGTH` | 100 | Moves without food allowed per cell of length |
| `TAIL_IS_SOLID` | true | Whether the tail blocks the head |

Changing any of them requires a [rebuild](building.md). A model trained under one set of rules may play worse under another, so keep the values with the model if you share it.
