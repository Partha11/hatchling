# The Beginning

Before an agent could learn anything, the game had to exist. I wrote it in C++17. The rules of Snake are small, but an agent needs millions of games to learn, and every microsecond per move adds up over that many games.

## The engine

The most important decision was to keep the engine separate from everything else. The engine knows the rules: where the snake is, where the food is and what happens on each move. It knows nothing about screens, keyboards or neural networks. That separation paid off again and again, because the same engine runs both the terminal game and the training environment, and nothing had to be written twice.

## A game you can play

On top of the engine sits a small terminal game for Linux and Windows. You steer with the arrow keys or WASD on a board 20 cells wide and 15 cells tall.

```text
+----------------------------------------+
|                                        |
|              *                         |
|                                        |
|                    o o o @             |
|                                        |
+----------------------------------------+
Score: 0
```

The head is `@`, the body is `o` and the food is `*`. That is the whole interface, and it is also exactly the picture the agent would one day learn from.

## The bridge to Python

Neural networks live in Python, so the engine needed a way in. I used pybind11 to expose it as a Python module called `snake_cpp`, wrapped in a standard Gymnasium environment.

Every move returns four things: the reward, whether the game is over, whether the step limit cut it off, and the score. The board comes out as a stack of grids, one per kind of thing on it. One grid marks the head, others mark the body, the food and so on. A neural network reads that directly, the way it reads the color channels of an image. The board is shared with Python without copying, so reading it costs almost nothing.

## Tools for later

Two features did not matter on day one but became important later. The engine can start a game with a snake of any length, laid out randomly on the board, which makes it possible to practice crowded boards without playing a whole game first. Later, it also learned to save and restore its complete state. That one became the key to replaying the agent's own mistakes in [Learning from Mistakes](06-learning-from-mistakes.md).
