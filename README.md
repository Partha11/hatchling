# Snake RL

A small Snake game written in C++17. The game rules live in a separate engine, while the terminal app handles input and drawing.

The engine returns a reward after each move. It also exposes the board as a byte buffer. This gives the project a starting point for future RL work. There is no training code or AI player yet.

## Current state

The project builds, but the terminal app still needs work. It currently exits after one game update because `return 0` is inside the game loop. The top and bottom borders are also shorter than the drawn rows.

## Requirements

- A C++17 compiler, such as GCC or Clang
- CMake 3.16 or newer
- Make or Ninja
- A terminal with ANSI escape support

The terminal app uses POSIX calls for keyboard input. Use Linux or another system with these calls. Windows is not supported by the current app.

## Build and run

From the project folder:

```sh
cmake -S . -B build
cmake --build build
./build/snake
```

Run the app in a terminal, not an IDE output panel.

## Controls

| Key | Action |
| --- | --- |
| W or Up arrow | Move up |
| S or Down arrow | Move down |
| A or Left arrow | Move left |
| D or Right arrow | Move right |
| Q or Escape | Quit |

The snake cannot turn straight back into itself. An opposite-direction input keeps it moving forward.

The board uses `@` for the head, `o` for the body, and `*` for food. Eating food adds one point. Hitting a wall or the body ends the game.

The app uses a 20 by 15 board and a 100 ms tick interval. These values are set near the start of `main()` in `src/app/main.cpp`.

## Project layout

```text
.
├── CMakeLists.txt
├── include/
│   └── snake/
│       └── engine.hpp
└── src/
    ├── app/
    │   └── main.cpp
    └── engine/
        └── engine.cpp
```

- `engine.hpp` defines the engine, actions, points, and step results.
- `engine.cpp` handles movement, food, collisions, and board state.
- `main.cpp` reads keys and draws the game in the terminal.

## Engine API

Create a `SnakeEngine` with a board width and height. Call `reset()` to start a new game. Each call to `step()` moves the snake once.

The engine accepts three actions:

- `Action::FORWARD`: keep the current direction
- `Action::LEFT`: turn left, then move
- `Action::RIGHT`: turn right, then move

Each step returns a `StepResult` with `reward`, `done`, and `score`.

| Result | Reward |
| --- | --- |
| Normal move | 0 |
| Eat food | 10 |
| Hit a wall or the body | -10 |
| Fill the board | 100 |

`getBoardBuffer()` returns three flat layers: head, body, and food. Each layer has `width * height` entries. A cell is `1` when occupied and `0` otherwise. The cell index within a layer is `y * width + x`.

## Build check

To check compilation after a change:

```sh
cmake --build build
```

There are no automated tests yet. A successful build only checks that the code compiles and links.
