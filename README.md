# Snake RL

A fast, lightweight Snake game and reinforcement learning (RL) environment written in C++17. The project cleanly separates core game mechanics and state representation from the interactive terminal interface, serving as a high-performance foundation for training RL agents.

---

## Features

- **Decoupled Architecture**: Pure C++ engine with zero GUI/terminal dependencies, ideal for headless simulation and batch rollout.
- **RL-Ready State Tensor**: Built-in 3-plane spatial state buffer (`head`, `body`, `food`) representing the board for convolutional neural networks.
- **Relative Action Space**: Supports egocentric relative turns (`FORWARD`, `LEFT`, `RIGHT`) to simplify policy learning.
- **Interactive Terminal App**: Real-time playable console app featuring ANSI escape rendering and non-blocking POSIX keyboard polling.

---

## Requirements

- **C++ Compiler**: C++17 compliant compiler (GCC 9+, Clang 10+)
- **Build System**: CMake 3.16+ and Make or Ninja
- **Operating System**: Linux or POSIX-compliant system (uses POSIX `termios` and `poll` for terminal input)

---

## Quick Start

### Build

```sh
cmake -S . -B build
cmake --build build
```

### Run Interactive Terminal Game

```sh
./build/snake
```

> **Note**: Run the binary directly inside a standard terminal emulator with ANSI escape sequence support rather than an embedded IDE output pane.

---

## Controls

| Key | Action |
| :--- | :--- |
| <kbd>W</kbd> or <kbd>&uarr;</kbd> | Move Up |
| <kbd>S</kbd> or <kbd>&darr;</kbd> | Move Down |
| <kbd>A</kbd> or <kbd>&larr;</kbd> | Move Left |
| <kbd>D</kbd> or <kbd>&rarr;</kbd> | Move Right |
| <kbd>Q</kbd> or <kbd>Esc</kbd> | Quit Game |

- The game runs at a default tick interval of 100 ms on a 20 &times; 15 board (configurable in [`src/app/main.cpp`](src/app/main.cpp)).
- The snake cannot reverse 180 degrees into itself; opposite-direction inputs are treated as moving forward.
- The board displays `@` for the snake head, `o` for body segments, and `*` for food.

---

## Reinforcement Learning Interface

### Action Space

The engine uses a discrete 3-action relative action space:

| Action | Enum | Description |
| :--- | :--- | :--- |
| `0` | `Action::FORWARD` | Continue moving in the current heading |
| `1` | `Action::LEFT` | Turn 90&deg; counter-clockwise, then move |
| `2` | `Action::RIGHT` | Turn 90&deg; clockwise, then move |

### Observation Representation

`SnakeEngine::getBoardBuffer()` returns a flattened `std::vector<uint8_t>` representing a 3-channel binary spatial tensor of dimensions **(3, Height, Width)**:

- **Channel 0**: Snake head position ($1$ where head is located, $0$ elsewhere)
- **Channel 1**: Snake body segments ($1$ on body segments, $0$ elsewhere)
- **Channel 2**: Food position ($1$ on food cell, $0$ elsewhere)

Index offset for coordinate `(x, y)` in channel `c`:
$$\text{index} = c \times (\text{width} \times \text{height}) + y \times \text{width} + x$$

### Reward Structure

Each invocation of `SnakeEngine::step(action)` returns a `StepResult` struct containing:
- `reward` (`float`)
- `done` (`bool`)
- `score` (`int`)

| Event | Reward | Terminal (`done`) |
| :--- | :---: | :---: |
| Normal step | `0.0` | `false` |
| Food collected | `+10.0` | `false` |
| Wall collision | `-10.0` | `true` |
| Self collision | `-10.0` | `true` |
| Board completed | `+100.0` | `true` |

---

## Project Structure

```text
.
├── CMakeLists.txt
├── include/
│   └── snake/
│       └── engine.hpp       # SnakeEngine class, enums, structs (Point, StepResult)
└── src/
    ├── app/
    │   └── main.cpp         # Terminal controller, ANSI renderer, game loop
    └── engine/
        └── engine.cpp       # Engine logic, state transitions, buffer updates
```

---

## Roadmap

- [ ] Python bindings via `pybind11` for direct NumPy tensor export.
- [ ] Gymnasium (OpenAI Gym) compatible Python environment wrapper.
- [ ] Deep Q-Network (DQN) / PPO reinforcement learning training pipeline.
- [ ] Automated unit test suite with CTest.
