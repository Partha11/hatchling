<div align="center">

  <h1>Snake RL</h1>
  
  <p>
    A fast, lightweight Snake game engine and reinforcement learning (RL) environment written in C++17.
  </p>
  
<!-- Badges -->
<p>
  <a href="https://github.com/Partha11/snake_rl_agent/stargazers">
    <img src="https://img.shields.io/github/stars/Partha11/snake_rl_agent" alt="stars" />
  </a>
  <a href="https://github.com/Partha11/snake_rl_agent/issues/">
    <img src="https://img.shields.io/github/issues/Partha11/snake_rl_agent" alt="open issues" />
  </a>
  <a href="https://isocpp.org/">
    <img src="https://img.shields.io/badge/C%2B%2B-17-00599C.svg?logo=c%2B%2B&logoColor=white" alt="C++17" />
  </a>
  <a href="https://cmake.org/">
    <img src="https://img.shields.io/badge/CMake-3.16+-064F8C.svg?logo=cmake&logoColor=white" alt="CMake" />
  </a>
</p>
   
<h4>
    <a href="https://github.com/Partha11/snake_rl_agent">Repository</a>
  <span> · </span>
    <a href="https://github.com/Partha11/snake_rl_agent#readme">Documentation</a>
  <span> · </span>
    <a href="https://github.com/Partha11/snake_rl_agent/releases">Download Binaries</a>
  <span> · </span>
    <a href="https://github.com/Partha11/snake_rl_agent/issues/">Report Bug</a>
</h4>
</div>

<br />

<!-- Table of Contents -->
# Table of Contents

- [Table of Contents](#table-of-contents)
  - [About the Project](#about-the-project)
    - [Screenshots](#screenshots)
    - [Tech Stack](#tech-stack)
    - [Features](#features)
  - [Getting Started](#getting-started)
    - [Download Pre-built Binaries](#download-pre-built-binaries)
    - [Prerequisites](#prerequisites)
    - [Run Locally](#run-locally)
    - [Python RL Environment Setup](#python-rl-environment-setup)
  - [Usage](#usage)
    - [Controls](#controls)
    - [Reinforcement Learning Interface](#reinforcement-learning-interface)
      - [Action Space](#action-space)
      - [Observation Space](#observation-space)
      - [Reward Function & Episode Termination](#reward-function--episode-termination)
      - [Curriculum & Length-Conditioned Reset](#curriculum--length-conditioned-reset)
    - [Training & Evaluation](#training--evaluation)
  - [Project Structure](#project-structure)
  - [Roadmap](#roadmap)
  - [Contributing](#contributing)
  - [License](#license)

<!-- About the Project -->
## About the Project

Snake RL is a high-performance, modular Snake simulation environment crafted in modern C++17. The project cleanly decouples simulation logic from frontend rendering, providing both a standalone cross-platform terminal game (Windows & Linux) and a zero-copy Python environment for training reinforcement learning (RL) agents.

<!-- Screenshots -->
### Screenshots

<!--
<div align="center"> 
  <img src="https://placehold.co/600x400?text=Snake+RL+Gameplay" alt="screenshot" />
</div>
-->

<!-- TechStack -->
### Tech Stack

<details>
  <summary>Core & Engine</summary>
  <ul>
    <li><a href="https://isocpp.org/">C++17</a> - Core simulation engine and data structures</li>
    <li><a href="https://cmake.org/">CMake (3.16+)</a> - Cross-platform build automation</li>
    <li><a href="https://github.com/pybind/pybind11">pybind11</a> - Seamless C++ to Python bindings with zero-copy NumPy buffers</li>
  </ul>
</details>

<details>
  <summary>Terminal Frontend</summary>
  <ul>
    <li><b>Linux / POSIX</b>: <code>termios</code> (raw mode) & <code>poll(2)</code> (non-blocking key polling)</li>
    <li><b>Windows</b>: Win32 Console API (<code>ENABLE_VIRTUAL_TERMINAL_PROCESSING</code>) & <code>conio.h</code> (<code>_kbhit</code>, <code>_getch</code>)</li>
    <li>ANSI Escape Sequences - Real-time terminal grid rendering and cursor management</li>
  </ul>
</details>

<details>
  <summary>Reinforcement Learning</summary>
  <ul>
    <li><a href="https://pytorch.org/">PyTorch</a> - Deep reinforcement learning models (ResNet-based Actor-Critic & Dueling DQN)</li>
    <li><a href="https://gymnasium.farama.org/">Gymnasium</a> - Standardized environment API</li>
    <li><a href="https://numpy.org/">NumPy</a> - Array manipulation and vectorized state transformations</li>
  </ul>
</details>

<!-- Features -->
### Features

- **Decoupled Architecture**: Clean separation between `SnakeEngine` and frontends, enabling headless execution and rapid vectorized rollouts.
- **Cross-Platform Standalone Game**: Native terminal game with non-blocking controls running out-of-the-box on both Linux and Windows.
- **5-Channel Spatial State Tensor**: Head, neck, body, body ordering (gradient decay), and food channels structured for convolutional neural networks.
- **Egocentric Observation Wrapper**: Optional rotated and centered observation view in Python aligning state representations with the snake's heading.
- **Curriculum Training Support**: Engine-level `resetWithLength()` using random walk or serpentine path generation for training across snake lengths.
- **Vectorized PPO & GAE**: Full Proximal Policy Optimization implementation with Generalized Advantage Estimation, residual convolutional trunk, and diagnostics.

<!-- Getting Started -->
## Getting Started

### Download Pre-built Binaries

Standalone executables are built and published for each release:

1. Go to the [Releases](https://github.com/Partha11/snake_rl_agent/releases) page.
2. Download the binary for your system:
   - **Linux**: `snake-linux-x86_64` (statically linked C++ runtime; run `chmod +x snake-linux-x86_64 && ./snake-linux-x86_64`)
   - **Windows**: `snake-windows-x86_64.exe` (run directly from Command Prompt or PowerShell)

---

### Prerequisites

For compiling from source:

- **C++ Compiler**: GCC 9+, Clang 10+, or MSVC 2019+ (C++17 support)
- **Build System**: CMake 3.16 or newer, along with Make, Ninja, or Visual Studio
- **Python (Optional, for RL)**: Python 3.10+ with `pip`

---

### Run Locally

1. Clone the repository:

```bash
git clone https://github.com/Partha11/snake_rl_agent.git
cd snake_rl_agent
```

2. Configure and build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

3. Launch the terminal game:

```bash
# On Linux / macOS
./build/snake

# On Windows
./build/Release/snake.exe
```

---

### Python RL Environment Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
```

2. Install dependencies using `requirements.in` and `pip-tools`:

```bash
pip install pip-tools
pip-compile requirements.in
pip install -r requirements.txt
```

3. Ensure the C++ Python module is compiled:

```bash
cmake --build build --target snake_cpp
```

<!-- Usage -->
## Usage

### Controls

| Key | Action |
| --- | --- |
| `W` or `Up Arrow` | Move Up |
| `S` or `Down Arrow` | Move Down |
| `A` or `Left Arrow` | Move Left |
| `D` or `Right Arrow` | Move Right |
| `Q` or `Escape` | Quit |

- The snake cannot reverse 180 degrees directly into itself; opposite inputs continue forward.
- The terminal grid displays `@` for the head, `o` for body segments, and `*` for food.
- Default configuration uses a 20 &times; 15 board with a 100 ms tick interval (configurable in `src/app/main.cpp`).

---

### Reinforcement Learning Interface

#### Action Space

The engine operates on a discrete 3-action relative action space:

| Action | Enum | Description |
| --- | --- | --- |
| `0` | `Action::FORWARD` | Maintain current movement direction |
| `1` | `Action::LEFT` | Turn 90 degrees counter-clockwise and move |
| `2` | `Action::RIGHT` | Turn 90 degrees clockwise and move |

#### Observation Space

`SnakeEngine::getBoardBuffer()` returns a flat buffer mapped to a 5-channel binary tensor of shape **(5, Height, Width)**:

- **Channel 0**: Head position (`255` at head coordinate, `0` elsewhere)
- **Channel 1**: Neck position (`255` at neck coordinate, `0` elsewhere)
- **Channel 2**: Body segments (`255` on occupied cells, `0` elsewhere)
- **Channel 3**: Body ordering gradient (`(255 * (length - i)) / length` from head to tail)
- **Channel 4**: Food position (`255` at food coordinate, `0` elsewhere)

The `SnakeEnv` wrapper (`rl/env.py`) converts these channels into a centered, egocentrically rotated 4-channel view of size `(4, view_size, view_size)` oriented in the direction of the snake's movement.

#### Reward Function & Episode Termination

Every step returns a `StepResult` with `reward`, `done` (terminated), `truncated`, and `score`:

| Event | Reward | `done` | `truncated` |
| --- | :---: | :---: | :---: |
| Move closer to food | `+0.1` | `false` | `false` |
| Move further from food | `-0.15` | `false` | `false` |
| Food consumed | `+10.0` | `false` | `false` |
| Wall collision | `-10.0` | `true` | `false` |
| Self collision | `-10.0` | `true` | `false` |
| Board completed | `+100.0` | `true` | `false` |
| Starvation (step limit exceeded) | `-1.0` | `false` | `true` |

#### Curriculum & Length-Conditioned Reset

`SnakeEngine::resetWithLength(length)` initializes the snake at an arbitrary length $L \ge 3$ using randomized walk generation (with a serpentine fallback), providing a curriculum mechanism to train policies on crowded end-game boards.

---

### Training & Evaluation

```bash
# Train using Proximal Policy Optimization (PPO)
python rl/train.py

# Evaluate and diagnose death causes (forced traps vs unforced mistakes)
python rl/diagnose.py --algo ppo --weights snake_ppo_best.pt --episodes 100

# Watch trained agent play in the terminal
python rl/play.py
```

<!-- Project Structure -->
## Project Structure

```text
.
├── .github/
│   └── workflows/
│       └── release.yml      # CI/CD workflow building and publishing Windows & Linux releases
├── CMakeLists.txt           # Build definitions for standalone game and Python module
├── requirements.in          # Abstract Python dependencies for pip-compile
├── include/
│   └── snake/
│       └── engine.hpp       # Pure C++ engine definitions, structs, and enums
├── src/
    ├── app/
    │   └── main.cpp         # Cross-platform interactive terminal game
    ├── bindings/
    │   └── bindings.cpp     # pybind11 module exporting snake_cpp
    └── engine/
        └── engine.cpp       # Core movement, collision, food spawn, and tensor generation
└── rl/
    ├── env.py               # Gymnasium environment with egocentric observation transformation
    ├── model.py             # Residual Actor-Critic & Dueling CNN network architectures
    ├── train.py             # Vectorized PPO training pipeline with GAE
    ├── play.py              # Visual terminal playback using trained weights
    └── diagnose.py          # Empirical death-cause analysis (forced vs unforced collisions)
```

<!-- Roadmap -->
## Roadmap

- [x] High-performance C++17 simulation engine
- [x] Multi-channel spatial observation buffer (5-channel tensor)
- [x] Cross-platform standalone terminal game (Linux & Windows)
- [x] Python bindings via `pybind11` for zero-copy NumPy buffers
- [x] Gymnasium environment wrapper with egocentric rotation
- [x] ResNet-based Actor-Critic and Dueling DQN architectures
- [x] Vectorized PPO training pipeline with GAE
- [x] Automated CI/CD release workflow for Linux and Windows binaries
- [ ] Automated unit test suite via CTest

<!-- Contributing -->
## Contributing

Contributions are always welcome!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- License -->
## License

Distributed under the MIT License. See `LICENSE` for more information.
