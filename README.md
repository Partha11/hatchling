<div align="center">

  <h1>Snake RL</h1>
  
  <p>
    A fast, lightweight Snake game engine and reinforcement learning (RL) environment written in C++17.
  </p>
  
<!-- Badges -->
<p>
  <a href="https://github.com/Partha11/snake_rl_agent/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/Partha11/snake_rl_agent" alt="contributors" />
  </a>
  <a href="https://github.com/Partha11/snake_rl_agent/commits/main">
    <img src="https://img.shields.io/github/last-commit/Partha11/snake_rl_agent" alt="last update" />
  </a>
  <a href="https://github.com/Partha11/snake_rl_agent/network/members">
    <img src="https://img.shields.io/github/forks/Partha11/snake_rl_agent" alt="forks" />
  </a>
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
    <a href="https://github.com/Partha11/snake_rl_agent/issues/">Report Bug</a>
  <span> · </span>
    <a href="https://github.com/Partha11/snake_rl_agent/issues/">Request Feature</a>
</h4>
</div>

<br />

<!-- Table of Contents -->
# Table of Contents

- [About the Project](#about-the-project)
  * [Screenshots](#screenshots)
  * [Tech Stack](#tech-stack)
  * [Features](#features)
- [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Run Locally](#run-locally)
- [Usage](#usage)
  * [Controls](#controls)
  * [Reinforcement Learning Interface](#reinforcement-learning-interface)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

<!-- About the Project -->
## About the Project

Snake RL is a high-performance, modular Snake simulation environment crafted in modern C++17. The project cleanly decouples simulation logic from frontend rendering, providing both an interactive terminal game and a foundation for training reinforcement learning (RL) agents.

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
  </ul>
</details>

<details>
  <summary>Terminal Frontend</summary>
  <ul>
    <li><a href="https://man7.org/linux/man-pages/man3/termios.3.html">POSIX termios</a> - Raw terminal mode keyboard handling</li>
    <li><a href="https://man7.org/linux/man-pages/man2/poll.2.html">POSIX poll</a> - Non-blocking input polling</li>
    <li>ANSI Escape Codes - In-place terminal drawing and cursor manipulation</li>
  </ul>
</details>

<!-- Features -->
### Features

- **Decoupled Architecture**: Clean separation between `SnakeEngine` and the terminal frontend for headless simulation.
- **3D Spatial Tensor Representation**: Direct extraction of a 3-channel binary state buffer (`head`, `body`, `food`) structured for convolutional neural networks.
- **Relative Action Model**: Egocentric turns (`FORWARD`, `LEFT`, `RIGHT`) to simplify policy training.
- **Interactive Terminal App**: Playable real-time console game with ANSI graphics and responsive non-blocking controls.
- **Deterministic Step API**: Standard RL step interface returning reward, terminal flag (`done`), and score.

<!-- Getting Started -->
## Getting Started

<!-- Prerequisites -->
### Prerequisites

Ensure you have the following installed on your system:

- C++17 compliant compiler (GCC 9+, Clang 10+)
- CMake 3.16 or newer
- Make or Ninja
- Linux or POSIX-compatible environment (for terminal raw mode and polling)

<!-- Run Locally -->
### Run Locally

1. Clone the repository

```bash
git clone https://github.com/Partha11/snake_rl_agent.git
```

2. Navigate to the project directory

```bash
cd snake_rl_agent
```

3. Configure the build directory

```bash
cmake -S . -B build
```

4. Build the executable

```bash
cmake --build build
```

5. Launch the game

```bash
./build/snake
```

> **Note**: Run the binary in a standard terminal emulator rather than an embedded IDE output panel.

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

- The snake cannot make a 180-degree turn directly into itself; reverse inputs continue forward.
- The grid is rendered with `@` for the head, `o` for the body, and `*` for food.
- Default configuration uses a 20 &times; 15 board with a 100 ms tick interval (configured in `src/app/main.cpp`).

### Reinforcement Learning Interface

#### Action Space

The engine operates on a discrete 3-action relative action space:

| Action | Enum | Description |
| --- | --- | --- |
| `0` | `Action::FORWARD` | Maintain current movement direction |
| `1` | `Action::LEFT` | Turn 90 degrees counter-clockwise and move |
| `2` | `Action::RIGHT` | Turn 90 degrees clockwise and move |

#### Observation Space

`SnakeEngine::getBoardBuffer()` returns a flat `std::vector<uint8_t>` representing a 3-channel binary tensor of shape **(3, Height, Width)**:

- **Channel 0**: Head position (`1` at head coordinates, `0` elsewhere)
- **Channel 1**: Body positions (`1` on occupied segments, `0` elsewhere)
- **Channel 2**: Food position (`1` on food coordinates, `0` elsewhere)

The linear index for cell `(x, y)` in channel `c` is computed as:
$$\text{index} = c \times (\text{width} \times \text{height}) + y \times \text{width} + x$$

#### Reward Function

Every call to `SnakeEngine::step(action)` returns a `StepResult`:

| Event | Reward | Terminal (`done`) |
| --- | :---: | :---: |
| Normal step | `0.0` | `false` |
| Food consumed | `+10.0` | `false` |
| Wall collision | `-10.0` | `true` |
| Self collision | `-10.0` | `true` |
| Board completed | `+100.0` | `true` |

<!-- Project Structure -->
## Project Structure

```text
.
├── CMakeLists.txt
├── include/
│   └── snake/
│       └── engine.hpp       # Engine class, enums (Action, Direction), and StepResult
└── src/
    ├── app/
    │   └── main.cpp         # Terminal controller, ANSI renderer, and game loop
    └── engine/
        └── engine.cpp       # Movement logic, food spawning, collision, and state buffer
```

<!-- Roadmap -->
## Roadmap

- [x] High-performance C++17 simulation engine
- [x] Multi-channel spatial observation buffer
- [x] Playable interactive ANSI terminal application
- [ ] Python bindings via `pybind11` for zero-copy NumPy buffers
- [ ] Gymnasium (OpenAI Gym) compatible environment wrapper
- [ ] Deep Q-Network (DQN) / PPO reinforcement learning training baseline
- [ ] Automated test suite via CTest

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

<!-- Contact -->
## Contact

Partha - abirpartha7@gmail.com

Project Link: [https://github.com/Partha11/snake_rl_agent](https://github.com/Partha11/snake_rl_agent)

<!-- Acknowledgments -->
## Acknowledgements

- [Awesome README Template](https://github.com/Louis3797/awesome-readme-template)
- [Shields.io](https://shields.io/)
