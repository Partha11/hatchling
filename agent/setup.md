# Setup

This page sets up everything you need to watch the trained agent play, train your own agent or run the diagnostics. It takes a few minutes, plus the time to download PyTorch.

## Requirements

| Requirement | Notes |
| --- | --- |
| Python | 3.10 or newer |
| A C++ toolchain and CMake | To build the `snake_cpp` module, see [Building](../game/building.md) |
| An NVIDIA GPU | Recommended for training, not needed to watch the agent play |

A typical training run used about 2.7 GB of GPU memory and 1.2 GB of system memory. Watching it play or running the diagnostics also works on a CPU, just more slowly.

## Install

Create a virtual environment and install the dependencies with pip-tools:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pip-tools
pip-compile requirements.in
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

## PyTorch and the GPU

The command above installs the default PyTorch build from PyPI. If you want to train on a GPU, make sure the installed build matches your CUDA driver. The [PyTorch installation selector](https://pytorch.org/get-started/locally/) gives you the exact command for your system. Run it after the steps above to replace the default build.

You can check that PyTorch sees your GPU with:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If it prints `False`, training still works, but it runs on the CPU and takes much longer.

## Build the engine module

With the environment active, build the Python module:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target snake_cpp
```

Building it inside the activated environment makes sure CMake picks up the right Python version.

## Watch the agent play

Download the trained weights, `snake_ppo_child_best.pt`, from the [releases](https://github.com/Partha11/hatchling/releases) page and put them in the repository folder. Then start a game:

```bash
python rl/play.py --weights snake_ppo_child_best.pt
```

`--delay` sets the pause between moves in seconds, so `--delay 0.01` plays fast and `--delay 0.1` lets you follow every move. The script reads the model's settings, such as its view, from the weights file, so no other options are needed.

When a game ends, `play.py` explains the last move:

```text
Game Over (collision)! Final Score: 289
Moves: 16220 | Snake Length: 292 | Weights: snake_ppo_child_best.pt | View: fixed | Order: relative
Last move options (snake length 292): Forward: body (265 steps from clearing) (chosen) | Left: 1 free cell | Right: 2 free cells
Network input around head: Forward: body | Left: empty | Right: food
Policy before last move: Up: 0.042 | Right: 0.903 | Down: 0.000 | Left: 0.055 | Value estimate: -17.48
```

| Line | Meaning |
| --- | --- |
| Last move options | What each move would have led to: free cells reachable, a wall, or body and how many steps until it clears |
| Network input around head | What the agent saw in the three cells around its head |
| Policy before last move | How likely the agent was to press each key |
| Value estimate | How good the agent thought its position was |

If the snake fills the board, `play.py` announces the win instead and shows the agent's last decision.
