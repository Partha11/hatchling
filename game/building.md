# Building

Building from source gives you the terminal game and the Python module the agent trains with. Both come from the same CMake project.

## Requirements

| Tool | Version |
| --- | --- |
| C++ compiler | GCC 9+, Clang 10+ or MSVC 2019+ |
| CMake | 3.16 or newer |
| Build tool | Make, Ninja or Visual Studio |
| Python | 3.10 or newer, only for the agent |

## Build the game

Clone the repository and build in release mode:

```bash
git clone https://github.com/Partha11/snake_rl_agent.git
cd snake_rl_agent
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Then start the game:

=== "Linux and macOS"

        :::bash
        ./build/snake

=== "Windows"

        :::powershell
        .\build\Release\snake.exe

## Build the Python module

The agent talks to the engine through a Python module called `snake_cpp`, built with pybind11. Build it with its own target:

```bash
cmake --build build --target snake_cpp
```

Build it inside the Python environment you will use for training, so that CMake finds the right Python version. The [AI setup](../agent/setup.md) walks through the full order of steps.

!!! warning "Rebuild after changing the rules"
    The game rules and rewards are constants in `include/snake/engine.hpp`. They are compiled into both the game and the Python module, so any change there needs a rebuild before it takes effect.

## Build targets

| Target | What it builds |
| --- | --- |
| `snake` | The terminal game |
| `snake_cpp` | The Python module for the agent |
