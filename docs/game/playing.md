# Playing

You can play Hatchling yourself in a terminal. No setup is needed if you use a prebuilt binary.

## Download

Every [release](https://github.com/Partha11/hatchling/releases) comes with ready-to-run binaries for Linux and Windows. Download the one for your system and start it from a terminal.

=== "Linux"

        :::bash
        chmod +x snake-linux-x86_64
        ./snake-linux-x86_64

    The Linux binary links the C++ runtime statically, so it runs on most distributions without extra packages.

=== "Windows"

        :::powershell
        .\snake-windows-x86_64.exe

    Run it from Command Prompt or PowerShell. Modern Windows terminals display the board correctly out of the box.

If you prefer to compile it yourself, see [Building](building.md).

## Controls

| Key | Action |
| --- | --- |
| `W` or `Up` | Move up |
| `S` or `Down` | Move down |
| `A` or `Left` | Move left |
| `D` or `Right` | Move right |
| `Q` or `Esc` | Quit |

The snake cannot turn back into itself. If you press the direction you came from, it simply keeps going straight.

## The board

The board is 20 cells wide and 15 cells tall. The snake starts with 3 cells in the middle of the board, moving right.

```text
+----------------------------------------+
|                                        |
|              *                         |
|                    o o o @             |
|                                        |
+----------------------------------------+
Score: 0
```

The head is `@`, the body is `o` and the food is `*`. Every food you eat adds one point to the score and one cell to the snake. The game ends when you hit a wall or your own body, and you win if the snake fills the whole board, which means a score of 297.

The game advances every 100 milliseconds. You can change the speed and the board size in `src/app/main.cpp` before building it yourself.
