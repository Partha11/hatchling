#include "snake/engine.hpp"

#include <iostream>
#include <chrono>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <conio.h>
#else
#include <termios.h>
#include <unistd.h>
#include <poll.h>
#endif

class TerminalController {
    public:
        TerminalController() {
#ifdef _WIN32
            hOut_ = GetStdHandle(STD_OUTPUT_HANDLE);
            if (hOut_ != INVALID_HANDLE_VALUE) {
                if (GetConsoleMode(hOut_, &orig_out_mode_)) {
#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif
                    SetConsoleMode(hOut_, orig_out_mode_ | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
                }
            }
            std::cout << "\033[?25l\033[2J\033[H" << std::flush;
#else
            tcgetattr(STDIN_FILENO, &orig_termios_);
            struct termios raw = orig_termios_;

            raw.c_lflag &= ~(ICANON | ECHO);

            raw.c_cc[VMIN] = 0;
            raw.c_cc[VTIME] = 0;

            tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);

            std::cout << "\033[?25l\033[2J\033[H" << std::flush;
#endif
        }

        ~TerminalController() {
            std::cout << "\033[?25h" << std::endl;
#ifdef _WIN32
            if (hOut_ != INVALID_HANDLE_VALUE) {
                SetConsoleMode(hOut_, orig_out_mode_);
            }
#else
            tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios_);
#endif
        }

        int readKey() {
#ifdef _WIN32
            if (_kbhit()) {
                int c = _getch();
                if (c == 0 || c == 224) {
                    switch (_getch()) {
                        case 72: return 'w'; // Up
                        case 80: return 's'; // Down
                        case 75: return 'a'; // Left
                        case 77: return 'd'; // Right
                        default: return -1;
                    }
                }
                if (c == 27) { // Escape
                    return 'q';
                }
                return c;
            }
            return -1;
#else
            struct pollfd pfd = {STDIN_FILENO, POLLIN, 0};
            int ret = poll(&pfd, 1, 0);

            if (ret > 0 && (pfd.revents & POLLIN)) {
                char c;
                
                if (read(STDIN_FILENO, &c, 1) == 1) {
                    if (c == '\033') {
                        char seq[2];
                        if (read(STDIN_FILENO, &seq[0], 1) == 1 && read(STDIN_FILENO, &seq[1], 1) == 1) {
                            if (seq[0] == '[') {
                                switch (seq[1]) {
                                    case 'A': return 'w';
                                    case 'B': return 's';
                                    case 'C': return 'd';
                                    case 'D': return 'a';
                                }
                            }
                        }
                        return 'q';
                    }
                    return c;
                }
            }
            return -1;
#endif
        }

    private:
#ifdef _WIN32
        HANDLE hOut_ = INVALID_HANDLE_VALUE;
        DWORD orig_out_mode_ = 0;
#else
        struct termios orig_termios_;
#endif
};

Action getRelativeAction(Direction currentDirection, char key) {
    Direction desiredDirection = currentDirection;

    if (key == 'w' || key == 'W') {
        desiredDirection = Direction::UP;
    } else if (key == 's' || key == 'S') {
        desiredDirection = Direction::DOWN;
    } else if (key == 'a' || key == 'A') {
        desiredDirection = Direction::LEFT;
    } else if (key == 'd' || key == 'D') {
        desiredDirection = Direction::RIGHT;
    } else {
        return Action::FORWARD;
    }

    uint8_t diff = (static_cast<uint8_t>(desiredDirection) + 4 - static_cast<uint8_t>(currentDirection)) % 4;

    switch (diff) {
        case 1:
            return Action::RIGHT;
        case 3:
            return Action::LEFT;
        default:
            return Action::FORWARD;
    }
}

void render(const SnakeEngine& game) {
    int width = game.getWidth();
    int height = game.getHeight();

    Point head = game.getSnakeHead();
    Point food = game.getFood();

    const auto& body = game.getSnakeBody();

    std::cout << "\033[H";

    std::cout << "+";
    for (int x = 0; x < width * 2; ++x) {
        std::cout << "-";
    }
    std::cout << "+\n";

    std::vector<std::string> rows(height, std::string(width * 2, ' '));

    if (food.x >= 0 && food.x < width && food.y >= 0 && food.y < height) {
        rows[food.y][food.x * 2] = '*';
    }

    for (const auto& segment: body) {
        if (segment.x >= 0 && segment.x < width && segment.y >= 0 && segment.y < height) {
            rows[segment.y][segment.x * 2] = 'o';
        }
    }

    if (head.x >= 0 && head.x < width && head.y >= 0 && head.y < height) {
        rows[head.y][head.x * 2] = '@';
    }

    for (int y = 0; y < height; ++y) {
        std::cout << "|" << rows[y] << "|\n";
    }

    std::cout << "+";
    for (int x = 0; x < width * 2; ++x) {
        std::cout << "-";
    }
    std::cout << "+\n";

    std::cout << "Score: " << game.getSnakeBody().size() - 2
              << " | Length: " << game.getSnakeBody().size() + 1
              << " | Head: (" << head.x << ", " << head.y << ")"
              << " | Food: (" << food.x << ", " << food.y << ")\n"
              << std::flush;
}

int main() {
    constexpr int WIDTH = 20;
    constexpr int HEIGHT = 15;
    constexpr auto TICK_INTERVAL = std::chrono::milliseconds(100);

    TerminalController terminalController;
    SnakeEngine game(WIDTH, HEIGHT);

    game.reset();
    render(game);

    bool running = true;
    int lastKey = -1;

    while (running) {
        auto tickStart = std::chrono::steady_clock::now();

        while (true) {
            int key = terminalController.readKey();
            if (key == -1) {
                break;
            }
            if (key == 'q' || key == 'Q') {
                running = false;
                break;
            }
            lastKey = key;
        }

        if (!running) {
            break;
        }

        Action action = getRelativeAction(game.getSnakeDirection(), lastKey);
        lastKey = -1;

        StepResult result = game.step(action);

        render(game);

        if (result.done) {
            std::cout << "\nGame Over! Final Score: " << result.score << "\n\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(1500));
            break;
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tickStart);
        if (elapsed < TICK_INTERVAL) {
            std::this_thread::sleep_for(TICK_INTERVAL - elapsed);
        }
    }

    return 0;
}
