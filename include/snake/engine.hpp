#pragma once

#include <deque>
#include <vector>
#include <cstdint>

enum class Action: uint8_t {
    FORWARD,
    LEFT,
    RIGHT
};

enum class Direction: uint8_t {
    UP = 0,
    DOWN = 2,
    LEFT = 3,
    RIGHT = 1
};

struct Point {
    int x;
    int y;

    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }

    bool operator!=(const Point& other) const {
        return !(*this == other);
    }
};

struct StepResult {
    bool done;
    int score;
    float reward;
};

class SnakeEngine {
    public:
        SnakeEngine(int width, int height);

        int getWidth() const;
        int getHeight() const;

        void reset();
        void spawnFood();
        StepResult step(Action action);

        const Point& getSnakeHead() const;
        const std::deque<Point>& getSnakeBody() const;
        Point getFood() const;
        Direction getSnakeDirection() const;

        const std::vector<uint8_t>& getBoardBuffer() const;

    private:
        void updateBoardBuffer();

        int width;
        int height;
        int score;

        Point head;
        Point food;
        Direction direction;

        std::deque<Point> body;
        std::vector<uint8_t> boardBuffer;
};