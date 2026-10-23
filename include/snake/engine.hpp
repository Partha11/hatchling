#pragma once

#include <cstdint>
#include <deque>
#include <vector>

enum class Action : uint8_t {
    FORWARD,
    LEFT,
    RIGHT
};

enum class Direction : uint8_t {
    UP = 0,
    RIGHT = 1,
    DOWN = 2,
    LEFT = 3
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
    bool truncated;
    int score;
    float reward;
};

class SnakeEngine {
    public:
        static constexpr int NUM_CHANNELS = 5;

        SnakeEngine(int width, int height, float gamma = 0.99f);

        int getWidth() const;
        int getHeight() const;
        float getGamma() const;
        void setGamma(float value);

        void reset();
        void resetWithLength(int length);
        void spawnFood();
        StepResult step(Action action);

        const Point& getSnakeHead() const;
        const std::deque<Point>& getSnakeBody() const;
        Point getFood() const;
        Direction getSnakeDirection() const;

        const std::vector<uint8_t>& getBoardBuffer() const;

    private:
        static constexpr int MAX_STEPS_PER_LENGTH = 100;
        static constexpr float FOOD_REWARD = 10.0f;
        static constexpr float DEATH_REWARD = -10.0f;
        static constexpr float WIN_REWARD = 100.0f;
        static constexpr float STEP_REWARD = -0.01f;

        int width;
        int height;
        float gamma;
        int score;
        int stepsWithoutFood;

        Point head;
        Point food;
        Direction direction;

        std::deque<Point> body;
        std::vector<uint8_t> boardBuffer;

        bool inBounds(const Point& p) const;
        bool buildRandomSnake(int length);
        void buildSerpentineSnake(int length);
        void placeSnake(const std::vector<Point>& path);
        void updateBoardBuffer();
};