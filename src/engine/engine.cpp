#include "snake/engine.hpp"

#include <algorithm>
#include <cstring>
#include <random>

namespace {
    thread_local std::mt19937 rng(std::random_device{}());
}

SnakeEngine::SnakeEngine(int width, int height)
    : width(width), height(height), score(0), direction(Direction::RIGHT) {
    boardBuffer.resize(
        4 * static_cast<size_t>(width) * static_cast<size_t>(height), 0);
    reset();
}

int SnakeEngine::getWidth() const {
    return width;
}

int SnakeEngine::getHeight() const {
    return height;
}

const Point& SnakeEngine::getSnakeHead() const {
    return head;
}

const std::deque<Point>& SnakeEngine::getSnakeBody() const {
    return body;
}

Point SnakeEngine::getFood() const {
    return food;
}

Direction SnakeEngine::getSnakeDirection() const {
    return direction;
}

const std::vector<uint8_t>& SnakeEngine::getBoardBuffer() const {
    return boardBuffer;
}

void SnakeEngine::reset() {
    score = 0;
    stepsWithoutFood = 0;
    direction = Direction::RIGHT;

    int startX = width / 2;
    int startY = height / 2;

    head = Point{startX, startY};

    body.clear();
    body.push_back(Point{startX - 1, startY});
    body.push_back(Point{startX - 2, startY});

    spawnFood();
    updateBoardBuffer();
}

void SnakeEngine::spawnFood() {
    std::uniform_int_distribution<int> distX(0, width - 1);
    std::uniform_int_distribution<int> distY(0, height - 1);

    while (true) {
        Point candidate{distX(rng), distY(rng)};

        if (candidate == head) {
            continue;
        }

        bool collidesWithBody = false;
        for (const auto& segment: body) {
            if (candidate == segment) {
                collidesWithBody = true;
                break;
            }
        }

        if (!collidesWithBody) {
            food = candidate;
            break;
        }
    }
}

StepResult SnakeEngine::step(Action action) {
    if (action == Action::RIGHT) {
        direction = static_cast<Direction>((static_cast<uint8_t>(direction) + 1) % 4);
    } else if (action == Action::LEFT) {
        direction = static_cast<Direction>((static_cast<uint8_t>(direction) + 3) % 4);
    }

    Point newHead = head;
    switch (direction) {
        case Direction::UP:
            newHead.y -= 1;
            break;
        case Direction::DOWN:
            newHead.y += 1;
            break;
        case Direction::LEFT:
            newHead.x -= 1;
            break;
        case Direction::RIGHT:
            newHead.x += 1;
            break;
    }

    StepResult result;
    result.done = false;
    result.score = score;
    result.reward = 0.0f;

    if (newHead.x < 0 || newHead.x >= width || newHead.y < 0 || newHead.y >= height) {
        result.done = true;
        result.reward = -10.0f;
        updateBoardBuffer();
        return result;
    }

    for (const auto& segment : body) {
        if (newHead == segment) {
            result.done = true;
            result.reward = -10.0f;
            updateBoardBuffer();
            return result;
        }
    }

    int oldDist = std::abs(head.x - food.x) + std::abs(head.y - food.y);
    int newDist = std::abs(newHead.x - food.x) + std::abs(newHead.y - food.y);

    body.push_front(head);
    head = newHead;
    stepsWithoutFood++;

    if (head == food) {
        score += 1;
        result.score = score;
        result.reward = 10.0f;
        stepsWithoutFood = 0;

        if (static_cast<int>(body.size()) + 1 >= width * height) {
            result.done = true;
            result.reward = 100.0f;
        } else {
            spawnFood();
        }
    } else {
        body.pop_back();

        if (newDist < oldDist) {
            result.reward = 0.1f;
        } else {
            result.reward = -0.15f;
        }

        if (stepsWithoutFood >= MAX_STEPS_WITHOUT_FOOD) {
            result.done = true;
            result.reward = -1.0f;
        }
    }

    updateBoardBuffer();
    return result;
}

void SnakeEngine::updateBoardBuffer() {
    std::fill(boardBuffer.begin(), boardBuffer.end(), static_cast<uint8_t>(0));

    const size_t planeSize = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t headChannelOffset = 0 * planeSize;
    const size_t neckChannelOffset = 1 * planeSize;
    const size_t bodyChannelOffset = 2 * planeSize;
    const size_t foodChannelOffset = 3 * planeSize;

    if (head.x >= 0 && head.x < width && head.y >= 0 && head.y < height) {
        boardBuffer[headChannelOffset + head.y * width + head.x] = 1;
    }

    if (!body.empty()) {
        const auto& neck = body.front();
        if (neck.x >= 0 && neck.x < width && neck.y >= 0 && neck.y < height) {
            boardBuffer[neckChannelOffset + neck.y * width + neck.x] = 1;
        }
    }

    for (const auto& segment: body) {
        if (segment.x >= 0 && segment.x < width && segment.y >= 0 && segment.y < height) {
            boardBuffer[bodyChannelOffset + segment.y * width + segment.x] = 1;
        }
    }

    if (food.x >= 0 && food.x < width && food.y >= 0 && food.y < height) {
        boardBuffer[foodChannelOffset + food.y * width + food.x] = 1;
    }
}