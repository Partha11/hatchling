#include "snake/engine.hpp"

#include <algorithm>
#include <array>
#include <random>

namespace {
    thread_local std::mt19937 rng(std::random_device{}());
}

SnakeEngine::SnakeEngine(int width, int height, float gamma)
    : width(width),
      height(height),
      gamma(gamma),
      score(0),
      stepsWithoutFood(0),
      direction(Direction::RIGHT) {
    boardBuffer.resize(
        static_cast<size_t>(NUM_CHANNELS) * static_cast<size_t>(width) * static_cast<size_t>(height), 0);
    reset();
}

int SnakeEngine::getWidth() const {
    return width;
}

int SnakeEngine::getHeight() const {
    return height;
}

float SnakeEngine::getGamma() const {
    return gamma;
}

void SnakeEngine::setGamma(float value) {
    gamma = value;
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

bool SnakeEngine::inBounds(const Point& p) const {
    return p.x >= 0 && p.x < width && p.y >= 0 && p.y < height;
}

void SnakeEngine::reset() {
    score = 0;
    stepsWithoutFood = 0;
    direction = Direction::RIGHT;

    const int startX = width / 2;
    const int startY = height / 2;

    head = Point{startX, startY};

    body.clear();
    body.push_back(Point{startX - 1, startY});
    body.push_back(Point{startX - 2, startY});

    spawnFood();
    updateBoardBuffer();
}

void SnakeEngine::resetWithLength(int length) {
    const int maxLength = width * height - 2;
    length = std::max(3, std::min(length, maxLength));

    if (length == 3) {
        reset();
        return;
    }

    bool built = false;
    for (int attempt = 0; attempt < 10 && !built; ++attempt) {
        built = buildRandomSnake(length);
    }
    if (!built) {
        buildSerpentineSnake(length);
    }

    score = length - 3;
    stepsWithoutFood = 0;
    spawnFood();
    updateBoardBuffer();
}

bool SnakeEngine::buildRandomSnake(int length) {
    static const int DX[4] = {0, 1, 0, -1};
    static const int DY[4] = {-1, 0, 1, 0};

    std::vector<uint8_t> used(static_cast<size_t>(width) * static_cast<size_t>(height), 0);
    auto cellIndex = [this](const Point& p) {
        return static_cast<size_t>(p.y) * static_cast<size_t>(width) + static_cast<size_t>(p.x);
    };

    std::uniform_int_distribution<int> distX(0, width - 1);
    std::uniform_int_distribution<int> distY(0, height - 1);

    std::vector<Point> path;
    std::vector<std::array<int, 4>> orders;
    std::vector<int> nextChoice;
    path.reserve(static_cast<size_t>(length));

    auto pushCell = [&](const Point& p) {
        std::array<int, 4> order = {0, 1, 2, 3};
        std::shuffle(order.begin(), order.end(), rng);
        path.push_back(p);
        orders.push_back(order);
        nextChoice.push_back(0);
        used[cellIndex(p)] = 1;
    };

    pushCell(Point{distX(rng), distY(rng)});

    long budget = 60L * length;
    while (!path.empty() && static_cast<int>(path.size()) < length && budget-- > 0) {
        const Point current = path.back();
        const int k = nextChoice.back();

        if (k == 4) {
            used[cellIndex(current)] = 0;
            path.pop_back();
            orders.pop_back();
            nextChoice.pop_back();
            continue;
        }

        nextChoice.back() = k + 1;
        const int dir = orders.back()[k];
        const Point candidate{current.x + DX[dir], current.y + DY[dir]};
        if (inBounds(candidate) && !used[cellIndex(candidate)]) {
            pushCell(candidate);
        }
    }

    if (static_cast<int>(path.size()) < length) {
        return false;
    }

    const Point& tip = path.back();
    bool hasExit = false;
    for (int dir = 0; dir < 4; ++dir) {
        const Point n{tip.x + DX[dir], tip.y + DY[dir]};
        if (inBounds(n) && !used[cellIndex(n)]) {
            hasExit = true;
            break;
        }
    }
    if (!hasExit) {
        return false;
    }

    placeSnake(path);
    return true;
}

void SnakeEngine::buildSerpentineSnake(int length) {
    std::vector<Point> order;
    order.reserve(static_cast<size_t>(width) * static_cast<size_t>(height));
    for (int y = 0; y < height; ++y) {
        if (y % 2 == 0) {
            for (int x = 0; x < width; ++x) {
                order.push_back(Point{x, y});
            }
        } else {
            for (int x = width - 1; x >= 0; --x) {
                order.push_back(Point{x, y});
            }
        }
    }

    std::uniform_int_distribution<int> flip(0, 1);
    const bool flipX = flip(rng) == 1;
    const bool flipY = flip(rng) == 1;
    for (auto& p : order) {
        if (flipX) {
            p.x = width - 1 - p.x;
        }
        if (flipY) {
            p.y = height - 1 - p.y;
        }
    }

    const int total = static_cast<int>(order.size());
    std::uniform_int_distribution<int> startDist(0, total - length - 1);
    const int start = startDist(rng);

    std::vector<Point> path(order.begin() + start, order.begin() + start + length);
    placeSnake(path);
}

void SnakeEngine::placeSnake(const std::vector<Point>& path) {
    head = path.back();
    body.clear();
    for (int i = static_cast<int>(path.size()) - 2; i >= 0; --i) {
        body.push_back(path[static_cast<size_t>(i)]);
    }

    const Point& neck = body.front();
    const int dx = head.x - neck.x;
    const int dy = head.y - neck.y;
    if (dx == 1) {
        direction = Direction::RIGHT;
    } else if (dx == -1) {
        direction = Direction::LEFT;
    } else if (dy == 1) {
        direction = Direction::DOWN;
    } else {
        direction = Direction::UP;
    }
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
        for (const auto& segment : body) {
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

    StepResult result{false, false, score, 0.0f};

    if (!inBounds(newHead)) {
        result.done = true;
        result.reward = DEATH_REWARD;
        updateBoardBuffer();
        return result;
    }

    const bool eating = (newHead == food);
    const size_t blocking = eating ? body.size() : body.size() - 1;

    for (size_t i = 0; i < blocking; ++i) {
        if (newHead == body[i]) {
            result.done = true;
            result.reward = DEATH_REWARD;
            updateBoardBuffer();
            return result;
        }
    }

    body.push_front(head);
    head = newHead;
    stepsWithoutFood++;

    if (eating) {
        score += 1;
        result.score = score;
        result.reward = FOOD_REWARD;
        stepsWithoutFood = 0;

        if (static_cast<int>(body.size()) + 1 >= width * height) {
            result.done = true;
            result.reward = WIN_REWARD;
        } else {
            spawnFood();
        }
    } else {
        body.pop_back();

        result.reward = STEP_REWARD;

        const int stepLimit = MAX_STEPS_PER_LENGTH * (static_cast<int>(body.size()) + 1);
        if (stepsWithoutFood >= stepLimit) {
            result.truncated = true;
        }
    }

    updateBoardBuffer();
    return result;
}

void SnakeEngine::updateBoardBuffer() {
    std::fill(boardBuffer.begin(), boardBuffer.end(), static_cast<uint8_t>(0));

    const size_t plane = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t headOffset = 0 * plane;
    const size_t neckOffset = 1 * plane;
    const size_t bodyOffset = 2 * plane;
    const size_t orderOffset = 3 * plane;
    const size_t foodOffset = 4 * plane;

    auto cell = [this](const Point& p) {
        return static_cast<size_t>(p.y) * static_cast<size_t>(width) + static_cast<size_t>(p.x);
    };

    if (inBounds(head)) {
        boardBuffer[headOffset + cell(head)] = 255;
    }

    if (!body.empty() && inBounds(body.front())) {
        boardBuffer[neckOffset + cell(body.front())] = 255;
    }

    const size_t n = body.size();
    for (size_t i = 0; i < n; ++i) {
        const Point& segment = body[i];
        if (!inBounds(segment)) {
            continue;
        }
        boardBuffer[bodyOffset + cell(segment)] = 255;
        boardBuffer[orderOffset + cell(segment)] =
            static_cast<uint8_t>((255 * (n - i) + n - 1) / n);
    }

    if (inBounds(food)) {
        boardBuffer[foodOffset + cell(food)] = 255;
    }
}