import argparse
import time
from collections import Counter, defaultdict

import numpy as np
import torch

from env import SnakeEnv
from model import DuelingCNN

WIDTH, HEIGHT = 20, 15
HEADINGS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_weights(model, path, retries=5):
    for attempt in range(retries):
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            return
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.0)


def head_and_heading(board):
    hy, hx = np.argwhere(board[0])[0]
    ny, nx = np.argwhere(board[1])[0]
    return (int(hy), int(hx)), HEADINGS.index((int(hy - ny), int(hx - nx)))


def predict_next_head(board, action):
    (hy, hx), idx = head_and_heading(board)
    if action == 1:
        idx = (idx + 3) % 4
    elif action == 2:
        idx = (idx + 1) % 4
    dy, dx = HEADINGS[idx]
    return hy + dy, hx + dx


def classify(board, action, history, score):
    py, px = predict_next_head(board, action)

    if py < 0 or py >= HEIGHT or px < 0 or px >= WIDTH:
        return "wall"

    if (py, px) == history[-(score + 3)]:
        return "tail"

    if board[2][py, px]:
        return "body"

    if score + 3 >= WIDTH * HEIGHT:
        return "win"

    return "timeout"


def run(weights, episodes):
    env = SnakeEnv(WIDTH, HEIGHT)
    model = DuelingCNN(*env.observation_space.shape, num_actions=3).to(device)
    load_weights(model, weights)
    model.eval()

    causes = Counter()
    scores_by_cause = defaultdict(list)
    all_scores = []

    for _ in range(episodes):
        state, _ = env.reset()
        board = np.array(env.engine.get_board(), copy=True)
        (hy, hx), _ = head_and_heading(board)
        history = [(hy, hx - 2), (hy, hx - 1), (hy, hx)]
        score = 0

        while True:
            with torch.no_grad():
                st = torch.from_numpy(state).unsqueeze(0).to(device)
                action = model(st).argmax(dim=-1).item()

            state, _, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                cause = classify(board, action, history, score)
                causes[cause] += 1
                scores_by_cause[cause].append(info["score"])
                all_scores.append(info["score"])
                break

            board = np.array(env.engine.get_board(), copy=True)
            history.append(head_and_heading(board)[0])
            score = info["score"]

    print(f"Weights: {weights} | Episodes: {episodes} | Mean score: {np.mean(all_scores):.2f} | Max: {np.max(all_scores)}")
    print(f"{'Cause':<10}{'Count':>7}{'Share':>9}{'Mean score':>13}")
    for cause in ["wall", "body", "tail", "timeout", "win"]:
        n = causes[cause]
        if n == 0:
            continue
        print(f"{cause:<10}{n:>7}{n / episodes:>8.0%}{np.mean(scores_by_cause[cause]):>13.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="snake_dqn_best.pt")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    run(args.weights, args.episodes)