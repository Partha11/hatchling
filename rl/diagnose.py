import argparse
import time
from collections import Counter, defaultdict

import numpy as np
import torch

from env import SnakeEnv
from model import ActorCritic, DuelingCNN

WIDTH, HEIGHT = 20, 15
HEADINGS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

DEFAULT_WEIGHTS = {"ppo": "snake_ppo_best.pt", "dqn": "snake_dqn_best.pt"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(algo, obs_shape):
    if algo == "ppo":
        return ActorCritic(*obs_shape, num_actions=3)
    return DuelingCNN(*obs_shape, num_actions=3)


def greedy_action(model, algo, st):
    out = model(st)
    if algo == "ppo":
        out = out[0]
    return out.argmax(dim=-1).item()


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


def blocked(board, y, x):
    if y < 0 or y >= HEIGHT or x < 0 or x >= WIDTH:
        return True
    if not board[2][y, x]:
        return False
    order = board[3]
    tail_value = order[order > 0].min()
    return order[y, x] != tail_value


def was_forced(board):
    (hy, hx), idx = head_and_heading(board)
    for turn in (0, 3, 1):
        dy, dx = HEADINGS[(idx + turn) % 4]
        if not blocked(board, hy + dy, hx + dx):
            return False
    return True


def classify(board, action, truncated, score):
    if truncated:
        return "timeout"

    py, px = predict_next_head(board, action)

    if py < 0 or py >= HEIGHT or px < 0 or px >= WIDTH:
        return "wall"

    if board[2][py, px]:
        return "body"

    if score + 3 >= WIDTH * HEIGHT:
        return "win"

    return "other"


def run(algo, weights, episodes):
    env = SnakeEnv(WIDTH, HEIGHT)
    model = build_model(algo, env.observation_space.shape).to(device)
    load_weights(model, weights)
    model.eval()

    causes = Counter()
    forced = Counter()
    scores_by_cause = defaultdict(list)
    all_scores = []

    for _ in range(episodes):
        state, _ = env.reset()
        board = np.array(env.engine.get_board(), copy=True)
        score = 0

        while True:
            with torch.no_grad():
                st = torch.from_numpy(state).unsqueeze(0).to(device)
                action = greedy_action(model, algo, st)

            state, _, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                cause = classify(board, action, truncated, score)
                causes[cause] += 1
                if cause in ("wall", "body") and was_forced(board):
                    forced[cause] += 1
                scores_by_cause[cause].append(info["score"])
                all_scores.append(info["score"])
                break

            board = np.array(env.engine.get_board(), copy=True)
            score = info["score"]

    print(f"Weights: {weights} | Episodes: {episodes} | Mean score: {np.mean(all_scores):.2f} | Max: {np.max(all_scores)}")
    print(f"{'Cause':<10}{'Count':>7}{'Share':>9}{'Mean score':>13}{'Forced':>9}{'Unforced':>11}")
    for cause in ["wall", "body", "timeout", "win", "other"]:
        n = causes[cause]
        if n == 0:
            continue
        line = f"{cause:<10}{n:>7}{n / episodes:>8.0%}{np.mean(scores_by_cause[cause]):>13.2f}"
        if cause in ("wall", "body"):
            line += f"{forced[cause]:>9}{n - forced[cause]:>11}"
        print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    run(args.algo, args.weights or DEFAULT_WEIGHTS[args.algo], args.episodes)