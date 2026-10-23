import argparse
import time
from collections import deque

import numpy as np

import torch

from env import SnakeEnv
from model import ActorCritic, DuelingCNN

WIDTH, HEIGHT = 20, 15
DEFAULT_WEIGHTS = {"ppo": "snake_ppo_best.pt", "dqn": "snake_dqn_best.pt"}
FRAME_DELAY = 0.08

HEADINGS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
MOVE_NAMES = ["Forward", "Left", "Right"]
MOVE_TURNS = [0, 3, 1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def free_mask(board):
    body = board[2] > 0
    order = board[3]
    if body.any():
        tail_value = order[order > 0].min()
        body = body & (order != tail_value)
    return ~body


def reachable_cells(free, start):
    h, w = free.shape
    sy, sx = start
    if not (0 <= sy < h and 0 <= sx < w) or not free[sy, sx]:
        return 0
    seen = np.zeros_like(free, dtype=bool)
    seen[sy, sx] = True
    queue = deque([start])
    count = 0
    while queue:
        y, x = queue.popleft()
        count += 1
        for dy, dx in HEADINGS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and free[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                queue.append((ny, nx))
    return count


def explain_death(board, chosen_action, snake_length):
    hy, hx = np.argwhere(board[0])[0]
    ny, nx = np.argwhere(board[1])[0]
    heading = HEADINGS.index((int(hy - ny), int(hx - nx)))
    free = free_mask(board)
    free[hy, hx] = False
    h, w = free.shape

    parts = []
    for action in range(3):
        dy, dx = HEADINGS[(heading + MOVE_TURNS[action]) % 4]
        ty, tx = int(hy + dy), int(hx + dx)
        if not (0 <= ty < h and 0 <= tx < w):
            text = "wall"
        elif not free[ty, tx]:
            text = "body"
        else:
            cells = reachable_cells(free, (ty, tx))
            text = f"{cells} free cell{'' if cells == 1 else 's'}"
        marker = " (chosen)" if action == chosen_action else ""
        parts.append(f"{MOVE_NAMES[action]}: {text}{marker}")

    print(f"Last move options (snake length {snake_length}): " + " | ".join(parts))


def render_terminal(engine, score):
    w, h = engine.get_width(), engine.get_height()
    grid = engine.get_board()
    head_mask = grid[0]
    body_mask = grid[2]
    food_mask = grid[4]

    out = ["\033[H+" + "--" * w + "+\n"]
    for y in range(h):
        row = ["|"]
        for x in range(w):
            if head_mask[y, x]:
                row.append("@ ")
            elif body_mask[y, x]:
                row.append("o ")
            elif food_mask[y, x]:
                row.append("* ")
            else:
                row.append("  ")
        row.append("|\n")
        out.append("".join(row))
    out.append("+" + "--" * w + "+\n")
    out.append(f"Score: {score}\n")
    print("".join(out), end="", flush=True)


def play(algo, weights, delay):
    env = SnakeEnv(WIDTH, HEIGHT)
    if algo == "ppo":
        model = ActorCritic(*env.observation_space.shape, num_actions=3).to(device)
    else:
        model = DuelingCNN(*env.observation_space.shape, num_actions=3).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    state, _ = env.reset()
    board = np.array(env.engine.get_board(), copy=True)
    score = 0
    moves = 0
    print("\033[2J")

    while True:
        render_terminal(env.engine, score)
        time.sleep(delay)

        with torch.no_grad():
            st = torch.from_numpy(state).unsqueeze(0).to(device)
            out = model(st)
            if algo == "ppo":
                out = out[0]
            action = out.argmax(dim=-1).item()

        state, _, terminated, truncated, info = env.step(action)
        score = info["score"]
        moves += 1

        if terminated or truncated:
            render_terminal(env.engine, score)
            reason = "timeout" if truncated else "collision"
            print(f"\nGame Over ({reason})! Final Score: {score}")
            print(f"Moves: {moves} | Snake Length: {score + 3} | Weights: {weights}")
            if terminated and not truncated:
                explain_death(board, action, score + 3)
            break

        board = np.array(env.engine.get_board(), copy=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--delay", type=float, default=FRAME_DELAY)
    args = parser.parse_args()
    play(args.algo, args.weights or DEFAULT_WEIGHTS[args.algo], args.delay)