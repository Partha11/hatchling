import argparse
import time
from collections import deque

import numpy as np

import torch

from checkpoint import load_checkpoint
from env import ABSOLUTE_NAMES, FIXED_NAMES, SnakeEnv
import snake_cpp
from model import ActorCritic, DuelingCNN

WIDTH, HEIGHT = 20, 15
DEFAULT_WEIGHTS = {"ppo": "snake_ppo_best.pt", "dqn": "snake_dqn_best.pt"}
FRAME_DELAY = 0.08
REWARD_SCALE = 0.1

HEADINGS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
MOVE_NAMES = ["Forward", "Left", "Right"]
MOVE_TURNS = [0, 3, 1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def free_mask(board, body):
    free = np.ones(board[2].shape, dtype=bool)
    for x, y in (body if snake_cpp.TAIL_IS_SOLID else body[:-1]):
        free[y, x] = False
    return free


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


def steps_until_free(body, y, x):
    return len(body) - body.index((x, y))


def explain_input(obs, view="ego", board=None):
    parts = []
    if view == "fixed":
        hy, hx = np.argwhere(board[0])[0]
        ny, nx = np.argwhere(board[1])[0]
        heading = HEADINGS.index((int(hy - ny), int(hx - nx)))
        for name, turn in zip(MOVE_NAMES, MOVE_TURNS):
            dy, dx = HEADINGS[(heading + turn) % 4]
            y, x = hy + dy + 1, hx + dx + 1
            present = [FIXED_NAMES[ch] for ch in range(obs.shape[0]) if obs[ch, y, x] > 0]
            parts.append(f"{name}: {present[0] if present else 'empty'}")
    else:
        c = obs.shape[1] // 2
        cells = {"Forward": (c - 1, c), "Left": (c, c - 1), "Right": (c, c + 1)}
        for name, (y, x) in cells.items():
            body = obs[0, y, x] > 0
            wall = obs[3, y, x] > 0
            food = obs[2, y, x] > 0
            what = "body" if body else ("wall" if wall else ("food" if food else "empty"))
            parts.append(f"{name}: {what}")
    print("Network input around head: " + " | ".join(parts))


def explain_danger(danger):
    parts = [f"{MOVE_NAMES[a]}: {danger[a]:.3f}" for a in range(3)]
    print("Predicted danger before last move: " + " | ".join(parts))


def explain_policy(probs, value):
    names = ABSOLUTE_NAMES if len(probs) == 4 else MOVE_NAMES
    parts = [f"{names[a]}: {probs[a]:.3f}" for a in range(len(probs))]
    print("Policy before last move: " + " | ".join(parts) + f" | Value estimate: {value:+.2f}")


def explain_death(board, body, chosen_action, snake_length):
    hy, hx = np.argwhere(board[0])[0]
    ny, nx = np.argwhere(board[1])[0]
    heading = HEADINGS.index((int(hy - ny), int(hx - nx)))
    free = free_mask(board, body)
    free[hy, hx] = False
    h, w = free.shape

    parts = []
    for action in range(3):
        dy, dx = HEADINGS[(heading + MOVE_TURNS[action]) % 4]
        ty, tx = int(hy + dy), int(hx + dx)
        if not (0 <= ty < h and 0 <= tx < w):
            text = "wall"
        elif not free[ty, tx]:
            text = f"body ({steps_until_free(body, ty, tx)} steps from clearing)"
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
    state_dict, meta = load_checkpoint(weights, device)
    order = meta.get("order", "relative")
    view = meta.get("view", "ego")
    env = SnakeEnv(WIDTH, HEIGHT, order_encoding=order, head_channel=meta.get("head_channel", False),
                    tail_channel=meta.get("tail_channel", False), view=view)
    num_actions = env.action_space.n
    has_danger = algo == "ppo" and meta.get("danger_head", False)
    if algo == "ppo":
        model = ActorCritic(*env.observation_space.shape, num_actions=num_actions, danger_head=has_danger).to(device)
    else:
        model = DuelingCNN(*env.observation_space.shape, num_actions=3).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    state, _ = env.reset()
    board = np.array(env.engine.get_board(), copy=True)
    body = env.engine.get_body()
    score = 0
    moves = 0
    print("\033[2J")

    while True:
        render_terminal(env.engine, score)
        time.sleep(delay)

        prev_state = state
        with torch.no_grad():
            st = torch.from_numpy(state).unsqueeze(0).to(device)
            out = model(st, return_danger=True) if has_danger else model(st)
            if has_danger:
                danger_probs = torch.sigmoid(out[2])[0].cpu().numpy()
                out = out[:2]
            if algo == "ppo":
                logits, value = out
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                last_value = value.item() / REWARD_SCALE
                out = logits
            action = out.argmax(dim=-1).item()

        state, reward, terminated, truncated, info = env.step(action)
        score = info["score"]
        moves += 1

        if terminated or truncated:
            render_terminal(env.engine, score)
            won = terminated and reward > 0
            if won:
                print(f"\nBoard filled - the snake won! Final Score: {score}")
            else:
                reason = "timeout" if truncated else "collision"
                print(f"\nGame Over ({reason})! Final Score: {score}")
            print(f"Moves: {moves} | Snake Length: {score + 3} | Weights: {weights} | View: {view} | Order: {order}")
            if won and algo == "ppo":
                explain_policy(probs, last_value)
            elif terminated and not truncated:
                explain_death(board, body, env.last_relative_action, score + 3)
                explain_input(prev_state, view, board)
                if algo == "ppo":
                    explain_policy(probs, last_value)
                if has_danger:
                    explain_danger(danger_probs)
            break

        board = np.array(env.engine.get_board(), copy=True)
        body = env.engine.get_body()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--delay", type=float, default=FRAME_DELAY)
    args = parser.parse_args()
    play(args.algo, args.weights or DEFAULT_WEIGHTS[args.algo], args.delay)