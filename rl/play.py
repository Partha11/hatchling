import time

import torch

from env import SnakeEnv
from model import ActorCritic, DuelingCNN

WIDTH, HEIGHT = 20, 15
ALGO = "ppo"
WEIGHTS = {"ppo": "snake_ppo_best.pt", "dqn": "snake_dqn_best.pt"}[ALGO]
FRAME_DELAY = 0.08

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def play():
    env = SnakeEnv(WIDTH, HEIGHT)
    if ALGO == "ppo":
        model = ActorCritic(*env.observation_space.shape, num_actions=3).to(device)
    else:
        model = DuelingCNN(*env.observation_space.shape, num_actions=3).to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.eval()

    state, _ = env.reset()
    score = 0
    moves = 0
    print("\033[2J")

    while True:
        render_terminal(env.engine, score)
        time.sleep(FRAME_DELAY)

        with torch.no_grad():
            st = torch.from_numpy(state).unsqueeze(0).to(device)
            out = model(st)
            if ALGO == "ppo":
                out = out[0]
            action = out.argmax(dim=-1).item()

        state, _, terminated, truncated, info = env.step(action)
        score = info["score"]
        moves += 1

        if terminated or truncated:
            render_terminal(env.engine, score)
            reason = "timeout" if truncated else "collision"
            print(f"\nGame Over ({reason})! Final Score: {score}")
            print(f"Moves: {moves} | Snake Length: {score + 3}")
            break


if __name__ == "__main__":
    play()