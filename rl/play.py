import time
import torch
import numpy as np
from env import SnakeEnv
from model import DuelingCNN

WIDTH, HEIGHT = 20, 15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def render_terminal(game_engine):
    w, h = game_engine.get_width(), game_engine.get_height()
    grid = game_engine.get_board()  # (3, H, W)
    head_mask = grid[0]
    body_mask = grid[1]
    food_mask = grid[2]

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
    print("".join(out), end="", flush=True)

def play():
    env = SnakeEnv(WIDTH, HEIGHT)
    model = DuelingCNN(3, HEIGHT, WIDTH, num_actions=3).to(device)
    model.load_state_dict(torch.load("snake_dqn.pt", map_location=device))
    model.eval()

    state, _ = env.reset()
    print("\033[2J")

    while True:
        render_terminal(env.engine)
        time.sleep(0.08)

        with torch.no_grad():
            st = torch.from_numpy(state).unsqueeze(0).to(device)
            action = model(st).argmax(dim=-1).item()

        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state

        if terminated or truncated:
            render_terminal(env.engine)
            print(f"\nGame Over! Final Score: {info['score']}")
            break

if __name__ == "__main__":
    play()