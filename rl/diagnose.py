import argparse
import time
from collections import Counter, defaultdict

import numpy as np
import torch

from checkpoint import load_checkpoint
from env import SnakeEnv
import snake_cpp
from model import ActorCritic, DuelingCNN

WIDTH, HEIGHT = 20, 15
HEADINGS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

DEFAULT_WEIGHTS = {"ppo": "snake_ppo_best.pt", "dqn": "snake_dqn_best.pt"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(algo, obs_shape, meta, num_actions=3):
    if algo == "ppo":
        return ActorCritic(*obs_shape, num_actions=num_actions, danger_head=meta.get("danger_head", False))
    return DuelingCNN(*obs_shape, num_actions=3)


def make_policy(model, algo, merge_into=None):
    def policy(states, blocked=None, forward=None):
        with torch.no_grad():
            st = torch.from_numpy(np.stack(states)).to(device)
            out = model(st)
            if algo == "ppo":
                out = out[0]
            if blocked is not None and forward is not None:
                probs = torch.softmax(out.float(), dim=-1).cpu().numpy()
                rows = np.arange(len(probs))
                probs[rows, forward] += probs[rows, blocked]
                probs[rows, blocked] = -1.0
                return probs.argmax(axis=1)
            scores = out.float().cpu().numpy()
        if blocked is not None:
            scores[np.arange(len(scores)), blocked] = -np.inf
        return scores.argmax(axis=1)
    return policy


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


def blocked(body, y, x):
    if y < 0 or y >= HEIGHT or x < 0 or x >= WIDTH:
        return True
    return (x, y) in (body if snake_cpp.TAIL_IS_SOLID else body[:-1])


def steps_until_free(body, y, x):
    return len(body) - body.index((x, y))


def crash_distance(board, body, action):
    py, px = predict_next_head(board, action)
    if py < 0 or py >= HEIGHT or px < 0 or px >= WIDTH or (px, py) not in body:
        return None
    return steps_until_free(body, py, px)


def was_forced(board, body):
    (hy, hx), idx = head_and_heading(board)
    for turn in (0, 3, 1):
        dy, dx = HEADINGS[(idx + turn) % 4]
        if not blocked(body, hy + dy, hx + dx):
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


def play_games(policy, make_env, episodes, parallel, fixed_view, mask_reverse, merge_reverse=False):
    stats = {
        "causes": Counter(), "forced": Counter(), "body_distances": [],
        "scores_by_cause": defaultdict(list), "all_scores": [],
        "reverse_presses": 0, "reverse_crashes": 0, "moves": 0,
    }
    n = max(1, min(parallel, episodes))
    envs = [make_env() for _ in range(n)]
    states, boards, bodies, scores = [], [], [], []
    for env in envs:
        state, _ = env.reset()
        states.append(state)
        boards.append(np.array(env.engine.get_board(), copy=True))
        bodies.append(env.engine.get_body())
        scores.append(0)
    active = [True] * n
    started, finished = n, 0

    while finished < episodes:
        idx = [i for i in range(n) if active[i]]
        directions = np.array([envs[i].engine.get_direction() for i in idx]) if fixed_view else None
        reverse_keys = (directions + 2) % 4 if fixed_view else None
        if merge_reverse:
            actions = policy([states[i] for i in idx], reverse_keys, directions)
        else:
            actions = policy([states[i] for i in idx], reverse_keys if mask_reverse else None)

        for k, i in enumerate(idx):
            env = envs[i]
            action = int(actions[k])
            pressed_reverse = fixed_view and action == reverse_keys[k]
            stats["reverse_presses"] += int(pressed_reverse)
            stats["moves"] += 1

            state, _, terminated, truncated, info = env.step(action)
            relative = env.last_relative_action

            if terminated or truncated:
                board, body = boards[i], bodies[i]
                cause = classify(board, relative, truncated, scores[i])
                stats["causes"][cause] += 1
                if cause in ("wall", "body") and pressed_reverse:
                    stats["reverse_crashes"] += 1
                if cause in ("wall", "body") and was_forced(board, body):
                    stats["forced"][cause] += 1
                if cause == "body":
                    distance = crash_distance(board, body, relative)
                    if distance is not None:
                        stats["body_distances"].append(distance)
                stats["scores_by_cause"][cause].append(info["score"])
                stats["all_scores"].append(info["score"])
                finished += 1

                if started >= episodes:
                    active[i] = False
                    continue
                started += 1
                state, _ = env.reset()
                scores[i] = 0
            else:
                scores[i] = info["score"]

            states[i] = state
            boards[i] = np.array(env.engine.get_board(), copy=True)
            bodies[i] = env.engine.get_body()

    return stats


def run(algo, weights, episodes, mask_reverse=False, parallel=50, merge_reverse=False):
    state_dict, meta = load_checkpoint(weights, device)
    order = meta.get("order", "relative")
    view = meta.get("view", "ego")
    fixed_view = view == "fixed"
    if (mask_reverse or merge_reverse) and not fixed_view:
        raise ValueError("--mask-reverse and --merge-reverse only apply to models trained with the fixed view")
    if mask_reverse and merge_reverse:
        raise ValueError("choose one of --mask-reverse and --merge-reverse")

    def make_env():
        return SnakeEnv(WIDTH, HEIGHT, order_encoding=order, head_channel=meta.get("head_channel", False),
                        tail_channel=meta.get("tail_channel", False), view=view)

    probe_env = make_env()
    model = build_model(algo, probe_env.observation_space.shape, meta, probe_env.action_space.n).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    start = time.time()
    stats = play_games(make_policy(model, algo), make_env, episodes, parallel, fixed_view, mask_reverse, merge_reverse)
    elapsed = time.time() - start

    causes, forced = stats["causes"], stats["forced"]
    body_distances, scores_by_cause = stats["body_distances"], stats["scores_by_cause"]
    all_scores = stats["all_scores"]
    reverse_presses, reverse_crashes, moves = stats["reverse_presses"], stats["reverse_crashes"], stats["moves"]

    print(
        f"Weights: {weights} | View: {meta.get('view', 'ego')}"
        f"{' (reverse key masked)' if mask_reverse else ''}{' (reverse merged into forward)' if merge_reverse else ''} | "
        f"Order: {order} | Episodes: {episodes} | Time: {elapsed / 60:.1f} min | "
        f"Mean score: {np.mean(all_scores):.2f} | Max: {np.max(all_scores)}"
    )
    scores = np.array(all_scores)
    q10, q25, q50, q75, q90 = np.percentile(scores, [10, 25, 50, 75, 90])
    print(
        f"Score distribution: min {scores.min()} | 10% {q10:.0f} | 25% {q25:.0f} | median {q50:.0f} | "
        f"75% {q75:.0f} | 90% {q90:.0f} | max {scores.max()}"
    )
    below = " | ".join(f"<{t}: {(scores < t).mean():.0%}" for t in (50, 100, 150))
    print(f"Games ending below: {below}")
    print("All scores (in order of finishing): " + " ".join(str(int(x)) for x in scores) + "\n")
    print(f"{'Cause':<10}{'Count':>7}{'Share':>9}{'Mean score':>13}{'Forced':>9}{'Unforced':>11}")
    for cause in ["wall", "body", "timeout", "win", "other"]:
        n = causes[cause]
        if n == 0:
            continue
        line = f"{cause:<10}{n:>7}{n / episodes:>8.0%}{np.mean(scores_by_cause[cause]):>13.2f}"
        if cause in ("wall", "body"):
            line += f"{forced[cause]:>9}{n - forced[cause]:>11}"
        print(line)

    if fixed_view:
        crashes = causes["wall"] + causes["body"]
        print(
            f"\nReverse key: pressed {reverse_presses} times ({reverse_presses / max(moves, 1):.2%} of moves), "
            f"caused {reverse_crashes} of {crashes} crashes ({reverse_crashes / max(crashes, 1):.0%})"
        )

    if body_distances:
        d = np.array(body_distances)
        buckets = [(1, 1), (2, 2), (3, 5), (6, 20), (21, 10_000)]
        print(f"\nBody crashes by steps until the hit segment would have cleared ({len(d)} crashes):")
        for lo, hi in buckets:
            n = int(((d >= lo) & (d <= hi)).sum())
            label = "1 (tail)" if lo == hi == 1 else (
                f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"))
            print(f"  {label:>8} steps: {n:4d} ({n / len(d):.0%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--mask-reverse", action="store_true")
    parser.add_argument("--parallel", type=int, default=50)
    parser.add_argument("--merge-reverse", action="store_true")
    args = parser.parse_args()
    run(args.algo, args.weights or DEFAULT_WEIGHTS[args.algo], args.episodes, args.mask_reverse, args.parallel,
        args.merge_reverse)