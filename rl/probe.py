import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import SnakeEnv
from model import ActorCritic, ResidualBlock, build_trunk, trunk_output_size
from checkpoint import load_checkpoint

WIDTH, HEIGHT = 20, 15
CATEGORIES = ["inline", "body_wall", "board_wall"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def forward_category(obs):
    c = obs.shape[1] // 2
    ahead_body = obs[0, c - 1, c] > 0
    ahead_wall = obs[3, c - 1, c] > 0
    if ahead_wall:
        return "board_wall"
    if not ahead_body:
        return None
    diag_left = obs[0, c - 1, c - 1] > 0 or obs[3, c - 1, c - 1] > 0
    diag_right = obs[0, c - 1, c + 1] > 0 or obs[3, c - 1, c + 1] > 0
    return "body_wall" if (diag_left or diag_right) else "inline"


def collect_random(n, long_share, seed):
    random.seed(seed)
    env = SnakeEnv(WIDTH, HEIGHT)
    obs_list, labels = [], []
    state, _ = env.reset()
    while len(obs_list) < n:
        obs_list.append(state)
        labels.append(env.danger())
        danger = labels[-1]
        safe = [a for a in range(3) if danger[a] == 0]
        action = random.choice(safe) if safe and random.random() < 0.95 else random.randrange(3)
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            if random.random() < long_share:
                state, _ = env.reset(options={"length": random.randint(20, 250)})
            else:
                state, _ = env.reset()
    return np.stack(obs_list), np.stack(labels)


def collect_agent(n, weights, n_envs=32):
    state_dict, meta = load_checkpoint(weights, device)
    envs = [SnakeEnv(WIDTH, HEIGHT, order_encoding=meta.get("order", "relative"),
                     head_channel=meta.get("head_channel", False)) for _ in range(n_envs)]
    model = ActorCritic(*envs[0].observation_space.shape, num_actions=3,
                        danger_head=meta.get("danger_head", False)).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    plain = [SnakeEnv(WIDTH, HEIGHT) for _ in range(n_envs)]
    states = [env.reset()[0] for env in envs]
    obs_list, labels = [], []
    while len(obs_list) < n:
        with torch.no_grad():
            logits, _ = model(torch.from_numpy(np.stack(states)).to(device))
            actions = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        for i, env in enumerate(envs):
            board = np.asarray(env.engine.get_board())
            obs_list.append(plain[i]._egocentric(board))
            labels.append(env.danger())
            state, _, terminated, truncated, _ = env.step(int(actions[i]))
            if terminated or truncated:
                state, _ = env.reset()
            states[i] = state
    return np.stack(obs_list[:n]), np.stack(labels[:n])


class CurrentNet(nn.Module):
    def __init__(self, in_channels, size):
        super().__init__()
        self.conv = build_trunk(in_channels, 64, 3, 32)
        out = trunk_output_size(self.conv, in_channels, size, size)
        self.head = nn.Sequential(nn.Linear(out, 512), nn.ReLU(), nn.Linear(512, 3))

    def forward(self, x):
        x = x.float() / 255.0
        return self.head(self.conv(x).flatten(1))


class FullResNet(nn.Module):
    def __init__(self, in_channels, size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            *[ResidualBlock(64) for _ in range(3)],
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
        )
        with torch.no_grad():
            out = self.conv(torch.zeros(1, in_channels, size, size)).flatten(1).size(1)
        self.head = nn.Sequential(nn.Linear(out, 512), nn.ReLU(), nn.Linear(512, 3))

    def forward(self, x):
        x = x.float() / 255.0
        return self.head(self.conv(x).flatten(1))


def evaluate(net, obs, labels, cats, batch=1024):
    net.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(obs), batch):
            logits = net(torch.from_numpy(obs[i:i + batch]).to(device))
            preds.append((logits > 0).cpu().numpy())
    net.train()
    preds = np.concatenate(preds).astype(bool)
    truth = labels.astype(bool)

    fatal = truth.sum()
    missed = (truth & ~preds).sum()
    false_alarm = (~truth & preds).sum()
    report = {
        "fatal_moves": int(fatal),
        "missed_fatal": int(missed),
        "miss_rate": missed / max(fatal, 1),
        "false_alarm_rate": false_alarm / max((~truth).sum(), 1),
    }
    for cat in CATEGORIES:
        mask = cats == cat
        n = int(mask.sum())
        miss = int((mask & ~preds[:, 0]).sum())
        report[f"forward_{cat}"] = (n, miss)
    return report


def train_net(name, net, train_obs, train_lab, test_obs, test_lab, test_cats, steps, batch, pos_weight):
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    pw = torch.full((3,), pos_weight, device=device)
    lab_t = torch.from_numpy(train_lab).float()
    start = time.time()
    for step in range(1, steps + 1):
        idx = np.random.randint(0, len(train_obs), size=batch)
        x = torch.from_numpy(train_obs[idx]).to(device)
        y = lab_t[idx].to(device)
        loss = F.binary_cross_entropy_with_logits(net(x), y, pos_weight=pw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(steps // 5, 1) == 0 or step == steps:
            r = evaluate(net, test_obs, test_lab, test_cats)
            print(
                f"[{name}] step {step:6d} | loss {loss.item():.4f} | miss rate {r['miss_rate']:.4f} "
                f"({r['missed_fatal']}/{r['fatal_moves']}) | false alarms {r['false_alarm_rate']:.4f} | "
                f"{time.time() - start:.0f}s"
            )
    return evaluate(net, test_obs, test_lab, test_cats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["random", "agent"], default="random")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--long-share", type=float, default=0.5)
    parser.add_argument("--nets", default="current,fullres")
    parser.add_argument("--oversample-inline", type=int, default=1)
    args = parser.parse_args()

    t0 = time.time()
    if args.source == "agent":
        if not args.weights:
            raise ValueError("--source agent needs --weights")
        obs, lab = collect_agent(args.samples, args.weights)
    else:
        obs, lab = collect_random(args.samples, args.long_share, seed=0)
    cats = np.array([forward_category(o) if l[0] else None for o, l in zip(obs, lab)], dtype=object)
    print(f"Collected {len(obs)} positions from {args.source} play in {time.time() - t0:.0f}s")

    perm = np.random.permutation(len(obs))
    split = int(len(obs) * 0.85)
    tr, te = perm[:split], perm[split:]
    if args.oversample_inline > 1:
        inline_tr = tr[cats[tr] == "inline"]
        tr = np.concatenate([tr] + [inline_tr] * (args.oversample_inline - 1))
        print(f"Inline training examples: {len(inline_tr)}, repeated {args.oversample_inline}x")
    print("Fatal forward moves in the test set by shape: " + ", ".join(
        f"{c}: {int((cats[te] == c).sum())}" for c in CATEGORIES))

    builders = {"current": CurrentNet, "fullres": FullResNet}
    results = {}
    for name in args.nets.split(","):
        net = builders[name](obs.shape[1], obs.shape[2])
        results[name] = train_net(name, net, obs[tr], lab[tr], obs[te], lab[te], cats[te],
                                  args.steps, args.batch, args.pos_weight)

    print("\nMissed fatal FORWARD moves by shape (missed / total):")
    header = f"{'shape':<12}" + "".join(f"{n:>18}" for n in results)
    print(header)
    for cat in CATEGORIES:
        row = f"{cat:<12}"
        for name, r in results.items():
            n, miss = r[f"forward_{cat}"]
            row += f"{f'{miss}/{n} ({miss / max(n, 1):.1%})':>18}"
        print(row)
    print("\nOverall:")
    for name, r in results.items():
        print(f"{name:<10} miss rate {r['miss_rate']:.4f} | false alarm rate {r['false_alarm_rate']:.4f}")


if __name__ == "__main__":
    main()