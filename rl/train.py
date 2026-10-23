import argparse
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from env import SnakeEnv
from model import ActorCritic

WIDTH, HEIGHT = 20, 15
N_ENVS = 32
TOTAL_STEPS = 5_000_000
LR = 2.5e-4
RESUME_LR = 1e-4
EPOCHS = 4
MINIBATCH_SIZE = 1024
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
MAX_GRAD_NORM = 0.5
REWARD_SCALE = 0.1
EVAL_FREQ = 100_000
EVAL_EPISODES = 50

DEFAULT_GAMMA = 0.998
DEFAULT_LAMBDA = 0.97
DEFAULT_ROLLOUT = 256
DEFAULT_LONG_START = 0.0
DEFAULT_LONG_MIN = 100
DEFAULT_LONG_MAX = 250

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_paths(tag: str):
    prefix = f"snake_ppo_{tag}" if tag else "snake_ppo"
    return f"{prefix}_best.pt", f"{prefix}_latest.pt", f"{prefix}.pt"


LONG_RANGE = [DEFAULT_LONG_MIN, DEFAULT_LONG_MAX]


def reset_env(env: SnakeEnv, long_start_prob: float):
    if long_start_prob > 0.0 and random.random() < long_start_prob:
        length = random.randint(LONG_RANGE[0], LONG_RANGE[1])
        state, _ = env.reset(options={"length": length})
        return state, True
    state, _ = env.reset()
    return state, False


def evaluate(model: nn.Module, envs):
    model.eval()
    states = [env.reset()[0] for env in envs]
    scores = [0] * len(envs)
    timeouts = 0
    active = list(range(len(envs)))

    while active:
        batch = torch.from_numpy(np.stack([states[i] for i in active])).to(device)
        with torch.no_grad():
            logits, _ = model(batch)
            actions = logits.argmax(dim=-1).cpu().numpy()

        still_active = []
        for j, i in enumerate(active):
            state, _, terminated, truncated, info = envs[i].step(int(actions[j]))
            states[i] = state
            scores[i] = info["score"]
            if truncated:
                timeouts += 1
            if not (terminated or truncated):
                still_active.append(i)
        active = still_active

    model.train()
    return float(np.mean(scores)), int(np.max(scores)), timeouts


def train(gamma, gae_lambda, rollout_len, long_start_prob, tag, resume_path, lr, total_steps):
    best_path, latest_path, final_path = checkpoint_paths(tag)

    envs = [SnakeEnv(WIDTH, HEIGHT, gamma) for _ in range(N_ENVS)]
    eval_envs = [SnakeEnv(WIDTH, HEIGHT, gamma) for _ in range(EVAL_EPISODES)]

    obs_shape = envs[0].observation_space.shape
    model = ActorCritic(*obs_shape, num_actions=3).to(device)
    if resume_path:
        model.load_state_dict(torch.load(resume_path, map_location=device))
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    print(
        f"gamma={gamma} lambda={gae_lambda} rollout={rollout_len} long_start={long_start_prob} "
        f"long_range={LONG_RANGE[0]}-{LONG_RANGE[1]} "
        f"lr={lr} steps={total_steps} resume={resume_path} -> {best_path}"
    )

    batch_size = N_ENVS * rollout_len
    num_updates = total_steps // batch_size

    obs_buf = torch.zeros((rollout_len, N_ENVS, *obs_shape), dtype=torch.uint8, device=device)
    act_buf = torch.zeros((rollout_len, N_ENVS), dtype=torch.long, device=device)
    logp_buf = torch.zeros((rollout_len, N_ENVS), device=device)
    rew_buf = torch.zeros((rollout_len, N_ENVS), device=device)
    done_buf = torch.zeros((rollout_len, N_ENVS), device=device)
    val_buf = torch.zeros((rollout_len, N_ENVS), device=device)

    states = []
    long_flags = []
    for env in envs:
        state, is_long = reset_env(env, long_start_prob)
        states.append(state)
        long_flags.append(is_long)
    states = np.stack(states)

    scores = deque(maxlen=100)
    long_start_games = 0
    best_eval = -1.0
    steps = 0
    next_eval = EVAL_FREQ

    for update in range(1, num_updates + 1):
        frac = 1.0 - (update - 1) / num_updates
        for group in optimizer.param_groups:
            group["lr"] = lr * frac

        for t in range(rollout_len):
            obs_t = torch.from_numpy(states).to(device)
            with torch.no_grad():
                logits, value = model(obs_t)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                logp = dist.log_prob(actions)

            obs_buf[t] = obs_t
            act_buf[t] = actions
            logp_buf[t] = logp
            val_buf[t] = value

            actions_np = actions.cpu().numpy()
            rewards = np.zeros(N_ENVS, dtype=np.float32)
            dones = np.zeros(N_ENVS, dtype=np.float32)
            truncated_idx = []
            truncated_obs = []

            for i, env in enumerate(envs):
                next_state, reward, terminated, truncated, info = env.step(int(actions_np[i]))
                rewards[i] = reward * REWARD_SCALE

                if terminated or truncated:
                    dones[i] = 1.0
                    if long_flags[i]:
                        long_start_games += 1
                    else:
                        scores.append(info["score"])
                    if truncated and not terminated:
                        truncated_idx.append(i)
                        truncated_obs.append(next_state)
                    next_state, long_flags[i] = reset_env(env, long_start_prob)

                states[i] = next_state

            if truncated_idx:
                with torch.no_grad():
                    _, final_values = model(torch.from_numpy(np.stack(truncated_obs)).to(device))
                for j, i in enumerate(truncated_idx):
                    rewards[i] += gamma * final_values[j].item()

            rew_buf[t] = torch.from_numpy(rewards).to(device)
            done_buf[t] = torch.from_numpy(dones).to(device)
            steps += N_ENVS

        with torch.no_grad():
            _, next_value = model(torch.from_numpy(states).to(device))

        advantages = torch.zeros_like(rew_buf)
        last_gae = torch.zeros(N_ENVS, device=device)
        for t in reversed(range(rollout_len)):
            following_value = next_value if t == rollout_len - 1 else val_buf[t + 1]
            not_done = 1.0 - done_buf[t]
            delta = rew_buf[t] + gamma * following_value * not_done - val_buf[t]
            last_gae = delta + gamma * gae_lambda * not_done * last_gae
            advantages[t] = last_gae
        returns = advantages + val_buf

        b_obs = obs_buf.reshape(batch_size, *obs_shape)
        b_act = act_buf.reshape(batch_size)
        b_logp = logp_buf.reshape(batch_size)
        b_adv = advantages.reshape(batch_size)
        b_ret = returns.reshape(batch_size)

        v_losses, entropies, kls, clipfracs = [], [], [], []

        for _ in range(EPOCHS):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, MINIBATCH_SIZE):
                idx = perm[start:start + MINIBATCH_SIZE]

                logits, values = model(b_obs[idx])
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(b_act[idx])
                entropy = dist.entropy().mean()

                log_ratio = new_logp - b_logp[idx]
                ratio = log_ratio.exp()

                adv = b_adv[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                pg_loss = torch.max(
                    -adv * ratio,
                    -adv * ratio.clamp(1.0 - CLIP_EPS, 1.0 + CLIP_EPS),
                ).mean()
                v_loss = 0.5 * (values - b_ret[idx]).pow(2).mean()
                loss = pg_loss + VF_COEF * v_loss - ENT_COEF * entropy

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                with torch.no_grad():
                    kls.append(((ratio - 1.0) - log_ratio).mean().item())
                    clipfracs.append(((ratio - 1.0).abs() > CLIP_EPS).float().mean().item())
                v_losses.append(v_loss.item())
                entropies.append(entropy.item())

        avg_score = np.mean(scores) if scores else 0.0
        line = (
            f"Step: {steps:8d} | Avg Score: {avg_score:5.2f} | "
            f"Entropy: {np.mean(entropies):.3f} | Value Loss: {np.mean(v_losses):.4f} | "
            f"KL: {np.mean(kls):.4f} | Clip: {np.mean(clipfracs):.3f}"
        )
        if long_start_prob > 0.0:
            line += f" | Long Starts: {long_start_games}"
        print(line)

        if steps >= next_eval:
            next_eval += EVAL_FREQ
            eval_mean, eval_max, eval_timeouts = evaluate(model, eval_envs)
            torch.save(model.state_dict(), latest_path)
            marker = ""
            if eval_mean > best_eval:
                best_eval = eval_mean
                torch.save(model.state_dict(), best_path)
                marker = " *"
            print(
                f"[Eval] Step: {steps:8d} | Greedy Mean: {eval_mean:6.2f} | Max: {eval_max:3d} | "
                f"Timeouts: {eval_timeouts}/{EVAL_EPISODES}{marker}"
            )

    torch.save(model.state_dict(), final_path)
    print(f"Training finished. Best greedy mean: {best_eval:.2f} ({best_path}), final weights: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--lam", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--rollout", type=int, default=DEFAULT_ROLLOUT)
    parser.add_argument("--long-start", type=float, default=DEFAULT_LONG_START)
    parser.add_argument("--long-min", type=int, default=DEFAULT_LONG_MIN)
    parser.add_argument("--long-max", type=int, default=DEFAULT_LONG_MAX)
    parser.add_argument("--tag", default="")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    args = parser.parse_args()

    LONG_RANGE[0], LONG_RANGE[1] = args.long_min, args.long_max
    lr = args.lr if args.lr is not None else (RESUME_LR if args.resume else LR)
    train(args.gamma, args.lam, args.rollout, args.long_start, args.tag, args.resume, lr, args.steps)