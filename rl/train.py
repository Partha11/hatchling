import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env import SnakeEnv
from model import DuelingCNN

WIDTH, HEIGHT = 20, 15
BATCH_SIZE = 64
LR = 1e-4
GAMMA = 0.99
MEMORY_SIZE = 100_000
WARMUP_STEPS = 10_000
TRAIN_FREQ = 4
TARGET_UPDATE_FREQ = 2_000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 250_000
TOTAL_STEPS = 1_500_000
LOG_FREQ = 5_000
EVAL_FREQ = 25_000
EVAL_EPISODES = 20
BEST_PATH = "snake_dqn_best.pt"
FINAL_PATH = "snake_dqn.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, capacity: int, shape: tuple):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, *shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, *shape), dtype=np.uint8)
        self.dones = np.zeros((capacity,), dtype=np.bool_)

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[idx]).to(device),
            torch.from_numpy(self.actions[idx]).to(device),
            torch.from_numpy(self.rewards[idx]).to(device),
            torch.from_numpy(self.next_states[idx]).to(device),
            torch.from_numpy(self.dones[idx]).to(device),
        )


def evaluate(model: nn.Module, env: SnakeEnv, episodes: int):
    model.eval()
    scores = []

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        score = 0

        while not done:
            with torch.no_grad():
                st = torch.from_numpy(state).unsqueeze(0).to(device)
                action = model(st).argmax(dim=-1).item()

            state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score = info["score"]

        scores.append(score)

    model.train()
    return float(np.mean(scores)), int(np.max(scores))


def train():
    env = SnakeEnv(WIDTH, HEIGHT)
    eval_env = SnakeEnv(WIDTH, HEIGHT)

    obs_shape = env.observation_space.shape
    in_channels, obs_h, obs_w = obs_shape

    replay_buffer = ReplayBuffer(MEMORY_SIZE, obs_shape)

    policy_net = DuelingCNN(in_channels, obs_h, obs_w, num_actions=3).to(device)
    target_net = DuelingCNN(in_channels, obs_h, obs_w, num_actions=3).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    criterion = nn.SmoothL1Loss()

    state, _ = env.reset()
    scores = deque(maxlen=50)
    losses = deque(maxlen=200)
    mean_qs = deque(maxlen=200)
    best_eval = -1.0

    for step in range(1, TOTAL_STEPS + 1):
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * max(
            0.0, (EPSILON_DECAY - step) / EPSILON_DECAY
        )

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                st = torch.from_numpy(state).unsqueeze(0).to(device)
                q_vals = policy_net(st)
                mean_qs.append(q_vals.max().item())
                action = q_vals.argmax(dim=-1).item()

        next_state, reward, terminated, truncated, info = env.step(action)

        replay_buffer.push(state, action, reward, next_state, terminated)
        state = next_state

        if terminated or truncated:
            scores.append(info["score"])
            state, _ = env.reset()

        if step > WARMUP_STEPS and step % TRAIN_FREQ == 0:
            b_state, b_action, b_reward, b_next_state, b_done = replay_buffer.sample(BATCH_SIZE)

            with torch.no_grad():
                next_action = policy_net(b_next_state).argmax(dim=-1, keepdim=True)
                next_q = target_net(b_next_state).gather(1, next_action).squeeze(1)
                target_q = b_reward + (1.0 - b_done.float()) * GAMMA * next_q

            current_q = policy_net(b_state).gather(1, b_action.unsqueeze(1)).squeeze(1)
            loss = criterion(current_q, target_q)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(loss.item())

        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if step % LOG_FREQ == 0:
            avg_score = np.mean(scores) if scores else 0.0
            avg_loss = np.mean(losses) if losses else 0.0
            avg_q = np.mean(mean_qs) if mean_qs else 0.0
            print(
                f"Step: {step:7d} | Eps: {epsilon:.3f} | "
                f"Avg Score: {avg_score:5.2f} | Max Q: {avg_q:+6.3f} | Loss: {avg_loss:.4f}"
            )

        if step % EVAL_FREQ == 0:
            eval_mean, eval_max = evaluate(policy_net, eval_env, EVAL_EPISODES)
            marker = ""
            if eval_mean > best_eval:
                best_eval = eval_mean
                torch.save(policy_net.state_dict(), BEST_PATH)
                marker = " *"
            print(f"[Eval] Step: {step:7d} | Greedy Mean: {eval_mean:5.2f} | Max: {eval_max:3d}{marker}")

    torch.save(policy_net.state_dict(), FINAL_PATH)
    print(f"Training finished. Best greedy mean: {best_eval:.2f} ({BEST_PATH}), final weights: {FINAL_PATH}")


if __name__ == "__main__":
    train()