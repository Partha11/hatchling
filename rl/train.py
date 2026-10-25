import argparse
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from checkpoint import load_checkpoint, save_checkpoint
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


@dataclass
class ReplayConfig:
    prob: float = 0.0
    back_min: int = 20
    back_max: int = 50
    every: int = 5
    pool_size: int = 5000
    envs: int = 0
    horizon: int = 200


class FailureReplay:
    def __init__(self, cfg: ReplayConfig, n_envs: int):
        self.cfg = cfg
        self.pool = deque(maxlen=cfg.pool_size)
        history_len = cfg.back_max // cfg.every + 2
        self.histories = [deque(maxlen=history_len) for _ in range(n_envs)]
        self.steps = [0] * n_envs
        self.added = 0
        self.used = 0

    @property
    def enabled(self):
        return self.cfg.prob > 0.0 or self.cfg.envs > 0

    def start(self, i, env):
        if not self.enabled:
            return
        self.steps[i] = 0
        self.histories[i].clear()
        self.histories[i].append((0, env.snapshot()))

    def record(self, i, env):
        if not self.enabled:
            return
        self.steps[i] += 1
        if self.steps[i] % self.cfg.every == 0:
            self.histories[i].append((self.steps[i], env.snapshot()))

    def crashed(self, i):
        if not self.enabled:
            return
        now = self.steps[i]
        candidates = [snap for t, snap in self.histories[i]
                      if self.cfg.back_min <= now - t <= self.cfg.back_max]
        if candidates:
            self.pool.append(random.choice(candidates))
            self.added += 1

    def pick(self, force=False):
        if not self.enabled or not self.pool:
            return None
        if not force and random.random() >= self.cfg.prob:
            return None
        self.used += 1
        return random.choice(self.pool)


@dataclass
class EndgameConfig:
    envs: int = 0
    min_length: int = 250
    every: int = 25
    pool_size: int = 2000


class EndgamePool:
    def __init__(self, cfg: EndgameConfig, n_envs: int):
        self.cfg = cfg
        self.pool = deque(maxlen=cfg.pool_size)
        self.moves = [0] * n_envs
        self.used = 0

    @property
    def enabled(self):
        return self.cfg.envs > 0

    def start(self, i):
        self.moves[i] = 0

    def record(self, i, env, score):
        if not self.enabled:
            return
        self.moves[i] += 1
        if score + 3 >= self.cfg.min_length and self.moves[i] % self.cfg.every == 0:
            self.pool.append(env.snapshot())

    def pick(self):
        if not self.enabled or not self.pool:
            return None
        self.used += 1
        return random.choice(self.pool)


def reset_env(env: SnakeEnv, long_start_prob: float, replay: FailureReplay, i: int, replay_slot: bool = False,
              endgame: "EndgamePool" = None, endgame_slot: bool = False):
    snapshot = endgame.pick() if (endgame is not None and endgame_slot) else None
    if snapshot is not None:
        state = env.restore(snapshot)
        if endgame is not None:
            endgame.start(i)
        replay.start(i, env)
        return state, "endgame"
    snapshot = replay.pick(force=replay_slot)
    if endgame is not None:
        endgame.start(i)
    if snapshot is not None:
        state = env.restore(snapshot)
        kind = "replay"
    elif long_start_prob > 0.0 and random.random() < long_start_prob:
        length = random.randint(LONG_RANGE[0], LONG_RANGE[1])
        state, _ = env.reset(options={"length": length})
        kind = "long"
    else:
        state, _ = env.reset()
        kind = "normal"
    replay.start(i, env)
    return state, kind


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


def train(
    gamma, gae_lambda, rollout_len, long_start_prob, tag, resume_path, lr, total_steps, order, danger_coef,
    head_channel, replay_cfg, tail_channel=False, view="ego", endgame_cfg=None,
):
    endgame_cfg = endgame_cfg or EndgameConfig()
    if replay_cfg.envs + endgame_cfg.envs >= N_ENVS:
        raise ValueError("replay and endgame slots must leave at least one normal game")
    best_path, latest_path, final_path = checkpoint_paths(tag)

    if view == "fixed" and danger_coef > 0.0:
        raise ValueError("the danger head is only available with the ego view")
    env_kwargs = dict(order_encoding=order, head_channel=head_channel, tail_channel=tail_channel, view=view)
    envs = [SnakeEnv(WIDTH, HEIGHT, gamma, **env_kwargs) for _ in range(N_ENVS)]
    eval_envs = [SnakeEnv(WIDTH, HEIGHT, gamma, **env_kwargs) for _ in range(EVAL_EPISODES)]
    use_danger = danger_coef > 0.0
    meta = {
        "order": order, "danger_head": use_danger, "head_channel": head_channel,
        "tail_channel": tail_channel, "view": view,
    }

    obs_shape = envs[0].observation_space.shape
    num_actions = envs[0].action_space.n
    model = ActorCritic(*obs_shape, num_actions=num_actions, danger_head=use_danger).to(device)
    if resume_path:
        state_dict, resume_meta = load_checkpoint(resume_path, device)
        if resume_meta.get("order", "relative") != order:
            raise ValueError(
                f"{resume_path} was trained with order encoding {resume_meta.get('order')!r}, not {order!r}"
            )
        if resume_meta.get("view", "ego") != view:
            raise ValueError(f"{resume_path} was trained with view={resume_meta.get('view', 'ego')!r}")
        if resume_meta.get("tail_channel", False) != tail_channel:
            raise ValueError(f"{resume_path} was trained with tail_channel={resume_meta.get('tail_channel', False)}")
        if resume_meta.get("head_channel", False) != head_channel:
            raise ValueError(f"{resume_path} was trained with head_channel={resume_meta.get('head_channel', False)}")
        same_heads = resume_meta.get("danger_head", False) == use_danger
        model.load_state_dict(state_dict, strict=same_heads)
        if not same_heads:
            print(f"Note: danger head {'added to' if use_danger else 'dropped from'} {resume_path}")
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    print(
        f"view={view} order={order} head_channel={head_channel} tail_channel={tail_channel} danger_coef={danger_coef} gamma={gamma} lambda={gae_lambda} rollout={rollout_len} long_start={long_start_prob} "
        f"long_range={LONG_RANGE[0]}-{LONG_RANGE[1]} "
        f"replay={replay_cfg.prob} replay_envs={replay_cfg.envs} horizon={replay_cfg.horizon} "
        f"endgame_envs={endgame_cfg.envs} endgame_length={endgame_cfg.min_length} "
        f"back={replay_cfg.back_min}-{replay_cfg.back_max} "
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
    dang_buf = torch.zeros((rollout_len, N_ENVS, 3), device=device)

    replay = FailureReplay(replay_cfg, N_ENVS)
    replay_slots = set(range(N_ENVS - min(replay_cfg.envs, N_ENVS), N_ENVS))
    endgame = EndgamePool(endgame_cfg, N_ENVS)
    endgame_first = N_ENVS - replay_cfg.envs - endgame_cfg.envs
    endgame_slots = set(range(endgame_first, N_ENVS - replay_cfg.envs))
    replay_steps = 0
    wins = 0
    states = []
    start_kinds = []
    for i, env in enumerate(envs):
        state, kind = reset_env(env, long_start_prob, replay, i, i in replay_slots, endgame, i in endgame_slots)
        states.append(state)
        start_kinds.append(kind)
    states = np.stack(states)
    dangers = np.stack([env.danger() for env in envs]) if use_danger else None

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
            if use_danger:
                dang_buf[t] = torch.from_numpy(dangers).to(device)
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
                if start_kinds[i] == "replay":
                    replay_steps += 1
                    if (i in replay_slots and not (terminated or truncated)
                            and replay.steps[i] + 1 >= replay_cfg.horizon):
                        truncated = True

                if terminated or truncated:
                    dones[i] = 1.0
                    if terminated and reward > 0:
                        wins += 1
                    if terminated and reward < 0:
                        replay.crashed(i)
                    if start_kinds[i] == "long":
                        long_start_games += 1
                    elif start_kinds[i] == "normal":
                        scores.append(info["score"])
                    if truncated and not terminated:
                        truncated_idx.append(i)
                        truncated_obs.append(next_state)
                    next_state, start_kinds[i] = reset_env(
                        env, long_start_prob, replay, i, i in replay_slots, endgame, i in endgame_slots)
                else:
                    replay.record(i, env)
                    endgame.record(i, env, info["score"])

                states[i] = next_state
                if use_danger:
                    dangers[i] = env.danger()

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
        b_dang = dang_buf.reshape(batch_size, 3)

        v_losses, entropies, kls, clipfracs = [], [], [], []
        d_losses, d_recalls = [], []

        for _ in range(EPOCHS):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, MINIBATCH_SIZE):
                idx = perm[start:start + MINIBATCH_SIZE]

                if use_danger:
                    logits, values, danger_logits = model(b_obs[idx], return_danger=True)
                else:
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
                if use_danger:
                    labels = b_dang[idx]
                    d_loss = F.binary_cross_entropy_with_logits(danger_logits, labels)
                    loss = loss + danger_coef * d_loss
                    d_losses.append(d_loss.item())
                    fatal = labels > 0.5
                    if fatal.any():
                        d_recalls.append(((danger_logits > 0) & fatal).sum().item() / fatal.sum().item())

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
        if use_danger:
            line += f" | Danger Loss: {np.mean(d_losses):.4f} | Danger Recall: {np.mean(d_recalls):.3f}"
        if long_start_prob > 0.0:
            line += f" | Long Starts: {long_start_games}"
        if replay.enabled:
            line += (
                f" | Replay Pool: {len(replay.pool)} | Replays: {replay.used} | "
                f"Replay Share: {replay_steps / max(steps, 1):.1%}"
            )
        if endgame.enabled:
            line += f" | Endgame Pool: {len(endgame.pool)} | Endgame Starts: {endgame.used}"
        line += f" | Wins: {wins}"
        print(line)

        if steps >= next_eval:
            next_eval += EVAL_FREQ
            eval_mean, eval_max, eval_timeouts = evaluate(model, eval_envs)
            save_checkpoint(latest_path, model, meta)
            marker = ""
            if eval_mean > best_eval:
                best_eval = eval_mean
                save_checkpoint(best_path, model, meta)
                marker = " *"
            print(
                f"[Eval] Step: {steps:8d} | Greedy Mean: {eval_mean:6.2f} | Max: {eval_max:3d} | "
                f"Timeouts: {eval_timeouts}/{EVAL_EPISODES}{marker}"
            )

    save_checkpoint(final_path, model, meta)
    print(f"Training finished. Best greedy mean: {best_eval:.2f} ({best_path}), final weights: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--lam", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--rollout", type=int, default=DEFAULT_ROLLOUT)
    parser.add_argument("--order", choices=["relative", "inverse"], default="relative")
    parser.add_argument("--danger-coef", type=float, default=0.0)
    parser.add_argument("--head-channel", action="store_true")
    parser.add_argument("--tail-channel", action="store_true")
    parser.add_argument("--view", choices=["ego", "fixed"], default="ego")
    parser.add_argument("--long-start", type=float, default=DEFAULT_LONG_START)
    parser.add_argument("--long-min", type=int, default=DEFAULT_LONG_MIN)
    parser.add_argument("--long-max", type=int, default=DEFAULT_LONG_MAX)
    parser.add_argument("--replay", type=float, default=0.0)
    parser.add_argument("--replay-back-min", type=int, default=20)
    parser.add_argument("--replay-back-max", type=int, default=50)
    parser.add_argument("--replay-pool", type=int, default=5000)
    parser.add_argument("--replay-envs", type=int, default=0)
    parser.add_argument("--replay-horizon", type=int, default=200)
    parser.add_argument("--endgame-envs", type=int, default=0)
    parser.add_argument("--endgame-length", type=int, default=250)
    parser.add_argument("--endgame-every", type=int, default=25)
    parser.add_argument("--endgame-pool", type=int, default=2000)
    parser.add_argument("--tag", default="")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    args = parser.parse_args()

    LONG_RANGE[0], LONG_RANGE[1] = args.long_min, args.long_max
    lr = args.lr if args.lr is not None else (RESUME_LR if args.resume else LR)
    train(
        args.gamma, args.lam, args.rollout, args.long_start, args.tag, args.resume, lr, args.steps, args.order,
        args.danger_coef, args.head_channel,
        ReplayConfig(
            args.replay, args.replay_back_min, args.replay_back_max, pool_size=args.replay_pool,
            envs=args.replay_envs, horizon=args.replay_horizon,
        ),
        args.tail_channel, args.view,
        EndgameConfig(args.endgame_envs, args.endgame_length, args.endgame_every, args.endgame_pool),
    )