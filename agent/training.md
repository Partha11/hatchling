# Training

Training runs `rl/train.py`. It uses PPO with 32 games in parallel and saves its progress as it goes, so long runs can be split into blocks and resumed.

## Train the final agent

This is the setup of the final agent: the fixed view with arrow keys, and 8 of the 32 parallel games reserved for failure replay.

```bash
python rl/train.py --tag child --view fixed --replay-envs 8
```

A run of 5 million steps takes a few hours on a mid-range GPU. The final agent trained for 30 million steps in six such blocks, each resumed from the end of the previous one:

```bash
python rl/train.py --tag child --view fixed --replay-envs 8 --resume snake_ppo_child.pt --steps 5000000
```

The first 15 million steps used a death penalty of -10. For the last 15 million, `DEATH_REWARD` in `include/snake/engine.hpp` was set to -50 and the module rebuilt. See [Rules and rewards](../game/rules.md).

!!! warning "Resuming overwrites the best checkpoint"
    A resumed run with the same tag starts its own record and overwrites `snake_ppo_<tag>_best.pt` at its first evaluation. Rename or copy your best file before resuming if you want to keep it.

## Files

| File | Content |
| --- | --- |
| `snake_ppo_<tag>_best.pt` | The checkpoint with the best evaluation so far |
| `snake_ppo_<tag>.pt` | The final weights at the end of the run |

Every checkpoint stores the settings it was trained with, such as the view and the extra channels. The other scripts read them from the file, so you never have to pass them again.

## What the log shows

Every 8,192 steps, training prints a line like this:

```text
Step:  4505600 | Avg Score: 129.47 | Entropy: 0.361 | Value Loss: 0.0579 | KL: 0.0012 | Clip: 0.010 | Replay Pool: 2417 | Replays: 7350 | Replay Share: 24.6%
```

| Column | Meaning |
| --- | --- |
| Avg Score | Average score of recent normal training games. Training games pick moves with some randomness, so this is lower than the evaluation score |
| Entropy | How random the agent's choices still are |
| Value Loss | How far the value estimates are from the actual results. Sudden jumps can signal unstable training |
| KL and Clip | How much the policy changed in the last update |
| Replay Pool | Failure positions collected so far |
| Replay Share | Share of all training moves that came from failure replays |
| Wins | Training games that filled the whole board. Shown by the current version of the script |

Every 100,000 steps, an evaluation line follows. It plays 50 games without randomness:

```text
[Eval] Step:  4505600 | Greedy Mean: 188.90 | Max: 225 | Timeouts: 0/50 *
```

The star marks a new best, which is saved as `snake_ppo_<tag>_best.pt`. Evaluation games are only for measurement and do not train the agent.

## Options

=== "Common"

    | Option | Default | Meaning |
    | --- | --- | --- |
    | `--tag` | | Name used in the checkpoint file names |
    | `--view` | `ego` | `fixed` for the board as a player sees it, `ego` for the snake centered view |
    | `--steps` | 5000000 | Total training steps of this run |
    | `--resume` | | Checkpoint to continue from |
    | `--lr` | 0.00025 | Learning rate. Resumed runs start at 0.0001 |
    | `--gamma` | 0.998 | Discount, how far ahead the agent looks |
    | `--lam` | 0.97 | GAE lambda |
    | `--rollout` | 256 | Moves per game before each update |

=== "Failure replay"

    | Option | Default | Meaning |
    | --- | --- | --- |
    | `--replay-envs` | 0 | Parallel games reserved for failure replay |
    | `--replay-horizon` | 200 | Maximum moves per replay |
    | `--replay-back-min` | 20 | Earliest snapshot before a crash |
    | `--replay-back-max` | 50 | Latest snapshot before a crash |
    | `--replay-pool` | 5000 | Number of failure positions kept |
    | `--replay` | 0.0 | Chance that a normal game starts from a replay instead |

=== "Endgame practice"

    | Option | Default | Meaning |
    | --- | --- | --- |
    | `--endgame-envs` | 0 | Parallel games that start from late game positions |
    | `--endgame-length` | 250 | Snake length from which positions are saved |
    | `--endgame-every` | 25 | Moves between saved positions |
    | `--endgame-pool` | 2000 | Number of late game positions kept |

    This made the agent worse in testing. See [Dead Ends](../story/10-what-did-not-work.md) before using it.

=== "Snake centered view"

    | Option | Default | Meaning |
    | --- | --- | --- |
    | `--order` | `relative` | Body order encoding, `relative` or `inverse` |
    | `--head-channel` | off | Adds a channel marking the head |
    | `--tail-channel` | off | Adds a channel marking the last 64 cells of the tail |
    | `--danger-coef` | 0.0 | Weight of the extra danger prediction task |
    | `--long-start` | 0.0 | Share of games that start with a long snake |
    | `--long-min` | 100 | Shortest snake for those games |
    | `--long-max` | 250 | Longest snake for those games |

    These options only apply to the snake centered view and are not used by the final agent.
