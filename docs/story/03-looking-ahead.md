# Looking Ahead

To get past the DQN agent, I switched to PPO, short for Proximal Policy Optimization. Everything that follows in this story is built on it.

## A different way to learn

Instead of scoring each move, PPO learns a policy directly: in this situation, how likely should each move be? A second output estimates how good the current position is, which is called the value. Training runs 32 games in parallel. The agent plays 256 moves in each, learns from all of them at once and then plays again. The first PPO agent averaged about 107 points, already a big step up from DQN.

## The discount

Every reinforcement learning agent has a setting called the discount, usually written as gamma. It decides how much the agent cares about the future. A reward right now counts fully, a reward one move later counts a little less, and a reward a thousand moves later counts almost nothing.

!!! note "How far ahead an agent looks"
    A discount of 0.995 means the agent mostly cares about the next 200 moves or so. With 0.998, that grows to roughly 500 moves.

The first PPO agent used 0.995. That sounds close to 1, but for Snake it is short. A long snake can trap itself with a decision made long before the trap closes. I raised the discount to 0.998 and adjusted a related setting, lambda, to 0.97.

The average jumped from 107 to 150.5. No new network and no new tricks, just an agent that cared about a longer future. It became the reference model for a long time, and I called it foresight.

## Variations

Once foresight worked, I tried a few variations around it. Each one changed a single thing and was measured over 200 games.

| Experiment | Average score |
| --- | --- |
| Foresight (gamma 0.998) | 150.5 |
| Longer rollouts (512 moves) | 137.9 |
| Starting some games with a long snake | 153.9 |
| A bigger death penalty for longer snakes | 157.4 |

Some helped a little and one made things worse. None came close to the jump from the discount.

## Measuring properly

Around this time, I built a diagnostic tool. It plays 200 games and reports how each one ended: wall or body, and whether the crash was forced with every move blocked, or whether a free move was left.

A single game can mislead you badly. Once, a new model lost its first two games at 38 and 56 points and looked worse than the old one. Over 200 games, it was the best model so far. From here on, every result in this story comes from a 200 game diagnostic, and the diagnostics were about to show something odd.
