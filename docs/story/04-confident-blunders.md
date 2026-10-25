# Confident Blunders

With the diagnostic tool in place, I could finally see how the agent died. Most deaths were crashes into the snake's own body, which was not surprising. How they happened was.

## Sure, and wrong

The agent was not trapped. There was usually free space right next to it. And it was not unsure either, because it chose the fatal move with high confidence, often 60 to 100 percent. Its own value estimate did not see it coming. Right before crashing, it typically rated its position as good.

About three out of four of these crashes hit a part of the body that was nowhere near clearing. It was solid body, in plain view, directly in front of the head.

## Asking the agent what it sees

To find out why, I gave the network an extra task. Next to choosing a move, it had to predict which of its three moves would kill it right away. The engine knows the correct answer for every move, so this could be trained directly. The network got very good at it and caught 99.7 percent of fatal moves during training. And yet, in the games where it crashed, the prediction failed exactly on the fatal move. In this one, body was directly ahead, the agent was 95 percent sure about going forward, and it rated the chance of dying there at half a percent:

```text
Last move options: Forward: body (191 steps from clearing) (chosen) | Left: 63 free cells | Right: 4 free cells
Policy before last move: Forward: 0.955 | Left: 0.045 | Right: 0.000
Predicted danger before last move: Forward: 0.005 | Left: 0.000 | Right: 0.079
```

## The same shape, again and again

When I put several of these crashes side by side, the same picture kept appearing. The head sat inside a straight line of its own body, with the neck directly behind it and another piece of body directly in front of it on the same line. In that one shape, the agent read the cell ahead as safe. It happened five times in a row in the games I watched.

I marked the head explicitly in the input, so the network always knew where it was. The blunder stayed, and the danger prediction did not remove it either.

## Rare, not invisible

So I ran a separate test outside of normal training. I collected hundreds of thousands of positions and trained a network only to spot fatal moves, with no rewards and no game. It learned almost perfectly and missed 6 fatal moves out of 85,018. When I showed it the straight line shape often enough, it got every case right.

The network could see the danger. The problem was how rarely the shape appeared:

=== "Random play"

    In 150,000 positions from random games with long snakes, the straight line shape appeared **39 times**.

=== "The agent's own games"

    In 200,000 positions from the agent's own games, it appeared **about 7 times**.

A situation that almost never comes up in training never gets learned. Giving the agent more practice with exactly those situations is what [Learning from Mistakes](06-learning-from-mistakes.md) is about.
