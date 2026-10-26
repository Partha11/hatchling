# Where It Stands

The final agent is the child mode model after 30 million steps of training. It sees the board, presses the arrow keys and learns from the score, with failure replay and a death penalty of -50.

## The final model

| | |
| --- | --- |
| Average score | 267.1 out of 297 |
| Median score | 289 |
| Games won | 21% |
| Games below 100 | 6% |
| Training | 30 million steps |

The trained weights are available on [Google Drive](https://drive.google.com/file/d/1LhTzv9P-ccbPSU1H_k5CRBCBj3OpRa9v/view?usp=sharing), so you can [watch it play](../agent/setup.md) yourself. Most games end with the board almost completely full, and about one in five ends with every cell filled.

## What could come next

The agent still loses most games in the last few percent of the board, usually because the free space has split into pockets it cannot fill. A few directions look promising.

=== "Starvation as a loss"

    A snake that goes too long without food would lose, exactly like a crash. Right now, a game that hits the step limit simply ends without a penalty. Making it a loss would remove any incentive to circle forever and should end the early game loops for good.

=== "Gentler endgame practice"

    The endgame practice in [Dead Ends](10-what-did-not-work.md) used six of 32 parallel games. Two would give the agent some practice at winning without destabilizing the rest of the game.

=== "Value estimates near the end"

    The agent still expects to lose right before it wins. Any change that lets it experience more wins during normal training should help it recognize winnable positions and play toward them.

The code, the trained weights and these notes are all open. If you take the hatchling further, I would love to see where it goes.
