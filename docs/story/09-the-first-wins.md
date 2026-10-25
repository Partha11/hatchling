# The First Wins

The snake starts with 3 cells and the board has 300, so a score of 297 means one thing: the snake covers every cell. The board is full and the game is won.

## The board fills up

The first wins showed up in the evaluation logs as a quiet line, `Max: 297`, and then again, and again. After another 5 million steps of training, the diagnostic counted 39 wins in 200 games. After 5 million more, the numbers settled here:

| | |
| --- | --- |
| Average score | 267.1 |
| Median score | 289 |
| Games won | 42 of 200 (21%) |
| Games below 100 | 6% |

Nine out of ten games reached at least 232 points, and one in five filled the entire board. Then I watched it win three games in a row.

```text
+----------------------------------------+
|o o o o o o o o o o o o o o o o o o o o |
|o o o o o o o o o o o o @ o o o o o o o |
|o o o o o o o o o o o o o o o o o o o o |
|o o o o o o o o o o o o o o o o o o o o |
|o o o o o o o o o o o o o o o o o o o o |
|o o o o o o o o o o o o o o o o o o o o |
|o o o o o o o o o o o o o o o o o o o o |
|o o o o o o o o o o o o o o o o o o o o |
|o o o o o o o o o o o o o o o o o o o o |
|o o o o o o o o o o o o o o o o o o o o |
|o o o o o o o o o o o o o o o o o o o o |
|o o o o o o o o o o o o o o o o o o o o |
|o o o o o o o o o o o o o o o o o o o o |
|o o o o o o o o o o o o o o o o o o o o |
|o o o o o o o o o o o o o o o o o o o o |
+----------------------------------------+
Score: 297

Board filled - the snake won! Final Score: 297
```

## Winning while expecting to lose

Those wins had one strange detail. Right before the winning move, the agent's own value estimates were -39.6, -29.4 and -50.4. It was one move away from a full board, and it was sure it was about to die.

I think I know why. Failure replay only collects positions from games that ended in a crash, so almost every nearly full board the agent ever practiced on came from a game it lost. In its experience, a crowded board meant death. It won anyway, because its moves were good, but it never learned that winning was possible.

## How the other games end

The games it loses look very different from the early blunders. Most of them end on a nearly full board, like this one with a 292 cell snake:

```text
+----------------------------------------+
|o o o o o o o o o o o o o o o o o o o o |
|o o o o o o o o o o o o o o o o o o o o |
|o o o o o o o o o o o o o o o o o o o o |
|o o o o o o o o o o o o o o o o o o o o |
|o o o o o o o o o o o o o o o o o o o o |
|o o o o o o o o o o o   o o o o o o o o |
|o o o o o o o o o o o   o o o o o o o o |
|o o o o o o o o o o o   o o o o o o o o |
|o o o o o o o o o o o   o o o o o o o o |
|o o o o o o o o o o o   o o o o o o o o |
|o o o o o o o o o o o o o o o o o o o o |
|o o o o o o o o o o o o   o o o o o o o |
|o o o o o o o o o o o o @ o o o o o o o |
|o o o o o o o o o o o   * o o o o o o o |
|o o o o o o o o o o o o o o o o o o o o |
+----------------------------------------+
```

Only 8 cells were free, and they were split into separate pockets: a column of 5 in the middle and a few single cells sealed in by body. Once the free space breaks into pieces like this, the board can never be filled. The real mistake happened long before, when the sweep left that column behind.

That is the hardest part of Snake. A perfect player fills the board along a single path that visits every cell, and the agent gets there one game in five purely by learning.
