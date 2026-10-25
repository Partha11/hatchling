# Dead Ends

Not every idea worked, and some made things clearly worse. These are the experiments that did not make it into the final agent.

## Different ways to show the body

The order of the body matters, because it tells you which parts clear first. I tried several ways to put that into the input. One version marked the cells closest to the tail most strongly. It reduced crashes right next to the tail but caused more crashes elsewhere, and the average fell to 132.4. A plain count of steps until each cell clears looked promising on paper. When I checked the actual values, it was almost identical to what the agent already had, so I dropped it before training.

## Marking the head and predicting danger

When the [confident blunders](04-confident-blunders.md) appeared, I marked the head as its own input layer, so the network would always know where it was. The blunders stayed exactly the same. An extra network output trained to predict fatal moves reached 99.7 percent accuracy, but it still missed the rare straight line shape, and the average stayed around 151.

## Practicing the endgame

After the [first wins](09-the-first-wins.md), I tried to teach the agent that winning was possible. During training, I saved positions from late in every game, whether that game was later won or lost. Six of the parallel games started from those positions and played on until the end.

From those positions, the agent won a lot, and training counted 4,918 wins. But everything else fell apart:

=== "Evaluation scores"

    Before the endgame practice started, evaluations reached 190 to 230. Afterwards, they collapsed to mostly between 40 and 130.

=== "Early loops"

    Loops at the start of the game came back. In some evaluations, up to 46 of 50 games timed out.

=== "Value estimates"

    The value loss, usually between 0.1 and 0.5, jumped to between 1 and 5. Training became unstable.

One network had to learn two very different worlds at once: games that end in a win after a few hundred moves, and games that take thousands of moves to get anywhere. It could not handle both.

## A model that wins more and plays worse

The best checkpoint from that run was saved just before the endgame practice started, so it was the 30 million step model plus 400,000 normal steps. It actually won more often, 26.5 percent of games instead of 21. But 16 percent of its games ended in an early loop, with the snake circling near the start and never going for the food, and its average fell to 233. I kept the 30 million step version.
