# Learning from Mistakes

The agent still made fatal mistakes in situations it rarely saw. Instead of changing the network again, I decided to show it those situations more often.

## Replaying the moment before a crash

The idea is called failure replay. During training, every game keeps a snapshot of the board every few moves. When a game ends in a crash, the snapshot from 20 to 50 moves earlier goes into a pool. New training games can then start from one of those positions. The food appears in a new random place, so each replay plays out differently, and if the agent crashes again, that position goes back into the pool too. It works like a player who keeps retrying the part of a level that keeps killing them.

## A mistake in my own design

The first version did almost nothing. After 1.6 million steps, only 75 games had started from the pool.

The problem was how I measured it. I had set a quarter of new games to start from a replay, but the agent's games now lasted thousands of moves, so new games were rare. All replays together added up to less than one percent of the training moves.

The fix was to reserve 8 of the 32 parallel games for replays. Each replay runs for at most 200 moves, and then the slot picks the next failure position. Now about a quarter of all training moves came from situations where the agent had once failed.

## The result

| | Best earlier model | With replay |
| --- | --- | --- |
| Average score | about 151 | 178.5 |
| Median score | 179 | 212 |
| Games below 100 | 28% | 18% |
| Best game | 240 | 256 |

/// caption
The earlier model is the last one measured over 200 games before the rule change. The replay agent continued from the model of [Changing the Rules](05-changing-the-rules.md).
///

The deaths changed character as well. A third of all games now ended at the wall, at an average score of 222, and most of those crashes were forced. The snake was boxed in on a nearly full board with nowhere left to go. That is a very different death from a careless blunder.

## Still a lot of help

The agent was strong now, but it had a lot of help. It saw the board rotated around its head, it got an extra input showing the order of its body, and it had an extra network output trained to spot danger. I started to wonder whether all of that was actually helping.
