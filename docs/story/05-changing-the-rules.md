# Changing the Rules

Over time, the game had picked up a special rule. The head was allowed to move into the cell where the tail currently was, because on the same move the tail moves away. Many Snake games work this way, and it sounds harmless.

## An exception nobody needed

For a learning agent, it created a contradiction. Most of the time, body in front of the head means death. But if that piece of body happens to be the tail, it means nothing. The agent had to tell those two cases apart, and the only hint was a tiny difference in the input.

I first suspected the agent had learned a shortcut from this. It often chased its own tail in a straight line, and in that situation the cell ahead was usually the tail. Maybe it had concluded that body straight ahead is often fine. That turned out not to explain the [confident blunders](04-confident-blunders.md), because the straight line blunder survived the rule change. But the exception was still a needless complication.

## A simpler rule

I wrote the rule the way it should have been all along:

| Move into | Reward |
| --- | --- |
| An empty cell | -0.01 |
| Food | +10 |
| Anything else | -10, and the game ends |

The tail is body, and body is fatal. I also removed the extra penalty for dying as a long snake, so every crash now costs the same.

## What happened

I trained a new agent from scratch under the new rules. They are stricter, so it struggled briefly at first. Then it overtook every earlier model. The training average climbed to 130 to 138, against about 105 for foresight. In evaluation, the best checkpoint averaged 180 points, and the best single game reached 244.

Something else disappeared too. Earlier models sometimes got stuck in endless loops at the start of a game. Under the new rules, those loops were gone in almost every evaluation.
