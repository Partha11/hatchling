# Raising the Stakes

Child mode played well, but late in the game it often died in situations that were not actually lost. The problem turned out to be in the rewards, not in the agent.

## Giving up too early

In one game, the snake had 246 cells and turned into its own body. It could have gone into a free corridor instead, and as it moved through, its tail would have cleared a way out. The agent did not see that, and the rewards gave it little reason to look.

Crashing right away cost -10. Surviving 18 more moves and then crashing cost a few hundredths in step costs plus a -10 that counts slightly less because it comes later. The difference was about 0.17 points. To the agent, staying alive a bit longer was worth almost nothing, so it rarely tried, and it rarely found out that some of those positions could be saved.

## Making death expensive

I raised the death penalty from -10 to -50 and left food at +10 and the step cost at -0.01. The same comparison was now worth about 1.7 points, ten times more. I continued training the child mode agent with the new penalty for another 5 million steps. Afterwards, half of all games ended with the snake covering at least 87 percent of the board.

| | Death penalty -10 | Death penalty -50 |
| --- | --- | --- |
| Average score | 192.8 | 224.9 |
| Median score | 220 | 262 |
| Games below 100 | 15% | 12% |
| Best game | 265 | 287 |

## A button that did nothing

Two games in a row ended the same way. The snake was moving down and the agent pressed Up. Pressing the reverse direction does nothing in Snake, so the snake kept going down, straight into its own body.

The diagnostic made it look serious. The agent pressed the reverse key in 29 percent of its moves, and 58 percent of its crashes came right after a reverse key press. So I tested it in two ways.

=== "Normal play"

    Average score **224.9**. The agent uses the reverse key as it learned to.

=== "Reverse key blocked"

    Average score **203.9**. Without the key, the agent falls back to its second choice, which is often a turn it never meant to make.

=== "Reverse merged into forward"

    Average score **225.2**. Forward and reverse do the same thing, so their probabilities are added together and treated as one move.

Merging the two keys changed nothing. The agent simply used the reverse key as a normal "keep going" key, and those crashes would have happened with the forward key too. Meanwhile, the merged test had produced a game of 291, just six cells short of a full board.
