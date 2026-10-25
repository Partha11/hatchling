# Like a Child

By now the agent had a lot of help: a rotated view, a body order input and a danger detector. I started to ask a different question. What if I took all of it away?

## How a child learns Snake

Picture a five year old with a Snake game. At first, they move around aimlessly. They hit a wall and the game ends, so they learn not to do that. They touch the food and the score goes up, so they learn to go for it. Much later, they start to plan and learn not to trap themselves. Nobody explains any of this to them. They see the screen, press the arrow keys and watch the score. So why was I giving my agent so much more?

## Taking the help away

I built a new mode and called it child mode. In child mode, the agent gets only what a player gets.

=== "The screen"

    No more rotating the board around the snake's head. The agent sees the fixed board with a wall around it, drawn as six simple layers: head, neck, body, tail, food and wall. Each cell is either one of those things or empty.

=== "The arrow keys"

    Up, down, left and right, just like the terminal game. Pressing the direction you came from does nothing, also just like the game.

=== "The score"

    The same rewards as before: food, death and a tiny cost per step. No body order input, no danger prediction and no rotated view.

The only thing I kept was failure replay. A child remembers the spot where it died last time and tries again, so that felt fair.

The neck and the tail get their own layers for a reason. A single still picture does not show which way the snake is moving or which end is the tail. With the neck visible, the direction follows from one frame, because the next cell continues the line from neck to head. With the tail visible, the agent knows which piece of body clears first. A child figures both out by watching the snake move, and the agent only ever gets one still picture.

## Slow, then steady

Child mode learned more slowly, as expected, because it had to figure out everything the old setup had handed to it. I trained it in blocks of 5 million steps and resumed each time from where it stopped.

| Training | Training average | Best evaluation |
| --- | --- | --- |
| 5 million steps | 67 | 73.5 |
| 10 million steps | 139 | 188.9 |
| 15 million steps | 148 | 210.1 |

At 15 million steps, the 200 game diagnostic gave an average of 192.8. That was better than every earlier model, including the one with all the help.

## A strategy nobody taught it

Then I watched it play. The snake moved across the board in tidy columns: up, over, down, over, up again. It filled the board in a careful sweep, keeping the free space in one piece and avoiding little holes where food could get stuck.

Nobody told it to do that. It only ever saw the screen, pressed the arrow keys and felt the score. The sweep was the strategy that kept it alive, so that is what it learned.
