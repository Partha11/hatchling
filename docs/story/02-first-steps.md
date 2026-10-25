# First Steps

The first agent used DQN, short for Deep Q-Network. It is one of the classic methods in reinforcement learning and a natural place to start.

## How it learns

For every situation, the network guesses how good each possible move is. It picks the move it thinks is best, sees what actually happened and corrects its guess. The agent had three moves: go straight, turn left or turn right. It got a reward for eating, a penalty for dying and a tiny cost for every step.

At the start it knows nothing, and its moves are random. Slowly the corrections pile up and patterns appear. Food is good. Walls are bad.

## The first bug

The early results were strange. Games ended at odd moments, and learning was slower than it should have been. The cause was a bug in the step counter. The engine counts how many moves the snake makes without eating and ends the game if that goes on too long. That counter was not working as intended, and it was distorting how games ended. Once it was fixed, the agent started to improve properly.

## Seeing the board from the snake's eyes

The biggest early jump did not come from the learning method. It came from how the agent saw the board.

At first, the agent saw the board as a fixed map, like a player looking at the screen. Then I rotated and shifted the view so that the snake's head was always in the center, always facing up. From the snake's point of view, "left" now always meant the same thing, and a wall ahead always looked the same, no matter where on the board the snake was.

At the same point in training, the score went from about 1 to about 7. That view stayed with the project for a long time. Much later, it would be questioned again.

## Where DQN ended

After the fixes, the DQN agent averaged around 55 to 60 points. It learned to chase food and avoid walls, but it kept dying as it grew. A long snake fills the board with its own body, and the agent had no real plan for that. It would eat, grow and eventually trap itself.
