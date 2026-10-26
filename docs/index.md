# Hatchling

This is the story of teaching a computer to play Snake. Not by writing rules for it, but by letting it play, fail and learn, millions of times over.

A hatchling is a snake fresh out of the egg. It knows nothing about the world yet, and everything it does, it has to learn. That is exactly where this agent starts.

## How it started

Snake is simple. You move a snake around a grid, eat food and grow. Hit a wall or your own body, and the game is over. The longer you survive, the harder it gets, because your own body fills the board.

I wanted to see if an AI could learn to play it well. Not just chase food for a while, but fill the whole board. So I built everything from scratch: a game engine in C++, a terminal version you can play yourself on Linux or Windows, Python bindings so an agent could play millions of games, and the agent itself.

The first agent was not very good. It scored around 55 points on average and kept running into its own body.

## Where it ended

The final agent plays on a 20 by 15 board. It sees what a player sees, the board, the snake and the food, and it presses the arrow keys. It learns from the score and nothing else.

It averages 267 points out of a possible 297. In about one game out of five, it fills every single cell of the board and wins. Nobody taught it how. It worked out its own strategy: a tidy back and forth sweep that keeps the board open until the very end.

/// echarts
    renderer: "svg"
    attrs:
        style: "width:100%;height:380px;"

const option = {
  tooltip: { trigger: "axis" },
  grid: { left: 50, right: 20, top: 30, bottom: 90 },
  xAxis: {
    type: "category",
    axisLabel: { rotate: 35 },
    data: ["DQN", "First PPO", "Foresight", "Length penalty", "Failure replay", "Child mode", "Death -50", "25M steps", "30M steps"]
  },
  yAxis: { type: "value", name: "Average score", max: 300 },
  series: [{
    type: "bar",
    data: [57, 107, 150.5, 157.4, 178.5, 192.8, 224.9, 247.5, 267.1],
    label: { show: true, position: "top" },
    itemStyle: { color: "#0891b2", borderRadius: [4, 4, 0, 0] }
  }]
};
///

/// caption
Average score over 200 games for each milestone. The DQN value is approximate. The maximum possible score is 297.
///

## The chapters

The story is told in eleven chapters. Some ideas worked and many did not, and the failures are part of the story too. You can read them in order or jump to the part that interests you.

| Chapter | What happens |
| --- | --- |
| [The Beginning](story/01-building-the-game.md) | The engine, the terminal game and the bridge to Python. |
| [First Steps](story/02-first-steps.md) | An agent that learned to eat, and then kept dying. |
| [Looking Ahead](story/03-looking-ahead.md) | Switching to PPO, and the setting that changed everything. |
| [Confident Blunders](story/04-confident-blunders.md) | Crashing into its own body while being sure it was safe. |
| [Changing the Rules](story/05-changing-the-rules.md) | A simpler game, and a better player. |
| [Learning from Mistakes](story/06-learning-from-mistakes.md) | Replaying failures until they stop happening. |
| [Like a Child](story/07-like-a-child.md) | Taking away every shortcut. |
| [Raising the Stakes](story/08-raising-the-stakes.md) | Making death expensive, and a button that did nothing. |
| [The First Wins](story/09-the-first-wins.md) | Filling the board, while expecting to lose. |
| [Dead Ends](story/10-what-did-not-work.md) | The experiments that made things worse. |
| [Where It Stands](story/11-where-it-stands.md) | The final model, and what could come next. |

## Try it yourself

You can [play the game](game/playing.md) yourself, [watch the trained agent](agent/setup.md) play, or [train your own](agent/training.md). Everything you need is in the repository, and the trained weights can be downloaded from [Google Drive](https://drive.google.com/file/d/1LhTzv9P-ccbPSU1H_k5CRBCBj3OpRa9v/view?usp=sharing) or GitHub [releases](https://github.com/Partha11/hatchling/releases).
