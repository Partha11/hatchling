import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "build"))

import snake_cpp


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"]}

    _ROTATIONS = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}

    def __init__(self, width: int = 20, height: int = 15):
        super().__init__()
        self.width = width
        self.height = height
        self.engine = snake_cpp.SnakeEngine(width, height)

        self.view_size = 2 * max(width, height) - 1
        self.center = self.view_size // 2

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, self.view_size, self.view_size), dtype=np.uint8
        )

        self._action_map = [
            snake_cpp.Action.FORWARD,
            snake_cpp.Action.LEFT,
            snake_cpp.Action.RIGHT,
        ]

    def _egocentric(self, board: np.ndarray) -> np.ndarray:
        H, W = self.height, self.width
        S, c = self.view_size, self.center

        hy, hx = np.argwhere(board[0])[0]
        ny, nx = np.argwhere(board[1])[0]
        heading = (int(hy - ny), int(hx - nx))

        obs = np.zeros((3, S, S), dtype=np.uint8)
        obs[2] = 1

        y0, x0 = c - hy, c - hx
        obs[0, y0:y0 + H, x0:x0 + W] = board[2]
        obs[1, y0:y0 + H, x0:x0 + W] = board[3]
        obs[2, y0:y0 + H, x0:x0 + W] = 0

        k = self._ROTATIONS[heading]
        return np.ascontiguousarray(np.rot90(obs, k=k, axes=(1, 2)))

    def _observe(self) -> np.ndarray:
        return self._egocentric(np.asarray(self.engine.get_board()))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.reset()
        return self._observe(), {}

    def step(self, action: int):
        result = self.engine.step(self._action_map[action])

        obs = self._observe()
        reward = float(result.reward)
        terminated = bool(result.done)
        truncated = False
        info = {"score": result.score}

        return obs, reward, terminated, truncated, info