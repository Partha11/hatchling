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

    def __init__(self, width: int = 20, height: int = 15, gamma: float = 0.99, view_size: int = 39):
        super().__init__()
        self.width = width
        self.height = height
        self.engine = snake_cpp.SnakeEngine(width, height, gamma)

        if view_size < 2 * max(width, height) - 1:
            raise ValueError(f"view_size {view_size} is too small for a {width}x{height} board")
        if view_size % 2 == 0:
            raise ValueError("view_size must be odd so the head sits in the center")
        self.view_size = view_size
        self.center = self.view_size // 2

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(4, self.view_size, self.view_size), dtype=np.uint8
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

        obs = np.zeros((4, S, S), dtype=np.uint8)
        obs[3] = 255

        y0, x0 = c - hy, c - hx
        obs[0, y0:y0 + H, x0:x0 + W] = board[2]
        obs[1, y0:y0 + H, x0:x0 + W] = board[3]
        obs[2, y0:y0 + H, x0:x0 + W] = board[4]
        obs[3, y0:y0 + H, x0:x0 + W] = 0

        k = self._ROTATIONS[heading]
        return np.ascontiguousarray(np.rot90(obs, k=k, axes=(1, 2)))

    def _observe(self) -> np.ndarray:
        return self._egocentric(np.asarray(self.engine.get_board()))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        length = (options or {}).get("length")
        if length:
            self.engine.reset_with_length(int(length))
        else:
            self.engine.reset()
        return self._observe(), {}

    def step(self, action: int):
        result = self.engine.step(self._action_map[action])

        obs = self._observe()
        reward = float(result.reward)
        terminated = bool(result.done)
        truncated = bool(result.truncated)
        info = {"score": result.score}

        return obs, reward, terminated, truncated, info