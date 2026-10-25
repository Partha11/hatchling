import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "build"))

import snake_cpp


TAIL_WINDOW = 64
TAIL_VALUES = (255 - 4 * np.arange(TAIL_WINDOW)).astype(np.uint8)


FIXED_CHANNELS = 6
FIXED_NAMES = ["head", "neck", "body", "tail", "food", "wall"]
ABSOLUTE_NAMES = ["Up", "Right", "Down", "Left"]


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"]}

    _ROTATIONS = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}

    def __init__(
        self,
        width: int = 20,
        height: int = 15,
        gamma: float = 0.99,
        view_size: int = 39,
        order_encoding: str = "relative",
        head_channel: bool = False,
        tail_channel: bool = False,
        view: str = "ego",
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.engine = snake_cpp.SnakeEngine(width, height, gamma)
        encodings = {
            "relative": snake_cpp.OrderEncoding.RELATIVE,
            "inverse": snake_cpp.OrderEncoding.INVERSE,
        }
        if order_encoding not in encodings:
            raise ValueError(f"unknown order_encoding {order_encoding!r}")
        self.order_encoding = order_encoding
        self.engine.order_encoding = encodings[order_encoding]

        if view_size < 2 * max(width, height) - 1:
            raise ValueError(f"view_size {view_size} is too small for a {width}x{height} board")
        if view_size % 2 == 0:
            raise ValueError("view_size must be odd so the head sits in the center")
        if view not in ("ego", "fixed"):
            raise ValueError(f"unknown view {view!r}")
        if view == "fixed" and (head_channel or tail_channel):
            raise ValueError("head_channel and tail_channel belong to the ego view")
        self.view = view
        self.last_relative_action = 0
        self.view_size = view_size
        self.head_channel = head_channel
        self.tail_channel = tail_channel
        self.num_channels = 4 + int(head_channel) + int(tail_channel)
        self.tail_index = 4 + int(head_channel)
        self.center = self.view_size // 2

        if view == "fixed":
            self.num_channels = FIXED_CHANNELS
            self.action_space = spaces.Discrete(4)
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(FIXED_CHANNELS, height + 2, width + 2), dtype=np.uint8
            )
        else:
            self.action_space = spaces.Discrete(3)
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.num_channels, self.view_size, self.view_size), dtype=np.uint8
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

        obs = np.zeros((self.num_channels, S, S), dtype=np.uint8)
        obs[3] = 255

        y0, x0 = c - hy, c - hx
        obs[0, y0:y0 + H, x0:x0 + W] = board[2]
        obs[1, y0:y0 + H, x0:x0 + W] = board[3]
        obs[2, y0:y0 + H, x0:x0 + W] = board[4]
        obs[3, y0:y0 + H, x0:x0 + W] = 0
        if self.head_channel:
            obs[4, c, c] = 255
        if self.tail_channel:
            tail = self.engine.get_tail_segments(TAIL_WINDOW)
            if len(tail):
                obs[self.tail_index, y0 + tail[:, 1], x0 + tail[:, 0]] = TAIL_VALUES[:len(tail)]

        k = self._ROTATIONS[heading]
        return np.ascontiguousarray(np.rot90(obs, k=k, axes=(1, 2)))

    def _fixed(self, board: np.ndarray) -> np.ndarray:
        H, W = self.height, self.width
        obs = np.zeros((FIXED_CHANNELS, H + 2, W + 2), dtype=np.uint8)
        inner = (slice(1, H + 1), slice(1, W + 1))

        obs[(0,) + inner] = board[0]
        obs[(1,) + inner] = board[1]
        body = board[2] > 0
        tail = self.engine.get_tail_segments(1)
        tail_mask = np.zeros((H, W), dtype=bool)
        if len(tail):
            tail_mask[tail[0, 1], tail[0, 0]] = True
        middle = body & ~(board[1] > 0) & ~tail_mask
        obs[(2,) + inner] = middle.astype(np.uint8) * 255
        obs[(3,) + inner] = tail_mask.astype(np.uint8) * 255
        obs[(4,) + inner] = board[4]
        obs[5] = 255
        obs[(5,) + inner] = 0
        return obs

    def _relative_from_absolute(self, absolute: int) -> int:
        diff = (int(absolute) - self.engine.get_direction()) % 4
        return {0: 0, 1: 2, 2: 0, 3: 1}[diff]

    def snapshot(self):
        return self.engine.clone()

    def restore(self, snapshot) -> np.ndarray:
        self.engine.restore(snapshot)
        return self._observe()

    def danger(self) -> np.ndarray:
        return np.array(self.engine.get_danger(), dtype=np.float32)

    def _observe(self) -> np.ndarray:
        board = np.asarray(self.engine.get_board())
        if self.view == "fixed":
            return self._fixed(board)
        return self._egocentric(board)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        length = (options or {}).get("length")
        if length:
            self.engine.reset_with_length(int(length))
        else:
            self.engine.reset()
        return self._observe(), {}

    def step(self, action: int):
        relative = self._relative_from_absolute(action) if self.view == "fixed" else int(action)
        self.last_relative_action = relative
        result = self.engine.step(self._action_map[relative])

        obs = self._observe()
        reward = float(result.reward)
        terminated = bool(result.done)
        truncated = bool(result.truncated)
        info = {"score": result.score}

        return obs, reward, terminated, truncated, info