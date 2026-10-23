import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + x)


def build_trunk(in_channels: int, channels: int, num_blocks: int, head_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        *[ResidualBlock(channels) for _ in range(num_blocks)],
        nn.Conv2d(channels, head_channels, kernel_size=1),
        nn.ReLU(),
    )


def trunk_output_size(trunk: nn.Module, in_channels: int, height: int, width: int) -> int:
    with torch.no_grad():
        dummy = torch.zeros(1, in_channels, height, width)
        return trunk(dummy).view(1, -1).size(1)


class DuelingCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        height: int,
        width: int,
        num_actions: int = 3,
        channels: int = 64,
        num_blocks: int = 3,
        head_channels: int = 32,
        hidden: int = 512,
    ):
        super().__init__()

        self.conv = build_trunk(in_channels, channels, num_blocks, head_channels)
        conv_out_size = trunk_output_size(self.conv, in_channels, height, width)

        self.val_stream = nn.Sequential(
            nn.Linear(conv_out_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        self.adv_stream = nn.Sequential(
            nn.Linear(conv_out_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        features = self.conv(x)
        features = features.view(features.size(0), -1)

        val = self.val_stream(features)
        adv = self.adv_stream(features)

        return val + (adv - adv.mean(dim=-1, keepdim=True))


class ActorCritic(nn.Module):
    def __init__(
        self,
        in_channels: int,
        height: int,
        width: int,
        num_actions: int = 3,
        channels: int = 64,
        num_blocks: int = 3,
        head_channels: int = 32,
        hidden: int = 512,
    ):
        super().__init__()

        self.conv = build_trunk(in_channels, channels, num_blocks, head_channels)
        conv_out_size = trunk_output_size(self.conv, in_channels, height, width)

        self.shared = nn.Sequential(
            nn.Linear(conv_out_size, hidden),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, num_actions)
        self.value = nn.Linear(hidden, 1)

        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.zeros_(self.policy.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    def forward(self, x: torch.Tensor):
        x = x.float() / 255.0
        features = self.conv(x)
        features = self.shared(features.view(features.size(0), -1))
        return self.policy(features), self.value(features).squeeze(-1)