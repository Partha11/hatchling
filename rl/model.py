import torch
import torch.nn as nn


class DuelingCNN(nn.Module):
    def __init__(self, in_channels: int, height: int, width: int, num_actions: int = 3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, height, width)
            conv_out_size = self.conv(dummy).view(1, -1).size(1)

        self.val_stream = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.adv_stream = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        features = self.conv(x)
        features = features.view(features.size(0), -1)

        val = self.val_stream(features)
        adv = self.adv_stream(features)

        return val + (adv - adv.mean(dim=-1, keepdim=True))