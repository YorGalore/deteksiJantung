"""ResNet-1D baseline for 12-lead ECG binary AMI classification."""

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """Two 1D convolutions with a residual connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=7, stride=stride, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=5, stride=1, padding=2, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)

        return out


class ResNet1D(nn.Module):
    """Baseline ResNet-1D for ECG input shaped (batch, 12, 1000)."""

    def __init__(self, in_channels: int = 12, num_classes: int = 1):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels, 32,
                kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(32, 32, blocks=2, stride=1)
        self.layer2 = self._make_layer(32, 64, blocks=2, stride=2)
        self.layer3 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer4 = self._make_layer(128, 256, blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(256, num_classes)

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = [
            ResidualBlock1D(in_channels, out_channels, stride=stride)
        ]

        for _ in range(1, blocks):
            layers.append(
                ResidualBlock1D(out_channels, out_channels)
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 12:
            raise ValueError(
                f"Expected input (batch, 12, time), got {tuple(x.shape)}"
            )

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)

        return self.classifier(x)


def resnet1d() -> ResNet1D:
    """Create the default binary AMI classifier."""
    return ResNet1D(in_channels=12, num_classes=1)
