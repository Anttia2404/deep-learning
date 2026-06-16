from __future__ import annotations

import torch
import torch.nn as nn


class MesoNet(nn.Module):
    def __init__(self, num_classes: int = 1, input_size: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 8, kernel_size=5, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(16, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),
        )
        flattened = 16 * (input_size // 64) * (input_size // 64)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(flattened, 16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        branch_channels = out_channels // 4
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, branch_channels, 1), nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, 1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x),
        ], dim=1)
        return self.bn(self.pool(out))


class MesoInception4(nn.Module):
    def __init__(self, num_classes: int = 1, input_size: int = 256) -> None:
        super().__init__()
        self.inception1 = InceptionBlock(3, 8)
        self.inception2 = InceptionBlock(8, 16)
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),
        )
        flattened = 16 * (input_size // 64) * (input_size // 64)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(flattened, 16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.classifier(x)
