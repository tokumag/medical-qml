"""
Classical Neural Network Baselines
==================================
1. ClassicalCNN: Operates on full 28x28 (784-dimensional) chest X-ray images (~50,000 parameters).
2. ClassicalMLP: Operates on 8-dimensional PCA-reduced Euclidean features (~2,000 parameters).

These baselines define the performance benchmarks against which quantum Hilbert-space 
mappings are comparatively evaluated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> ReLU block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ClassicalCNN(nn.Module):
    """
    Full-resolution classical CNN benchmark for PneumoniaMNIST.
    Input : (B, 1, 28, 28)
    Output: (B, 2) logits [Normal, Pneumonia]
    """

    def __init__(self, dropout_p: float = 0.3):
        super().__init__()
        self.conv1 = ConvBlock(1, 32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = ConvBlock(64, 128)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.gap(self.conv3(x))
        return self.classifier(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts the 64-dimensional pre-classifier latent vector."""
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.gap(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.classifier[1](x))
        return x


class ClassicalMLP(nn.Module):
    """
    Shallow MLP baseline consuming 8-dimensional PCA features.
    Dimension matches n_qubits = 8 for direct Euclidean vs. Hilbert-space comparison.
    Input : (B, 8)
    Output: (B, 2) logits
    """

    def __init__(self, input_dim: int = 8, hidden: list = None, dropout_p: float = 0.2):
        super().__init__()
        if hidden is None:
            hidden = [64, 32]

        layers = []
        prev = input_dim
        for h in hidden:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p)
            ])
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
