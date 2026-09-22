"""CNN (Convolutional Neural Network) architecture template.

Best suited for image classification / feature extraction.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from skynet.architectures.registry import register_architecture


@register_architecture("cnn")
class CNN(nn.Module):
    """Configurable convolutional neural network for image data.

    Builds a stack of Conv2d → BatchNorm → Activation → MaxPool blocks,
    followed by an adaptive average pool and a fully-connected classifier.

    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB).
        num_classes: Number of output classes.
        conv_channels: List of output channel counts for each conv block.
            Defaults to ``[32, 64, 128]``.
        kernel_size: Convolution kernel size (applied to all blocks).
        dropout: Dropout probability before the final linear layer.
        activation: Activation function — 'relu', 'gelu', or 'silu'.
    """

    _ACTIVATIONS = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        conv_channels: Optional[list[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        if conv_channels is None:
            conv_channels = [32, 64, 128]

        act_cls = self._ACTIVATIONS.get(activation.lower())
        if act_cls is None:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Choose from: {list(self._ACTIVATIONS.keys())}"
            )

        # Build convolutional feature extractor
        conv_layers: list[nn.Module] = []
        prev_ch = in_channels
        for ch in conv_channels:
            conv_layers.extend([
                nn.Conv2d(prev_ch, ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(ch),
                act_cls(),
                nn.MaxPool2d(2),
            ])
            prev_ch = ch

        self.features = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(conv_channels[-1], 256),
            act_cls(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, channels, height, width)``.

        Returns:
            Logits of shape ``(batch, num_classes)``.
        """
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
