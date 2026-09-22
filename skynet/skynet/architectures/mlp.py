"""MLP (Multi-Layer Perceptron) architecture template.

Best suited for tabular / structured data or as a general-purpose baseline.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from skynet.architectures.registry import register_architecture


@register_architecture("mlp")
class MLP(nn.Module):
    """Configurable fully-connected feed-forward network.

    Args:
        input_dim: Number of input features.
        output_dim: Number of output classes / regression targets.
        hidden_dims: List of hidden layer sizes.  Defaults to ``[128, 64]``.
        activation: Activation function — 'relu', 'gelu', or 'silu'.
        dropout: Dropout probability applied after each hidden layer.
        batch_norm: Whether to apply batch normalization.
    """

    _ACTIVATIONS = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[list[int]] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        act_cls = self._ACTIVATIONS.get(activation.lower())
        if act_cls is None:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Choose from: {list(self._ACTIVATIONS.keys())}"
            )

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.

        Returns:
            Logits of shape ``(batch, output_dim)``.
        """
        return self.network(x)
