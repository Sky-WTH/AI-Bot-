"""Autoencoder architecture template.

Symmetric encoder–decoder MLP for unsupervised representation learning,
dimensionality reduction, and anomaly detection.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from skynet.architectures.registry import register_architecture


@register_architecture("autoencoder")
class Autoencoder(nn.Module):
    """Symmetric autoencoder with configurable depth.

    The encoder compresses ``input_dim`` → ``latent_dim`` through a series
    of hidden layers.  The decoder mirrors the encoder to reconstruct the
    input.  Use MSE loss for training.

    Args:
        input_dim: Dimensionality of the input features.
        latent_dim: Dimensionality of the bottleneck (latent space).
        hidden_dims: Sizes of hidden layers in the encoder (decoder mirrors).
            Defaults to ``[128, 64]``.
        activation: Activation function — 'relu', 'gelu', or 'silu'.
        dropout: Dropout probability.
    """

    _ACTIVATIONS = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dims: Optional[list[int]] = None,
        activation: str = "relu",
        dropout: float = 0.0,
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

        # ── Encoder ────────────────────────────────────────────────
        enc_layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            enc_layers.append(nn.Linear(prev_dim, h_dim))
            enc_layers.append(act_cls())
            if dropout > 0:
                enc_layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        enc_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # ── Decoder (mirror) ───────────────────────────────────────
        dec_layers: list[nn.Module] = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            dec_layers.append(nn.Linear(prev_dim, h_dim))
            dec_layers.append(act_cls())
            if dropout > 0:
                dec_layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        dec_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input into the latent space.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.

        Returns:
            Latent representation of shape ``(batch, latent_dim)``.
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to input space.

        Args:
            z: Latent tensor of shape ``(batch, latent_dim)``.

        Returns:
            Reconstruction of shape ``(batch, input_dim)``.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode — full reconstruction pass.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.

        Returns:
            Reconstructed tensor of shape ``(batch, input_dim)``.
        """
        return self.decode(self.encode(x))
