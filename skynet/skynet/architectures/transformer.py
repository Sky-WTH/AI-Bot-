"""Lightweight Transformer encoder architecture template.

Best suited for text classification and sequence modelling tasks.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from skynet.architectures.registry import register_architecture


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Tensor of shape ``(batch, seq_len, d_model)``.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


@register_architecture("transformer")
class TransformerClassifier(nn.Module):
    """Lightweight Transformer encoder for classification.

    Embeds tokens, applies positional encoding, passes through N encoder
    layers, then mean-pools and classifies.

    Args:
        vocab_size: Size of the token vocabulary.
        d_model: Embedding / hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        num_classes: Number of output classes.
        max_seq_len: Maximum sequence length.
        dim_feedforward: Feed-forward dimension inside each encoder layer.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 2,
        max_seq_len: int = 256,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization for embeddings and linear layers."""
        nn.init.xavier_uniform_(self.embedding.weight)
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Token IDs of shape ``(batch, seq_len)``.
            src_key_padding_mask: Boolean mask where ``True`` indicates
                padding positions.

        Returns:
            Logits of shape ``(batch, num_classes)``.
        """
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Mean-pool over the sequence dimension (ignoring padding)
        if src_key_padding_mask is not None:
            mask = ~src_key_padding_mask  # True for real tokens
            mask = mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        return self.classifier(x)
