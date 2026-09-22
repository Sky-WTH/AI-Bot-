"""Pydantic configuration models for Skynet.

Provides strict validation for training hyperparameters, data settings,
and export options. All configs are serializable to/from YAML.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


# ── Training Configuration ────────────────────────────────────────────


class TrainConfig(BaseModel):
    """Hyperparameters and settings for a training run.

    Attributes:
        architecture: Name of the architecture template (e.g. 'mlp', 'cnn').
        learning_rate: Optimizer learning rate.
        batch_size: Mini-batch size for training.
        epochs: Maximum number of training epochs.
        optimizer: Optimizer name — 'adam', 'sgd', or 'adamw'.
        loss_fn: Loss function name — 'cross_entropy', 'mse', or 'bce'.
        device: Compute device — 'auto', 'cpu', 'cuda', or 'mps'.
        checkpoint_dir: Directory to store checkpoints.
        checkpoint_every: Save a checkpoint every N epochs.
        mixed_precision: Enable AMP (automatic mixed precision) on CUDA.
        early_stopping_patience: Stop after N epochs with no improvement.
            Set to 0 to disable early stopping.
        weight_decay: L2 regularization coefficient.
        seed: Random seed for reproducibility. ``None`` for non-deterministic.
        arch_params: Extra keyword arguments forwarded to the architecture
            constructor (e.g. ``hidden_dims``, ``dropout``).
    """

    architecture: str = "mlp"
    learning_rate: float = Field(default=1e-3, gt=0)
    batch_size: int = Field(default=32, gt=0)
    epochs: int = Field(default=10, gt=0)
    optimizer: Literal["adam", "sgd", "adamw"] = "adam"
    loss_fn: Literal["cross_entropy", "mse", "bce"] = "cross_entropy"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    checkpoint_dir: str = "./checkpoints"
    checkpoint_every: int = Field(default=1, gt=0)
    mixed_precision: bool = False
    early_stopping_patience: int = Field(default=0, ge=0)
    weight_decay: float = Field(default=0.0, ge=0)
    seed: Optional[int] = None
    arch_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("architecture", mode="before")
    @classmethod
    def _lowercase_arch(cls, v: str) -> str:
        return v.strip().lower()

    # ── Serialization helpers ──────────────────────────────────────

    def to_yaml(self, path: str | Path) -> None:
        """Write this config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        """Load a config from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        """Write this config to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        """Load a config from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


# ── Data Configuration ─────────────────────────────────────────────


class DataConfig(BaseModel):
    """Settings for data loading and preprocessing.

    Attributes:
        data_type: The modality — 'text', 'image', or 'tabular'.
        data_path: Path to the data source (file or directory).
        target_column: For tabular data, the column to predict.
        image_size: Resize images to this square dimension.
        vocab_size: Maximum vocabulary size for text tokenization.
        max_seq_len: Maximum token sequence length for text.
        val_split: Fraction of data to use for validation.
        num_workers: Number of DataLoader worker processes.
        augment: Enable data augmentation (images only).
    """

    data_type: Literal["text", "image", "tabular"] = "tabular"
    data_path: str = "./data"
    target_column: Optional[str] = None
    image_size: int = Field(default=64, gt=0)
    vocab_size: int = Field(default=10000, gt=0)
    max_seq_len: int = Field(default=256, gt=0)
    val_split: float = Field(default=0.2, ge=0.0, le=1.0)
    num_workers: int = Field(default=0, ge=0)
    augment: bool = False

    @field_validator("data_type", mode="before")
    @classmethod
    def _lowercase_type(cls, v: str) -> str:
        return v.strip().lower()

    def to_yaml(self, path: str | Path) -> None:
        """Write this config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DataConfig":
        """Load a config from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)


# ── Export Configuration ───────────────────────────────────────────


class ExportConfig(BaseModel):
    """Settings for exporting trained model weights.

    Attributes:
        output_path: Destination path for the exported ``.pth`` file.
        include_optimizer: If ``True``, include optimizer state in the export.
        include_metadata: If ``True``, include training metadata.
    """

    output_path: str = "./model.pth"
    include_optimizer: bool = False
    include_metadata: bool = True


# ── Unified config loader ──────────────────────────────────────────


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a unified YAML config file that may contain ``train`` and ``data`` sections.

    Returns:
        A dictionary with keys ``'train'`` (TrainConfig) and ``'data'`` (DataConfig).
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    result: dict[str, Any] = {}
    if "train" in raw:
        result["train"] = TrainConfig(**raw["train"])
    else:
        result["train"] = TrainConfig(**{k: v for k, v in raw.items() if k != "data"})

    if "data" in raw:
        result["data"] = DataConfig(**raw["data"])

    return result
