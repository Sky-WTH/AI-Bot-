"""
Skynet — Train personalized neural networks from scratch, entirely on-device.

Skynet gives any user a simple SDK and CLI to select an architecture template,
feed in their own datasets, and train a model locally on their own hardware.
Training is checkpointed and fully user-controlled.

Quick Start (SDK):
    >>> from skynet import Trainer, TrainConfig, DataPipeline
    >>> config = TrainConfig(architecture="mlp", epochs=10)
    >>> model = config.build_model(input_dim=784, output_dim=10)
    >>> trainer = Trainer(model, config)
    >>> trainer.fit(train_loader)
    >>> trainer.export("my_model.pth")

Quick Start (CLI):
    $ skynet train --config config.yaml --data ./data/
    $ skynet export --checkpoint ./checkpoints/best.pt --output model.pth
"""

__version__ = "0.1.0"

from skynet.config import TrainConfig, DataConfig, ExportConfig
from skynet.architectures.registry import get_architecture, list_architectures
from skynet.data.loader import DataPipeline
from skynet.trainer.engine import Trainer
from skynet.trainer.checkpoint import save_checkpoint, load_checkpoint, export_weights
from skynet.trainer.callbacks import (
    Callback,
    EarlyStopping,
    MetricsLogger,
    CheckpointCallback,
)
from skynet.utils.device import get_best_device, device_info

__all__ = [
    # Config
    "TrainConfig",
    "DataConfig",
    "ExportConfig",
    # Architectures
    "get_architecture",
    "list_architectures",
    # Data
    "DataPipeline",
    # Training
    "Trainer",
    "save_checkpoint",
    "load_checkpoint",
    "export_weights",
    # Callbacks
    "Callback",
    "EarlyStopping",
    "MetricsLogger",
    "CheckpointCallback",
    # Utils
    "get_best_device",
    "device_info",
]
