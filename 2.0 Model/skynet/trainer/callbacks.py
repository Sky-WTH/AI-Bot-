"""Callback system for the Skynet training loop.

Callbacks observe training events and can trigger side-effects like
logging, checkpointing, and early stopping without modifying the core
training loop.
"""

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import torch.nn as nn

from skynet.utils.logger import SkynetLogger


class Callback(ABC):
    """Abstract base class for training callbacks.

    Override any of the hook methods to add custom behaviour. All hooks
    receive a ``state`` dict containing the current training context:
    ``model``, ``optimizer``, ``epoch``, ``loss``, ``metrics``, etc.
    """

    def on_train_begin(self, state: dict[str, Any]) -> None:
        """Called once at the start of training."""

    def on_train_end(self, state: dict[str, Any]) -> None:
        """Called once at the end of training."""

    def on_epoch_begin(self, state: dict[str, Any]) -> None:
        """Called at the start of each epoch."""

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        """Called at the end of each epoch."""

    def on_batch_begin(self, state: dict[str, Any]) -> None:
        """Called at the start of each batch."""

    def on_batch_end(self, state: dict[str, Any]) -> None:
        """Called at the end of each batch."""


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving.

    Args:
        patience: Number of epochs with no improvement before stopping.
        metric: Name of the metric to monitor (default: ``'val_loss'``).
        mode: ``'min'`` to stop when metric stops decreasing, ``'max'``
            when it stops increasing.
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(
        self,
        patience: int = 5,
        metric: str = "val_loss",
        mode: str = "min",
        min_delta: float = 0.0,
    ) -> None:
        super().__init__()
        self.patience = patience
        self.metric = metric
        self.min_delta = min_delta
        self.mode = mode

        self.best_value: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def _is_improvement(self, current: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return current < self.best_value - self.min_delta
        return current > self.best_value + self.min_delta

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        metrics = state.get("metrics", {})
        current = metrics.get(self.metric)

        if current is None:
            return

        if self._is_improvement(current):
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger = SkynetLogger(verbose=True)
                logger.warning(
                    f"Early stopping triggered — {self.metric} did not improve "
                    f"for {self.patience} epochs."
                )


class MetricsLogger(Callback):
    """Log training metrics to the console using Rich.

    Prints a one-line summary at the end of each epoch.
    """

    def __init__(self, verbose: bool = True) -> None:
        super().__init__()
        self.logger = SkynetLogger(verbose=verbose)
        self.history: list[dict[str, float]] = []

    def on_train_begin(self, state: dict[str, Any]) -> None:
        self.logger.banner()
        self.logger.info(
            f"Training on device: [bold]{state.get('device', 'cpu')}[/bold]"
        )
        self.logger.info(
            f"Architecture: [bold]{state.get('architecture', 'unknown')}[/bold]"
        )

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        epoch = state.get("epoch", 0)
        total = state.get("total_epochs", 0)
        metrics = state.get("metrics", {})

        record = {"epoch": epoch, **metrics}
        self.history.append(record)

        self.logger.epoch_summary(epoch, total, metrics)

    def on_train_end(self, state: dict[str, Any]) -> None:
        self.logger.success("Training complete!")
        if len(self.history) > 1:
            self.logger.metrics_table(self.history)


class CheckpointCallback(Callback):
    """Save checkpoints at regular intervals and track the best model.

    Args:
        checkpoint_dir: Directory to save checkpoints.
        every_n_epochs: Save every N epochs.
        save_best: If ``True``, also maintain a ``best.pt`` checkpoint
            based on validation loss.
    """

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        every_n_epochs: int = 1,
        save_best: bool = True,
    ) -> None:
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.every_n_epochs = every_n_epochs
        self.save_best = save_best
        self.best_loss: Optional[float] = None
        self.logger = SkynetLogger(verbose=True)

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        from skynet.trainer.checkpoint import build_training_state, save_checkpoint

        epoch = state.get("epoch", 0)
        model = state["model"]
        optimizer = state["optimizer"]
        loss = state.get("metrics", {}).get("train_loss", 0.0)
        scheduler = state.get("scheduler")
        scaler = state.get("scaler")

        # Periodic checkpoint
        if epoch % self.every_n_epochs == 0:
            ckpt_state = build_training_state(
                model, optimizer, epoch, loss, scheduler=scheduler, scaler=scaler
            )
            path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(ckpt_state, path)
            self.logger.info(f"Checkpoint saved: [dim]{path}[/dim]")

        # Best model
        if self.save_best:
            val_loss = state.get("metrics", {}).get("val_loss")
            compare_loss = val_loss if val_loss is not None else loss
            if self.best_loss is None or compare_loss < self.best_loss:
                self.best_loss = compare_loss
                ckpt_state = build_training_state(
                    model, optimizer, epoch, compare_loss,
                    scheduler=scheduler, scaler=scaler,
                )
                best_path = self.checkpoint_dir / "best.pt"
                save_checkpoint(ckpt_state, best_path)
                self.logger.info(f"Best model updated: [dim]{best_path}[/dim]")
