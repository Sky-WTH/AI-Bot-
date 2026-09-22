"""Core training engine for Skynet.

Provides the :class:`Trainer` class — the central SDK entry point that
manages the training loop, checkpointing, device placement, mixed
precision, and callback dispatch.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from skynet.config import TrainConfig
from skynet.trainer.callbacks import (
    Callback,
    CheckpointCallback,
    EarlyStopping,
    MetricsLogger,
)
from skynet.trainer.checkpoint import (
    build_training_state,
    export_weights,
    load_checkpoint,
    restore_rng_states,
    save_checkpoint,
)
from skynet.utils.device import get_best_device
from skynet.utils.logger import SkynetLogger


class Trainer:
    """High-level training controller.

    Wraps a PyTorch model and manages the full training lifecycle:
    fitting, evaluation, checkpointing, resumption, and export.

    Args:
        model: An ``nn.Module`` to train.
        config: A :class:`TrainConfig` with hyperparameters.
        callbacks: Optional list of :class:`Callback` instances.  If
            ``None``, default callbacks (MetricsLogger, CheckpointCallback,
            and optionally EarlyStopping) are added automatically.

    Usage::

        trainer = Trainer(model, config)
        trainer.fit(train_loader, val_loader)
        trainer.export("model.pth")
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        callbacks: Optional[list[Callback]] = None,
    ) -> None:
        self.config = config
        self.logger = SkynetLogger()

        # ── Device ─────────────────────────────────────────────────
        self.device = get_best_device(config.device)
        self.model = model.to(self.device)

        # ── Optimizer ──────────────────────────────────────────────
        self.optimizer = self._build_optimizer()

        # ── Loss function ──────────────────────────────────────────
        self.criterion = self._build_criterion()

        # ── Learning rate scheduler ────────────────────────────────
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, verbose=False
        )

        # ── Mixed precision ────────────────────────────────────────
        self.use_amp = config.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # ── Callbacks ──────────────────────────────────────────────
        if callbacks is not None:
            self.callbacks = list(callbacks)
        else:
            self.callbacks = self._default_callbacks()

        # ── Training state ─────────────────────────────────────────
        self.start_epoch = 1
        self.history: list[dict[str, float]] = []

        # ── Seed ───────────────────────────────────────────────────
        if config.seed is not None:
            self._set_seed(config.seed)

    # ==================================================================
    #  PUBLIC API
    # ==================================================================

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> list[dict[str, float]]:
        """Run the training loop.

        Args:
            train_loader: DataLoader for training data.
            val_loader: Optional DataLoader for validation data.

        Returns:
            List of per-epoch metric dictionaries.
        """
        state = self._make_state()
        self._fire("on_train_begin", state)

        try:
            for epoch in range(self.start_epoch, self.config.epochs + 1):
                state["epoch"] = epoch

                self._fire("on_epoch_begin", state)

                # ── Train one epoch ────────────────────────────────
                train_loss, train_acc = self._train_epoch(train_loader, epoch)

                metrics: dict[str, float] = {
                    "train_loss": train_loss,
                }
                if train_acc is not None:
                    metrics["train_acc"] = train_acc

                # ── Validate ───────────────────────────────────────
                if val_loader is not None:
                    val_loss, val_acc = self._evaluate_epoch(val_loader)
                    metrics["val_loss"] = val_loss
                    if val_acc is not None:
                        metrics["val_acc"] = val_acc

                self.history.append(metrics)
                state["metrics"] = metrics

                # Step LR scheduler
                step_metric = metrics.get("val_loss", train_loss)
                self.scheduler.step(step_metric)

                self._fire("on_epoch_end", state)

                # Check early stopping
                if self._should_stop():
                    break

        except KeyboardInterrupt:
            self.logger.warning(
                "Training interrupted by user. Saving emergency checkpoint…"
            )
            emergency_state = build_training_state(
                self.model,
                self.optimizer,
                state.get("epoch", 0),
                self.history[-1].get("train_loss", 0.0) if self.history else 0.0,
                scheduler=self.scheduler,
                scaler=self.scaler,
            )
            emergency_path = Path(self.config.checkpoint_dir) / "interrupted.pt"
            save_checkpoint(emergency_state, emergency_path)
            self.logger.success(f"Emergency checkpoint saved: {emergency_path}")

        self._fire("on_train_end", state)
        return self.history

    def evaluate(self, data_loader: DataLoader) -> dict[str, float]:
        """Evaluate the model on a dataset.

        Args:
            data_loader: DataLoader to evaluate on.

        Returns:
            Dictionary with ``'loss'`` and optionally ``'accuracy'``.
        """
        loss, acc = self._evaluate_epoch(data_loader)
        result: dict[str, float] = {"loss": loss}
        if acc is not None:
            result["accuracy"] = acc
        return result

    def export(self, path: str | Path, metadata: Optional[dict] = None) -> Path:
        """Export only the model weights to a ``.pth`` file.

        Args:
            path: Destination file path.
            metadata: Optional metadata to include.

        Returns:
            Path to the exported file.
        """
        meta = {
            "architecture": self.config.architecture,
            "arch_params": self.config.arch_params,
            **(metadata or {}),
        }
        result_path = export_weights(self.model, path, metadata=meta)
        self.logger.success(f"Model exported to: {result_path}")
        return result_path

    def resume(self, checkpoint_path: str | Path) -> None:
        """Resume training from a checkpoint.

        Restores model weights, optimizer state, scheduler state, epoch
        counter, and RNG states.

        Args:
            checkpoint_path: Path to the checkpoint file.
        """
        checkpoint = load_checkpoint(checkpoint_path, device=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.start_epoch = checkpoint.get("epoch", 0) + 1
        restore_rng_states(checkpoint)

        self.logger.success(
            f"Resumed from epoch {checkpoint.get('epoch', '?')} "
            f"(loss: {checkpoint.get('loss', '?'):.4f})"
        )

    # ==================================================================
    #  TRAINING LOOP INTERNALS
    # ==================================================================

    def _train_epoch(
        self, loader: DataLoader, epoch: int
    ) -> tuple[float, Optional[float]]:
        """Train for one epoch.

        Returns:
            Tuple of (average_loss, accuracy_or_None).
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        is_classification = self.config.loss_fn in ("cross_entropy", "bce")

        progress = self.logger.training_progress(len(loader))
        with progress:
            task = progress.add_task(
                f"Epoch {epoch}/{self.config.epochs}", total=len(loader)
            )
            state = self._make_state()

            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                state["batch_idx"] = batch_idx
                self._fire("on_batch_begin", state)

                self.optimizer.zero_grad()

                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                    self.scaler.scale(loss).backward()  # type: ignore[union-attr]
                    self.scaler.step(self.optimizer)  # type: ignore[union-attr]
                    self.scaler.update()  # type: ignore[union-attr]
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

                if is_classification:
                    preds = outputs.argmax(dim=-1)
                    correct += (preds == targets).sum().item()

                state["batch_loss"] = loss.item()
                self._fire("on_batch_end", state)
                progress.advance(task)

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = correct / max(total_samples, 1) if is_classification else None
        return avg_loss, accuracy

    @torch.no_grad()
    def _evaluate_epoch(
        self, loader: DataLoader
    ) -> tuple[float, Optional[float]]:
        """Evaluate the model for one epoch.

        Returns:
            Tuple of (average_loss, accuracy_or_None).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        is_classification = self.config.loss_fn in ("cross_entropy", "bce")

        for inputs, targets in loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            if is_classification:
                preds = outputs.argmax(dim=-1)
                correct += (preds == targets).sum().item()

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = correct / max(total_samples, 1) if is_classification else None
        return avg_loss, accuracy

    # ==================================================================
    #  HELPERS
    # ==================================================================

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Construct the optimizer from config."""
        name = self.config.optimizer.lower()
        params = self.model.parameters()
        lr = self.config.learning_rate
        wd = self.config.weight_decay

        if name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=wd)
        if name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        if name == "sgd":
            return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)

        raise ValueError(f"Unknown optimizer: '{name}'. Choose from: adam, adamw, sgd.")

    def _build_criterion(self) -> nn.Module:
        """Construct the loss function from config."""
        name = self.config.loss_fn.lower()
        if name == "cross_entropy":
            return nn.CrossEntropyLoss()
        if name == "mse":
            return nn.MSELoss()
        if name == "bce":
            return nn.BCEWithLogitsLoss()

        raise ValueError(
            f"Unknown loss function: '{name}'. Choose from: cross_entropy, mse, bce."
        )

    def _default_callbacks(self) -> list[Callback]:
        """Build the default callback stack."""
        cbs: list[Callback] = [
            MetricsLogger(),
            CheckpointCallback(
                checkpoint_dir=self.config.checkpoint_dir,
                every_n_epochs=self.config.checkpoint_every,
            ),
        ]
        if self.config.early_stopping_patience > 0:
            cbs.append(
                EarlyStopping(patience=self.config.early_stopping_patience)
            )
        return cbs

    def _fire(self, hook: str, state: dict[str, Any]) -> None:
        """Dispatch a callback hook to all registered callbacks."""
        for cb in self.callbacks:
            getattr(cb, hook)(state)

    def _should_stop(self) -> bool:
        """Check if any EarlyStopping callback wants to stop."""
        for cb in self.callbacks:
            if isinstance(cb, EarlyStopping) and cb.should_stop:
                return True
        return False

    def _make_state(self) -> dict[str, Any]:
        """Build the shared state dict passed to callbacks."""
        return {
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "scaler": self.scaler,
            "device": str(self.device),
            "architecture": self.config.architecture,
            "total_epochs": self.config.epochs,
            "epoch": self.start_epoch,
            "metrics": {},
        }

    @staticmethod
    def _set_seed(seed: int) -> None:
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
