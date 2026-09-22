"""Checkpoint management — save, load, export, and list checkpoints.

Checkpoints include the full training state (model, optimizer, scheduler,
epoch, loss, RNG states) so training can be resumed exactly where it stopped.
"""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn


def save_checkpoint(
    state: dict[str, Any],
    path: str | Path,
) -> Path:
    """Atomically save a checkpoint dictionary to disk.

    Uses a temp file + rename strategy to prevent corruption if the process
    is interrupted mid-write.

    Args:
        state: Dictionary containing model/optimizer state dicts, epoch, etc.
        path: Destination file path.

    Returns:
        The resolved :class:`Path` of the saved checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp file in the same directory, then atomic rename
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.close(fd)
        torch.save(state, tmp_path)
        # On Windows, target must not exist for rename
        if path.exists():
            path.unlink()
        Path(tmp_path).rename(path)
    except Exception:
        # Clean up temp file on failure
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()
        raise

    return path


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint from disk.

    Args:
        path: Path to the checkpoint file.
        device: Device to map tensors to.

    Returns:
        The checkpoint dictionary.

    Raises:
        FileNotFoundError: If the checkpoint file doesn't exist.
        ValueError: If the file doesn't look like a valid Skynet checkpoint.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Expected checkpoint to be a dict, got {type(checkpoint).__name__}. "
            "This may not be a valid Skynet checkpoint."
        )

    return checkpoint


def export_weights(
    model: nn.Module,
    path: str | Path,
    metadata: Optional[dict[str, Any]] = None,
) -> Path:
    """Export only the model weights as a clean ``.pth`` file.

    This creates a portable file containing just ``model.state_dict()``
    (and optional metadata) — no optimizer or training state.

    Args:
        model: The trained model.
        path: Destination ``.pth`` file path.
        metadata: Optional metadata dict to include (e.g. architecture name,
            training config, input dimensions).

    Returns:
        The resolved :class:`Path` of the exported file.
    """
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
    }
    if metadata:
        payload["metadata"] = metadata

    return save_checkpoint(payload, path)


def list_checkpoints(directory: str | Path) -> list[dict[str, Any]]:
    """List all checkpoint files in a directory, sorted by epoch.

    Args:
        directory: Path to the checkpoints directory.

    Returns:
        List of dicts with keys ``'path'``, ``'epoch'``, ``'loss'``
        for each discovered checkpoint.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []

    results: list[dict[str, Any]] = []
    for f in sorted(directory.glob("*.pt")):
        try:
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                results.append({
                    "path": str(f),
                    "epoch": ckpt.get("epoch", -1),
                    "loss": ckpt.get("loss", float("nan")),
                })
        except Exception:
            continue

    results.sort(key=lambda r: r["epoch"])
    return results


def build_training_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a comprehensive checkpoint state dictionary.

    Captures all information needed to resume training exactly:
    model weights, optimizer state, scheduler state, RNG states, etc.

    Args:
        model: The model being trained.
        optimizer: The optimizer.
        epoch: Current epoch number.
        loss: Current loss value.
        scheduler: Optional learning rate scheduler.
        scaler: Optional AMP GradScaler.
        extra: Any extra metadata to include.

    Returns:
        A dictionary ready to be passed to :func:`save_checkpoint`.
    """
    state: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        # RNG states for reproducibility
        "rng_state_python": random.getstate(),
        "rng_state_numpy": np.random.get_state(),
        "rng_state_torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["rng_state_cuda"] = torch.cuda.get_rng_state_all()

    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()

    if extra:
        state.update(extra)

    return state


def restore_rng_states(checkpoint: dict[str, Any]) -> None:
    """Restore random number generator states from a checkpoint.

    Args:
        checkpoint: A loaded checkpoint dictionary.
    """
    if "rng_state_python" in checkpoint:
        random.setstate(checkpoint["rng_state_python"])
    if "rng_state_numpy" in checkpoint:
        np.random.set_state(checkpoint["rng_state_numpy"])
    if "rng_state_torch" in checkpoint:
        torch.set_rng_state(checkpoint["rng_state_torch"])
    if "rng_state_cuda" in checkpoint and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["rng_state_cuda"])
