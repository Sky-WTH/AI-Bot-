"""Tests for the training engine, checkpointing, and callbacks."""

import csv
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Ensure architectures are registered
import skynet.architectures.mlp  # noqa: F401

from skynet.architectures.registry import get_architecture
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
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from skynet.trainer.engine import Trainer


# ── Helpers ────────────────────────────────────────────────────────


def make_synthetic_loader(
    input_dim: int = 20,
    output_dim: int = 3,
    n_samples: int = 100,
    batch_size: int = 16,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader with random data for testing."""
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, output_dim, (n_samples,))
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ── Checkpoint tests ──────────────────────────────────────────────


class TestCheckpoint:

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        state = {"epoch": 5, "loss": 0.42, "data": torch.randn(3, 3)}
        path = tmp_path / "test_ckpt.pt"
        save_checkpoint(state, path)
        loaded = load_checkpoint(path)
        assert loaded["epoch"] == 5
        assert abs(loaded["loss"] - 0.42) < 1e-6
        assert torch.allclose(loaded["data"], state["data"])

    def test_load_nonexistent_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "nonexistent.pt")

    def test_export_weights(self, tmp_path: Path):
        model = get_architecture("mlp", input_dim=10, output_dim=2)
        path = tmp_path / "exported.pth"
        export_weights(model, path, metadata={"arch": "mlp"})

        loaded = load_checkpoint(path)
        assert "model_state_dict" in loaded
        assert loaded["metadata"]["arch"] == "mlp"

    def test_build_training_state(self):
        model = get_architecture("mlp", input_dim=10, output_dim=2)
        optimizer = torch.optim.Adam(model.parameters())
        state = build_training_state(model, optimizer, epoch=3, loss=0.5)

        assert state["epoch"] == 3
        assert "model_state_dict" in state
        assert "optimizer_state_dict" in state
        assert "rng_state_torch" in state

    def test_list_checkpoints(self, tmp_path: Path):
        # Create some checkpoint files
        for epoch in [1, 5, 10]:
            save_checkpoint(
                {"epoch": epoch, "loss": 1.0 / epoch},
                tmp_path / f"epoch_{epoch:04d}.pt",
            )

        results = list_checkpoints(tmp_path)
        assert len(results) == 3
        assert results[0]["epoch"] == 1
        assert results[-1]["epoch"] == 10

    def test_atomic_save_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "nested" / "deep" / "ckpt.pt"
        save_checkpoint({"test": True}, path)
        assert path.exists()


# ── Callback tests ────────────────────────────────────────────────


class TestCallbacks:

    def test_early_stopping_triggers(self):
        es = EarlyStopping(patience=3, metric="val_loss", mode="min")

        # Simulate non-improving epochs
        for i in range(5):
            es.on_epoch_end({"metrics": {"val_loss": 1.0}})

        assert es.should_stop is True

    def test_early_stopping_resets_on_improvement(self):
        es = EarlyStopping(patience=3, metric="val_loss", mode="min")

        es.on_epoch_end({"metrics": {"val_loss": 1.0}})
        es.on_epoch_end({"metrics": {"val_loss": 1.0}})
        es.on_epoch_end({"metrics": {"val_loss": 0.5}})  # Improvement!

        assert es.counter == 0
        assert es.should_stop is False

    def test_custom_callback(self):
        class TrackingCallback(Callback):
            def __init__(self):
                self.events = []

            def on_train_begin(self, state):
                self.events.append("begin")

            def on_epoch_end(self, state):
                self.events.append("epoch")

            def on_train_end(self, state):
                self.events.append("end")

        cb = TrackingCallback()
        cb.on_train_begin({})
        cb.on_epoch_end({})
        cb.on_epoch_end({})
        cb.on_train_end({})

        assert cb.events == ["begin", "epoch", "epoch", "end"]

    def test_checkpoint_callback_saves(self, tmp_path: Path):
        ckpt_cb = CheckpointCallback(
            checkpoint_dir=str(tmp_path), every_n_epochs=1, save_best=True
        )
        model = get_architecture("mlp", input_dim=10, output_dim=2)
        optimizer = torch.optim.Adam(model.parameters())

        state = {
            "epoch": 1,
            "model": model,
            "optimizer": optimizer,
            "metrics": {"train_loss": 0.5},
        }
        ckpt_cb.on_epoch_end(state)

        # Check that files were created
        assert (tmp_path / "epoch_0001.pt").exists()
        assert (tmp_path / "best.pt").exists()


# ── Trainer tests ─────────────────────────────────────────────────


class TestTrainer:

    def test_fit_basic(self):
        model = get_architecture("mlp", input_dim=20, output_dim=3)
        config = TrainConfig(
            architecture="mlp",
            epochs=3,
            learning_rate=0.01,
            batch_size=16,
            checkpoint_dir="./test_checkpoints",
        )
        train_loader = make_synthetic_loader(input_dim=20, output_dim=3)

        # Disable checkpointing for test speed
        trainer = Trainer(model, config, callbacks=[MetricsLogger(verbose=False)])
        history = trainer.fit(train_loader)

        assert len(history) == 3
        assert "train_loss" in history[0]

    def test_fit_with_validation(self):
        model = get_architecture("mlp", input_dim=20, output_dim=3)
        config = TrainConfig(epochs=2, learning_rate=0.01)

        train_loader = make_synthetic_loader(input_dim=20, output_dim=3, n_samples=80)
        val_loader = make_synthetic_loader(input_dim=20, output_dim=3, n_samples=20)

        trainer = Trainer(model, config, callbacks=[MetricsLogger(verbose=False)])
        history = trainer.fit(train_loader, val_loader)

        assert "val_loss" in history[0]

    def test_evaluate(self):
        model = get_architecture("mlp", input_dim=20, output_dim=3)
        config = TrainConfig(epochs=1)
        loader = make_synthetic_loader(input_dim=20, output_dim=3)

        trainer = Trainer(model, config, callbacks=[])
        results = trainer.evaluate(loader)

        assert "loss" in results

    def test_export_and_reload(self, tmp_path: Path):
        model = get_architecture("mlp", input_dim=10, output_dim=2)
        config = TrainConfig(architecture="mlp")
        trainer = Trainer(model, config, callbacks=[])

        export_path = tmp_path / "model.pth"
        trainer.export(export_path)
        assert export_path.exists()

        # Reload weights into a new model
        new_model = get_architecture("mlp", input_dim=10, output_dim=2)
        ckpt = load_checkpoint(export_path)
        new_model.load_state_dict(ckpt["model_state_dict"])

        # Verify weights match
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2)

    def test_resume_training(self, tmp_path: Path):
        model = get_architecture("mlp", input_dim=20, output_dim=3)
        config = TrainConfig(
            epochs=5,
            checkpoint_dir=str(tmp_path),
            checkpoint_every=1,
        )
        train_loader = make_synthetic_loader(input_dim=20, output_dim=3)

        # Train for 2 epochs
        config_short = TrainConfig(
            epochs=2,
            checkpoint_dir=str(tmp_path),
            checkpoint_every=1,
        )
        trainer = Trainer(model, config_short, callbacks=[
            CheckpointCallback(str(tmp_path), every_n_epochs=1, save_best=False)
        ])
        trainer.fit(train_loader)

        # Resume from last checkpoint
        ckpt_path = tmp_path / "epoch_0002.pt"
        assert ckpt_path.exists()

        new_model = get_architecture("mlp", input_dim=20, output_dim=3)
        new_trainer = Trainer(new_model, config, callbacks=[
            MetricsLogger(verbose=False)
        ])
        new_trainer.resume(ckpt_path)
        assert new_trainer.start_epoch == 3

    def test_loss_decreases(self):
        """Sanity check: loss should decrease on a trivially learnable dataset."""
        torch.manual_seed(42)

        # Create a linearly separable dataset
        X = torch.randn(200, 10)
        w = torch.randn(10)
        y = (X @ w > 0).long()
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        model = get_architecture("mlp", input_dim=10, output_dim=2, hidden_dims=[32])
        config = TrainConfig(
            epochs=20,
            learning_rate=0.01,
            loss_fn="cross_entropy",
            seed=42,
        )
        trainer = Trainer(model, config, callbacks=[])
        history = trainer.fit(loader)

        # Loss should be lower at the end than at the start
        assert history[-1]["train_loss"] < history[0]["train_loss"]
