"""Skynet CLI — command-line interface for training neural networks.

A thin Typer wrapper over the Skynet SDK. All logic is delegated to the
core library so that the same operations are available programmatically.

Entry point registered in ``pyproject.toml``::

    [project.scripts]
    skynet = "skynet.cli:app"
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from skynet.utils.logger import SkynetLogger, console

# Ensure all architectures are registered at import time
import skynet.architectures.mlp  # noqa: F401
import skynet.architectures.cnn  # noqa: F401
import skynet.architectures.transformer  # noqa: F401
import skynet.architectures.autoencoder  # noqa: F401

app = typer.Typer(
    name="skynet",
    help="⚡ Skynet — Train personalized neural networks from scratch, on-device.",
    add_completion=False,
    rich_markup_mode="rich",
)

logger = SkynetLogger()


# ── list-architectures ─────────────────────────────────────────────


@app.command("list-architectures")
def list_architectures_cmd() -> None:
    """Show all available architecture templates."""
    from skynet.architectures.registry import list_architectures, get_architecture_class
    import inspect

    archs = list_architectures()

    table = Table(title="Available Architectures", border_style="cyan")
    table.add_column("Name", style="bold cyan")
    table.add_column("Class", style="white")
    table.add_column("Description", style="dim")

    for name in archs:
        cls = get_architecture_class(name)
        doc = (cls.__doc__ or "").strip().split("\n")[0]
        table.add_row(name, cls.__name__, doc)

    console.print(table)


# ── init ───────────────────────────────────────────────────────────


@app.command()
def init(
    arch: str = typer.Option("mlp", "--arch", "-a", help="Architecture template name."),
    output: str = typer.Option("./config.yaml", "--config", "-c", help="Output config file path."),
) -> None:
    """Initialize a new Skynet project with a config file."""
    from skynet.config import TrainConfig, DataConfig

    train_config = TrainConfig(architecture=arch)
    data_config = DataConfig()

    # Build a unified config dict
    import yaml

    unified = {
        "train": train_config.model_dump(),
        "data": data_config.model_dump(),
    }

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(unified, f, default_flow_style=False, sort_keys=False)

    logger.success(f"Config created: {path}")
    logger.info(f"Architecture: [bold]{arch}[/bold]")
    logger.info("Edit the config file, then run: [bold]skynet train --config config.yaml[/bold]")


# ── train ──────────────────────────────────────────────────────────


@app.command()
def train(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file."),
    data: Optional[str] = typer.Option(None, "--data", "-d", help="Override data path."),
    resume_from: Optional[str] = typer.Option(
        None, "--resume", "-r", help="Resume from a checkpoint file."
    ),
) -> None:
    """Start or resume training from a config file."""
    from skynet.config import load_config
    from skynet.architectures.registry import get_architecture
    from skynet.data.loader import DataPipeline
    from skynet.trainer.engine import Trainer

    # Load config
    configs = load_config(config)
    train_config = configs["train"]
    data_config = configs.get("data")

    if data_config is None:
        logger.error("No 'data' section found in config file.")
        raise typer.Exit(1)

    # Override data path if provided
    if data:
        data_config.data_path = data

    # Build data pipeline
    logger.info("Loading data…")
    pipeline = DataPipeline(data_config, batch_size=train_config.batch_size)
    train_loader, val_loader = pipeline.build()
    input_info = pipeline.get_input_info()
    logger.success(f"Data loaded: {len(train_loader.dataset)} samples")

    # Build model from architecture + data info
    arch_kwargs = {**input_info, **train_config.arch_params}
    logger.info(f"Building model: [bold]{train_config.architecture}[/bold]")
    model = get_architecture(train_config.architecture, **arch_kwargs)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: [bold]{n_params:,}[/bold]")

    # Create trainer
    trainer = Trainer(model, train_config)

    # Resume if requested
    if resume_from:
        trainer.resume(resume_from)

    # Train
    trainer.fit(train_loader, val_loader)


# ── resume ─────────────────────────────────────────────────────────


@app.command()
def resume(
    checkpoint: str = typer.Option(..., "--checkpoint", "-k", help="Path to checkpoint file."),
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file."),
    data: Optional[str] = typer.Option(None, "--data", "-d", help="Override data path."),
) -> None:
    """Resume training from a specific checkpoint."""
    # Delegate to the train command with resume flag
    train(config=config, data=data, resume_from=checkpoint)


# ── export ─────────────────────────────────────────────────────────


@app.command()
def export(
    checkpoint: str = typer.Option(..., "--checkpoint", "-k", help="Path to checkpoint file."),
    output: str = typer.Option("./model.pth", "--output", "-o", help="Output .pth file path."),
) -> None:
    """Export clean model weights from a checkpoint."""
    from skynet.trainer.checkpoint import load_checkpoint, save_checkpoint

    logger.info(f"Loading checkpoint: {checkpoint}")
    ckpt = load_checkpoint(checkpoint)

    if "model_state_dict" not in ckpt:
        logger.error("Checkpoint does not contain model weights.")
        raise typer.Exit(1)

    # Build a clean export
    export_data = {
        "model_state_dict": ckpt["model_state_dict"],
        "metadata": {
            "source_checkpoint": checkpoint,
            "epoch": ckpt.get("epoch", "unknown"),
            "loss": ckpt.get("loss", "unknown"),
        },
    }

    path = save_checkpoint(export_data, output)
    logger.success(f"Weights exported to: {path}")


# ── evaluate ───────────────────────────────────────────────────────


@app.command()
def evaluate(
    checkpoint: str = typer.Option(..., "--checkpoint", "-k", help="Path to checkpoint file."),
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file."),
    data: Optional[str] = typer.Option(None, "--data", "-d", help="Override data path."),
) -> None:
    """Evaluate a model checkpoint on test data."""
    from skynet.config import load_config
    from skynet.architectures.registry import get_architecture
    from skynet.data.loader import DataPipeline
    from skynet.trainer.checkpoint import load_checkpoint
    from skynet.trainer.engine import Trainer

    configs = load_config(config)
    train_config = configs["train"]
    data_config = configs.get("data")

    if data_config is None:
        logger.error("No 'data' section found in config file.")
        raise typer.Exit(1)

    if data:
        data_config.data_path = data

    # Build data pipeline (use entire dataset for eval — no split)
    data_config.val_split = 0.0
    pipeline = DataPipeline(data_config, batch_size=train_config.batch_size)
    eval_loader, _ = pipeline.build()
    input_info = pipeline.get_input_info()

    # Build model and load weights
    arch_kwargs = {**input_info, **train_config.arch_params}
    model = get_architecture(train_config.architecture, **arch_kwargs)

    ckpt = load_checkpoint(checkpoint)
    model.load_state_dict(ckpt["model_state_dict"])

    # Evaluate
    trainer = Trainer(model, train_config, callbacks=[])
    results = trainer.evaluate(eval_loader)

    table = Table(title="Evaluation Results", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    for name, value in results.items():
        table.add_row(name, f"{value:.4f}")
    console.print(table)


# ── info ───────────────────────────────────────────────────────────


@app.command()
def info(
    checkpoint: str = typer.Option(..., "--checkpoint", "-k", help="Path to checkpoint file."),
) -> None:
    """Show metadata about a checkpoint file."""
    from skynet.trainer.checkpoint import load_checkpoint

    ckpt = load_checkpoint(checkpoint)

    table = Table(title="Checkpoint Info", border_style="cyan")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Epoch", str(ckpt.get("epoch", "N/A")))
    table.add_row("Loss", f"{ckpt.get('loss', 'N/A')}")
    table.add_row("Has model weights", str("model_state_dict" in ckpt))
    table.add_row("Has optimizer state", str("optimizer_state_dict" in ckpt))
    table.add_row("Has scheduler state", str("scheduler_state_dict" in ckpt))
    table.add_row("Has scaler state", str("scaler_state_dict" in ckpt))

    if "metadata" in ckpt:
        for k, v in ckpt["metadata"].items():
            table.add_row(f"meta.{k}", str(v))

    console.print(table)


# ── device ─────────────────────────────────────────────────────────


@app.command()
def device() -> None:
    """Show available compute devices."""
    from skynet.utils.device import device_info

    console.print(device_info())


# ── Entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
