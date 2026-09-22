"""Rich-powered logging for Skynet."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.panel import Panel
from rich.text import Text


# Shared console instance
console = Console()


class SkynetLogger:
    """Structured, colourful logger built on Rich.

    Provides methods for training status, metric tables, and progress bars
    so the user gets clear feedback during training.
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.console = console

    # ------------------------------------------------------------------
    # Banners
    # ------------------------------------------------------------------

    def banner(self) -> None:
        """Print the Skynet startup banner."""
        banner_text = Text()
        banner_text.append("⚡ SKYNET ", style="bold cyan")
        banner_text.append("v0.1.0", style="dim")
        banner_text.append(
            " — Personal Neural Network Training", style="italic white"
        )
        self.console.print(
            Panel(banner_text, border_style="cyan", padding=(0, 2))
        )

    # ------------------------------------------------------------------
    # Status messages
    # ------------------------------------------------------------------

    def info(self, msg: str) -> None:
        """Print an informational message."""
        if self.verbose:
            self.console.print(f"[cyan]ℹ[/cyan]  {msg}")

    def success(self, msg: str) -> None:
        """Print a success message."""
        self.console.print(f"[green]✓[/green]  {msg}")

    def warning(self, msg: str) -> None:
        """Print a warning message."""
        self.console.print(f"[yellow]⚠[/yellow]  {msg}")

    def error(self, msg: str) -> None:
        """Print an error message."""
        self.console.print(f"[red]✗[/red]  {msg}")

    # ------------------------------------------------------------------
    # Training metrics
    # ------------------------------------------------------------------

    def epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        metrics: dict[str, float],
    ) -> None:
        """Print a single-line epoch summary.

        Args:
            epoch: Current epoch number (1-indexed).
            total_epochs: Total number of epochs.
            metrics: Dictionary of metric name → value pairs.
        """
        parts = [f"[bold]Epoch {epoch}/{total_epochs}[/bold]"]
        for name, value in metrics.items():
            parts.append(f"{name}: {value:.4f}")
        self.console.print("  ".join(parts))

    def metrics_table(self, history: list[dict[str, float]]) -> None:
        """Print a table of training history.

        Args:
            history: List of dicts, one per epoch, containing metric values.
        """
        if not history:
            return

        table = Table(title="Training History", border_style="cyan")
        table.add_column("Epoch", justify="right", style="bold")
        # Add columns from the first entry
        metric_names = [k for k in history[0] if k != "epoch"]
        for name in metric_names:
            table.add_column(name, justify="right")

        for record in history:
            epoch_str = str(record.get("epoch", "?"))
            values = [f"{record.get(m, 0):.4f}" for m in metric_names]
            table.add_row(epoch_str, *values)

        self.console.print(table)

    # ------------------------------------------------------------------
    # Progress bar
    # ------------------------------------------------------------------

    def training_progress(self, total: int) -> Progress:
        """Create a Rich progress bar for batch-level iteration.

        Args:
            total: Total number of batches.

        Returns:
            A ``rich.progress.Progress`` context manager.
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
