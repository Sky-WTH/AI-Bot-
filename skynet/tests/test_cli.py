"""Tests for the Skynet CLI."""

import pytest
from typer.testing import CliRunner

# Ensure architectures are registered
import skynet.architectures.mlp  # noqa: F401
import skynet.architectures.cnn  # noqa: F401
import skynet.architectures.transformer  # noqa: F401
import skynet.architectures.autoencoder  # noqa: F401

from skynet.cli import app


runner = CliRunner()


class TestCLI:

    def test_list_architectures(self):
        result = runner.invoke(app, ["list-architectures"])
        assert result.exit_code == 0
        assert "mlp" in result.stdout.lower()
        assert "cnn" in result.stdout.lower()

    def test_init_creates_config(self, tmp_path):
        config_path = str(tmp_path / "config.yaml")
        result = runner.invoke(app, ["init", "--arch", "mlp", "--config", config_path])
        assert result.exit_code == 0
        assert (tmp_path / "config.yaml").exists()

    def test_init_default_arch(self, tmp_path):
        config_path = str(tmp_path / "config.yaml")
        result = runner.invoke(app, ["init", "--config", config_path])
        assert result.exit_code == 0

        # Verify config contents
        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert config["train"]["architecture"] == "mlp"

    def test_device_command(self):
        result = runner.invoke(app, ["device"])
        assert result.exit_code == 0
        assert "device" in result.stdout.lower() or "cpu" in result.stdout.lower()

    def test_info_command(self, tmp_path):
        """Test the info command with a real checkpoint."""
        import torch

        ckpt_path = tmp_path / "test.pt"
        torch.save(
            {"epoch": 5, "loss": 0.3, "model_state_dict": {}},
            ckpt_path,
        )
        result = runner.invoke(app, ["info", "--checkpoint", str(ckpt_path)])
        assert result.exit_code == 0
        assert "5" in result.stdout

    def test_export_command(self, tmp_path):
        """Test the export command."""
        import torch

        ckpt_path = tmp_path / "checkpoint.pt"
        torch.save(
            {
                "epoch": 3,
                "loss": 0.5,
                "model_state_dict": {"weight": torch.randn(3, 3)},
            },
            ckpt_path,
        )

        output_path = tmp_path / "model.pth"
        result = runner.invoke(app, [
            "export",
            "--checkpoint", str(ckpt_path),
            "--output", str(output_path),
        ])
        assert result.exit_code == 0
        assert output_path.exists()
