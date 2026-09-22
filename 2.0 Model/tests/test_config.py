"""Tests for Skynet configuration models."""

import json
from pathlib import Path

import pytest
import yaml

from skynet.config import DataConfig, ExportConfig, TrainConfig, load_config


class TestTrainConfig:

    def test_defaults(self):
        config = TrainConfig()
        assert config.architecture == "mlp"
        assert config.learning_rate == 1e-3
        assert config.batch_size == 32
        assert config.epochs == 10
        assert config.device == "auto"

    def test_architecture_lowercase(self):
        config = TrainConfig(architecture="  MLP  ")
        assert config.architecture == "mlp"

    def test_invalid_learning_rate(self):
        with pytest.raises(Exception):
            TrainConfig(learning_rate=-0.1)

    def test_yaml_roundtrip(self, tmp_path: Path):
        original = TrainConfig(
            architecture="cnn",
            learning_rate=0.005,
            epochs=50,
            batch_size=64,
        )
        yaml_path = tmp_path / "train.yaml"
        original.to_yaml(yaml_path)

        loaded = TrainConfig.from_yaml(yaml_path)
        assert loaded.architecture == "cnn"
        assert loaded.learning_rate == 0.005
        assert loaded.epochs == 50
        assert loaded.batch_size == 64

    def test_json_roundtrip(self, tmp_path: Path):
        original = TrainConfig(architecture="transformer", epochs=20)
        json_path = tmp_path / "train.json"
        original.to_json(json_path)

        loaded = TrainConfig.from_json(json_path)
        assert loaded.architecture == "transformer"
        assert loaded.epochs == 20


class TestDataConfig:

    def test_defaults(self):
        config = DataConfig()
        assert config.data_type == "tabular"
        assert config.val_split == 0.2
        assert config.image_size == 64

    def test_data_type_lowercase(self):
        config = DataConfig(data_type="  IMAGE  ")
        assert config.data_type == "image"

    def test_val_split_bounds(self):
        with pytest.raises(Exception):
            DataConfig(val_split=1.5)
        with pytest.raises(Exception):
            DataConfig(val_split=-0.1)

    def test_yaml_roundtrip(self, tmp_path: Path):
        original = DataConfig(data_type="image", image_size=128, augment=True)
        yaml_path = tmp_path / "data.yaml"
        original.to_yaml(yaml_path)

        loaded = DataConfig.from_yaml(yaml_path)
        assert loaded.data_type == "image"
        assert loaded.image_size == 128
        assert loaded.augment is True


class TestExportConfig:

    def test_defaults(self):
        config = ExportConfig()
        assert config.output_path == "./model.pth"
        assert config.include_optimizer is False
        assert config.include_metadata is True


class TestLoadConfig:

    def test_unified_config(self, tmp_path: Path):
        config_data = {
            "train": {
                "architecture": "cnn",
                "epochs": 30,
            },
            "data": {
                "data_type": "image",
                "data_path": "./images",
                "image_size": 128,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        result = load_config(config_path)
        assert result["train"].architecture == "cnn"
        assert result["train"].epochs == 30
        assert result["data"].data_type == "image"
        assert result["data"].image_size == 128

    def test_flat_config(self, tmp_path: Path):
        """Flat config without train/data sections should still work."""
        config_data = {
            "architecture": "mlp",
            "epochs": 15,
        }
        config_path = tmp_path / "flat.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        result = load_config(config_path)
        assert result["train"].architecture == "mlp"
        assert result["train"].epochs == 15
