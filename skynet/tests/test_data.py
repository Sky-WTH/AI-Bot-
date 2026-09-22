"""Tests for data loading and preprocessing."""

import csv
import os
import tempfile
from pathlib import Path

import pytest
import torch

from skynet.config import DataConfig
from skynet.data.loader import DataPipeline
from skynet.data.tabular import TabularDataset
from skynet.data.text import TextDataset, Vocabulary
from skynet.data.incremental import IncrementalDataset


# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def tabular_csv(tmp_path: Path) -> Path:
    """Create a small CSV file for tabular tests."""
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["age", "salary", "city", "bought"])
        for i in range(50):
            writer.writerow([
                25 + i % 30,
                30000 + i * 1000,
                ["NYC", "LA", "Chicago"][i % 3],
                ["yes", "no"][i % 2],
            ])
    return csv_path


@pytest.fixture
def text_csv(tmp_path: Path) -> Path:
    """Create a small CSV file for text tests."""
    csv_path = tmp_path / "texts.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        texts = [
            ("I love this product", "positive"),
            ("This is terrible", "negative"),
            ("Great quality and fast shipping", "positive"),
            ("Waste of money", "negative"),
            ("Excellent service", "positive"),
            ("Very disappointing", "negative"),
        ] * 5
        for text, label in texts:
            writer.writerow([text, label])
    return csv_path


@pytest.fixture
def text_file(tmp_path: Path) -> Path:
    """Create a simple tab-delimited text file."""
    txt_path = tmp_path / "corpus.txt"
    lines = [
        "hello world\tgreet",
        "good morning\tgreet",
        "buy now\tcommand",
        "stop that\tcommand",
    ] * 5
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return txt_path


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    """Create a minimal image directory structure with tiny PNGs."""
    from PIL import Image
    import numpy as np

    for class_name in ["cat", "dog"]:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for i in range(5):
            img = Image.fromarray(
                np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            )
            img.save(class_dir / f"img_{i:03d}.png")

    return tmp_path


# ── Vocabulary tests ───────────────────────────────────────────────


class TestVocabulary:

    def test_build_and_encode(self):
        vocab = Vocabulary(max_size=50)
        vocab.build(["hello world", "hello there"])
        ids = vocab.encode("hello world", max_len=5)
        assert len(ids) == 5
        assert ids[0] != 1  # "hello" should not be UNK

    def test_unknown_token(self):
        vocab = Vocabulary(max_size=50)
        vocab.build(["hello"])
        ids = vocab.encode("xyz_unknown", max_len=3)
        assert ids[0] == 1  # UNK_ID

    def test_padding(self):
        vocab = Vocabulary(max_size=50)
        vocab.build(["a b c"])
        ids = vocab.encode("a", max_len=5)
        assert ids[-1] == 0  # PAD_ID
        assert len(ids) == 5


# ── TextDataset tests ─────────────────────────────────────────────


class TestTextDataset:

    def test_load_csv(self, text_csv: Path):
        ds = TextDataset(text_csv, max_seq_len=20, vocab_size=100)
        assert len(ds) == 30  # 6 * 5
        x, y = ds[0]
        assert x.shape == (20,)
        assert y.dtype == torch.long

    def test_load_txt(self, text_file: Path):
        ds = TextDataset(text_file, max_seq_len=10, vocab_size=50)
        assert len(ds) == 20  # 4 * 5
        x, y = ds[0]
        assert x.shape == (10,)

    def test_label_mapping(self, text_csv: Path):
        ds = TextDataset(text_csv, max_seq_len=10)
        assert len(ds.label_map) == 2  # positive, negative


# ── TabularDataset tests ──────────────────────────────────────────


class TestTabularDataset:

    def test_load_and_shape(self, tabular_csv: Path):
        ds = TabularDataset(tabular_csv, target_column="bought")
        assert len(ds) == 50
        x, y = ds[0]
        assert x.shape[0] > 0  # Has features
        assert y.dtype == torch.long  # Classification

    def test_input_dim(self, tabular_csv: Path):
        ds = TabularDataset(tabular_csv, target_column="bought")
        # 2 numeric (age, salary) + 1 categorical (city)
        assert ds.input_dim == 3

    def test_auto_task_detection(self, tabular_csv: Path):
        ds = TabularDataset(tabular_csv, target_column="bought")
        assert ds.task == "classification"
        assert ds.num_classes == 2

    def test_unsupervised_mode(self, tabular_csv: Path):
        ds = TabularDataset(tabular_csv, target_column=None)
        assert ds.task == "unsupervised"


# ── ImageDataset tests ────────────────────────────────────────────


class TestImageDataset:

    def test_load_and_shape(self, image_dir: Path):
        from skynet.data.image import ImageDataset

        ds = ImageDataset(image_dir, image_size=32)
        assert len(ds) == 10  # 5 cats + 5 dogs
        x, y = ds[0]
        assert x.shape == (3, 32, 32)
        assert y.dtype == torch.long

    def test_num_classes(self, image_dir: Path):
        from skynet.data.image import ImageDataset

        ds = ImageDataset(image_dir)
        assert ds.num_classes == 2

    def test_augmentation(self, image_dir: Path):
        from skynet.data.image import ImageDataset

        ds = ImageDataset(image_dir, image_size=32, augment=True)
        x, y = ds[0]
        assert x.shape[0] == 3  # Still 3 channels


# ── IncrementalDataset tests ──────────────────────────────────────


class TestIncrementalDataset:

    def test_basic_wrapping(self, tabular_csv: Path):
        base = TabularDataset(tabular_csv, target_column="bought")
        incremental = IncrementalDataset(base, buffer_size=10)
        # Before buffer update, length = base length + 0 buffer
        assert len(incremental) == len(base)

    def test_buffer_update(self, tabular_csv: Path):
        base = TabularDataset(tabular_csv, target_column="bought")
        incremental = IncrementalDataset(base, buffer_size=10)
        incremental.update_buffer()
        assert len(incremental.get_buffer()) <= 10
        assert len(incremental) == len(base) + len(incremental.get_buffer())

    def test_getitem(self, tabular_csv: Path):
        base = TabularDataset(tabular_csv, target_column="bought")
        incremental = IncrementalDataset(base, buffer_size=5)
        incremental.update_buffer()
        x, y = incremental[0]
        assert isinstance(x, torch.Tensor)


# ── DataPipeline tests ────────────────────────────────────────────


class TestDataPipeline:

    def test_tabular_pipeline(self, tabular_csv: Path):
        config = DataConfig(
            data_type="tabular",
            data_path=str(tabular_csv),
            target_column="bought",
            val_split=0.2,
        )
        pipeline = DataPipeline(config, batch_size=8)
        train_loader, val_loader = pipeline.build()

        assert train_loader is not None
        assert val_loader is not None

        batch_x, batch_y = next(iter(train_loader))
        assert batch_x.shape[0] <= 8

    def test_text_pipeline(self, text_csv: Path):
        config = DataConfig(
            data_type="text",
            data_path=str(text_csv),
            max_seq_len=20,
            vocab_size=100,
            val_split=0.2,
        )
        train_loader, val_loader = DataPipeline.from_config(config, batch_size=4)
        assert train_loader is not None

    def test_image_pipeline(self, image_dir: Path):
        config = DataConfig(
            data_type="image",
            data_path=str(image_dir),
            image_size=32,
            val_split=0.2,
        )
        train_loader, val_loader = DataPipeline.from_config(config, batch_size=4)
        assert train_loader is not None

    def test_input_info_tabular(self, tabular_csv: Path):
        config = DataConfig(
            data_type="tabular",
            data_path=str(tabular_csv),
            target_column="bought",
        )
        pipeline = DataPipeline(config)
        pipeline.build()
        info = pipeline.get_input_info()
        assert "input_dim" in info
        assert "num_classes" in info
