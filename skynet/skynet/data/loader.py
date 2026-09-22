"""Unified data pipeline — the single entry point for all data loading.

Routes to the correct modality-specific dataset based on :class:`DataConfig`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from skynet.config import DataConfig
from skynet.data.image import ImageDataset
from skynet.data.tabular import TabularDataset
from skynet.data.text import TextDataset


class DataPipeline:
    """Factory that builds train and validation DataLoaders from a :class:`DataConfig`.

    Usage::

        pipeline = DataPipeline(data_config, batch_size=32)
        train_loader, val_loader = pipeline.build()

    Attributes:
        dataset: The full dataset (before splitting).
        train_dataset: Training split.
        val_dataset: Validation split (may be ``None`` if ``val_split=0``).
    """

    def __init__(
        self,
        config: DataConfig,
        batch_size: int = 32,
    ) -> None:
        self.config = config
        self.batch_size = batch_size
        self.dataset: Optional[Dataset] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def build(self) -> tuple[DataLoader, Optional[DataLoader]]:
        """Build DataLoaders for training and (optionally) validation.

        Returns:
            Tuple of ``(train_loader, val_loader)``.  ``val_loader`` is
            ``None`` when ``config.val_split == 0``.
        """
        self.dataset = self._create_dataset()
        self.train_dataset, self.val_dataset = self._split(self.dataset)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        val_loader = None
        if self.val_dataset is not None and len(self.val_dataset) > 0:  # type: ignore[arg-type]
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=torch.cuda.is_available(),
            )

        return train_loader, val_loader

    @classmethod
    def from_config(
        cls,
        config: DataConfig,
        batch_size: int = 32,
    ) -> tuple[DataLoader, Optional[DataLoader]]:
        """Convenience class method to build loaders in one call.

        Args:
            config: A :class:`DataConfig` instance.
            batch_size: Mini-batch size.

        Returns:
            Tuple of ``(train_loader, val_loader)``.
        """
        pipeline = cls(config, batch_size=batch_size)
        return pipeline.build()

    # ── Internal helpers ───────────────────────────────────────────

    def _create_dataset(self) -> Dataset:
        """Instantiate the correct dataset class based on ``data_type``."""
        data_type = self.config.data_type
        data_path = Path(self.config.data_path)

        if data_type == "text":
            return TextDataset(
                data_path=data_path,
                max_seq_len=self.config.max_seq_len,
                vocab_size=self.config.vocab_size,
            )

        if data_type == "image":
            return ImageDataset(
                root=data_path,
                image_size=self.config.image_size,
                augment=self.config.augment,
            )

        if data_type == "tabular":
            return TabularDataset(
                data_path=data_path,
                target_column=self.config.target_column,
            )

        raise ValueError(
            f"Unknown data_type '{data_type}'. Choose from: text, image, tabular."
        )

    def _split(
        self, dataset: Dataset
    ) -> tuple[Dataset, Optional[Dataset]]:
        """Split dataset into train and validation sets."""
        val_split = self.config.val_split
        if val_split <= 0 or val_split >= 1:
            return dataset, None

        total = len(dataset)  # type: ignore[arg-type]
        val_size = max(1, int(total * val_split))
        train_size = total - val_size

        return random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

    # ── Introspection helpers ──────────────────────────────────────

    def get_input_info(self) -> dict:
        """Return useful metadata about the loaded dataset.

        Must be called after :meth:`build`.

        Returns:
            Dictionary with keys like ``input_dim``, ``num_classes``, etc.
        """
        if self.dataset is None:
            raise RuntimeError("Call build() before get_input_info().")

        info: dict = {}

        if isinstance(self.dataset, TabularDataset):
            info["input_dim"] = self.dataset.input_dim
            info["num_classes"] = self.dataset.num_classes
            info["task"] = self.dataset.task

        elif isinstance(self.dataset, ImageDataset):
            info["in_channels"] = self.dataset.num_channels
            info["num_classes"] = self.dataset.num_classes
            info["image_size"] = self.config.image_size

        elif isinstance(self.dataset, TextDataset):
            info["vocab_size"] = len(self.dataset.vocab)
            info["num_classes"] = len(self.dataset.label_map)
            info["max_seq_len"] = self.config.max_seq_len

        return info
