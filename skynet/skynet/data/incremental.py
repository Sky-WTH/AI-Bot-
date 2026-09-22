"""Incremental dataset wrapper with replay buffer.

Wraps any base dataset and mixes in exemplars from previous training
rounds to mitigate catastrophic forgetting during incremental learning.
"""

from __future__ import annotations

import random
from typing import Any, Optional

import torch
from torch.utils.data import Dataset


class IncrementalDataset(Dataset):
    """Replay-buffer wrapper for continual / incremental learning.

    Stores a configurable number of exemplars from previous datasets and
    interleaves them with the current dataset during iteration.

    Args:
        current_dataset: The new dataset for the current training round.
        buffer_size: Maximum number of exemplars to keep from previous rounds.
        buffer: Existing buffer from a previous round (list of ``(x, y)`` tuples).
        mix_ratio: Fraction of each batch that should come from the buffer
            (approximately).  ``0.0`` = no mixing, ``1.0`` = all buffer.
    """

    def __init__(
        self,
        current_dataset: Dataset,
        buffer_size: int = 500,
        buffer: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        mix_ratio: float = 0.3,
    ) -> None:
        super().__init__()
        self.current_dataset = current_dataset
        self.buffer_size = buffer_size
        self.mix_ratio = max(0.0, min(1.0, mix_ratio))

        # Initialize or carry forward the replay buffer
        self._buffer: list[tuple[torch.Tensor, torch.Tensor]] = list(buffer) if buffer else []

    # ── Buffer management ──────────────────────────────────────────

    def update_buffer(self, dataset: Optional[Dataset] = None) -> None:
        """Update the replay buffer with exemplars from the given dataset.

        Uses reservoir sampling so that older exemplars are gradually replaced
        as new data comes in.

        Args:
            dataset: Dataset to sample from.  Defaults to ``current_dataset``.
        """
        dataset = dataset or self.current_dataset
        n = len(dataset)  # type: ignore[arg-type]

        for i in range(n):
            sample = dataset[i]
            # Detach tensors to avoid holding computation graphs
            x = sample[0].clone().detach() if isinstance(sample[0], torch.Tensor) else sample[0]
            y = sample[1].clone().detach() if isinstance(sample[1], torch.Tensor) else sample[1]

            if len(self._buffer) < self.buffer_size:
                self._buffer.append((x, y))
            else:
                # Reservoir sampling
                j = random.randint(0, i)
                if j < self.buffer_size:
                    self._buffer[j] = (x, y)

    def get_buffer(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return the current replay buffer for persistence."""
        return list(self._buffer)

    # ── Dataset interface ──────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.current_dataset) + len(self._buffer)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        current_len = len(self.current_dataset)  # type: ignore[arg-type]

        # Should we sample from the buffer?
        if self._buffer and random.random() < self.mix_ratio and idx < current_len:
            buf_idx = random.randint(0, len(self._buffer) - 1)
            return self._buffer[buf_idx]

        if idx < current_len:
            return self.current_dataset[idx]
        else:
            buf_idx = idx - current_len
            return self._buffer[buf_idx]
