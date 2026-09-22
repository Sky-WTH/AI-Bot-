"""Text dataset — tokenization, vocabulary building, and sequence encoding.

Supports two formats:
- **Line-delimited text files** (``*.txt``): one sample per line, with an
  optional label separated by a tab character.
- **CSV files** (``*.csv``): requires a ``text`` column and an optional
  ``label`` column.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1


class Vocabulary:
    """Word-level vocabulary with frequency-based pruning.

    Args:
        max_size: Maximum number of tokens (including special tokens).
    """

    def __init__(self, max_size: int = 10000) -> None:
        self.max_size = max_size
        self.token2id: dict[str, int] = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        self.id2token: dict[int, str] = {PAD_ID: PAD_TOKEN, UNK_ID: UNK_TOKEN}

    def build(self, texts: list[str]) -> None:
        """Build the vocabulary from a corpus of texts.

        Args:
            texts: List of raw text strings.
        """
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(self._tokenize(text))

        # Keep the most common tokens (up to max_size − 2 for specials)
        for token, _ in counter.most_common(self.max_size - 2):
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token

    def encode(self, text: str, max_len: int) -> list[int]:
        """Tokenize and encode a string into a fixed-length ID sequence.

        Args:
            text: Raw text string.
            max_len: Maximum sequence length (pad or truncate).

        Returns:
            List of integer token IDs.
        """
        tokens = self._tokenize(text)
        ids = [self.token2id.get(t, UNK_ID) for t in tokens[:max_len]]
        # Pad to max_len
        ids += [PAD_ID] * (max_len - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.token2id)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer."""
        text = text.lower().strip()
        # Split on whitespace and separate punctuation
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return tokens


class TextDataset(Dataset):
    """PyTorch Dataset for text classification.

    Supports ``.txt`` (tab-separated ``text\\tlabel``) and ``.csv`` files
    with ``text`` and ``label`` columns.

    Args:
        data_path: Path to the data file.
        vocab: Pre-built :class:`Vocabulary` instance.  If ``None``, a new
            vocabulary is built from the data.
        max_seq_len: Maximum token sequence length.
        vocab_size: Maximum vocabulary size (used only when building a new vocab).
        text_column: Column name for text (CSV only).
        label_column: Column name for labels (CSV only).
    """

    def __init__(
        self,
        data_path: str | Path,
        vocab: Optional[Vocabulary] = None,
        max_seq_len: int = 256,
        vocab_size: int = 10000,
        text_column: str = "text",
        label_column: str = "label",
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.max_seq_len = max_seq_len
        self.text_column = text_column
        self.label_column = label_column

        # Load raw data
        self.texts, self.labels, self.label_map = self._load_data()

        # Build or reuse vocabulary
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = Vocabulary(max_size=vocab_size)
            self.vocab.build(self.texts)

    # ── Data loading ───────────────────────────────────────────────

    def _load_data(self) -> tuple[list[str], list[int], dict[str, int]]:
        """Load texts and labels from the data file.

        Returns:
            Tuple of (texts, encoded_labels, label_map).
        """
        suffix = self.data_path.suffix.lower()
        if suffix == ".csv":
            return self._load_csv()
        return self._load_txt()

    def _load_csv(self) -> tuple[list[str], list[int], dict[str, int]]:
        texts: list[str] = []
        raw_labels: list[str] = []

        with open(self.data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                texts.append(row[self.text_column])
                if self.label_column in row:
                    raw_labels.append(row[self.label_column])

        label_map = self._build_label_map(raw_labels)
        labels = [label_map.get(l, 0) for l in raw_labels] if raw_labels else [0] * len(texts)
        return texts, labels, label_map

    def _load_txt(self) -> tuple[list[str], list[int], dict[str, int]]:
        texts: list[str] = []
        raw_labels: list[str] = []

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "\t" in line:
                    parts = line.split("\t", maxsplit=1)
                    texts.append(parts[0])
                    raw_labels.append(parts[1])
                else:
                    texts.append(line)

        label_map = self._build_label_map(raw_labels)
        labels = [label_map.get(l, 0) for l in raw_labels] if raw_labels else [0] * len(texts)
        return texts, labels, label_map

    @staticmethod
    def _build_label_map(raw_labels: list[str]) -> dict[str, int]:
        unique = sorted(set(raw_labels))
        return {label: idx for idx, label in enumerate(unique)}

    # ── Dataset interface ──────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.vocab.encode(self.texts[idx], self.max_seq_len)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
