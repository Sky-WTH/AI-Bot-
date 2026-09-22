"""Tabular dataset — CSV/JSON loading with automatic type detection.

Handles both classification and regression by auto-detecting the target
column type. Categorical features are label-encoded; numeric features are
standardized (zero mean, unit variance).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """PyTorch Dataset for tabular (structured) data.

    Loads CSV or JSON files, auto-detects column types, encodes categoricals,
    standardizes numerics, and produces ``(features, target)`` tensors.

    Args:
        data_path: Path to a ``.csv`` or ``.json`` data file.
        target_column: Name of the column to predict.  If ``None``, the
            dataset operates in unsupervised mode (features only).
        categorical_columns: Explicit list of categorical column names.
            If ``None``, columns with dtype ``object`` are treated as
            categorical.
        encoders: Pre-fitted label encoders (column_name → {value: int}).
            If ``None``, encoders are built from the data.
        stats: Pre-computed ``(mean, std)`` arrays for numeric columns.
            If ``None``, stats are computed from the data.
    """

    def __init__(
        self,
        data_path: str | Path,
        target_column: Optional[str] = None,
        categorical_columns: Optional[list[str]] = None,
        encoders: Optional[dict[str, dict[str, int]]] = None,
        stats: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.target_column = target_column

        # Load data
        df = self._load_dataframe()

        # Separate target
        if target_column and target_column in df.columns:
            target_series = df.pop(target_column)
        else:
            target_series = None

        # Identify column types
        if categorical_columns is not None:
            self.cat_columns = [c for c in categorical_columns if c in df.columns]
        else:
            self.cat_columns = list(df.select_dtypes(include=["object", "category", "str", "string"]).columns)
        self.num_columns = [c for c in df.columns if c not in self.cat_columns]

        # ── Encode categoricals ────────────────────────────────────
        if encoders is not None:
            self.encoders = encoders
        else:
            self.encoders = {}
            for col in self.cat_columns:
                unique = sorted(df[col].dropna().unique())
                self.encoders[col] = {v: i for i, v in enumerate(unique)}

        for col in self.cat_columns:
            mapping = self.encoders[col]
            df[col] = df[col].map(mapping).fillna(0).astype(float)

        # ── Standardize numerics ───────────────────────────────────
        num_data = df[self.num_columns].values.astype(np.float32) if self.num_columns else np.zeros((len(df), 0), dtype=np.float32)
        if stats is not None:
            self.mean, self.std = stats
        else:
            self.mean = num_data.mean(axis=0) if num_data.shape[1] > 0 else np.array([])
            self.std = num_data.std(axis=0) if num_data.shape[1] > 0 else np.array([])
            # Prevent division by zero
            self.std[self.std == 0] = 1.0

        if num_data.shape[1] > 0:
            num_data = (num_data - self.mean) / self.std

        # ── Combine features ───────────────────────────────────────
        cat_data = df[self.cat_columns].values.astype(np.float32) if self.cat_columns else np.zeros((len(df), 0), dtype=np.float32)
        self.features = np.hstack([num_data, cat_data]).astype(np.float32)

        # ── Process target ─────────────────────────────────────────
        if target_series is not None:
            if target_series.dtype == object or str(target_series.dtype) in ("string", "str", "object"):
                # Classification target
                unique_targets = sorted(target_series.dropna().unique())
                self.target_encoder = {v: i for i, v in enumerate(unique_targets)}
                self.targets = target_series.map(self.target_encoder).fillna(0).values.astype(np.int64)
                self.task = "classification"
                self.num_classes = len(unique_targets)
            else:
                # Regression target
                self.target_encoder = None
                self.targets = target_series.values.astype(np.float32)
                self.task = "regression"
                self.num_classes = 1
        else:
            self.targets = np.zeros(len(df), dtype=np.float32)
            self.target_encoder = None
            self.task = "unsupervised"
            self.num_classes = 0

    # ── Data loading ───────────────────────────────────────────────

    def _load_dataframe(self) -> pd.DataFrame:
        suffix = self.data_path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(self.data_path)
        elif suffix == ".json":
            return pd.read_json(self.data_path)
        else:
            raise ValueError(f"Unsupported file type: '{suffix}'. Use .csv or .json.")

    # ── Dataset interface ──────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        if self.task == "classification":
            y = torch.tensor(self.targets[idx], dtype=torch.long)
        else:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y

    @property
    def input_dim(self) -> int:
        """Number of input features after preprocessing."""
        return self.features.shape[1]

    def get_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """Return fitted (mean, std) arrays for reuse on test data."""
        return self.mean, self.std
