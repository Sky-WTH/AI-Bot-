"""Image dataset — directory-based image loading with transforms.

Expects a folder structure::

    root/
    ├── class_a/
    │   ├── img001.png
    │   └── img002.jpg
    └── class_b/
        ├── img003.png
        └── img004.jpg

Each sub-directory name becomes a class label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


# Supported image file extensions
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    """PyTorch Dataset for image classification from a directory tree.

    Args:
        root: Root directory with class sub-folders.
        image_size: Resize all images to ``(image_size, image_size)``.
        augment: If ``True``, apply random augmentations (flip, rotation, jitter).
        normalize: If ``True``, apply ImageNet-style normalization.
        transform: An optional custom ``torchvision.transforms`` pipeline that
            overrides the built-in one.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int = 64,
        augment: bool = False,
        normalize: bool = True,
        transform: Optional[T.Compose] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)

        # Discover classes and image paths
        self.classes, self.class_to_idx = self._discover_classes()
        self.samples = self._discover_samples()

        if not self.samples:
            raise FileNotFoundError(
                f"No images found in '{self.root}'. Expected sub-folders "
                f"with image files ({', '.join(_IMAGE_EXTENSIONS)})."
            )

        # Build transform pipeline
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._build_transform(image_size, augment, normalize)

    # ── Discovery ──────────────────────────────────────────────────

    def _discover_classes(self) -> tuple[list[str], dict[str, int]]:
        """Find class sub-directories.

        Returns:
            Tuple of (sorted class names, class_name → index mapping).
        """
        classes = sorted(
            d.name for d in self.root.iterdir() if d.is_dir()
        )
        class_to_idx = {name: idx for idx, name in enumerate(classes)}
        return classes, class_to_idx

    def _discover_samples(self) -> list[tuple[Path, int]]:
        """Walk class directories and collect (image_path, label) pairs."""
        samples: list[tuple[Path, int]] = []
        for class_name, idx in self.class_to_idx.items():
            class_dir = self.root / class_name
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in _IMAGE_EXTENSIONS:
                    samples.append((img_path, idx))
        return samples

    # ── Transforms ─────────────────────────────────────────────────

    @staticmethod
    def _build_transform(
        image_size: int, augment: bool, normalize: bool
    ) -> T.Compose:
        """Build a torchvision transform pipeline.

        Args:
            image_size: Target square size.
            augment: Whether to include random augmentations.
            normalize: Whether to include ImageNet normalization.
        """
        ops: list = []

        if augment:
            ops.extend([
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])
        else:
            ops.extend([
                T.Resize((image_size, image_size)),
            ])

        ops.append(T.ToTensor())

        if normalize:
            ops.append(
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )

        return T.Compose(ops)

    # ── Dataset interface ──────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    @property
    def num_classes(self) -> int:
        """Number of classes discovered from the directory structure."""
        return len(self.classes)

    @property
    def num_channels(self) -> int:
        """Number of image channels (always 3 — RGB)."""
        return 3
