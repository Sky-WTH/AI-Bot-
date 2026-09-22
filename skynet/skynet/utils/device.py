"""Device detection and management utilities."""

from __future__ import annotations

import torch


def get_best_device(preference: str = "auto") -> torch.device:
    """Return the best available device for training.

    Priority order: CUDA > MPS > CPU (when preference is 'auto').

    Args:
        preference: One of 'auto', 'cpu', 'cuda', 'mps'.
            If a specific device is requested but unavailable, falls back to CPU.

    Returns:
        A ``torch.device`` instance.
    """
    preference = preference.lower().strip()

    if preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if preference == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if preference == "cpu":
        return torch.device("cpu")

    # Auto-detect
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info() -> str:
    """Return a human-readable summary of available compute devices.

    Returns:
        Multi-line string describing CUDA, MPS, and CPU availability.
    """
    lines: list[str] = []
    best = get_best_device()
    lines.append(f"Best device : {best}")

    # CUDA info
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory
            mem_gb = mem / (1024**3)
            lines.append(f"CUDA:{i}      : {name} ({mem_gb:.1f} GB)")
    else:
        lines.append("CUDA         : not available")

    # MPS info
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        lines.append("MPS          : available (Apple Silicon)")
    else:
        lines.append("MPS          : not available")

    # CPU info
    lines.append(f"CPU threads  : {torch.get_num_threads()}")

    return "\n".join(lines)
