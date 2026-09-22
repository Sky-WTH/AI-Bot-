"""Architecture registry — discover, register, and instantiate model templates.

The registry uses a decorator pattern so new architectures can be added by
simply decorating a class with ``@register_architecture("name")``.
"""

from __future__ import annotations

from typing import Any, Type

import torch.nn as nn

# ── Global registry ────────────────────────────────────────────────

_ARCHITECTURE_REGISTRY: dict[str, Type[nn.Module]] = {}


def register_architecture(name: str):
    """Decorator that registers an ``nn.Module`` subclass in the global registry.

    Usage::

        @register_architecture("my_model")
        class MyModel(nn.Module):
            def __init__(self, input_dim, output_dim):
                ...

    Args:
        name: Lowercase string key used to look up this architecture.
    """
    name = name.strip().lower()

    def _decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in _ARCHITECTURE_REGISTRY:
            raise ValueError(
                f"Architecture '{name}' is already registered "
                f"(class: {_ARCHITECTURE_REGISTRY[name].__name__}). "
                f"Cannot re-register with {cls.__name__}."
            )
        _ARCHITECTURE_REGISTRY[name] = cls
        return cls

    return _decorator


def get_architecture(name: str, **kwargs: Any) -> nn.Module:
    """Instantiate a registered architecture by name.

    Args:
        name: Registry key (case-insensitive).
        **kwargs: Arguments forwarded to the architecture's ``__init__``.

    Returns:
        An instantiated ``nn.Module``.

    Raises:
        KeyError: If ``name`` is not found in the registry.
    """
    name = name.strip().lower()
    if name not in _ARCHITECTURE_REGISTRY:
        available = ", ".join(sorted(_ARCHITECTURE_REGISTRY.keys()))
        raise KeyError(
            f"Architecture '{name}' not found. Available: [{available}]"
        )
    return _ARCHITECTURE_REGISTRY[name](**kwargs)


def list_architectures() -> list[str]:
    """Return sorted list of all registered architecture names."""
    return sorted(_ARCHITECTURE_REGISTRY.keys())


def get_architecture_class(name: str) -> Type[nn.Module]:
    """Return the class (not an instance) of a registered architecture.

    Useful for introspecting constructor signatures.
    """
    name = name.strip().lower()
    if name not in _ARCHITECTURE_REGISTRY:
        available = ", ".join(sorted(_ARCHITECTURE_REGISTRY.keys()))
        raise KeyError(
            f"Architecture '{name}' not found. Available: [{available}]"
        )
    return _ARCHITECTURE_REGISTRY[name]
