"""Native MLX FlexibleUNet; public names load on first access."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from flexible_unet_mlx.specification import specification

if TYPE_CHECKING:
    from flexible_unet_mlx.inference import forward_patch as forward_patch
    from flexible_unet_mlx.network import FlexibleUNet as FlexibleUNet
    from flexible_unet_mlx.weights import load_model as load_model

__version__ = "0.2.5"
__all__ = [
    "FlexibleUNet",
    "forward_patch",
    "load_model",
    "specification",
]


def __getattr__(name: str) -> object:
    """Import a public name from its submodule on first access and cache it."""
    modules = {
        "FlexibleUNet": "network",
        "forward_patch": "inference",
        "load_model": "weights",
    }
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(f"flexible_unet_mlx.{modules[name]}"), name)
    globals()[name] = value
    return value
