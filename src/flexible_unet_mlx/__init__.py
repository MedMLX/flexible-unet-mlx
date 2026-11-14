"""Endoscopic surgical tool segmentation with MONAI FlexibleUNet in native MLX.

``segment`` runs one frame; ``load_model`` keeps the network loaded for several.
Names load on first access, so importing the package does not import MLX.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from flexible_unet_mlx.specification import specification

if TYPE_CHECKING:
    from flexible_unet_mlx.inference import forward_patch as forward_patch
    from flexible_unet_mlx.medmlx import FlexibleUNetModel as FlexibleUNetModel
    from flexible_unet_mlx.medmlx import load_model as load_model
    from flexible_unet_mlx.medmlx import segment as segment
    from flexible_unet_mlx.network import FlexibleUNet as FlexibleUNet
    from flexible_unet_mlx.weights import load_network as load_network

__version__ = "0.2.7"
__all__ = [
    "FlexibleUNet",
    "FlexibleUNetModel",
    "forward_patch",
    "load_model",
    "load_network",
    "segment",
    "specification",
]


def __getattr__(name: str) -> object:
    """Import a public name from its submodule on first access and cache it."""
    modules = {
        "FlexibleUNet": "network",
        "FlexibleUNetModel": "medmlx",
        "forward_patch": "inference",
        "load_model": "medmlx",
        "load_network": "weights",
        "segment": "medmlx",
    }
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(f"flexible_unet_mlx.{modules[name]}"), name)
    globals()[name] = value
    return value
