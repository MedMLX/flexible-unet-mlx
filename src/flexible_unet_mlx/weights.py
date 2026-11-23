"""Validate and load native safetensors weights."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from medmlx_core.errors import InvalidInputError
from medmlx_core.runtime import import_mlx
from medmlx_core.typing import MlxRuntime
from medmlx_core.weights import load_tensors

from flexible_unet_mlx.network import FlexibleUNet
from flexible_unet_mlx.runtime import require_apple_silicon
from flexible_unet_mlx.specification import specification


def load_network(weights_path: Path) -> FlexibleUNet:
    """Load native ``.safetensors`` weights into a :class:`FlexibleUNet`.

    For the published bundle, pass its ``flexibleunet.safetensors`` file path.
    The file must contain exactly the converted parameter names (no
    ``num_batches_tracked``) with the specified shapes and float32 dtype, all finite.
    Neither Torch nor MONAI is imported. The weights are evaluated eagerly so the
    returned network holds materialized arrays.

    Raises:
        InvalidInputError: The file does not have the ``.safetensors`` suffix.
        AssetNotReadyError: The file is unreadable or does not match the pinned graph.
        MissingDependencyError: The shared MLX runtime or Metal admission fails.
    """
    if weights_path.suffix != ".safetensors":
        raise InvalidInputError("Native FlexibleUNet weights must use the .safetensors suffix")
    require_apple_silicon()
    mx = cast(MlxRuntime, import_mlx())
    shapes = {
        name: tuple(spec["shape"])
        for name, spec in specification()["parameters"].items()
        if not name.endswith("num_batches_tracked")
    }
    weights = load_tensors(
        weights_path, model_id="flexible-unet", label="FlexibleUNet", shapes=shapes
    )
    return FlexibleUNet(weights, mx=mx)
