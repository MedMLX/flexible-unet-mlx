"""Validate and load native safetensors weights."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from medmlx_core.errors import InvalidInputError
from medmlx_core.runtime import import_mlx

from flexible_unet_mlx._typing import NativeRuntime
from flexible_unet_mlx.network import FlexibleUNet
from flexible_unet_mlx.runtime import require_apple_silicon
from flexible_unet_mlx.specification import specification

if TYPE_CHECKING:
    from mlx.core import array as NativeArray

MODEL_ID = "endoscopic_tool_segmentation"


def _validate(state: Mapping[str, NativeArray], mx: NativeRuntime) -> dict[str, NativeArray]:
    """Check names, shapes, dtypes and finiteness against ``specification.json``.

    Checks run on the loaded MLX arrays, so each tensor is materialized once from
    the file and becomes the model weight without a host copy. Training-only
    BatchNorm counters are not part of the native checkpoint.
    """
    expected = {
        key: value
        for key, value in specification()["parameters"].items()
        if not key.endswith("num_batches_tracked")
    }
    if set(state) != set(expected):
        raise InvalidInputError(f"{MODEL_ID} checkpoint names differ from the pinned source graph")
    dtypes = {"float32": mx.float32}
    for name, spec in expected.items():
        value = state[name]
        if list(value.shape) != spec["shape"] or value.dtype != dtypes.get(spec["dtype"]):
            raise InvalidInputError(f"Invalid {MODEL_ID} checkpoint parameter: {name}")
    arrays = {name: state[name] for name in expected}
    # One evaluation loads every tensor and its finiteness flag together.
    finite = mx.stack([mx.all(mx.isfinite(value)) for value in arrays.values()])
    mx.eval(arrays, finite)
    for name, ok in zip(arrays, np.asarray(finite).tolist(), strict=True):
        if not ok:
            raise InvalidInputError(f"Invalid {MODEL_ID} checkpoint parameter: {name}")
    return arrays


def load_model(weights_path: Path) -> FlexibleUNet:
    """Load native ``.safetensors`` weights into a :class:`FlexibleUNet`.

    For the published bundle, pass its ``flexibleunet.safetensors`` file path.
    The file must contain exactly the converted parameter names (no
    ``num_batches_tracked``) with the specified shapes and float32 dtype, all finite.
    Neither Torch nor MONAI is imported. The weights are evaluated eagerly so the
    returned network holds materialized arrays.

    Raises:
        InvalidInputError: The loaded mapping or its entries fail validation.
        RuntimeError: MLX cannot open or decode the weights file.
        MissingDependencyError: The shared MLX runtime or Metal admission fails.
    """
    if weights_path.suffix != ".safetensors":
        raise InvalidInputError("Native FlexibleUNet weights must use the .safetensors suffix")
    require_apple_silicon()
    mx = cast(NativeRuntime, import_mlx())
    return FlexibleUNet(_validate(mx.load(str(weights_path), format="safetensors"), mx), mx=mx)
