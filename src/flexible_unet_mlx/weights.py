"""Validate and load native safetensors weights."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
from medmlx_core.checkpoints import HostArray, tensor_mapping_from_payload
from medmlx_core.errors import InvalidInputError
from medmlx_core.runtime import import_mlx
from medmlx_core.typing import ArrayFactory

from flexible_unet_mlx._typing import NativeRuntime
from flexible_unet_mlx.network import FlexibleUNet
from flexible_unet_mlx.runtime import require_apple_silicon
from flexible_unet_mlx.specification import specification

MODEL_ID = "endoscopic_tool_segmentation"


def _validate(
    payload: Mapping[str, object] | Mapping[object, object],
) -> dict[str, HostArray]:
    """Check names, shapes, dtypes and finiteness against ``specification.json``.

    Training-only BatchNorm counters are not part of the native checkpoint.
    """
    expected = {
        key: value
        for key, value in specification()["parameters"].items()
        if not key.endswith("num_batches_tracked")
    }
    if set(payload) != set(expected):
        raise InvalidInputError(f"{MODEL_ID} checkpoint names differ from the pinned source graph")
    state = tensor_mapping_from_payload(payload, what="endoscopic_tool_segmentation")
    if set(state) != set(expected):
        raise InvalidInputError(f"{MODEL_ID} checkpoint values must all be tensors")
    arrays: dict[str, HostArray] = {}
    for name, spec in expected.items():
        value = state[name]
        if (
            list(value.shape) != spec["shape"]
            or str(value.dtype) != spec["dtype"]
            or not np.isfinite(value).all()
        ):
            raise InvalidInputError(f"Invalid {MODEL_ID} checkpoint parameter: {name}")
        if not name.endswith("num_batches_tracked"):
            arrays[name] = np.ascontiguousarray(value)
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
    payload = mx.load(str(weights_path), format="safetensors")
    arrays = _validate(
        {name: cast(HostArray, np.asarray(value)) for name, value in payload.items()},
    )
    array = cast(ArrayFactory, mx.array)
    weights = {name: array(value) for name, value in arrays.items()}
    mx.eval(weights)
    return FlexibleUNet(weights, mx=mx)
