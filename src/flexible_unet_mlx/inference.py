"""Synchronized FP32 network-window inference."""

from __future__ import annotations

from typing import cast

import numpy as np
from medmlx_core.errors import InvalidInputError
from medmlx_core.typing import ArrayFactory

from flexible_unet_mlx._typing import FloatArray, HostArray
from flexible_unet_mlx.network import FlexibleUNet
from flexible_unet_mlx.runtime import require_metal


def forward_patch(network: FlexibleUNet, values: HostArray) -> FloatArray:
    """Run one synchronized FP32 forward pass on Metal and return raw logits.

    Args:
        network: Network from :func:`flexible_unet_mlx.load_model`.
        values: Preprocessed batch of shape ``(B, 3, 736, 480)``, ``B >= 1``, float32
            and finite. The two spatial axes follow the bundle's reader convention:
            axis 2 is image width (resized to 736) and axis 3 is image height (480).
            Intensities are expected in [0, 1] (``ScaleIntensity``), but this is not
            checked. Resizing and intensity scaling are the caller's job.

    Returns:
        Logits of shape ``(B, 2, 736, 480)``, float32 and finite, with the same spatial
        axes as the input: channel 0 is background and channel 1 is tool. No softmax,
        argmax or inverse resize is applied.

    Raises:
        InvalidInputError: ``values`` is not finite float32.
        MissingDependencyError: Apple Silicon or the Metal GPU is unavailable.
        ValueError: ``values`` does not have the shape above (raised by the network).
        RuntimeError: The outputs are not finite float32.
    """
    values = np.asarray(values)
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise InvalidInputError("Network input must be finite float32")
    mx = network.mx
    require_metal(mx)
    array = cast(ArrayFactory, mx.array)
    result = network(array(values))
    # MLX is lazy: force the graph to run, and wait for the GPU, before reading back.
    mx.eval(result)
    mx.synchronize()
    host = cast(HostArray, np.asarray(result))
    if host.dtype != np.float32 or not np.isfinite(host).all():
        raise RuntimeError("Network produced invalid float32 outputs")
    return cast(FloatArray, host)
