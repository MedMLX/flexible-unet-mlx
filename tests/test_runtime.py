"""Public execution admits only Apple Silicon with MLX on Metal."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from medmlx_core.errors import MissingDependencyError
from reference_data import recording

from flexible_unet_mlx import (
    FlexibleUNet,
    forward_patch,
    load_model,
    medmlx,
    runtime,
)
from flexible_unet_mlx._typing import NativeRuntime

if TYPE_CHECKING:
    from mlx.core import array as NativeArray


@pytest.mark.parametrize("host", [("unsupported", "arm64"), ("Darwin", "unsupported")])
def test_public_execution_requires_apple_silicon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host: tuple[str, str]
) -> None:
    """Reject public execution on unsupported operating systems or architectures."""
    monkeypatch.setattr(runtime.platform, "system", lambda: host[0])
    monkeypatch.setattr(runtime.platform, "machine", lambda: host[1])
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"host admission precedes checkpoint loading")
    image = tmp_path / "small.jpg"
    image.write_bytes(recording("small")[0]["jpeg"].tobytes())
    with pytest.raises(MissingDependencyError, match="macOS arm64"):
        load_model(weights)
    with pytest.raises(MissingDependencyError, match="macOS arm64"):
        medmlx.run(
            image_path=str(image), weights_path=str(weights), output_dir=str(tmp_path / "out")
        )
    with pytest.raises(MissingDependencyError, match="macOS arm64"):
        FlexibleUNet({}, mx=cast(NativeRuntime, None))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("metal_available", [False, True])
def test_direct_network_requires_metal_default_device(
    monkeypatch: pytest.MonkeyPatch, metal_available: bool
) -> None:
    """Reject direct network construction without the Metal default device."""
    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "arm64")
    mx = SimpleNamespace(
        metal=SimpleNamespace(is_available=lambda: metal_available),
        gpu="gpu",
        default_device=lambda: SimpleNamespace(type="host"),
    )
    with pytest.raises(MissingDependencyError, match="Metal GPU"):
        FlexibleUNet({}, mx=cast(NativeRuntime, mx))


def test_network_refuses_execution_after_metal_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject execution if Metal becomes unavailable after construction."""
    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "arm64")
    mx = SimpleNamespace(
        metal=SimpleNamespace(is_available=lambda: True),
        gpu="gpu",
        default_device=lambda: SimpleNamespace(type="gpu"),
    )
    network = FlexibleUNet({}, mx=cast(NativeRuntime, mx))
    mx.metal.is_available = lambda: False
    with pytest.raises(MissingDependencyError, match="Metal GPU"):
        network(cast("NativeArray", None))
    with pytest.raises(MissingDependencyError, match="Metal GPU"):
        forward_patch(network, np.zeros((1, 3, 736, 480), np.float32))
