"""Apple Silicon and Metal admission before shared MLX execution."""

from __future__ import annotations

import platform

from medmlx_core.errors import MissingDependencyError

from flexible_unet_mlx._typing import NativeRuntime


def require_apple_silicon() -> None:
    """Raise ``MissingDependencyError`` unless running on macOS arm64."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise MissingDependencyError(
            "flexible-unet-mlx requires macOS arm64 (Apple Silicon)",
            hint="Run on an Apple Silicon Mac with MLX on Metal.",
        )


def require_metal(mx: NativeRuntime) -> None:
    """Raise ``MissingDependencyError`` unless Metal is available and MLX's default device.

    ``mx`` is the module returned by ``medmlx_core.import_mlx()``. The check is repeated
    on every forward pass, so changing the default device after construction is caught.
    """
    require_apple_silicon()
    if not mx.metal.is_available() or mx.default_device().type != mx.gpu:
        raise MissingDependencyError(
            "flexible-unet-mlx requires an available Metal GPU as the MLX default device",
            hint="Initialize MLX with medmlx_core.import_mlx() on an Apple Silicon Mac.",
        )
