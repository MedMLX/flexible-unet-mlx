"""Probe Metal in a child process so host checks can collect safely."""

from __future__ import annotations

import subprocess
import sys
from typing import cast

import pytest

from flexible_unet_mlx._typing import NativeRuntime


@pytest.fixture(scope="session")
def mx() -> NativeRuntime:
    """Admit the Apple Silicon Metal runtime or skip native tests when unavailable."""
    from flexible_unet_mlx.runtime import require_apple_silicon

    require_apple_silicon()
    code = "import mlx.core as mx; raise SystemExit(0 if mx.metal.is_available() else 1)"
    probe = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if probe.returncode:
        pytest.skip(
            f"Native MLX runtime unavailable (probe exit {probe.returncode}); requires Metal"
        )
    from medmlx_core.runtime import import_mlx

    return cast(NativeRuntime, import_mlx())
