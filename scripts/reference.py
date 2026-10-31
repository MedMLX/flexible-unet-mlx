"""Pinned upstream admission shared by the offline reference scripts."""

from __future__ import annotations

import json
import platform
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, cast

from source_admission import require_git_sources
from type_contracts import MonaiApi

if TYPE_CHECKING:
    from flexible_unet_mlx._typing import Specification

ROOT = Path(__file__).resolve().parents[1]


def source_specification() -> Specification:
    """Read ``specification.json`` from the working tree without importing the package."""
    return cast(
        "Specification", json.loads((ROOT / "src/flexible_unet_mlx/specification.json").read_text())
    )


def require_upstream(bundle: Path, source_repository: Path | None = None) -> Specification:
    """Admit the MONAI revision and exact pinned bundle configuration bytes.

    Git-less bundle downloads require a local verification repository via
    ``source_repository``. This checks configuration source, not LFS weights.
    """
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Record upstream references on an Apple Silicon Mac (macOS arm64)")

    monai = cast(MonaiApi, import_module("monai"))

    spec = source_specification()
    if monai.__version__ != spec["upstream"]["version"]:
        raise RuntimeError(f"Reference generation requires MONAI {spec['upstream']['version']}")
    if monai.__revision_id__ != spec["upstream"]["revision"]:
        raise RuntimeError(
            f"Reference generation requires MONAI revision {spec['upstream']['revision']}"
        )
    require_git_sources(
        bundle,
        spec["bundle"]["revision"],
        {"configs/inference.json": "configs/inference.json"},
        source_repository,
    )
    return spec
