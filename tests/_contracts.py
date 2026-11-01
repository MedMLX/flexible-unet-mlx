"""Schemas and tensor boundaries for the independent retained recordings."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Self, TypedDict

from flexible_unet_mlx._typing import Bundle, HostArray, Upstream


class ValuesDescription(TypedDict):
    files: list[str]
    shape: list[int]


class RecordingMetadata(TypedDict):
    schema_version: int
    upstream: Upstream
    bundle: Bundle
    reference_platform: str
    weights_seed: int


class FrameMetadata(RecordingMetadata):
    salt: int
    image: ValuesDescription
    restored: ValuesDescription
    png_mode: str
    png_size: list[int]


class OracleReceipt(TypedDict):
    schema_version: int
    upstream: Upstream
    bundle_revision: str
    outputs: list[str]


class ArrayArchive(Protocol):
    @property
    def files(self) -> list[str]:
        """List the array names in the archive."""
        ...
    def __getitem__(self, key: str) -> HostArray:
        """Read a named array from the archive."""
        ...
    def __enter__(self) -> Self:
        """Enter the archive context."""
        ...
    def __exit__(self, *args: object) -> None:
        """Close the archive context."""
        ...


class SafetensorsApi(Protocol):
    def save_file(
        self,
        tensor_dict: dict[str, HostArray],
        filename: str | Path,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Write named host tensors to a safetensors file."""
        ...
