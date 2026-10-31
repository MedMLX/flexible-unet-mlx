"""Schemas and tensor boundaries for the independent retained recordings."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, Self, TypedDict

from flexible_unet_mlx._typing import Bundle, HostArray, RunResult, Upstream


class ValuesDescription(TypedDict):
    files: list[str]
    shape: list[int]


class RecordingMetadata(TypedDict):
    schema_version: int
    upstream: Upstream
    bundle: Bundle
    reference_platform: str
    seed: int
    weights_seed: int
    torch_version: str
    numpy_version: str
    pillow_version: str
    threads: int
    device: str


class FrameMetadata(RecordingMetadata):
    case: str
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


class RunnerReport(TypedDict):
    forbidden: list[str]
    result: RunResult


class Tensor(Protocol):
    def detach(self) -> Self:
        """Detach the reference tensor from gradient tracking."""
        ...
    def cpu(self) -> Self:
        """Move the reference tensor to the CPU."""
        ...
    def numpy(self) -> HostArray:
        """Convert the reference tensor to a host array."""
        ...


class TorchArrayApi(Protocol):
    def from_numpy(self, ndarray: HostArray) -> Tensor:
        """Wrap a host array as a Torch tensor."""
        ...


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


type HostState = Mapping[str, HostArray]


class RunArguments(TypedDict):
    image_path: str
    weights_path: str
    output_dir: str


class SafetensorsApi(Protocol):
    def save_file(
        self,
        tensor_dict: dict[str, HostArray],
        filename: str | Path,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Write named host tensors to a safetensors file."""
        ...
