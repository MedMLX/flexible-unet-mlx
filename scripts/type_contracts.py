"""Interfaces of the pinned MONAI 1.4 and Torch reference recorder APIs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self

if TYPE_CHECKING:
    from flexible_unet_mlx._typing import FloatArray, HostArray


class Tensor(Protocol):
    """Torch tensor surface used to read reference outputs and state-dict tensors."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the tensor dimensions."""
        ...
    def numpy(self) -> HostArray:
        """Return the tensor as a NumPy array."""
        ...
    def detach(self) -> Self:
        """Detach the tensor from gradient tracking."""
        ...
    def cpu(self) -> Self:
        """Move the tensor to the CPU."""
        ...


class TorchArrayApi(Protocol):
    """Torch module functions used to build and seed reference tensors."""

    def from_numpy(self, ndarray: HostArray) -> Tensor:
        """Wrap a NumPy array as a Torch tensor."""
        ...
    def zeros_like(self, input: Tensor) -> Tensor:
        """Create a zero tensor with the input shape and dtype."""
        ...


class Padding(Protocol):
    """Source SAME-padding module exposing the widths applied to the last two axes.

    ``padding`` is ``(left, right, top, bottom)`` in pixels, the order
    ``torch.nn.ConstantPad2d`` expects.
    """

    @property
    def padding(self) -> tuple[int, ...]:
        """Return the padding widths in source order."""
        ...


class MBConv(Protocol):
    """Upstream MBConv block metadata compared against ``specification.json``."""

    @property
    def stride(self) -> int:
        """Return the depthwise convolution stride."""
        ...
    @property
    def in_channels(self) -> int:
        """Return the block input channel count."""
        ...
    @property
    def out_channels(self) -> int:
        """Return the block output channel count."""
        ...
    @property
    def expand_ratio(self) -> int:
        """Return the block expansion ratio."""
        ...
    @property
    def _depthwise_conv_padding(self) -> Padding:
        """Return the source depthwise padding module."""
        ...


class BlockGroup(Protocol):
    """Upstream container of the named MBConv blocks in one encoder stack."""

    def named_children(self) -> Iterable[tuple[str, MBConv]]:
        """Iterate named MBConv blocks in the group."""
        ...


class Encoder(Protocol):
    """Upstream EfficientNet-B2 encoder surface read while verifying ``specification.json``."""

    @property
    def _conv_stem_padding(self) -> Padding:
        """Return the source stem padding module."""
        ...
    @property
    def extract_stacks(self) -> list[int]:
        """Return the encoder stacks used as decoder features."""
        ...
    @property
    def _blocks(self) -> Iterable[BlockGroup]:
        """Iterate source encoder block groups."""
        ...


class ReferenceNetwork(Protocol):
    """Upstream FlexibleUNet surface used to load weights and run reference forwards."""

    @property
    def encoder(self) -> Encoder:
        """Return the source EfficientNet encoder."""
        ...
    def eval(self) -> Self:
        """Switch the source model to inference mode."""
        ...
    def float(self) -> Self:
        """Convert the source model parameters to float32."""
        ...
    def state_dict(self) -> Mapping[str, Tensor]:
        """Return the source model tensor mapping."""
        ...
    def load_state_dict(self, state_dict: Mapping[str, Tensor], strict: bool) -> object:
        """Load the source model tensor mapping with the selected strictness."""
        ...
    def __call__(self, input: Tensor) -> Tensor:
        """Run the source model on an input tensor."""
        ...


type ImageData = dict[str, object]


class DataTransform(Protocol):
    """Source dictionary transform over the bundle's image data."""

    def __call__(self, data: ImageData) -> ImageData:
        """Apply the source dictionary transform."""
        ...


class DataPipeline(DataTransform, Protocol):
    """Source ``Compose`` pipeline exposing its ordered transforms."""

    @property
    def transforms(self) -> list[DataTransform]:
        """Return the ordered source transforms."""
        ...


class ConfigParser(Protocol):
    """MONAI bundle configuration parser for one inference configuration."""

    def read_config(self, f: str) -> None:
        """Read one source bundle configuration file."""
        ...
    def __setitem__(self, id: str, config: object) -> None:
        """Set a source configuration value by identifier."""
        ...
    def get_parsed_content(self, id: str) -> object:
        """Instantiate or return parsed source configuration content."""
        ...


class ConfigParserApi(Protocol):
    """Module surface exposing the MONAI bundle ``ConfigParser`` constructor."""

    def ConfigParser(self) -> ConfigParser:
        """Create a MONAI bundle configuration parser."""
        ...


class MonaiDataApi(Protocol):
    """``monai.data`` surface used to attach MONAI metadata to reference tensors."""

    def MetaTensor(self, x: Tensor) -> Tensor:
        """Wrap a tensor with MONAI metadata."""
        ...


class MonaiApi(Protocol):
    """``monai`` module metadata surface checked during upstream admission."""

    @property
    def __revision_id__(self) -> str:
        """Return the installed MONAI source revision identifier."""
        ...
    @property
    def __version__(self) -> str:
        """Return the installed MONAI version."""
        ...
    @property
    def __file__(self) -> str:
        """Return the installed MONAI module path."""
        ...


class ArrayArchive(Protocol):
    """Compressed NumPy archive opened for reading as a context manager."""

    @property
    def files(self) -> list[str]:
        """List stored array names."""
        ...
    def __getitem__(self, key: str) -> HostArray:
        """Read a named array from the archive."""
        ...
    def __enter__(self) -> Self:
        """Enter the open-array archive context."""
        ...
    def __exit__(self, *args: object) -> None:
        """Close the array archive context."""
        ...


class ReferenceOutput(Protocol):
    """Callable source network returning one tensor or named output tensors."""

    def __call__(self, input: Tensor) -> Tensor | dict[str, Tensor]:
        """Run the source model and return tensor output or named tensors."""
        ...


class FloatTensor(Tensor, Protocol):
    """Reference tensor whose NumPy conversion is float32."""

    def numpy(self) -> FloatArray:
        """Return the tensor as a float32 NumPy array."""
        ...


class NumpyArchiveApi(Protocol):
    """NumPy module surface used to write compressed named arrays."""

    def savez_compressed(self, file: str | Path, **kwds: HostArray) -> None:
        """Write named arrays to a compressed NumPy archive."""
        ...
