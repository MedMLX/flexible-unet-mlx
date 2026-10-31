"""Native tensor and external data contracts used by this pinned graph."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, NotRequired, Protocol, TypedDict

import numpy as np
import numpy.typing as npt
from medmlx_core.typing import MlxRuntime

if TYPE_CHECKING:
    from mlx.core import StreamOrDevice, array

type HostArray = npt.NDArray[np.generic]
type FloatArray = npt.NDArray[np.float32]
type BoolArray = npt.NDArray[np.bool_]
type IntArray = npt.NDArray[np.int64]


class Parameter(TypedDict):
    """Expected shape and NumPy dtype name of one upstream state-dict tensor."""

    shape: list[int]
    dtype: str


class Upstream(TypedDict):
    """Pinned MONAI source release and commit."""

    repository: str
    version: str
    revision: str


class Bundle(TypedDict):
    """Pinned Hugging Face bundle revision and its weight files."""

    repo_id: str
    revision: str
    gated: bool
    weights: list[str]


class Block(TypedDict):
    """One MBConv block of the EfficientNet-B2 encoder.

    ``prefix`` is the state-dict key prefix. ``stack`` is the 1-based EfficientNet
    stack the block belongs to. ``depthwise_padding`` is ``[left, right, top, bottom]``
    in pixels, applied before the depthwise convolution (the same order as
    ``torch.nn.functional.pad`` on the last two axes).
    """

    prefix: str
    stack: int
    stride: int
    in_channels: int
    out_channels: int
    expand_ratio: int
    depthwise_padding: list[int]


class Specification(TypedDict):
    """Layout of ``specification.json``, the single definition of the pinned graph.

    ``parameters`` lists every tensor of the upstream state dict in upstream order.
    ``extract_stacks`` are the encoder stacks whose outputs feed the decoder as skip
    connections, and ``stem_padding`` is ``[left, right, top, bottom]`` for the stem.
    """

    schema_version: int
    upstream: Upstream
    bundle: Bundle
    parameters: dict[str, Parameter]
    extract_stacks: list[int]
    stem_padding: list[int]
    blocks: list[Block]


type NativeTree = array | Sequence[NativeTree] | Mapping[str, NativeTree]


class NativeRuntime(MlxRuntime, Protocol):
    def load(self, file: str, /, format: Literal["safetensors"]) -> dict[str, array]:
        """Read named MLX arrays from a safetensors checkpoint."""
        ...

    def save_safetensors(self, file: str, arrays: dict[str, array]) -> None:
        """Write named MLX arrays as safetensors."""
        ...

    def eval(self, *values: NativeTree) -> None:
        """Evaluate MLX arrays in the supplied trees."""
        ...

    def compile(self, fun: Callable[[array], array]) -> Callable[[array], array]:
        """Compile an array-to-array inference function."""
        ...

    def contiguous(self, x: array, /, *, stream: StreamOrDevice = None) -> array:
        """Return a contiguous copy when the array is not contiguous."""
        ...

    def synchronize(self, stream: StreamOrDevice = None) -> None:
        """Wait for work on the specified or default stream to finish."""
        ...

    def full(self, shape: Sequence[int], vals: float) -> array:
        """Create an array filled with the specified scalar."""
        ...


class ImageSize(TypedDict):
    """Original frame size in pixels."""

    width: int
    height: int


class Seconds(TypedDict):
    """Wall-clock seconds per runner stage, rounded to 4 decimals."""

    load_inputs_and_weights: float
    preprocess: float
    inference: float
    postprocess_and_write: float


class RunDetails(TypedDict):
    """JSON-serializable metadata returned by ``medmlx.run``.

    ``network_input_shape`` is the two spatial sizes of the network input, ``[736, 480]``.
    ``tool_pixels`` counts mask pixels and ``tool_fraction`` is that count divided by
    the number of frame pixels (0 to 1, rounded to 6 decimals). ``mlx_device`` is the
    string form of MLX's default device. ``requires_human_review`` is always ``True``.
    """

    weights_revision: NotRequired[str | None]
    bundle: str
    labels: dict[str, str]
    image_size: ImageSize
    network_input_shape: list[int]
    tool_pixels: int
    tool_fraction: float
    mlx_device: str
    seconds: Seconds
    requires_human_review: bool


class RunResult(TypedDict):
    """Return value of ``medmlx.run``; ``outputs["mask"]`` is the absolute POSIX PNG path."""

    outputs: dict[str, str]
    details: RunDetails
