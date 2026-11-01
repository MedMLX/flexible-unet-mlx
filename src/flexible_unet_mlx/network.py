"""Native EfficientNet-B2/FlexibleUNet graph from MONAI bundle 0.6.2."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from medmlx_core.channel_last import ChannelLastGraph

from flexible_unet_mlx._kernels import depthwise_bn_silu, nearest2x_concat
from flexible_unet_mlx._typing import Block, NativeRuntime
from flexible_unet_mlx.runtime import require_metal
from flexible_unet_mlx.specification import specification

if TYPE_CHECKING:
    from mlx.core import array as NativeArray


class FlexibleUNet(ChannelLastGraph):
    """EfficientNet-B2 encoder and FlexibleUNet decoder for 2 classes, FP32, on Metal.

    ``weights`` maps upstream state-dict names to MLX arrays (batch-norm counters
    removed). :func:`flexible_unet_mlx.load_model` validates these arrays and returns
    a constructed model. Convolution
    kernels keep the upstream ``(out, in, kH, kW)`` checkpoint layout. The first
    call materializes convolution layouts and BN scales once, then compiles the
    fixed inference graph. Weights are immutable for the lifetime of this model;
    construct a new instance when replacing them.
    Activations are channel-last (NHWC) inside the graph. Batch norm uses the stored
    running statistics (inference only) with epsilon 1e-3 passed explicitly, to match
    the upstream graph; the shared default is 1e-5.

    The input is ``(B, 3, 736, 480)`` float32 and the output ``(B, 2, 736, 480)``
    logits, both channel-first at the boundary. Axes 2 and 3 are the image's width
    and height in the bundle's reader convention (see ``transforms.read_rgb``).
    """

    mx: NativeRuntime
    w: Mapping[str, NativeArray]

    if TYPE_CHECKING:
        # The pinned core methods operate on the injected native MLX arrays.
        def relu(self, x: NativeArray) -> NativeArray:
            """Apply the inherited elementwise ReLU operation."""
            ...
        def swish(self, x: NativeArray) -> NativeArray:
            """Apply the inherited elementwise swish operation."""
            ...
        def require_input(self, x: NativeArray, shape: tuple[int, ...]) -> None:
            """Validate the native array against the expected input shape."""
            ...

    def __init__(self, weights: Mapping[str, NativeArray], *, mx: NativeRuntime) -> None:
        """Store the weights; raise ``MissingDependencyError`` unless Metal is the default device.

        Weight names and shapes are not checked here; ``load_model`` validates them.
        """
        require_metal(mx)
        super().__init__(dict(weights), mx=mx)
        self._convolutions: dict[str, NativeArray] = {}
        self._scales: dict[str, NativeArray] = {}
        self._compiled: Callable[[NativeArray], NativeArray] | None = None

    def _prepare(self) -> None:
        """Materialize immutable model constants before tracing the inference graph."""
        self._convolutions = {
            name.removesuffix(".weight"): self.mx.contiguous(self.mx.transpose(w, (0, 2, 3, 1)))
            for name, w in self.w.items() if name.endswith(".weight") and w.ndim == 4
        }
        self._scales = {
            name.removesuffix(".running_var"): (
                self.w[name.removesuffix(".running_var") + ".weight"] * self.mx.rsqrt(w + 1e-3)
            )
            for name, w in self.w.items() if name.endswith(".running_var")
        }
        self.mx.eval(self._convolutions, self._scales)
        self._compiled = self.mx.compile(self._forward)

    def conv(
        self, x: NativeArray, name: str, *, padding: int = 0, stride: int = 1, groups: int = 1
    ) -> NativeArray:
        """Convolve using the prepared OHWI weight, retaining the checkpoint bias."""
        if self._compiled is None:
            self._prepare()
        y = self.mx.conv2d(
            x, self._convolutions[name], stride=stride, padding=padding, groups=groups
        )
        bias = self.w.get(f"{name}.bias")
        return y if bias is None else y + bias

    def batch_norm(self, x: NativeArray, name: str, *, eps: float = 1e-5) -> NativeArray:
        """Apply inference BN, using the cached scale when ``eps`` is 1e-3."""
        if self._compiled is None:
            self._prepare()
        scale = (self._scales[name] if eps == 1e-3 else
                 self.w[f"{name}.weight"] * self.mx.rsqrt(self.w[f"{name}.running_var"] + eps))
        return (x - self.w[f"{name}.running_mean"]) * scale + self.w[f"{name}.bias"]

    def pad(self, x: NativeArray, padding: list[int]) -> NativeArray:
        """Zero-pad an NHWC array by ``[left, right, top, bottom]`` pixels.

        Width gets left/right and height gets top/bottom. The amounts are the ones
        recorded from the upstream graph in ``specification.json`` (some are
        asymmetric), not derived from the input size. The stem uses this helper;
        depthwise kernels implement their padding directly during input reads.
        """
        left, right, top, bottom = padding
        return self.mx.pad(x, [(0, 0), (top, bottom), (left, right), (0, 0)])

    def mbconv(self, x: NativeArray, block: Block) -> NativeArray:
        """Apply one MBConv block (expand, depthwise, squeeze-excite, project) to NHWC ``x``."""
        if self._compiled is None:
            self._prepare()
        name = block["prefix"]
        y = x
        if block["expand_ratio"] != 1:
            y = self.swish(
                self.batch_norm(self.conv(y, f"{name}._expand_conv"), f"{name}._bn0", eps=1e-3)
            )
        left, right, top, bottom = block["depthwise_padding"]
        y = depthwise_bn_silu(
            y, self.w[f"{name}._depthwise_conv.weight"],
            self.w[f"{name}._bn1.running_mean"], self._scales[f"{name}._bn1"],
            self.w[f"{name}._bn1.bias"],
            padding=(left, right, top, bottom),
            stride=block["stride"],
            mx=self.mx,
        )
        # Squeeze-and-excitation: global spatial mean, bottleneck, sigmoid channel gate.
        squeezed = self.mx.mean(y, axis=(1, 2), keepdims=True)
        squeezed = self.conv(
            self.swish(self.conv(squeezed, f"{name}._se_reduce")), f"{name}._se_expand"
        )
        y = y * self.mx.sigmoid(squeezed)
        y = self.batch_norm(self.conv(y, f"{name}._project_conv"), f"{name}._bn2", eps=1e-3)
        return (
            y + x if block["stride"] == 1 and block["in_channels"] == block["out_channels"] else y
        )

    def __call__(self, x: NativeArray) -> NativeArray:
        """Return ``(B, 2, 736, 480)`` float32 logits for a ``(B, 3, 736, 480)`` input.

        Raises:
            MissingDependencyError: Metal is no longer the default device.
            ValueError: ``x`` is not float32 of shape ``(B, 3, 736, 480)`` with ``B >= 1``.
        """
        require_metal(self.mx)
        self.require_input(x, (3, 736, 480))
        if self._compiled is None:
            self._prepare()
        compiled = self._compiled
        if compiled is None:
            raise RuntimeError("FlexibleUNet compilation was not initialized")
        return compiled(x)

    def _forward(self, x: NativeArray) -> NativeArray:
        """Execute the pure array graph; public validation stays outside compilation."""
        spec = specification()
        # MLX convolutions are channel-last: NCHW -> NHWC here, and back at the end.
        x = self.mx.transpose(x, (0, 2, 3, 1))
        y = self.conv(self.pad(x, spec["stem_padding"]), "encoder._conv_stem", stride=2)
        y = self.swish(self.batch_norm(y, "encoder._bn0", eps=1e-3))
        # Skip features are the last block output of each extracted stack, at strides
        # 2, 4, 8, 16 and 32; the final one is the decoder's starting point.
        features: list[NativeArray] = []
        blocks = spec["blocks"]
        for i, block in enumerate(blocks):
            y = self.mbconv(y, block)
            end = i == len(blocks) - 1 or blocks[i + 1]["stack"] != block["stack"]
            if end and block["stack"] in spec["extract_stacks"]:
                features.append(y)
        # Five decoder levels, each upsampling 2x by nearest-neighbor repetition. The
        # first four concatenate the next-shallower skip feature (skip first, then the
        # upsampled tensor); the last runs at full resolution without a skip.
        for level in range(5):
            y = nearest2x_concat(y, features[-level - 2] if level < 4 else None, mx=self.mx)
            for conv in range(2):
                name = f"decoder.blocks.{level}.convs.conv_{conv}"
                y = self.relu(
                    self.batch_norm(
                        self.conv(y, f"{name}.conv", padding=1), f"{name}.adn.N", eps=1e-3
                    )
                )
        return self.mx.transpose(self.conv(y, "segmentation_head.0", padding=1), (0, 3, 1, 2))
