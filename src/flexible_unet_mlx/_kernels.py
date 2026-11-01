"""FP32 Metal fusions for the pinned FlexibleUNet encoder and decoder.

Depthwise convolution consumes Torch-layout weights and applies inference batch
normalization before SiLU. Decoder resize writes skip channels first, matching
MONAI's concatenation order. Neither operation changes the model's precision.
"""

from __future__ import annotations

from functools import cache
from math import prod
from typing import TYPE_CHECKING, cast

from medmlx_core.typing import MetalKernel, MlxRuntime

if TYPE_CHECKING:
    from mlx.core import array as Array

_DEPTHWISE = """
uint i = thread_position_in_grid.x;
if (i >= N * OH * OW * C) return;
uint c = i % C;
uint ox = (i / C) % OW;
uint oy = (i / (C * OW)) % OH;
uint n = i / (C * OW * OH);
float acc = 0.0f;
for (int ky = 0; ky < K; ++ky) {
    int iy = int(oy) * S + ky - PT;
    for (int kx = 0; kx < K; ++kx) {
        int ix = int(ox) * S + kx - PL;
        if (iy >= 0 && iy < H && ix >= 0 && ix < W)
            acc = metal::fma(inp[((n * H + iy) * W + ix) * C + c],
                             weight[(c * K + ky) * K + kx], acc);
    }
}
float v = (acc - mean[c]) * scale[c] + beta[c];
out[i] = v / (1.0f + metal::exp(-v));
"""

_RESIZE = """
uint i = thread_position_in_grid.x;
if (i >= N * H * W * 4 * (CI + CS)) return;
uint c = i % (CI + CS);
uint pixel = i / (CI + CS);
if (c < CS) {
    out[i] = skip[pixel * CS + c];
} else {
    uint ox = pixel % (2 * W);
    uint oy = (pixel / (2 * W)) % (2 * H);
    uint n = pixel / (4 * H * W);
    out[i] = inp[((n * H + oy / 2) * W + ox / 2) * CI + c - CS];
}
"""


@cache
def _kernel(mx: MlxRuntime, depthwise: bool) -> MetalKernel:
    """Build (once per runtime) the depthwise+BN+SiLU kernel or the resize+concat kernel.

    ``math_mode="safe"`` avoids fast-math approximations so ``exp`` and the fused
    multiply-adds stay at full FP32 accuracy.
    """
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="flexible_unet_depthwise_bn_silu" if depthwise else "flexible_unet_resize_concat",
            input_names=["inp", "weight", "mean", "scale", "beta"]
            if depthwise else ["inp", "skip"],
            output_names=["out"],
            source=_DEPTHWISE if depthwise else _RESIZE,
            compile_options={"math_mode": "safe"},
        ),
    )


def _count(shape: tuple[int, ...]) -> int:
    """Element count of ``shape``; the kernels index with 32-bit ``uint``/``int``."""
    count = prod(shape)
    if min(shape) < 1 or count > 2**31 - 1:
        raise ValueError("FlexibleUNet fusion requires positive dimensions within Metal limits")
    return count


def depthwise_bn_silu(
    x: Array,
    weight: Array,
    mean: Array,
    scale: Array,
    beta: Array,
    *,
    padding: tuple[int, int, int, int],
    stride: int,
    mx: MlxRuntime,
) -> Array:
    """Fuse NHWC depthwise convolution, inference BN and SiLU with asymmetric padding.

    ``weight`` is FP32 ``(C, 1, K, K)`` for K=3 or 5. The channel vectors use
    ``scale = gamma / sqrt(running_var + epsilon)``. Padding is left/right/top/bottom;
    zero padding is handled during reads, without allocating a padded tensor.
    Unsupported shapes, strides, padding or dtypes raise before kernel dispatch.
    """
    if x.ndim != 4 or weight.ndim != 4:
        raise ValueError("Depthwise fusion requires NHWC input and rank-four weights")
    if any(value.dtype != mx.float32 for value in (x, weight, mean, scale, beta)):
        raise TypeError("Depthwise fusion requires float32 tensors")
    n, h, w, c = x.shape
    k = weight.shape[2]
    if weight.shape != (c, 1, k, k) or k not in (3, 5):
        raise ValueError("Depthwise fusion requires (C, 1, K, K) weights with K=3 or 5")
    if any(value.shape != (c,) for value in (mean, scale, beta)):
        raise ValueError("Depthwise batch-normalization vectors must match the input channels")
    if stride not in (1, 2) or min(padding) < 0:
        raise ValueError("Depthwise fusion requires stride 1 or 2 and nonnegative padding")
    left, right, top, bottom = padding
    shape = (n, (h + top + bottom - k) // stride + 1, (w + left + right - k) // stride + 1, c)
    count = _count(shape)
    _count(tuple(x.shape))
    _count(tuple(weight.shape))
    return _kernel(mx, True)(
        inputs=[x, weight, mean, scale, beta],
        template=list(dict(N=n, H=h, W=w, C=c, OH=shape[1], OW=shape[2],
                           K=k, S=stride, PT=top, PL=left).items()),
        grid=(count, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shape],
        output_dtypes=[mx.float32],
        stream=mx.gpu,
    )[0]


def nearest2x_concat(x: Array, skip: Array | None, *, mx: MlxRuntime) -> Array:
    """Write NHWC nearest 2x resize and optional skip-first concatenation in one pass.

    The skip must share the input batch and have doubled spatial dimensions.
    Inputs must be FP32 with positive dimensions; invalid shapes fail before Metal.
    """
    if x.ndim != 4 or (skip is not None and skip.ndim != 4):
        raise ValueError("Decoder fusion requires rank-four NHWC tensors")
    if x.dtype != mx.float32 or (skip is not None and skip.dtype != mx.float32):
        raise TypeError("Decoder fusion requires float32 tensors")
    n, h, w, ci = x.shape
    cs = 0 if skip is None else skip.shape[-1]
    if skip is not None and (skip.shape[:3] != (n, 2 * h, 2 * w) or cs < 1):
        raise ValueError("Decoder skip must match the batch and doubled spatial dimensions")
    _count(tuple(x.shape))
    shape = (n, 2 * h, 2 * w, ci + cs)
    count = _count(shape)
    # With CS=0 the shader never reads skip; reuse x instead of allocating a dummy buffer.
    return _kernel(mx, False)(
        inputs=[x, x if skip is None else skip],
        template=list(dict(N=n, H=h, W=w, CI=ci, CS=cs).items()),
        grid=(count, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shape],
        output_dtypes=[mx.float32],
        stream=mx.gpu,
    )[0]
