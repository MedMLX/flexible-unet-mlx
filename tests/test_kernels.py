"""Independent coordinate and channel-order checks for the fused FP32 operations."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from medmlx_core.typing import ArrayFactory

from flexible_unet_mlx._kernels import depthwise_bn_silu, nearest2x_concat
from flexible_unet_mlx._typing import FloatArray, NativeRuntime


@pytest.mark.parametrize("kernel,stride", [(3, 1), (3, 2), (5, 2)])
def test_depthwise_fusion_preserves_asymmetric_padding(
    mx: NativeRuntime, kernel: int, stride: int
) -> None:
    """Compare fused depthwise convolution with reference asymmetric padding."""
    rng = np.random.default_rng(18)
    x = rng.normal(size=(2, 5, 7, 3)).astype(np.float32)
    weight = rng.normal(size=(3, 1, kernel, kernel)).astype(np.float32)
    mean, scale, beta = rng.normal(size=(3, 3)).astype(np.float32)
    left, right, top, bottom = (0, kernel - 1, 1, kernel - 2)
    padded = np.pad(x, ((0, 0), (top, bottom), (left, right), (0, 0)))
    oh = (padded.shape[1] - kernel) // stride + 1
    ow = (padded.shape[2] - kernel) // stride + 1
    expected = np.empty((2, oh, ow, 3), dtype=np.float32)
    spatial_weight = weight[:, 0].transpose(1, 2, 0)
    for y in range(oh):
        for xx in range(ow):
            patch = padded[:, y * stride:y * stride + kernel, xx * stride:xx * stride + kernel]
            expected[:, y, xx] = np.sum(patch * spatial_weight, axis=(1, 2))
    expected = (expected - mean) * scale + beta
    expected = expected / (1 + np.exp(-expected))
    array = cast(ArrayFactory, mx.array)
    actual = depthwise_bn_silu(
        array(x), array(weight), array(mean), array(scale), array(beta),
        padding=(left, right, top, bottom), stride=stride, mx=mx,
    )
    mx.eval(actual)
    assert actual.dtype == mx.float32
    np.testing.assert_allclose(cast(FloatArray, np.asarray(actual)), expected, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("with_skip", [False, True])
def test_decoder_fusion_preserves_batches_and_skip_first_order(
    mx: NativeRuntime, with_skip: bool
) -> None:
    """Compare fused upsampling and skip concatenation with reference arrays."""
    array = cast(ArrayFactory, mx.array)
    host = np.arange(2 * 3 * 5 * 7, dtype=np.float32).reshape(2, 3, 5, 7)
    skip = np.full((2, 6, 10, 2), -1, dtype=np.float32)
    expected = np.repeat(np.repeat(host, 2, axis=1), 2, axis=2)
    if with_skip:
        expected = np.concatenate([skip, expected], axis=-1)
    # Feed a real strided NHWC view to exercise the custom-kernel contiguity boundary.
    x = mx.transpose(array(host.transpose(0, 3, 1, 2).copy()), (0, 2, 3, 1))
    actual = nearest2x_concat(x, array(skip) if with_skip else None, mx=mx)
    mx.eval(actual)
    assert actual.dtype == mx.float32
    np.testing.assert_array_equal(cast(FloatArray, np.asarray(actual)), expected)
    with pytest.raises(ValueError, match="doubled spatial"):
        nearest2x_concat(x, array(skip[:, :-1]), mx=mx)
