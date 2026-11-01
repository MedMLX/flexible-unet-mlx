"""NumPy ports of the MONAI 1.4.0 transforms in the pinned endoscopic tool bundle.

The bundle's ``configs/inference.json`` (revision f0416a2, pinned in
``specification.json``) reads an RGB frame with MONAI's PIL reader, moves channels
first, resizes to 736 x 480 bilinearly, rescales intensities to [0, 1], and after the
network resizes the logits back, takes the argmax and saves the tool channel as an
8-bit PNG scaled by 255. These functions reproduce those steps without Torch or
MONAI. Bilinear resizing follows ``torch.nn.functional.interpolate``
(``align_corners=False``, no antialiasing) in float32. It emulates the fused
multiply-adds of the Torch CPU build used to record the test fixtures (darwin-arm64);
on that build the tests observe zero error, but other builds may round differently.

Axis convention: the bundle's PIL reader and channel-first transform produce
``(channel, width, height)`` arrays, so the "spatial" axes of every array here are
``(width, height)``, not the usual ``(height, width)``. ``SPATIAL_SIZE`` and the
network input follow that order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from flexible_unet_mlx._typing import IntArray

type Float32Array = npt.NDArray[np.float32]

# The bundle's Resized(spatial_size=[736, 480]) on the reader's (width, height) axes:
# the image is resized to 736 wide by 480 high regardless of its aspect ratio.
SPATIAL_SIZE = (736, 480)


def read_rgb(path: Path) -> Float32Array:
    """``LoadImage`` with MONAI's PIL reader, then ``EnsureChannelFirst(channel_dim=-1)``.

    PIL's (height, width, channel) array has its first two axes swapped, as MONAI's
    ``reverse_indexing`` does, and channels are moved first: (3, width, height).

    Returns:
        Float32 array of shape ``(3, width, height)`` holding the raw 0-255 pixel
        values (not scaled).

    Raises:
        InvalidInputError: The file cannot be read as an image, or its PIL mode is not
            ``RGB`` (grayscale, palette and alpha images are rejected, not converted).
    """
    from medmlx_core.errors import InvalidInputError
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise InvalidInputError(
                    f"The endoscopic tool bundle reads RGB frames; {path.name} is {image.mode}"
                )
            data = np.moveaxis(np.asarray(image), 0, 1)
    except (OSError, UnidentifiedImageError) as error:
        raise InvalidInputError(f"Unreadable image: {path}") from error
    return np.moveaxis(data.astype(np.float32), -1, 0)


def resize_bilinear(data: Float32Array, size: tuple[int, int]) -> Float32Array:
    """``Resize(size, mode="bilinear")``: Torch's ``interpolate`` on float32 channels.

    Args:
        data: Float32 array of shape ``(C, n0, n1)``. In this pipeline the spatial axes
            are ``(width, height)`` (see the module docstring).
        size: Output ``(n0, n1)``. Interpolation is per axis with ``align_corners=False``
            and no antialiasing.

    Returns:
        A new float32 array of shape ``(C, size[0], size[1])``; a copy when the size
        is unchanged.
    """
    if tuple(data.shape[1:]) == tuple(size):
        return data.astype(np.float32, copy=True)
    row0, row1, row_low, row_high = _linear_indices(data.shape[1], size[0])
    col0, col1, col_low, col_high = _linear_indices(data.shape[2], size[1])
    low, high = data[:, row0], data[:, row1]
    top = _combine(low[:, :, col0], col_low, low[:, :, col1], col_high)
    bottom = _combine(high[:, :, col0], col_low, high[:, :, col1], col_high)
    return _combine(top, row_low[:, None], bottom, row_high[:, None])


def scale_intensity(data: Float32Array) -> Float32Array:
    """``ScaleIntensity()``: rescale the whole array to [0, 1] in float32.

    The minimum and maximum are taken over all channels together, not per channel.
    A finite constant array maps to all zeros. The caller supplies finite input;
    this helper does not validate it.
    """
    low, high = data.min(), data.max()
    if low == high:
        return (data * np.float32(0.0)).astype(np.float32)
    normalized = (data - low) / (high - low)
    # Multiply by 1 and add 0 mirror MONAI's `norm * (maxv - minv) + minv` for the
    # default [0, 1] range; they leave the float32 values unchanged.
    return (normalized * np.float32(1.0) + np.float32(0.0)).astype(np.float32, copy=False)


def tool_mask(logits: Float32Array) -> npt.NDArray[np.bool_]:
    """``AsDiscrete(argmax=True, to_onehot=2)`` then ``x[1:]``: where class 1 wins.

    Args:
        logits: Array of shape ``(2, n0, n1)``; channel 0 is background, 1 is tool.

    Returns:
        Boolean array of shape ``(n0, n1)``. Ties go to class 0 (first maximum).
    """
    return np.argmax(logits, axis=0) == 1


def write_png(path: Path, mask: npt.NDArray[np.bool_]) -> None:
    """MONAI's ``PILWriter`` with ``scale=255``: 0/255 8-bit grayscale PNG.

    ``mask`` has the reader's ``(width, height)`` axes and is transposed so the saved
    image is ``height`` rows by ``width`` columns, like the input frame.
    """
    from PIL import Image

    pixels = (np.float32(255) * mask.astype(np.float32)).astype(np.uint8)
    Image.fromarray(np.ascontiguousarray(pixels.T), mode="L").save(path, format="PNG")


def _fma(
    left: Float32Array | np.float32,
    right: Float32Array | np.float32,
    addend: Float32Array | np.float32,
) -> Float32Array:
    """Emulate a float32 fused multiply-add, ``left * right + addend``.

    The product of two float32 values is exact in float64. The float64 sum is then
    rounded to float32. That is two roundings, so in rare double-rounding cases it can
    differ from a hardware fma, which rounds once.
    """
    return (
        np.asarray(left, np.float64) * np.asarray(right, np.float64)
        + np.asarray(addend, np.float64)
    ).astype(np.float32)


def _combine(
    first: Float32Array,
    first_weight: Float32Array,
    second: Float32Array,
    second_weight: Float32Array,
) -> Float32Array:
    """ATen's ``t0 * w0 + t1 * w1``, compiled as ``fma(t0, w0, t1 * w1)``.

    This form reproduces the rounding of the darwin-arm64 Torch build used to record
    the fixtures; it is not guaranteed for other builds.
    """
    return _fma(first, first_weight, (second * second_weight).astype(np.float32))


def _linear_indices(
    input_size: int,
    output_size: int,
) -> tuple[IntArray, IntArray, Float32Array, Float32Array]:
    """``compute_source_index_and_lambda`` for ``align_corners=False``.

    Returns:
        ``(low, high, low_weight, high_weight)`` for each of ``output_size`` output
        positions: source indices into the input axis (int64) and the float32
        interpolation weights, which sum to one. Source coordinates are
        ``(i + 0.5) * input_size / output_size - 0.5``, clamped to be non-negative and
        with ``high`` clamped to the last input index.
    """
    if input_size == output_size:
        index = np.arange(output_size)
        return index, index, np.ones(output_size, np.float32), np.zeros(output_size, np.float32)
    scale = np.float32(input_size) / np.float32(output_size)
    source = _fma(
        scale, np.arange(output_size, dtype=np.float32) + np.float32(0.5), np.float32(-0.5)
    )
    source = np.where(source < 0, np.float32(0.0), source).astype(np.float32)
    low = np.minimum(np.floor(source).astype(np.int64), input_size - 1)
    high_weight = np.minimum(
        np.maximum(source - low.astype(np.float32), np.float32(0.0)), np.float32(1.0)
    ).astype(np.float32)
    high = low + (low < input_size - 1)
    return low, high, (np.float32(1.0) - high_weight).astype(np.float32), high_weight
