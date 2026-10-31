"""Released-source parity and strict converted-checkpoint admission."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import safetensors.numpy as safetensors
from _contracts import OracleReceipt, SafetensorsApi
from medmlx_core.errors import InvalidInputError
from reference_data import recording, seeded_weights

from flexible_unet_mlx import (
    forward_patch,
    load_model,
    specification,
)
from flexible_unet_mlx._typing import FloatArray, NativeRuntime

if TYPE_CHECKING:
    from mlx.core import array as NativeArray

save_file = cast(SafetensorsApi, safetensors).save_file


def test_converted_model_matches_independent_source(tmp_path: Path, mx: NativeRuntime) -> None:
    """Compare an explicitly supplied released checkpoint with independent source outputs."""
    checkpoint = os.environ.get("MODEL_WEIGHTS")
    reference = os.environ.get("MODEL_SOURCE_OUTPUT")
    input_path = os.environ.get("MODEL_NETWORK_INPUT")
    if checkpoint is None or reference is None or input_path is None:
        pytest.skip("Set MODEL_WEIGHTS, MODEL_NETWORK_INPUT and MODEL_SOURCE_OUTPUT")
    receipt = cast(OracleReceipt, json.loads((Path(reference) / "source.json").read_text()))
    spec = specification()
    assert receipt["schema_version"] == 1
    assert receipt["upstream"] == spec["upstream"]
    assert receipt["bundle_revision"] == spec["bundle"]["revision"]
    network = load_model(Path(checkpoint))
    values = np.load(input_path, allow_pickle=False)
    assert values.shape == (1, 3, 736, 480)
    result = forward_patch(network, values)
    outputs = result if isinstance(result, dict) else {"logits": result}
    for name, actual in outputs.items():
        assert name in receipt["outputs"]
        reference_path = Path(reference) / f"{name}.npy"
        expected = np.load(reference_path, allow_pickle=False)
        assert expected.shape == actual.shape
        assert actual.dtype == np.float32 and np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
        np.testing.assert_array_equal(actual.argmax(axis=1), expected.argmax(axis=1))
    invalid = values.copy()
    invalid.flat[0] = np.nan
    with pytest.raises(InvalidInputError, match="finite float32"):
        forward_patch(network, invalid)


@pytest.fixture(scope="module")
def converted_weights(
    tmp_path_factory: pytest.TempPathFactory, mx: NativeRuntime
) -> dict[str, NativeArray]:
    """Create and validate a seeded converted checkpoint for admission cases."""
    destination = tmp_path_factory.mktemp("converted") / "model.safetensors"
    _, metadata = recording("network")
    save_file(seeded_weights(metadata["weights_seed"]), destination)
    load_model(destination)
    return mx.load(str(destination), format="safetensors")


def test_load_model_rejects_npz(tmp_path: Path, mx: NativeRuntime) -> None:
    """Reject checkpoints that do not use the safetensors format."""
    path = tmp_path / "weights.npz"
    np.savez(path, sample=np.zeros(1, dtype=np.float32))
    with pytest.raises(InvalidInputError, match=r"must use the \.safetensors suffix"):
        load_model(path)


@pytest.mark.parametrize("corruption", ["missing", "extra", "shape", "dtype", "nonfinite"])
def test_converted_checkpoint_admission(
    tmp_path: Path, converted_weights: dict[str, NativeArray], corruption: str, mx: NativeRuntime
) -> None:
    """Reject missing, extra, malformed, or non-finite checkpoint tensors."""
    weights = dict(converted_weights)
    name = next(key for key, value in weights.items() if value.dtype == mx.float32)
    value = weights[name]
    if corruption == "missing":
        del weights[name]
    elif corruption == "extra":
        weights["unexpected.weight"] = mx.zeros((1,))
    elif corruption == "shape":
        weights[name] = mx.zeros((1,))
    elif corruption == "dtype":
        weights[name] = value.astype(mx.float16)
    else:
        weights[name] = mx.full(value.shape, float("nan"))
    destination = tmp_path / "invalid.safetensors"
    mx.save_safetensors(str(destination), weights)
    with pytest.raises(InvalidInputError):
        load_model(destination)


def test_seeded_network_matches_upstream(tmp_path: Path, mx: NativeRuntime) -> None:
    """Compare seeded native network logits and decisions with source recordings."""
    arrays, metadata = recording("network")
    destination = tmp_path / "seeded.safetensors"
    save_file(seeded_weights(metadata["weights_seed"]), destination)
    from flexible_unet_mlx import medmlx, transforms

    frame, _ = recording("small")
    image_path = tmp_path / "small.jpg"
    image_path.write_bytes(frame["jpeg"].tobytes())
    values = medmlx.preprocess(transforms.read_rgb(image_path))
    actual = cast(FloatArray, forward_patch(load_model(destination), values))
    expected = cast(FloatArray, arrays["logits"])
    assert actual.dtype == np.float32 and np.isfinite(actual).all()
    # FP32 reductions differ between Torch and MLX, so compare within a tolerance.
    # This seeded-weights bound is tighter than the released-checkpoint gate (1e-4).
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)
    np.testing.assert_array_equal(actual.argmax(axis=1), expected.argmax(axis=1))
