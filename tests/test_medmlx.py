"""Bundle transform parity with MONAI, an end-to-end run and argument rejection.

The parity fixtures were recorded outside this repository by running the pinned
bundle's upstream MONAI 1.4.0 transforms on seeded frames.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import safetensors.numpy as safetensors
from _contracts import FrameMetadata, SafetensorsApi
from medmlx_core.errors import AssetNotReadyError, InvalidInputError
from PIL import Image
from reference_data import recorded_values, recording, seeded_weights

from flexible_unet_mlx import medmlx, transforms
from flexible_unet_mlx._typing import FloatArray, NativeRuntime
from flexible_unet_mlx.medmlx import run

save_file = cast(SafetensorsApi, safetensors).save_file


def _logits(shape: tuple[int, ...], salt: int) -> FloatArray:
    """The fixture script's deterministic logits (integer hashing, exact float32 division)."""
    index = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    hashed = (index * 2654435761 + salt * 40503) % 2003
    return ((hashed - 1001).astype(np.float32) / np.float32(257.0)).astype(np.float32)


def _frame(tmp_path: Path, case: str) -> Path:
    """Write a recorded JPEG frame into the test directory."""
    arrays, _ = recording(case)
    path = tmp_path / f"{case}.jpg"
    path.write_bytes(arrays["jpeg"].tobytes())
    return path


@pytest.mark.parametrize("case", ["small", "tall", "wide"])
def test_transforms_match_the_bundle(tmp_path: Path, case: str) -> None:
    """Compare preprocessing and restored masks with recorded MONAI transforms."""
    arrays, metadata = recording(case)
    record = cast(FrameMetadata, metadata)
    frame = transforms.read_rgb(_frame(tmp_path, case))
    image = medmlx.preprocess(frame)[0]
    assert image.dtype == np.float32
    # Observed maximum error is zero against the recording (Torch 2.13.0, Apple
    # Silicon). One float32 epsilon permits rounding in the normalized [0, 1] range
    # across NumPy/Torch arithmetic.
    np.testing.assert_allclose(
        image,
        recorded_values(record["image"]),
        atol=cast(float, np.finfo(np.float32).eps),
        rtol=0,
        equal_nan=False,
        strict=True,
    )
    logits = _logits((2, *image.shape[1:]), record["salt"])
    restored = transforms.resize_bilinear(logits, frame.shape[1:])
    # Observed maximum error is zero against the recording; one epsilon times the
    # synthetic logits' maximum magnitude (1001/257) allows float32 rounding,
    # including cancellation.
    np.testing.assert_allclose(
        restored,
        recorded_values(record["restored"]),
        atol=cast(float, np.finfo(np.float32).eps * (1001 / 257)),
        rtol=0,
        equal_nan=False,
        strict=True,
    )
    mask = medmlx.postprocess(logits, frame.shape[1:])
    path = tmp_path / f"{case}.png"
    transforms.write_png(path, mask)
    reference = io.BytesIO(arrays["png"].tobytes())
    with Image.open(path) as written, Image.open(reference) as saved:
        assert written.mode == saved.mode == record["png_mode"]
        assert list(written.size) == list(saved.size) == record["png_size"]
        assert np.array_equal(np.asarray(written), np.asarray(saved))


def test_run_matches_upstream(tmp_path: Path, mx: NativeRuntime) -> None:
    """The runner's mask from a seeded weights folder matches the upstream runner mask."""
    frame = _frame(tmp_path, "small")
    arrays, metadata = recording("runner")
    folder = tmp_path / "model"
    folder.mkdir()
    # Match a Hugging Face snapshot link to an extensionless cache blob.
    blob = tmp_path / "cached-blob"
    save_file(seeded_weights(metadata["weights_seed"]), blob)
    (folder / "flexibleunet.safetensors").symlink_to(blob)
    result = run(image_path=str(frame), weights_path=str(folder), output_dir=str(tmp_path / "out"))
    with Image.open(result["outputs"]["mask"]) as written, Image.open(
        io.BytesIO(arrays["png"].tobytes())
    ) as reference:
        assert written.mode == reference.mode == "L"
        np.testing.assert_array_equal(np.asarray(written), np.asarray(reference))


@pytest.mark.parametrize(
    ("problem", "error"),
    [("torch_weights", AssetNotReadyError), ("overwrite_not_bool", InvalidInputError),
     ("existing_output", InvalidInputError)],
)
def test_run_rejects_invalid_arguments_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str, error: type[Exception],
) -> None:
    """Reject each invalid argument category before any weights are downloaded."""

    def no_download(model_id: str) -> Path:
        """Fail the test if the runner reaches the weights download."""
        raise AssertionError(f"downloaded {model_id}")

    monkeypatch.setattr(medmlx, "weights_dir", no_download)
    image, output = _frame(tmp_path, "small"), tmp_path / "out"
    weights, overwrite = None, False
    if problem == "torch_weights":
        (tmp_path / "model.pt").write_bytes(b"unconverted")
        weights = str(tmp_path / "model.pt")
    elif problem == "overwrite_not_bool":
        overwrite = cast(bool, "true")
    else:
        (output / "small").mkdir(parents=True)
        (output / "small/small_trans.png").write_bytes(b"existing output")
    with pytest.raises(error):
        run(
            image_path=str(image), weights_path=weights, output_dir=str(output),
            overwrite=overwrite,
        )
