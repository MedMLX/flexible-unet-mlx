"""medmlx declaration, bundle transform parity with MONAI, and an end-to-end run.

The parity fixture is recorded by scripts/make_medmlx_fixtures.py, which runs the
pinned bundle's upstream MONAI 1.4.0 transforms on seeded frames.
"""

from __future__ import annotations

import importlib
import io
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import safetensors.numpy as safetensors
from _contracts import FrameMetadata, RunArguments, RunnerReport, SafetensorsApi
from medmlx_core.errors import AssetNotReadyError, InvalidInputError
from PIL import Image
from reference_data import recorded_values, recording, seeded_weights

from flexible_unet_mlx import medmlx, transforms
from flexible_unet_mlx._typing import FloatArray, HostArray, NativeRuntime
from flexible_unet_mlx.medmlx import MODEL, run

save_file = cast(SafetensorsApi, safetensors).save_file

ROOT = Path(__file__).resolve().parents[1]


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


def _weights(path: Path, seed: int) -> Path:
    """Seeded weights in the converted format, with positive BatchNorm variances."""
    save_file(seeded_weights(seed), path)
    return path


def test_entry_point_declares_the_runner() -> None:
    """Check the package entry point and registered runner declaration."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["entry-points"]["medmlx.models"] == {
        "flexible-unet": "flexible_unet_mlx.medmlx:MODEL"
    }
    assert set(MODEL) == {"id", "label", "task", "run", "summary"}
    assert MODEL["id"] == "flexible-unet"
    assert MODEL["task"] == "segmentation"
    assert MODEL["run"] == "flexible_unet_mlx.medmlx:run"
    module, _, attribute = MODEL["run"].partition(":")
    assert getattr(importlib.import_module(module), attribute) is run


def test_reader_swaps_pil_axes_and_moves_channels_first(tmp_path: Path) -> None:
    """Check RGB reader axis order and channel placement."""
    pixels = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    path = tmp_path / "frame.png"
    Image.fromarray(pixels).save(path)
    frame = transforms.read_rgb(path)
    assert frame.dtype == np.float32
    assert frame.shape == (3, 6, 4)
    assert frame[2, 5, 3] == pixels[3, 5, 2]


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


def test_run_matches_upstream_without_reference_frameworks(
    tmp_path: Path, mx: NativeRuntime
) -> None:
    """Compare seeded native runner output while blocking MONAI and Torch imports."""
    frame = _frame(tmp_path, "small")
    _, metadata = recording("runner")
    weights = _weights(tmp_path / "weights.safetensors", metadata["weights_seed"])
    # Match a Hugging Face snapshot link to an extensionless cache blob.
    blob = tmp_path / "cached-blob"
    weights.rename(blob)
    weights.symlink_to(blob)
    output_dir = tmp_path / "out"
    code = """
import importlib.abc, json, sys

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "monai"}:
            raise ModuleNotFoundError(f"Forbidden runtime dependency: {fullname}")

sys.meta_path.insert(0, Block())
from flexible_unet_mlx.medmlx import run

result = run(**json.loads(sys.argv[1]))
loaded = sorted(m for m in sys.modules if m.split(".")[0] in {"torch", "monai"})
print(json.dumps({"result": result, "forbidden": loaded}))
"""
    arguments: RunArguments = {
        "image_path": str(frame),
        "weights_path": str(weights),
        "output_dir": str(output_dir),
    }
    completed = subprocess.run(
        [sys.executable, "-c", code, json.dumps(arguments)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = cast(RunnerReport, json.loads(completed.stdout.strip().splitlines()[-1]))
    assert report["forbidden"] == []
    result = report["result"]
    mask_path = output_dir.resolve() / "small/small_trans.png"
    assert result["outputs"] == {"mask": mask_path.as_posix()}
    with Image.open(mask_path) as written:
        pixels = np.asarray(written)
        assert written.mode == "L"
        assert written.size == (120, 90)
    arrays, _ = recording("runner")
    with Image.open(io.BytesIO(arrays["png"].tobytes())) as reference:
        np.testing.assert_array_equal(pixels, np.asarray(reference))
    assert set(np.unique(pixels).tolist()) <= {0, 255}
    details = result["details"]
    assert details["image_size"] == {"width": 120, "height": 90}
    assert details["network_input_shape"] == [736, 480]
    assert details["tool_pixels"] == int((pixels == 255).sum())
    with pytest.raises(InvalidInputError, match="overwrite"):
        run(**arguments)
    again = run(**arguments, overwrite=True)
    assert again["outputs"] == result["outputs"]


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("absent", "image_path not found"),
        ("gray", "reads RGB frames; frame.png is L"),
        ("text", "Unreadable image"),
    ],
)
def test_run_rejects_invalid_frames(tmp_path: Path, corruption: str, message: str) -> None:
    """Reject missing, grayscale, or unreadable input frames."""
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"checked after the frame")
    path = tmp_path / "frame.png"
    if corruption == "gray":
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(path)
    elif corruption == "text":
        path.write_text("not an image")
    with pytest.raises(InvalidInputError, match=message):
        run(image_path=str(path), weights_path=str(weights), output_dir=str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("weights", "message"),
    [("absent", "weights not found"), ("torch", "must be converted MLX safetensors")],
)
def test_run_requires_matching_converted_weights(
    tmp_path: Path,
    weights: str,
    message: str,
) -> None:
    """Reject missing or unconverted checkpoint paths."""
    if weights == "absent":
        path = tmp_path / "missing.safetensors"
    else:
        path = tmp_path / "model.pt"
        path.write_bytes(b"unconverted")
    with pytest.raises(AssetNotReadyError, match=message):
        run(
            image_path=str(_frame(tmp_path, "small")),
            weights_path=str(path),
            output_dir=str(tmp_path / "out"),
        )
    assert not (tmp_path / "out").exists()


def test_run_rejects_mismatched_weights(tmp_path: Path, mx: NativeRuntime) -> None:
    """Reject a checkpoint that does not match the released graph."""
    path = tmp_path / "other.safetensors"
    save_file({"segmentation_head.0.weight": np.zeros((2, 16, 3, 3), np.float32)}, path)
    with pytest.raises(AssetNotReadyError, match="do not match"):
        run(
            image_path=str(_frame(tmp_path, "small")),
            weights_path=str(path),
            output_dir=str(tmp_path / "out"),
        )
    assert not (tmp_path / "out").exists()


def test_run_uses_upstream_output_layout_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise runner IO on upstream logits; native forward has its own Metal gate."""
    frame = _frame(tmp_path, "small")
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"loading is isolated in this runner-IO check")
    arrays, _ = recording("network")

    def load_network(path: Path) -> object:
        """Replace network loading with a lightweight fixture object."""
        return object()

    monkeypatch.setattr(medmlx, "_load_network", load_network)

    def predict(network: object, image: FloatArray) -> tuple[HostArray, str]:
        """Replay recorded logits for the runner output test."""
        return arrays["logits"], "test"

    monkeypatch.setattr(medmlx, "_predict", predict)
    arguments: RunArguments = {
        "image_path": str(frame),
        "weights_path": str(weights),
        "output_dir": str(tmp_path / "out"),
    }
    path = tmp_path / "out/small/small_trans.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"existing output")
    with pytest.raises(InvalidInputError, match="overwrite"):
        run(**arguments, overwrite=False)
    assert path.read_bytes() == b"existing output"
    for invalid in ("false", "true", 0, 1, None):
        with pytest.raises(InvalidInputError, match="overwrite must be boolean"):
            run(**arguments, overwrite=cast(bool, invalid))
        assert path.read_bytes() == b"existing output"
    result = run(**arguments, overwrite=True)
    assert result["outputs"] == {"mask": path.resolve().as_posix()}
    reference, _ = recording("runner")
    with Image.open(path) as written, Image.open(io.BytesIO(reference["png"].tobytes())) as saved:
        assert written.mode == saved.mode == "L"
        np.testing.assert_array_equal(np.asarray(written), np.asarray(saved))
    assert json.loads(json.dumps(result)) == result
    assert result["details"]["network_input_shape"] == [736, 480]
    with Image.open(path) as mask:
        assert mask.size == (120, 90)
        assert result["details"]["tool_pixels"] == int((np.asarray(mask) == 255).sum())


def test_run_rejects_output_dir_that_is_a_file(tmp_path: Path) -> None:
    """Reject an output directory path that is an existing file."""
    target = tmp_path / "taken"
    target.write_text("")
    with pytest.raises(InvalidInputError, match="not a directory"):
        run(image_path="frame.jpg", weights_path="weights.safetensors", output_dir=str(target))


def test_public_imports_load_no_native_or_reference_frameworks() -> None:
    """Check public imports stay lazy and avoid source frameworks."""
    code = """
import importlib.abc, sys

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx', 'torch', 'monai'}:
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, Block())
import flexible_unet_mlx as package
import flexible_unet_mlx.medmlx
for name in package.__all__:
    assert callable(getattr(package, name)), name
assert not {'mlx', 'torch', 'monai'} & {m.split('.')[0] for m in sys.modules}
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_run_reports_unreadable_converted_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classify unreadable converted weights as an asset error."""
    from flexible_unet_mlx import weights

    def unreadable(path: Path) -> None:
        """Raise a simulated checkpoint read failure."""
        raise OSError("unreadable converted weights")

    monkeypatch.setattr(weights, "load_model", unreadable)
    path = tmp_path / "weights.safetensors"
    path.write_bytes(b"unreadable")
    with pytest.raises(AssetNotReadyError, match="unreadable converted weights"):
        run(
            image_path=str(_frame(tmp_path, "small")),
            weights_path=str(path),
            output_dir=str(tmp_path / "out"),
        )
    assert not (tmp_path / "out").exists()
