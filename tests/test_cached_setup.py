"""Offline reuse of weights prepared through the shared download contract."""

from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest
from medmlx_core import weights as shared_weights
from PIL import Image

from flexible_unet_mlx import medmlx, weights


class CheckpointReached(Exception):
    """Stop at native checkpoint I/O before learned inference."""


def test_preloaded_weights_are_reused_with_the_hub_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-download by registered ID, then reach the same checkpoint without network."""
    offline = False

    def model_info(_repo_id: str) -> SimpleNamespace:
        assert not offline, "Pre-downloaded weights must be reusable with the Hub offline"
        return SimpleNamespace(sha="a" * 40)

    def snapshot_download(_repo_id: str, *, local_dir: Path, **_options: object) -> str:
        assert not offline, "An offline run must not download another model folder"
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "flexibleunet.safetensors").write_bytes(b"native checkpoint placeholder")
        return str(local_dir)

    hub = SimpleNamespace(
        HfApi=lambda: SimpleNamespace(model_info=model_info),
        snapshot_download=snapshot_download,
    )

    def import_hub(_name: str) -> SimpleNamespace:
        return hub

    monkeypatch.setenv("MEDMLX_ASSET_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(shared_weights, "import_module", import_hub)
    model_id = "flexible-unet"
    preloaded = shared_weights.weights_dir(model_id)
    assert shared_weights.weights_revision(preloaded) == "a" * 40
    offline = True
    selected: list[Path] = []

    def load_checkpoint(path: Path, **_options: object) -> NoReturn:
        selected.append(Path(path))
        raise CheckpointReached("native checkpoint I/O reached")

    monkeypatch.setattr(weights, "load_model", load_checkpoint)
    image = tmp_path / "frame.png"
    Image.new("RGB", (8, 8), (40, 60, 80)).save(image)
    with pytest.raises(CheckpointReached, match="native checkpoint I/O reached"):
        medmlx.run(image_path=str(image), output_dir=str(tmp_path / "out"))
    assert selected == [preloaded / "flexibleunet.safetensors"]
