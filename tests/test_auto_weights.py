"""Automatic weights and explicit folders reach the native loader unchanged."""

from pathlib import Path
from typing import NoReturn

import pytest

from flexible_unet_mlx import medmlx


class LoadingReached(Exception):
    """Stop before inference while observing the checkpoint selected by the runner."""


@pytest.mark.parametrize("automatic", [True, False])
def test_weights_folder_selects_native_file_without_resolving_cache_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, automatic: bool,
) -> None:
    """Check automatic and explicit folders select the named symlinked checkpoint."""
    folder = tmp_path / "model"
    folder.mkdir()
    blob = tmp_path / "suffixless-blob"
    blob.write_bytes(b"fixture")
    checkpoint = folder / "flexibleunet.safetensors"
    checkpoint.symlink_to(blob)
    image = tmp_path / "input.nii.gz"
    image.write_bytes(b"input is not read before the loader")
    downloads: list[str] = []
    selected: list[Path] = []

    def download(model_id: str) -> Path:
        """Record the requested model repository and return its fixture folder."""
        downloads.append(model_id)
        return folder

    def load(path: str | Path) -> NoReturn:
        """Record the checkpoint path and stop before network loading."""
        selected.append(Path(path))
        raise LoadingReached

    monkeypatch.setattr(medmlx, "weights_dir", download)
    monkeypatch.setattr(medmlx, "_weights_file", load)
    with pytest.raises(LoadingReached):
        medmlx.run(
            image_path=str(image), output_dir=str(tmp_path / "out"),
            weights_path=None if automatic else str(folder),
        )
    assert downloads == (["flexible-unet"] if automatic else [])
    assert selected == [checkpoint]
    assert checkpoint.is_symlink()
