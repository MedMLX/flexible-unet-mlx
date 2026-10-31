"""Read fixed Mac upstream recordings; generation scripts stay outside tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
from _contracts import ArrayArchive, RecordingMetadata, ValuesDescription

from flexible_unet_mlx import specification
from flexible_unet_mlx._typing import FloatArray, HostArray

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/darwin-arm64"


def recording(name: str) -> tuple[dict[str, HostArray], RecordingMetadata]:
    """Load a named reference recording and its metadata."""
    with cast(ArrayArchive, np.load(FIXTURES / f"{name}.npz", allow_pickle=False)) as data:
        arrays = {key: data[key] for key in data.files if key != "metadata"}
        metadata = cast(RecordingMetadata, json.loads(str(data["metadata"])))
    spec = specification()
    assert metadata["schema_version"] == 2
    assert metadata["upstream"] == spec["upstream"]
    assert metadata["bundle"]["revision"] == spec["bundle"]["revision"]
    assert metadata["reference_platform"] == "darwin-arm64"
    return arrays, metadata


def recorded_values(description: ValuesDescription) -> FloatArray:
    """Reassemble a recorded array from its named compressed parts."""
    parts = [recording(Path(filename).stem)[0]["values"] for filename in description["files"]]
    return cast(FloatArray, np.concatenate(parts).reshape(description["shape"]))


def seeded_state(seed: int) -> dict[str, HostArray]:
    """Rebuild the recorded weights, retaining the original NumPy seed/order."""
    rng = np.random.default_rng(seed)
    arrays: dict[str, HostArray] = {}
    for name, spec in specification()["parameters"].items():
        if name.endswith("num_batches_tracked"):
            arrays[name] = np.zeros(spec["shape"], dtype=np.int64)
            continue
        values = (
            rng.uniform(0.5, 1.5, spec["shape"])
            if name.endswith("running_var")
            else rng.normal(0.0, 0.05, spec["shape"])
        )
        arrays[name] = values.astype(np.float32)
    return arrays


def seeded_weights(seed: int) -> dict[str, HostArray]:
    """Build native test weights from the recorded seed without training counters."""
    return {
        name: value
        for name, value in seeded_state(seed).items()
        if not name.endswith("num_batches_tracked")
    }
