# flexible-unet-mlx technical notes

Full details behind the [README](../README.md).

## Overview

Native FP32 MLX [FlexibleUNet](https://huggingface.co/MONAI/endoscopic_tool_segmentation)
inference for RGB endoscopy frames on Apple Silicon. Part of
[MedMLX](https://github.com/MedMLX), with shared runtime and arithmetic in `medmlx-core`.
The supported environment is macOS arm64 with MLX on Metal.

The reference is upstream [MONAI 1.4.0](https://github.com/Project-MONAI/MONAI/tree/46a5272196a6c2590ca2589029eed8e4d56ff008)
(commit `46a5272196a6c2590ca2589029eed8e4d56ff008`) and the official endoscopic tool
bundle revision `f0416a27d59516d9c1b6bec5181c9c33002c84fa`. Tensor
specifications and encoder settings have
one canonical definition in `src/flexible_unet_mlx/specification.json`.

## Install

Python 3.13–3.14 is supported. Runtime inference uses MLX, NumPy and Pillow;
The runtime never imports Torch or MONAI.
Importing the package, its public names or its model declaration does not import
MLX. Native execution requires macOS arm64 and an available Metal GPU, initialized
through `medmlx_core.import_mlx()`. Missing Metal raises `MissingDependencyError`.
The uv resolution environment is restricted to macOS arm64 in `pyproject.toml`.

## Use

For the low-level network API, download `flexibleunet.safetensors` from
[`MedMLX/flexible-unet-effnetb2-mlx`](https://huggingface.co/MedMLX/flexible-unet-effnetb2-mlx/tree/ccbbc596f2a9da8477e25cba0d7548cbb2191feb)
at revision `ccbbc596f2a9da8477e25cba0d7548cbb2191feb`. No conversion is needed.
Weights are not included in this repository.

```python
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
from flexible_unet_mlx import load_model, forward_patch

weights = hf_hub_download(
    "MedMLX/flexible-unet-effnetb2-mlx", "flexibleunet.safetensors",
    revision="ccbbc596f2a9da8477e25cba0d7548cbb2191feb",
)
network = load_model(Path(weights))
logits = forward_patch(network, np.zeros((1, 3, 736, 480), dtype=np.float32))
```

`forward_patch` takes finite FP32 NCHW data of shape `(B, 3, 736, 480)`, with
`B >= 1`, and returns raw FP32 logits of shape `(B, 2, 736, 480)` (channel 1 is the
tool class). Following the bundle's reader, the two spatial axes are the image's width
(736) and height (480). The caller owns preprocessing and postprocessing at this
boundary. `FlexibleUNet(weights, mx=mx)`
supports direct MLX-array execution with Metal as the default device. Construction
and forward execution both enforce that requirement.

`load_model` validates the weights against the network's names, shapes, dtypes
and finite values before constructing the network. This does not establish the
checkpoint's identity or origin.

## Use with medmlx

The `medmlx.models` entry point declares `flexible-unet` through
`flexible_unet_mlx.medmlx:MODEL`. Its keyword-only runner accepts:

| Argument | Meaning |
| --- | --- |
| `image_path` | One RGB image Pillow can read; the bundle used JPEG |
| `weights_path` | Optional model folder containing `flexibleunet.safetensors`, or an explicit native checkpoint file. Omit to download `MedMLX/flexible-unet-effnetb2-mlx`. |
| `output_dir` | Output root, created if missing |
| `overwrite` | Default `false`; refuses an existing mask unless true |

Using the downloaded `weights` path above:

```python
from flexible_unet_mlx.medmlx import run

result = run(image_path="frame.jpg", weights_path=weights, output_dir="out")
```

For `frame.jpg`, it writes `out/frame/frame_trans.png`, following the bundle's
`SaveImaged(output_postfix="trans", separate_folder=True)`. The PNG is 8-bit
grayscale at the original frame size, with 255 for tools and 0 for background.
This 2D model uses images rather than NIfTI volumes. The return value is
`{"outputs": {"mask": absolute_posix_path}, "details": {...}}`, with JSON values
for labels, frame size, network input shape, tool counts, runtime
and timings. Invalid frames raise `InvalidInputError`; unavailable or mismatched
weights raise `AssetNotReadyError`; native inference failures raise
`ModelExecutionError`. Missing native dependencies raise `MissingDependencyError`.

The runner ports the pinned `configs/inference.json`: the PIL reader swaps row
and column axes, channels move first, bilinear resizing produces `(3, 736, 480)`
on width/height axes, intensities scale to [0, 1], and one FP32 forward pass runs.
Logits resize back bilinearly, argmax selects the first maximal class, and the
tool channel becomes the PNG. `transforms.py` reproduces Torch CPU float32
interpolation, including fused multiply-add rounding emulated in float64.

## Verify

Keep development environments and caches outside the checkout:

```bash
export UV_CACHE_DIR="$TMPDIR/uv-cache"
uv venv "$TMPDIR/venv-flexible-unet-mlx" --python 3.13
uv pip install --python "$TMPDIR/venv-flexible-unet-mlx/bin/python" -e ".[dev]"
PYTHONDONTWRITEBYTECODE=1 HYPOTHESIS_STORAGE_DIRECTORY="$TMPDIR/flexible-unet-hypothesis" \
  "$TMPDIR/venv-flexible-unet-mlx/bin/python" -m pytest -q \
  -o cache_dir="$TMPDIR/flexible-unet-pytest-cache"
RUFF_CACHE_DIR="$TMPDIR/flexible-unet-ruff-cache" \
  "$TMPDIR/venv-flexible-unet-mlx/bin/ruff" check .
```

The recorded upstream fixtures cover three seeded JPEG sizes, every prepared
input value, every restored synthetic logit, saved PNG pixels, the full seeded
network output, and the upstream runner mask.
Tests do not import fixture-generation scripts or MONAI. Property-based safetensors round trips need no Metal. Native tests
probe the runtime in a child process and explicitly skip if it cannot initialize.
A missing reference file fails. Tests load the fixed `tests/fixtures/darwin-arm64/`
recordings directly, independent of the installed Torch version.

Preprocessing and inverse resizing have zero observed error against the recordings
(made with Torch 2.13.0 and Pillow 12.3.0 on Apple Silicon). Their assertions allow
float32 rounding: absolute bounds of one float32 epsilon for normalized inputs and
one epsilon times `1001/257` for synthetic logits, with `rtol=0`. PNG pixels, mode
and geometry match exactly. The seeded native-network gate remains `atol=1e-7, rtol=1e-6`
plus exact argmax; the optional released-checkpoint gate remains
`atol=rtol=1e-4` plus exact argmax. The released-checkpoint test requires
`MODEL_WEIGHTS`, `MODEL_NETWORK_INPUT` and `MODEL_SOURCE_OUTPUT` to select the
checkpoint and independent reference arrays. Seeded references do not qualify
pretrained accuracy or clinical performance.

## Regenerating reference fixtures

On an Apple Silicon Mac, use a separate upstream environment with MONAI 1.4.0
and Torch 2.13.0. MONAI's
published dependency metadata requires `numpy<2`; the recorded run used NumPy 2.5.3
and MONAI source because the pinned wheel was unavailable. No numerical upstream
source was modified.

```bash
uv venv "$TMPDIR/flexible-unet-reference" --python 3.12
uv pip install --python "$TMPDIR/flexible-unet-reference/bin/python" \
  "monai==1.4.0" "numpy<2" "torch>=2.13,<2.14" "Pillow>=11,<13" huggingface_hub
"$TMPDIR/flexible-unet-reference/bin/hf" download MONAI/endoscopic_tool_segmentation \
  configs/inference.json --revision f0416a27d59516d9c1b6bec5181c9c33002c84fa \
  --local-dir "$TMPDIR/flexible-unet-bundle"
# Verification objects are separate from the downloaded bundle; no weights are checked out.
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/MONAI/endoscopic_tool_segmentation \
  "$TMPDIR/flexible-unet-source"
PYTHONDONTWRITEBYTECODE=1 "$TMPDIR/flexible-unet-reference/bin/python" \
  scripts/make_medmlx_fixtures.py --bundle "$TMPDIR/flexible-unet-bundle" \
  --source-repository "$TMPDIR/flexible-unet-source"
```

The generator imports only the upstream reference stack, requires macOS arm64,
and writes `tests/fixtures/darwin-arm64/`. It verifies MONAI's version/revision and byte-compares `configs/inference.json`
against the declared bundle Git revision before recording. A Git checkout uses
its own objects; an archive or Hugging Face download needs `--source-repository`
pointing to a local Git repository containing that revision and blob. Verification
never fetches missing objects. An unverifiable or modified source fails explicitly;
directory names and download metadata are not accepted as proof. This verifies
configuration source, not checkpoint weights: Git LFS pointers do not establish
weight-byte identity. The oracle still uses the operator-supplied checkpoint.
It checks all tensor specifications and encoder settings against the actual upstream
network. It retains frame seed `7081`, sizes `(90,120)`, `(500,150)`, `(120,760)`
(height, width), and NumPy weights seed `3`. Metadata includes `schema_version`,
upstream revision, bundle revision, reference platform, seeds,
framework/library versions, Torch execution device and thread count. Schema 2
stores full lossless FP32 arrays in chunks; all 37 current fixtures are under
488 KB each (12.7 MB total). Regenerate on another Apple Silicon Mac using the
same command. Tests compare numerical values and decoded pixels, so Pillow/zlib
encoding differences do not require a separate recording.

Optional real-checkpoint parity uses an independent upstream Torch oracle on the Mac:

```bash
"$TMPDIR/flexible-unet-reference/bin/hf" download MONAI/endoscopic_tool_segmentation \
  configs/inference.json models/model.pt \
  --revision f0416a27d59516d9c1b6bec5181c9c33002c84fa --local-dir "$TMPDIR/flexible-unet-bundle"
# Prepare any finite FP32 (1, 3, 736, 480) input.npy outside this checkout.
"$TMPDIR/flexible-unet-reference/bin/python" scripts/source_oracle.py \
  --bundle "$TMPDIR/flexible-unet-bundle" --input "$TMPDIR/input.npy" \
  --output-dir "$TMPDIR/flexible-unet-reference-output" \
  --source-repository "$TMPDIR/flexible-unet-source"
MODEL_CHECKPOINT="$TMPDIR/flexible-unet-bundle/models/model.pt" \
MODEL_NETWORK_INPUT="$TMPDIR/input.npy" MODEL_SOURCE_OUTPUT="$TMPDIR/flexible-unet-reference-output" \
  "$TMPDIR/venv-flexible-unet-mlx/bin/python" -m pytest -q
```

The oracle writes `logits.npy` and a versioned `source.json` receipt.
Without the three declared assets the optional real-checkpoint test skips.
Seeded fixtures and checkpoint admission tests require no released checkpoint.
