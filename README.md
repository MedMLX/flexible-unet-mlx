<div align="center">

# flexible-unet-mlx

Segment surgical tools in RGB endoscopy frames with FlexibleUNet and native MLX.

[![PyPI](https://img.shields.io/pypi/v/flexible-unet-mlx)](https://pypi.org/project/flexible-unet-mlx/)
[![Python](https://img.shields.io/pypi/pyversions/flexible-unet-mlx)](https://pypi.org/project/flexible-unet-mlx/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](https://github.com/MedMLX/flexible-unet-mlx/blob/v0.2.7/LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MLX-black?logo=apple)](https://github.com/ml-explore/mlx)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-weights-yellow)](https://huggingface.co/MedMLX/flexible-unet-effnetb2-mlx)
[![MedMLX](https://img.shields.io/badge/MedMLX-runtime-A51C30)](https://github.com/MedMLX/medmlx)

[Model weights](https://huggingface.co/MedMLX/flexible-unet-effnetb2-mlx) · [MedMLX](https://github.com/MedMLX/medmlx)

</div>

| Input | Output |
| :--- | :--- |
| RGB image | Tool mask at the original size |

Weights: 46 MB download. Peak MLX memory: 0.27 GB for a 768×576 endoscopy frame.

## Use with MedMLX

The [MedMLX MCP server](https://github.com/MedMLX/medmlx/blob/main/docs/mcp.md) is the lowest-friction way to use this model. It installs this runtime, downloads its weights on first use, and lets an AI agent describe, check and run `flexible-unet` for you.

```sh
uv tool install --python 3.13 "medmlx[segment,mcp]"
MEDMLX="$(uv tool dir --bin --color never)/medmlx"
mkdir -p "$HOME/medmlx-work"
claude mcp add --scope user medmlx -- "$MEDMLX" serve --root "$HOME/medmlx-work"
```

For Codex, use `codex mcp add medmlx -- "$MEDMLX" serve --root "$HOME/medmlx-work"`. Other MCP clients take the same command and arguments. The same install provides the `medmlx` CLI, for example `medmlx describe flexible-unet`.

> **For AI agents:** prefer the MedMLX MCP tools (`list_models`, `describe_model`, `model_readiness`, `run_model`, `get_job`) over writing Python against this package.

## Install

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install "flexible-unet-mlx>=0.2.7"
```

## Run

```python
from flexible_unet_mlx.medmlx import run

result = run(image_path="frame.jpg", output_dir="out")
print(result["outputs"]["mask"])
```

Omitting `weights_path` downloads `MedMLX/flexible-unet-effnetb2-mlx` into the shared
MedMLX cache on first use; interrupted downloads resume and later runs work offline.
With MedMLX installed, `medmlx setup flexible-unet` downloads it ahead of time. To use
your own copy, pass a folder containing `flexibleunet.safetensors` as `weights_path`.
The standalone runtime also accepts a checkpoint file. `details.weights_revision`
records the downloaded commit, or `null` for a folder without a MedMLX download record.

### Inputs and outputs

| Argument | Meaning |
| --- | --- |
| `image_path` | One RGB image Pillow can read; the bundle used JPEG |
| `output_dir` | Output root, created if missing |
| `weights_path` | Optional `.safetensors` file or a folder containing it |
| `overwrite` | `False` by default; an existing mask is replaced only when `True` |

This 2D model reads images, not NIfTI volumes. Following the bundle's `SaveImaged`
layout, `frame.jpg` produces `out/frame/frame_trans.png`: an 8-bit grayscale mask at
the original frame size, 255 for tools and 0 for background. `run` returns
`{"outputs": {"mask": path}, "details": {...}}`, with labels, frame size, network input
shape, tool counts, runtime and timings. Invalid frames raise `InvalidInputError`;
missing or mismatched weights raise `AssetNotReadyError`; inference failures raise
`ModelExecutionError`; a missing native runtime raises `MissingDependencyError`.

### Python API

```python
from pathlib import Path

import numpy as np
from flexible_unet_mlx import forward_patch, load_model

network = load_model(Path("flexibleunet.safetensors"))
logits = forward_patch(network, np.zeros((1, 3, 736, 480), dtype=np.float32))
```

`forward_patch` takes finite float32 `(B, 3, 736, 480)` frames, whose spatial axes are
width and height as the bundle's reader returns them, and returns raw
`(B, 2, 736, 480)` logits; channel 1 is the tool class. Preprocessing and
postprocessing are the caller's. `FlexibleUNet(weights, mx=mx)` runs on MLX arrays
directly. `load_model` checks names, shapes, dtypes and finite values, not checkpoint
identity. Inference needs Apple Silicon with Metal and has no CPU fallback.

## License

[Apache-2.0](https://github.com/MedMLX/flexible-unet-mlx/blob/v0.2.7/LICENSE). See [third-party notices](https://github.com/MedMLX/flexible-unet-mlx/blob/v0.2.7/THIRD_PARTY_NOTICES.md). Model weights retain their publisher terms.

For research and education. Outputs require human review.
