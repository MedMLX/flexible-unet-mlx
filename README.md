# flexible-unet-mlx

Segment surgical tools in RGB endoscopy frames with FlexibleUNet and native MLX.

[Docs](https://github.com/MedMLX/flexible-unet-mlx/blob/main/docs/technical.md) · [Model weights](https://huggingface.co/MedMLX/flexible-unet-effnetb2-mlx) · [MedMLX](https://github.com/MedMLX)

| Input | Output |
| :--- | :--- |
| RGB image | Tool mask at the original size |

**Apple Silicon · Metal · Python 3.13 / 3.14** · [uv](https://docs.astral.sh/uv/)

## Install

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install "flexible-unet-mlx>=0.2.5"
```

## Run

```python
from flexible_unet_mlx.medmlx import run

result = run(image_path="frame.jpg", output_dir="out")
print(result["outputs"]["mask"])
```

Omitting `weights_path` downloads `MedMLX/flexible-unet-effnetb2-mlx` once and reuses
its local cache. To use your own model folder, pass `weights_path="/path/to/model"`;
the runtime finds `flexibleunet.safetensors` inside it. Explicit checkpoint-file
paths remain accepted by the standalone runtime.

Downloaded commits are returned in `details.weights_revision`; explicit local
weights report `null` unless the folder carries a MedMLX download revision.

## License

[Apache-2.0](https://github.com/MedMLX/flexible-unet-mlx/blob/main/LICENSE). See [third-party notices](https://github.com/MedMLX/flexible-unet-mlx/blob/main/THIRD_PARTY_NOTICES.md). Model weights retain their publisher terms.

For research and education. Outputs require human review.
