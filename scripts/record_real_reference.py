"""Record pinned MONAI bundle outputs on a chest X-ray outside Git."""

import argparse
import json
from pathlib import Path

import monai
import numpy as np
import torch
from monai.bundle import ConfigParser
from monai.data import MetaTensor
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_REVISION = "f0416a27d59516d9c1b6bec5181c9c33002c84fa"


def logits(shape, salt):
    """Recreate the deterministic two-channel postprocessing probe."""
    index = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    return ((index * 2654435761 + salt * 40503) % 2003 - 1001).astype(np.float32) / np.float32(257)


def main():
    """Run upstream preprocessing, postprocessing, and seeded network on the X-ray."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--xray", type=Path, default=Path.home() / ".cache/medmlx/test-data/chest_xray.jpg"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "work/real-reference")
    args = parser.parse_args()
    if monai.__version__ != "1.4.0" or not torch.__version__.startswith("2.13."):
        raise ValueError("MONAI 1.4.0 and Torch 2.13 are required")
    if args.bundle.name != BUNDLE_REVISION:
        raise ValueError("Supply the pinned bundle snapshot")
    args.output.mkdir(parents=True, exist_ok=True)
    config = ConfigParser()
    config.read_config(str(args.bundle / "configs/inference.json"))
    config["device"] = "$torch.device('cpu')"
    config["output_dir"] = str(args.output / "masks")
    preprocess = config.get_parsed_content("preprocessing")
    postprocess = config.get_parsed_content("postprocessing")
    network = config.get_parsed_content("network_def").cpu().float().eval()
    specification = json.loads((ROOT / "src/flexible_unet_mlx/specification.json").read_text())
    rng = np.random.default_rng(3)
    state = network.state_dict()
    for name, value in specification["parameters"].items():
        if name.endswith("num_batches_tracked"):
            continue
        data = (
            rng.uniform(0.5, 1.5, value["shape"])
            if name.endswith("running_var")
            else rng.normal(0.0, 0.05, value["shape"])
        )
        state[name] = torch.from_numpy(data.astype(np.float32))
    network.load_state_dict(state, strict=True)
    torch.set_num_threads(4)
    with Image.open(args.xray) as source:
        rgb = source.convert("RGB")
        for case, size, salt in (
            ("small", (120, 90), 0),
            ("tall", (150, 500), 1),
            ("wide", (760, 120), 2),
        ):
            rgb.resize(size, Image.Resampling.LANCZOS).save(args.output / f"{case}.jpg", quality=95)
            data = preprocess({"image": str(args.output / f"{case}.jpg")})
            image = data["image"].as_tensor().numpy()
            probe = logits((2, *image.shape[1:]), salt)
            data["pred"] = MetaTensor(torch.from_numpy(probe))
            postprocess(data)
            mask_path = next((args.output / "masks" / case).glob("*.png"))
            with Image.open(mask_path) as png:
                mask_bytes = np.frombuffer(mask_path.read_bytes(), dtype=np.uint8)
                mode, png_size = png.mode, list(png.size)
            # Invertd's bilinear interpolation is the restored continuous reference.
            restored = torch.nn.functional.interpolate(
                torch.from_numpy(probe)[None], size=size, mode="bilinear", align_corners=False
            )[0].numpy()
            metadata = {
                "schema_version": 2,
                "upstream": specification["upstream"],
                "bundle": specification["bundle"],
                "reference_platform": "darwin-arm64",
                "weights_seed": 3,
                "salt": salt,
                "image": {"files": [f"{case}_image_0.npz"], "shape": list(image.shape)},
                "restored": {"files": [f"{case}_restored_0.npz"], "shape": list(restored.shape)},
                "png_mode": mode,
                "png_size": png_size,
            }
            np.savez_compressed(
                args.output / f"{case}.npz", png=mask_bytes, metadata=json.dumps(metadata)
            )
            np.savez_compressed(
                args.output / f"{case}_image_0.npz",
                values=image.ravel(),
                metadata=json.dumps(metadata),
            )
            np.savez_compressed(
                args.output / f"{case}_restored_0.npz",
                values=restored.ravel(),
                metadata=json.dumps(metadata),
            )
            if case == "small":
                with torch.inference_mode():
                    prediction = network(torch.from_numpy(image)[None])[0]
                runner_data = preprocess({"image": str(args.output / f"{case}.jpg")})
                runner_data["pred"] = MetaTensor(prediction)
                postprocess(runner_data)
                runner_png = np.frombuffer(mask_path.read_bytes(), dtype=np.uint8)
                np.savez_compressed(
                    args.output / "network.npz",
                    logits=prediction.numpy()[None],
                    metadata=json.dumps(metadata),
                )
                np.savez_compressed(
                    args.output / "runner.npz", png=runner_png, metadata=json.dumps(metadata)
                )


if __name__ == "__main__":
    main()
