"""Record pinned MONAI 1.4.0 transforms, seeded network logits and runner PNGs.

Run in a separate reference environment (MONAI and Torch installed); the MLX port is
imported only for type annotations. The JPEG seed (7081), frame sizes and weights
seed (3) are fixed so the recordings can be regenerated. Every prepared input,
restored logit and network logit is recorded as lossless FP32, in the bundle's
``(channel, width, height)`` axis order.
"""

from __future__ import annotations

import io
import json
import tempfile
from argparse import ArgumentParser
from collections.abc import Callable, Iterable
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import PIL
import torch
from PIL import Image
from reference import ROOT, require_upstream
from type_contracts import (
    BlockGroup,
    ConfigParser,
    ConfigParserApi,
    DataPipeline,
    FloatTensor,
    ImageData,
    MonaiDataApi,
    NumpyArchiveApi,
    Padding,
    ReferenceNetwork,
    Tensor,
    TorchArrayApi,
)

if TYPE_CHECKING:
    from flexible_unet_mlx._typing import Block, FloatArray, HostArray, Specification

parser_api = cast(ConfigParserApi, import_module("monai.bundle"))
data_api = cast(MonaiDataApi, import_module("monai.data"))

upstream_attribute: Callable[[object, str], object] = getattr

SEED = 7081
WEIGHTS_SEED = 3
# Synthetic JPEG sizes as (height, width) in pixels.
FRAMES = {"small": (90, 120), "tall": (500, 150), "wide": (120, 760)}


def logits(shape: tuple[int, ...], salt: int) -> FloatArray:
    """Deterministic synthetic float32 logits from integer hashing.

    Values lie in about [-3.9, 3.9] (``(hashed - 1001) / 257``); ``salt`` varies the
    pattern per frame. These stand in for network output when recording the
    postprocessing (inverse resize and argmax) reference.
    """
    index = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    hashed = (index * 2654435761 + salt * 40503) % 2003
    return ((hashed - 1001).astype(np.float32) / np.float32(257.0)).astype(np.float32)


def _jpeg(rng: np.random.Generator, height: int, width: int) -> bytes:
    """Encode a seeded RGB test frame (10 x 10 pixel random blocks plus noise) as JPEG."""
    base = rng.integers(0, 256, (height // 10 + 1, width // 10 + 1, 3), dtype=np.uint8)
    frame = np.kron(base, np.ones((10, 10, 1), dtype=np.uint8))[:height, :width]
    noise = rng.integers(-20, 21, frame.shape)
    frame = np.clip(frame.astype(np.int64) + noise, 0, 255).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(frame).save(stream, format="JPEG", quality=92)
    return stream.getvalue()


def _parser(bundle: Path, root: Path) -> ConfigParser:
    """Load the bundle configuration and select CPU and temporary output paths."""
    parser = parser_api.ConfigParser()
    parser.read_config(str(bundle / "configs/inference.json"))
    parser["output_dir"] = str(root / "out")
    parser["device"] = "$torch.device('cpu')"
    return parser


def _network(parser: ConfigParser, spec: Specification) -> ReferenceNetwork:
    """Build the upstream network in eval mode with seeded random weights.

    The weights are random, not the released checkpoint: normal(0, 0.05), except
    ``running_var``, drawn from uniform(0.5, 1.5) so variances are positive.
    """
    source = cast(ReferenceNetwork, parser.get_parsed_content("network_def")).eval().float()
    rng = np.random.default_rng(WEIGHTS_SEED)
    state: dict[str, Tensor] = {}
    for name, value in source.state_dict().items():
        expected = spec["parameters"][name]
        if list(value.shape) != expected["shape"] or str(value.numpy().dtype) != expected["dtype"]:
            raise RuntimeError(f"Upstream state specification differs: {name}")
        if name.endswith("num_batches_tracked"):
            state[name] = cast(TorchArrayApi, torch).zeros_like(value)
            continue
        values = (
            rng.uniform(0.5, 1.5, value.shape)
            if name.endswith("running_var")
            else rng.normal(0.0, 0.05, value.shape)
        )
        state[name] = cast(TorchArrayApi, torch).from_numpy(values.astype(np.float32))
    if list(state) != list(spec["parameters"]):
        raise RuntimeError("Upstream state names/order differ from the specification")
    source.load_state_dict(state, strict=True)
    _check_graph(source, spec)
    return source


def _check_graph(source: ReferenceNetwork, spec: Specification) -> None:
    """Raise if the upstream encoder differs from ``specification.json``."""
    encoder = source.encoder
    if (
        list(cast(Padding, upstream_attribute(encoder, "_conv_stem_padding")).padding)
        != spec["stem_padding"]
    ):
        raise RuntimeError("Upstream stem padding differs")
    if encoder.extract_stacks != spec["extract_stacks"]:
        raise RuntimeError("Upstream feature stacks differ")
    blocks: list[Block] = []
    for stack, group in enumerate(
        cast(Iterable[BlockGroup], upstream_attribute(encoder, "_blocks"))
    ):
        for index, block in group.named_children():
            blocks.append(
                {
                    "prefix": f"encoder._blocks.{stack}.{index}",
                    "stack": stack + 1,
                    "stride": block.stride,
                    "in_channels": block.in_channels,
                    "out_channels": block.out_channels,
                    "expand_ratio": block.expand_ratio,
                    "depthwise_padding": list(
                        cast(Padding, upstream_attribute(block, "_depthwise_conv_padding")).padding
                    ),
                }
            )
    if blocks != spec["blocks"]:
        raise RuntimeError("Upstream MBConv graph differs from the specification")


def _png(
    parser: ConfigParser,
    data: ImageData,
    values: FloatArray,
    root: Path,
) -> tuple[bytes, FloatArray]:
    """Run the bundle's postprocessing on ``values``, float32 (2, 736, 480) logits.

    Returns the PNG bytes the bundle's ``SaveImaged`` wrote under ``root / "out"`` and
    the logits after the first postprocessing transform (the resize back to frame size).
    """
    data["pred"] = data_api.MetaTensor(cast(TorchArrayApi, torch).from_numpy(values))
    transforms = cast(DataPipeline, parser.get_parsed_content("postprocessing")).transforms
    data = transforms[0](data)
    restored = np.asarray(data["pred"], dtype=np.float32).copy()
    for transform in transforms[1:]:
        data = transform(data)
    (written,) = (root / "out").rglob("*.png")
    return written.read_bytes(), restored


def _save(destination: Path, arrays: dict[str, HostArray], metadata: ImageData) -> None:
    """Save recorded arrays with metadata and enforce the fixture size bound."""
    arrays["metadata"] = np.array(json.dumps(metadata, sort_keys=True))
    cast(NumpyArchiveApi, np).savez_compressed(destination, **arrays)
    if destination.stat().st_size >= 750_000:
        raise RuntimeError(f"Reference fixture is too large: {destination}")
    print(f"{destination.relative_to(ROOT)}: {destination.stat().st_size} bytes", flush=True)


def _save_values(
    destination: Path, name: str, values: FloatArray, metadata: ImageData
) -> dict[str, object]:
    """Split lossless values so each numerical fixture stays well below 1 MB."""
    flattened = np.ascontiguousarray(values).reshape(-1)
    files: list[str] = []
    for index, start in enumerate(range(0, flattened.size, 131_072)):
        filename = f"{name}_{index}.npz"
        _save(destination / filename, {"values": flattened[start : start + 131_072]}, metadata)
        files.append(filename)
    return {"shape": list(values.shape), "files": files}


def record(bundle: Path, source_repository: Path | None = None) -> None:
    """Write all fixtures under ``tests/fixtures/darwin-arm64`` using the bundle's configs."""
    spec = require_upstream(bundle, source_repository)
    # Fixed thread count and deterministic kernels; both are recorded in the metadata.
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    destination = ROOT / "tests/fixtures/darwin-arm64"
    destination.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "schema_version": 2,
        "upstream": spec["upstream"],
        "bundle": spec["bundle"],
        "reference_platform": "darwin-arm64",
        "seed": SEED,
        "weights_seed": WEIGHTS_SEED,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pillow_version": PIL.__version__,
        "threads": 8,
        "device": "cpu",
    }
    rng = np.random.default_rng(SEED)
    for salt, (case, (height, width)) in enumerate(FRAMES.items()):
        payload = _jpeg(rng, height, width)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = root / f"{case}.jpg"
            frame.write_bytes(payload)
            parser = _parser(bundle, root)
            data = cast(DataPipeline, parser.get_parsed_content("preprocessing"))(
                {"image": str(frame)}
            )
            image = np.ascontiguousarray(np.asarray(data["image"], dtype=np.float32))
            png, restored = _png(parser, data, logits((2, *image.shape[1:]), salt), root)
            arrays: dict[str, HostArray] = {
                "jpeg": np.frombuffer(payload, np.uint8),
                "png": np.frombuffer(png, np.uint8),
            }
            with Image.open(io.BytesIO(png)) as saved:
                mode, size = saved.mode, list(saved.size)
            case_metadata: dict[str, object] = {
                **metadata,
                "case": case,
                "salt": salt,
                "image": _save_values(destination, f"{case}_image", image, metadata),
                "restored": _save_values(destination, f"{case}_restored", restored, metadata),
                "png_mode": mode,
                "png_size": size,
            }
            _save(destination / f"{case}.npz", arrays, case_metadata)
            if case != "small":
                continue
            source = _network(parser, spec)
            with torch.inference_mode():
                output = cast(
                    FloatTensor, source(cast(TorchArrayApi, torch).from_numpy(image[None]))
                ).numpy()
            _save(destination / "network.npz", {"logits": output}, metadata)
            data = cast(DataPipeline, parser.get_parsed_content("preprocessing"))(
                {"image": str(frame)}
            )
            (root / "out/small/small_trans.png").unlink()
            runner_png, _ = _png(parser, data, output[0], root)
            _save(
                destination / "runner.npz", {"png": np.frombuffer(runner_png, np.uint8)}, metadata
            )


def main() -> None:
    """Record source fixtures from a bundle and optional source repository."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source-repository", type=Path)
    args = parser.parse_args()
    record(args.bundle.resolve(), args.source_repository)


if __name__ == "__main__":
    main()
