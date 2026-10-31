#!/usr/bin/env python3
"""Capture the pinned MONAI 1.4 released network's FP32 outputs on an Apple Silicon Mac.

Loads ``models/model.pt`` from ``--bundle`` into the upstream network on CPU and runs one
forward pass on ``--input``, a finite float32 ``.npy`` of shape (1, 3, 736, 480) that is
already preprocessed. Writes ``<name>.npy`` per network output (``logits.npy`` here) and
a ``source.json`` receipt to ``--output-dir``, which must not exist yet.
"""

from __future__ import annotations

import argparse
import json
import time
from importlib import import_module
from pathlib import Path
from typing import cast

import numpy as np
import torch
from reference import require_upstream
from type_contracts import (
    ConfigParserApi,
    MonaiApi,
    ReferenceNetwork,
    ReferenceOutput,
    Tensor,
    TorchArrayApi,
)

monai = cast(MonaiApi, import_module("monai"))
parser_api = cast(ConfigParserApi, import_module("monai.bundle.config_parser"))


def main() -> None:
    """Run the pinned source oracle using the requested input and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-repository", type=Path)
    args = parser.parse_args()
    spec = require_upstream(args.bundle, args.source_repository)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    config = parser_api.ConfigParser()
    config.read_config(str(args.bundle / "configs/inference.json"))
    config["bundle_root"] = str(args.bundle)
    config["device"] = "cpu"
    source = cast(ReferenceNetwork, config.get_parsed_content("network_def")).eval().float()
    checkpoint = args.bundle / "models/model.pt"
    source.load_state_dict(
        cast(dict[str, Tensor], torch.load(checkpoint, map_location="cpu", weights_only=True)),
        strict=True,
    )
    values = cast(np.typing.NDArray[np.float32], np.load(args.input, allow_pickle=False))
    if (
        values.shape != (1, 3, 736, 480)
        or values.dtype != np.float32
        or not np.isfinite(values).all()
    ):
        raise ValueError("Source oracle input must be finite FP32 of shape (1, 3, 736, 480)")
    started = time.perf_counter()
    with torch.inference_mode():
        torch_module = cast(TorchArrayApi, torch)
        result = cast(ReferenceOutput, source)(torch_module.from_numpy(values))
    elapsed = time.perf_counter() - started
    outputs = result if isinstance(result, dict) else {"logits": result}
    for name, value in outputs.items():
        np.save(args.output_dir / f"{name}.npy", value.detach().cpu().numpy(), allow_pickle=False)
    receipt = {
        "schema_version": 1,
        "upstream": spec["upstream"],
        "bundle_revision": spec["bundle"]["revision"],
        "monai_version": monai.__version__,
        "torch_version": torch.__version__,
        "monai_source": monai.__file__,
        "device": "cpu",
        "dtype": "float32",
        "threads": 8,
        "single_forward_seconds": elapsed,
        "outputs": sorted(outputs),
    }
    (args.output_dir / "source.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
