"""medmlx runner following the pinned MONAI endoscopic tool bundle."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from flexible_unet_mlx._typing import BoolArray, FloatArray, RunResult

if TYPE_CHECKING:
    from flexible_unet_mlx.network import FlexibleUNet

import numpy as np
from medmlx_core.errors import AssetNotReadyError, InvalidInputError, ModelExecutionError
from medmlx_core.outputs import artifact
from medmlx_core.weights import weights_dir, weights_revision

from flexible_unet_mlx import transforms

BUNDLE = "endoscopic_tool_segmentation"
LABELS = {0: "background", 255: "tools"}

MODEL = {
    "id": "flexible-unet",
    "label": "FlexibleUNet endoscopic tools",
    "task": "segmentation",
    "run": "flexible_unet_mlx.medmlx:run",
    "cohort": "flexible_unet_mlx.medmlx:prepare",
    # Measured MLX peak for a 768x576 endoscopy frame; larger inputs need more.
    "peak_memory_gb": 0.3,
    "summary": "MONAI endoscopic_tool_segmentation: surgical tools in RGB endoscopy frames.",
    "labels": {str(value): name for value, name in LABELS.items()},
}


def run(
    *,
    image_path: str | Path,
    weights_path: str | Path | None = None,
    output_dir: str | Path,
    overwrite: bool = False,
) -> RunResult:
    """Segment surgical tools in one RGB endoscopy frame and write the mask PNG.

    The frame is resized to the network's fixed input size, run through one FP32
    forward pass, and the logits are resized back to the frame size, so the mask has
    the frame's own width and height. This is research software: the details always
    carry ``requires_human_review: True``.

    Args:
        image_path: RGB image PIL can read (the bundle used JPEG frames). Grayscale
            and alpha images are rejected.
        weights_path: Optional model folder containing ``flexibleunet.safetensors``.
            Omit to download and reuse ``MedMLX/flexible-unet-effnetb2-mlx`` automatically.
            Explicit native ``.safetensors`` checkpoint paths are also accepted.
        output_dir: Output root. It is created when missing, after inference succeeds.
        overwrite: Replace an existing mask. Must be exactly ``True`` or ``False``.

    Returns:
        ``{"outputs": {"mask": <absolute POSIX path>}, "artifacts": {"mask": ...},
        "details": {...}}`` with JSON-serializable values (see ``RunDetails``). The
        ``mask`` artifact is a ``binary_mask`` in ``pixel`` space on the
        ``image_path`` frame's pixel grid, with labels 0 background and 255 tools.
        The mask is
        ``<output_dir>/<stem>/<stem>_trans.png`` where ``<stem>`` is the image file
        name after resolving symlinks and removing its suffix: 8-bit grayscale,
        255 for tools and 0 elsewhere, following the bundle's ``SaveImaged`` layout.
        Written atomically; the overwrite check is preflight and is not race-proof.

    Raises:
        InvalidInputError: ``overwrite`` is not a bool, ``output_dir`` is a file, the
            mask exists and ``overwrite`` is false, or the image is missing, unreadable
            or not RGB.
        AssetNotReadyError: The weights file is missing, is not ``.safetensors``, or
            does not match the pinned graph.
        MissingDependencyError: Apple Silicon or the Metal GPU is unavailable.
        ModelExecutionError: The forward pass fails.
    """
    image_file, mask_path = _destination(image_path, output_dir, overwrite)
    weights = _resolve_weights(weights_path)
    started = time.perf_counter()
    frame = transforms.read_rgb(image_file)
    network = _load_network(weights)
    return _segment(network, weights_revision(weights.parent), frame, mask_path, started)


class FrameSession:
    """One loaded FlexibleUNet reused for serial frames until :meth:`close`.

    Loading, checkpoint admission and graph compilation happen once; each
    :meth:`run` call then has the one-shot :func:`run` contract for a single
    frame. The session owns only the network: frames, logits and masks belong to
    each call. ``weights_revision`` is read at preparation, and later edits to the
    checkpoint file do not affect the loaded network. Per-frame
    ``load_inputs_and_weights`` covers reading the frame only. Frames run one at
    a time; MedMLX cohorts bound the sequence length.
    """

    def __init__(self, *, weights_path: str | Path | None = None) -> None:
        """Admit and load the checkpoint, raising like :func:`run`."""
        weights = _resolve_weights(weights_path)
        self._revision = weights_revision(weights.parent)
        self._network: FlexibleUNet | None = _load_network(weights)

    def run(
        self, *, image_path: str | Path, output_dir: str | Path, overwrite: bool = False
    ) -> RunResult:
        """Segment one frame with the session's network; see :func:`run`.

        Raises:
            RuntimeError: The session is closed.
        """
        network = self._network
        if network is None:
            raise RuntimeError("FlexibleUNet session is closed")
        image_file, mask_path = _destination(image_path, output_dir, overwrite)
        started = time.perf_counter()
        frame = transforms.read_rgb(image_file)
        return _segment(network, self._revision, frame, mask_path, started)

    def close(self) -> None:
        """Release the network; later :meth:`run` calls raise ``RuntimeError``."""
        self._network = None

    def __enter__(self) -> FrameSession:
        """Return this session for a ``with`` block that closes it on exit."""
        return self

    def __exit__(self, *args: object) -> None:
        """Close the session."""
        self.close()


def prepare(*, weights_path: str | Path | None = None) -> FrameSession:
    """Load FlexibleUNet once for MedMLX's explicit cohort protocol or a frame loop.

    ``weights_path`` has the meaning it has in :func:`run`. Close the returned
    session, or use it as a context manager, to release the model.
    """
    return FrameSession(weights_path=weights_path)


def _destination(
    image_path: str | Path, output_dir: str | Path, overwrite: bool
) -> tuple[Path, Path]:
    """Return the resolved image and mask paths after the preflight input and output checks."""
    if type(overwrite) is not bool:
        raise InvalidInputError("overwrite must be boolean")
    image_file = Path(image_path).expanduser().resolve()
    if not image_file.is_file():
        raise InvalidInputError(f"image_path not found: {image_file}")
    output_root = Path(output_dir).expanduser().resolve()
    mask_path = output_root / image_file.stem / f"{image_file.stem}_trans.png"
    if output_root.exists() and not output_root.is_dir():
        raise InvalidInputError(f"output_dir is not a directory: {output_root}")
    if mask_path.exists() and not overwrite:
        raise InvalidInputError(f"Output exists: {mask_path}; pass overwrite=true to replace it")
    return image_file, mask_path


def _resolve_weights(weights_path: str | Path | None) -> Path:
    """Return the admitted checkpoint file, downloading the default bundle when omitted."""
    bundle_dir = (
        weights_dir("flexible-unet")
        if weights_path is None else Path(weights_path).expanduser()
    )
    # Preserve explicit checkpoint-file callers and named cache symlinks.
    checkpoint = bundle_dir / "flexibleunet.safetensors" if bundle_dir.is_dir() else bundle_dir
    return _weights_file(str(checkpoint))


def _segment(
    network: FlexibleUNet, revision: str | None, frame: FloatArray, mask_path: Path,
    started: float,
) -> RunResult:
    """Predict, restore and publish one read frame; ``started`` precedes the frame read."""
    loaded = time.perf_counter()
    image = preprocess(frame)
    prepared = time.perf_counter()
    logits, device = _predict(network, image)
    predicted = time.perf_counter()
    mask = postprocess(logits[0], frame.shape[1:])
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(mask_path, mask)
    finished = time.perf_counter()
    width, height = frame.shape[1:]
    return {
        "outputs": {"mask": mask_path.as_posix()},
        "artifacts": {
            "mask": artifact(
                kind="binary_mask",
                coordinate_space="pixel",
                grid_of="image_path",
                labels=[{"value": value, "name": name} for value, name in LABELS.items()],
            )
        },
        "details": {
            "weights_revision": revision,
            "bundle": BUNDLE,
            "labels": {str(value): name for value, name in LABELS.items()},
            "image_size": {"width": int(width), "height": int(height)},
            "network_input_shape": list(image.shape[2:]),
            "tool_pixels": int(mask.sum()),
            "tool_fraction": round(float(mask.mean()), 6),
            "mlx_device": device,
            "seconds": {
                "load_inputs_and_weights": round(loaded - started, 4),
                "preprocess": round(prepared - loaded, 4),
                "inference": round(predicted - prepared, 4),
                "postprocess_and_write": round(finished - predicted, 4),
            },
            "requires_human_review": True,
        },
    }


def preprocess(frame: FloatArray) -> FloatArray:
    """``Resized([736, 480], "bilinear")`` then ``ScaleIntensityd``.

    Args:
        frame: Float32 ``(3, width, height)`` array of 0-255 values from
            ``transforms.read_rgb``.

    Returns:
        Float32 ``(1, 3, 736, 480)`` in [0, 1]: width resized to 736 and height to 480,
        then one min/max rescale over the whole resized array.
    """
    resized = transforms.resize_bilinear(frame, transforms.SPATIAL_SIZE)
    return transforms.scale_intensity(resized)[None]


def postprocess(logits: FloatArray, size: tuple[int, ...]) -> BoolArray:
    """``Invertd`` (bilinear resize of the logits back), argmax, the tool channel.

    Args:
        logits: Float32 ``(2, 736, 480)`` network logits for one frame (batch axis
            removed); channel 1 is tool.
        size: ``(width, height)`` of the original frame, i.e. ``frame.shape[1:]``.
            Only the first two entries are used.

    Returns:
        Boolean ``(width, height)`` mask, true where the tool channel wins the argmax.
    """
    restored = transforms.resize_bilinear(logits, (int(size[0]), int(size[1])))
    return transforms.tool_mask(restored)


def _weights_file(weights_path: str) -> Path:
    """Require an existing ``.safetensors`` file; the path itself is kept unresolved."""
    # Preserve the named Hub-cache symlink for suffix admission.
    path = Path(weights_path).expanduser().absolute()
    hint = (
        "Download flexibleunet.safetensors from MedMLX/flexible-unet-effnetb2-mlx."
    )
    if not path.is_file():
        raise AssetNotReadyError(
            f"FlexibleUNet weights not found: {path}", reason=f"missing {path}", hint=hint
        )
    if path.suffix != ".safetensors":
        raise AssetNotReadyError(
            f"FlexibleUNet weights must be converted MLX safetensors, got {path.name}",
            reason=f"unconverted {path.name}",
            hint=hint,
        )
    return path


def _load_network(weights: Path) -> FlexibleUNet:
    """Load the checkpoint; a mismatch with the pinned graph raises ``AssetNotReadyError``."""

    from flexible_unet_mlx.weights import load_model

    try:
        return load_model(weights)
    except (OSError, InvalidInputError, ValueError, RuntimeError) as error:
        raise AssetNotReadyError(
            f"FlexibleUNet weights {weights} do not match the pinned graph: {error}",
            reason=f"mismatched {weights.name}",
            hint="Download flexibleunet.safetensors from MedMLX/flexible-unet-effnetb2-mlx.",
        ) from error


def _predict(network: FlexibleUNet, image: FloatArray) -> tuple[FloatArray, str]:
    """The bundle's ``SimpleInferer``: one synchronized FP32 forward pass.

    Returns the ``(1, 2, 736, 480)`` logits and the string form of MLX's default device.
    """
    from flexible_unet_mlx.inference import forward_patch

    try:
        logits = forward_patch(network, np.ascontiguousarray(image, dtype=np.float32))
    except (RuntimeError, ValueError) as error:
        raise ModelExecutionError(f"FlexibleUNet MLX inference failed: {error}") from error
    return logits, str(network.mx.default_device())


def _write_atomic(path: Path, mask: BoolArray) -> None:
    """Write a PNG through a temporary file and atomically replace its destination."""
    # Write next to the target, then rename, so an interrupted run cannot leave a
    # truncated PNG at the final path.
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".png", dir=path.parent)
    os.close(descriptor)
    try:
        transforms.write_png(Path(name), mask)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


__all__ = ["MODEL", "FrameSession", "prepare", "run"]
