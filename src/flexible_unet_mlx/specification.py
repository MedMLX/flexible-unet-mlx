"""Immutable released-source graph specification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from flexible_unet_mlx._typing import Specification


def specification() -> Specification:
    """Return the pinned graph specification parsed from ``specification.json``.

    The file is read on every call, so each caller gets its own mutable copy.
    """
    return cast(Specification, json.loads(Path(__file__).with_suffix(".json").read_text()))
