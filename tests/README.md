The transform and seeded-network tests use a chest X-ray at
`~/.cache/medmlx/test-data/chest_xray.jpg`. Their image-derived inputs and MONAI
outputs stay in `work/real-reference`, outside Git. Supply an RGB chest X-ray
using `--xray` if it is stored elsewhere.

To regenerate the local references, run `scripts/record_real_reference.py` in
an environment with MONAI 1.4.0, Torch 2.13, NumPy and Pillow, and pass
`--bundle` pointing to the `f0416a27d59516d9c1b6bec5181c9c33002c84fa`
endoscopic-tool bundle snapshot. The recorder resizes the X-ray to three frame
sizes, runs the bundle transforms, and records the seeded network and runner
outputs. Run `python -m pytest tests/test_medmlx.py tests/test_released_checkpoint.py`
in the project environment afterward. The released-checkpoint parity case also
requires its documented model and source-output environment variables.
