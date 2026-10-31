"""Offline admission tests using existing Git objects, without importing reference frameworks."""

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scripts.source_admission import require_git_sources

ROOT = Path(__file__).resolve().parents[1]


def _git(*arguments: str) -> bytes:
    """Run a Git command against the temporary source repository and return its output."""
    return subprocess.run(
        ("git", "--no-lazy-fetch", "--no-replace-objects", "-C", str(ROOT), *arguments),
        check=True,
        capture_output=True,
    ).stdout


@pytest.mark.parametrize(
    "case", ["matching", "modified", "missing", "unavailable", "gitless", "wrong_path"]
)
def test_snapshot_source_admission(tmp_path: Path, case: str) -> None:
    """Accept exact archive copies; reject mutation or unavailable proof without fetching."""
    revision = _git("rev-parse", "HEAD").decode().strip()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    source = snapshot / "source.txt"
    expected = _git("show", f"{revision}:README.md")
    source.write_bytes(expected + (b"changed" if case == "modified" else b""))
    if case == "missing":
        source.unlink()
    if case == "unavailable":
        revision = "0" * 40
    mapping = {"source.txt": "absent/source.txt" if case == "wrong_path" else "README.md"}
    if case == "matching":
        require_git_sources(snapshot, revision, mapping, ROOT)
        assert source.read_bytes() == expected
    else:
        with pytest.raises(
            RuntimeError, match=r"Missing pinned|differs from pinned|source-repository"
        ):
            require_git_sources(snapshot, revision, mapping, None if case == "gitless" else ROOT)


@given(st.binary(max_size=128))
@settings(max_examples=12, deadline=None)
def test_mutated_snapshot_never_admitted(contents: bytes) -> None:
    """No arbitrary byte mutation can pass as the selected committed source blob."""
    revision = _git("rev-parse", "HEAD").decode().strip()
    expected = _git("show", f"{revision}:README.md")
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "source.txt").write_bytes(contents)
        if contents == expected:
            require_git_sources(root, revision, {"source.txt": "README.md"}, ROOT)
        else:
            with pytest.raises(RuntimeError, match="differs from pinned"):
                require_git_sources(root, revision, {"source.txt": "README.md"}, ROOT)
