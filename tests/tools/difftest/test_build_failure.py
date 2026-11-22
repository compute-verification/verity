"""A `lake build` that fails raises a `BuildError` that carries the end of Lake's stdout and of its stderr, so the next failure says why."""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

import verity_difftest as D


def _lake(tmp_path: Path, monkeypatch, body: str) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    lake = bin_dir / "lake"
    lake.write_text("#!/bin/sh\n" + body)
    lake.chmod(lake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.defpath}")
    monkeypatch.setattr(D, "BUILD_LOCK", tmp_path / ".lake" / "harness-build.lock")
    D.build.cache_clear()
    return tmp_path / "pkg"


def test_a_failed_build_raises_the_line_each_stream_ended_with(tmp_path, monkeypatch):
    pkg = _lake(tmp_path, monkeypatch, 'echo "error: stdout line"; echo "error: stderr line" >&2; exit 1\n')
    pkg.mkdir()
    with pytest.raises(D.BuildError) as e:
        D.build(pkg, "some-exe")
    assert "failed (1)" in str(e.value) and "error: stdout line" in str(e.value) and "error: stderr line" in str(e.value)


def test_a_long_output_is_cut_from_the_front_and_says_so(tmp_path, monkeypatch):
    pkg = _lake(tmp_path, monkeypatch,
                'i=0; while [ $i -lt 2000 ]; do echo "compiling module $i"; i=$((i+1)); done; echo "the cause" >&2; exit 2\n')
    pkg.mkdir()
    with pytest.raises(D.BuildError) as e:
        D.build(pkg, "some-exe")
    said = str(e.value)
    assert "failed (2)" in said and "the cause" in said and "compiling module 1999" in said
    assert "compiling module 0\n" not in said and "…" in said
    assert len(said) < 2 * D.BUILD_TAIL + 600
