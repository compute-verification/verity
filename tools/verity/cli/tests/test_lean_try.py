"""`verity lean try FILE` without `--on`: `lake env lean FILE` in the directory of the nearest lakefile above FILE, with
lean's exit status passed through, against a stand-in `lake` on PATH, so no Lean toolchain is needed."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import verity_cli


def stand_in_lake(tmp_path: Path, monkeypatch, rc: int) -> Path:
    """A `lake` that writes its working directory and arguments to the file it returns, and exits with ``rc``."""
    called, bin_ = tmp_path / "called", tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "lake").write_text(f'#!/bin/sh\n{{ pwd -P; echo "$@"; }} > "{called}"\nexit {rc}\n')
    (bin_ / "lake").chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_), prepend=os.pathsep)
    return called


def test_try_runs_lake_env_lean_in_the_nearest_package_and_passes_its_status(tmp_path, monkeypatch, capsys):
    (tmp_path / "lakefile.toml").write_text('name = "Outer"\n')
    pkg = tmp_path / "Outer" / "Inner"
    (pkg / "Inner" / "Area").mkdir(parents=True)
    (pkg / "lakefile.lean").write_text("")
    file = pkg / "Inner" / "Area" / "File.lean"
    file.write_text("theorem t : True := trivial\n")
    called = stand_in_lake(tmp_path, monkeypatch, 3)
    assert verity_cli.main(["lean", "try", str(file)]) == 3
    assert called.read_text().splitlines() == [str(pkg.resolve()), "env lean Inner/Area/File.lean"]
    assert f"verity lean try: lake env lean Inner/Area/File.lean (in {pkg.resolve()})" in capsys.readouterr().out


def test_try_refuses_a_file_in_no_package_and_remote_options_without_on(tmp_path, monkeypatch, capsys):
    called = stand_in_lake(tmp_path, monkeypatch, 0)
    loose = tmp_path / "Loose.lean"
    loose.write_text("")
    with pytest.raises(SystemExit, match="Loose.lean is not a .lean file in a Lake package"):
        verity_cli.main(["lean", "try", str(loose)])
    with pytest.raises(SystemExit):
        verity_cli.main(["lean", "try", str(loose), "--pool", "P"])
    assert "--pool and --as go with --on" in capsys.readouterr().err
    assert not called.exists()
