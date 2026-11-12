"""`machine-model lean-headline`: the committed Lean files are what the command writes, byte for byte."""

from __future__ import annotations

import pytest

from verity_machine_model import lean_headline as LH
from verity_machine_model.cli import main


def test_the_committed_instances_are_what_the_command_writes(tmp_path, monkeypatch, capsys):
    """`--check` passes on the tree's files and writes nothing; written elsewhere, the two files are the committed
    bytes; and `--check` names a file that is missing."""
    files = (LH.SM120.out, LH.OUT_M)
    committed = {p.name: p.read_bytes() for p in files}
    assert main(["lean-headline", "--check"]) == 0
    assert "wrote" not in capsys.readouterr().out and {p.name: p.read_bytes() for p in files} == committed
    out = tmp_path / "security" / "Definitions" / "Accounting" / "Headline"
    monkeypatch.setattr(LH, "ROOT", tmp_path)
    monkeypatch.setattr(LH, "OUT", out / LH.OUT.name)
    monkeypatch.setattr(LH, "OUT_M", out / LH.OUT_M.name)
    assert main(["lean-headline"]) == 0
    assert {p.name: p.read_bytes() for p in out.iterdir()} == committed
    assert capsys.readouterr().out.count("wrote security/Definitions/Accounting/Headline/Sm120") == 2
    (out / LH.OUT_M.name).unlink()
    assert main(["lean-headline", "--check"]) == 1
    assert "Sm120Measured.lean is not what `machine-model lean-headline` writes" in capsys.readouterr().err


def test_the_instances_are_of_the_packages_own_data(tmp_path):
    with pytest.raises(SystemExit, match="this package's own data"):
        main(["--device", "sm_90", "lean-headline", "--check"])
    with pytest.raises(SystemExit, match="this package's own data"):
        main(["--data", str(tmp_path), "lean-headline"])
    assert not list(tmp_path.iterdir())
