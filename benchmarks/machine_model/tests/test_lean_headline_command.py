"""`machine-model lean-headline`: the files it writes are the files its `--check` passes. That the tree's committed Lean
files are those bytes is `benchmarks/pouw/tests/`'s to hold, whose suite reads them: this one's reads none."""

from __future__ import annotations

import pytest
from verity_machine_model import lean_headline as LH
from verity_machine_model.cli import main


def test_check_passes_on_what_the_command_writes_and_names_a_file_that_is_not(tmp_path, monkeypatch, capsys):
    out = tmp_path / "security" / "Definitions" / "Accounting" / "Headline"
    monkeypatch.setattr(LH, "ROOT", tmp_path)
    monkeypatch.setattr(LH, "OUT", out / LH.OUT.name)
    monkeypatch.setattr(LH, "OUT_M", out / LH.OUT_M.name)
    assert main(["lean-headline", "--check"]) == 1 and not out.exists()
    assert capsys.readouterr().err.count("is not what `machine-model lean-headline` writes") == 2
    assert main(["lean-headline"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["Sm120.lean", "Sm120Measured.lean"]
    assert capsys.readouterr().out.count("wrote security/Definitions/Accounting/Headline/Sm120") == 2
    written = {p.name: p.read_bytes() for p in out.iterdir()}
    assert main(["lean-headline", "--check"]) == 0
    assert "wrote" not in capsys.readouterr().out and {p.name: p.read_bytes() for p in out.iterdir()} == written
    (out / LH.OUT_M.name).write_text(written[LH.OUT_M.name].decode() + "\n")
    assert main(["lean-headline", "--check"]) == 1
    err = capsys.readouterr().err
    assert "Sm120Measured.lean is not what `machine-model lean-headline` writes" in err and "Sm120.lean is" not in err


def test_the_instances_are_of_the_packages_own_data(tmp_path):
    with pytest.raises(SystemExit, match="this package's own data"):
        main(["--device", "sm_90", "lean-headline", "--check"])
    with pytest.raises(SystemExit, match="this package's own data"):
        main(["--data", str(tmp_path), "lean-headline"])
    assert not list(tmp_path.iterdir())
