"""`machine-model list`: the named vectors."""
from __future__ import annotations

from verity_machine_model.cli import main


def test_list(capsys):
    assert main(["--device", "sm_120", "list"]) == 0
    out = capsys.readouterr().out
    assert "VF3i-strict-r8" in out and "superseded by VF3i-strict" in out
