"""`machine-model census`: a class's u on each row at an epoch."""
from __future__ import annotations

from verity_machine_model.cli import main


def test_census(capsys):
    assert main(["--device", "sm_120", "census", "VF3j", "--kind", "FFMA"]) == 0
    assert capsys.readouterr().out.startswith("FFMA: ")
