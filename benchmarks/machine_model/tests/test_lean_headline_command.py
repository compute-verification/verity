"""`machine-model lean-headline`: the files it writes are the files its `--check` passes. That the tree's committed Lean
files are those bytes is `benchmarks/pouw/tests/`'s to hold, whose suite reads them: this one's reads none."""

from __future__ import annotations

import json

import pytest
from verity_machine_model import lean_headline as LH
from verity_machine_model import lean_headline_dc as D
from verity_machine_model.cli import main

SEVEN = ["Headline/Sm120.lean", "Headline/Sm120Measured.lean", "HeadlineDc/Sm100.lean", "HeadlineDc/Sm100Measured.lean",
         "HeadlineDc/Sm90.lean", "HeadlineDc/Sm90Measured.lean", "HeadlineDc/Sm90MeasuredI3.lean"]


@pytest.fixture
def out(tmp_path, monkeypatch):
    """A tree of its own for the command to write into: `tmp_path`'s `security/Definitions/Accounting/`."""
    out = tmp_path / "security" / "Definitions" / "Accounting"
    monkeypatch.setattr(LH, "ROOT", tmp_path)
    monkeypatch.setattr(LH, "OUT", out / "Headline" / LH.OUT.name)
    monkeypatch.setattr(LH, "OUT_M", out / "Headline" / LH.OUT_M.name)
    monkeypatch.setattr(D, "OUT", out / "HeadlineDc")
    return out


@pytest.mark.slow
def test_check_passes_on_what_the_command_writes_and_names_a_file_that_is_not(out, capsys):
    not_written = "is not what `machine-model lean-headline` writes"
    assert main(["lean-headline", "--check"]) == 1 and not out.exists()
    assert capsys.readouterr().err.count(not_written) == 7
    assert main(["lean-headline"]) == 0
    written = {str(p.relative_to(out)): p.read_bytes() for p in out.glob("*/*")}
    assert sorted(written) == SEVEN
    assert capsys.readouterr().out.count("wrote security/Definitions/Accounting/Headline") == 7
    assert main(["lean-headline", "--check"]) == 0
    assert "wrote" not in capsys.readouterr().out
    assert {str(p.relative_to(out)): p.read_bytes() for p in out.glob("*/*")} == written
    for name in ("Headline/Sm120Measured.lean", "HeadlineDc/Sm90.lean"):
        (out / name).write_bytes(written[name] + b"\n")
    assert main(["lean-headline", "--check"]) == 1
    err = capsys.readouterr().err
    assert err.count(not_written) == 2
    assert f"Headline/Sm120Measured.lean {not_written}" in err and f"HeadlineDc/Sm90.lean {not_written}" in err


def test_numbers_prints_the_data_centre_readings_and_writes_no_lean_file(out, tmp_path, capsys):
    assert main(["lean-headline", "--numbers", "--json", str(tmp_path / "readings.json")]) == 0
    printed = capsys.readouterr().out
    assert "wrote" not in printed and "DS-120" not in printed
    assert [p.name for p in tmp_path.iterdir()] == ["readings.json"]
    rows = json.loads((tmp_path / "readings.json").read_text())
    assert {r["vector"] for r in rows} == {"DS-90", "DC-90-ri", "DS-100", "DC-100-ri"}
    assert all(f"{r['design']} on {r['vector']}" in printed for r in rows)
    with pytest.raises(SystemExit):
        main(["lean-headline", "--numbers", "--check"])


def test_the_instances_are_of_the_packages_own_data(tmp_path):
    with pytest.raises(SystemExit, match="this package's own data"):
        main(["--device", "sm_90", "lean-headline", "--check"])
    with pytest.raises(SystemExit, match="this package's own data"):
        main(["--data", str(tmp_path), "lean-headline"])
    assert not list(tmp_path.iterdir())
