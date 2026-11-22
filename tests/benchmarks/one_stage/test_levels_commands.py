"""The audit command's `SPOT_LEVELS`: it refuses a level count it doesn't run, by name, before reading anything, rather
than running another, two levels by `levels.TWO_LEVELS_REFUSED`; and the boundary record's format is the one `levels`
reads as two levels."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "benchmarks" / "one_stage"))
A5 = importlib.import_module("a5")
SC = importlib.import_module("served_cases")
LV = importlib.import_module("verity.protocols.proof.spot.verifier.levels")

A5_ARGS = ["a5.py", "--partition", "p", "--template-sets", "t", "--m0-pythonpath", "m", "--bin", "b", "--lean", "l",
           "--out", "o", "--bundle", "x"]


def test_a5_refuses_two_levels_by_name(monkeypatch, capsys):
    monkeypatch.setenv("SPOT_LEVELS", "2")
    monkeypatch.setattr(sys, "argv", A5_ARGS)
    with pytest.raises(SystemExit) as e:
        A5.main()
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert LV.TWO_LEVELS_REFUSED in err


@pytest.mark.parametrize("raw,reason", [("3", "levels-not-built"), ("one", "levels-not-integer")])
def test_a5_refuses_a_count_not_built_or_not_a_number(monkeypatch, capsys, raw, reason):
    monkeypatch.setenv("SPOT_LEVELS", raw)
    monkeypatch.setattr(sys, "argv", A5_ARGS)
    with pytest.raises(SystemExit):
        A5.main()
    assert reason in capsys.readouterr().err


def test_the_boundary_record_format_is_the_one_levels_reads_as_two():
    from verity_vllm.pipeline.boundary_rows import FORMAT

    assert LV.BOUNDARY_RECORD == FORMAT


def test_sampled_case_is_at_one_level():
    case = {"params": {"proofs": {}}, "evidence": b'{"proofs":[]}'.hex(), "commitment": "00"}
    got = SC.with_levels(case)
    assert got["params"]["levels"] == 1
    with pytest.raises(SC.Refused):
        SC.with_levels(got)
