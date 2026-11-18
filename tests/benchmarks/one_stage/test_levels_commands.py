"""The audit commands' `SPOT_LEVELS`: each refuses a level count it doesn't run, by name, before reading anything,
rather than running another; and the boundary record's format is the one `levels` reads as two levels."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "benchmarks" / "one_stage"))
A5 = importlib.import_module("a5")
TS = importlib.import_module("two_stage")
SC = importlib.import_module("served_cases")
LV = importlib.import_module("verity.protocols.proofs.spot.verifier.levels")

A5_ARGS = ["a5.py", "--partition", "p", "--template-sets", "t", "--m0-pythonpath", "m", "--bin", "b", "--lean", "l",
           "--out", "o", "--bundle", "x"]


def test_a5_refuses_two_levels_and_names_the_two_level_command(monkeypatch, capsys):
    monkeypatch.setenv("SPOT_LEVELS", "2")
    monkeypatch.setattr(sys, "argv", A5_ARGS)
    with pytest.raises(SystemExit) as e:
        A5.main()
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "levels-no-path" in err and "two_stage.py" in err


@pytest.mark.parametrize("raw,reason", [("3", "levels-not-built"), ("one", "levels-not-integer")])
def test_a5_refuses_a_count_not_built_or_not_a_number(monkeypatch, capsys, raw, reason):
    monkeypatch.setenv("SPOT_LEVELS", raw)
    monkeypatch.setattr(sys, "argv", A5_ARGS)
    with pytest.raises(SystemExit):
        A5.main()
    assert reason in capsys.readouterr().err


@pytest.mark.parametrize("raw,reason", [("1", "levels-no-path"), ("3", "levels-not-built")])
def test_two_stage_refuses_any_count_but_two(monkeypatch, capsys, tmp_path, raw, reason):
    monkeypatch.setenv("SPOT_LEVELS", raw)
    with pytest.raises(SystemExit) as e:
        TS.main(["first", "--boundary", str(tmp_path / "none"), "--check", "c", "--replay", "rate:1/2", "--k", "1",
                 "--challenger", "local", "--main", "m", "--out", str(tmp_path / "out")])
    assert e.value.code == 2
    assert reason in capsys.readouterr().err


def test_two_stage_refuses_a_boundary_record_serving_one_level(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("SPOT_LEVELS", raising=False)
    d = tmp_path / "w0" / "p16-o16"
    d.mkdir(parents=True)
    (d / TS.BR.COMMIT).write_text('{"levels": 1, "run": "r", "window": 0}')
    with pytest.raises(SystemExit):
        TS.main(["first", "--boundary", str(d), "--check", "c", "--replay", "rate:1/2", "--k", "1",
                 "--challenger", "local", "--main", "m", "--out", str(tmp_path / "out")])
    assert "serves 1 level(s), not the two this command audits" in capsys.readouterr().err


def test_the_boundary_record_format_is_the_one_levels_reads_as_two():
    from verity_vllm.records import boundary_commit

    assert LV.BOUNDARY_RECORD == boundary_commit.SCHEMA


def test_sampled_case_is_at_one_level():
    case = {"params": {"proofs": {}}, "evidence": b'{"proofs":[]}'.hex(), "commitment": "00"}
    got = SC.with_levels(case)
    assert got["params"]["form"] == "one-stage"
    with pytest.raises(SC.Refused):
        SC.with_levels(got)
