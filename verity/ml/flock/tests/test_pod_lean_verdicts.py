"""The pod jobs' Lean step fails closed (`verity/ml/flock/pod/lean_verdicts.py`, which 85-rec-reprice.sh and 86-zk-cpu-steps.sh
run after their parallel `verify` lines): a part passes on its exit and its log's VERDICT together, a forgery only when Lean
refuses it, and a part that never ran, or no part at all, fails the step."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

POD = Path(__file__).resolve().parents[1] / "pod"
sys.path.insert(0, str(POD))
import lean_verdicts as LV  # noqa: E402


def _part(d: Path, name: str, rc: int | None, *accepted: bool, why: str = "") -> None:
    """``PART.log`` with one VERDICT per entry of ``accepted`` and ``time-PART.txt`` as ``/usr/bin/time -v`` writes it."""
    lines = [json.dumps({"session": f"/s/{name}/{i}", "accepted": a, "why": "" if a else why}, separators=(",", ":"))
             for i, a in enumerate(accepted)]
    (d / f"{name}.log").write_text("".join(f"VERDICT {line}\n" for line in lines))
    if rc is not None:
        (d / f"time-{name}.txt").write_text(f'\tCommand being timed: "flock-verify verify"\n\tExit status: {rc}\n')


def _run(d: Path, *parts: str, capsys) -> tuple[int, list[str]]:
    rc = LV.main([str(d), *parts])
    return rc, capsys.readouterr().out.splitlines()


def test_every_part_accepted_passes(tmp_path, capsys):
    for p in ("L0", "L1", "alg-p0"):
        _part(tmp_path, p, 0, True)
    rc, out = _run(tmp_path, "L0", "L1", "alg-p0", capsys=capsys)
    assert rc == 0 and out[-1] == "lean: 3/3 accepted"
    assert out[0] == "lean L0: exit 0, accepted true, why ''"


def test_one_refusal_fails_the_step(tmp_path, capsys):
    _part(tmp_path, "L0", 0, True)
    _part(tmp_path, "L1", 1, False, why="refused: a session without a draw (D12)")
    rc, out = _run(tmp_path, "L0", "L1", capsys=capsys)
    assert rc == 1 and out[-1] == "lean: 1/2 accepted: the step fails"
    assert "why 'refused: a session without a draw (D12)'  <- FAILS" in out[1]


@pytest.mark.parametrize("rc, accepted", [(1, (True,)), (0, (False,)), (0, (True, False)), (0, ()), (2, ()), (None, (True,))],
                         ids=["exit-1-accepted", "exit-0-refused", "one-session-of-two-refused", "no-verdict",
                              "inputs-refused", "no-exit-record"])
def test_an_exit_or_a_verdict_alone_is_not_enough(tmp_path, capsys, rc, accepted):
    _part(tmp_path, "L0", rc, *accepted, why="x")
    assert _run(tmp_path, "L0", capsys=capsys)[0] == 1


def test_a_part_that_never_ran_fails(tmp_path, capsys):
    _part(tmp_path, "L0", 0, True)
    rc, out = _run(tmp_path, "L0", "L1", capsys=capsys)
    assert rc == 1 and out[1] == "lean L1: exit None, accepted [], why 'no VERDICT'  <- FAILS"


def test_no_part_fails(tmp_path, capsys):
    rc, out = _run(tmp_path, capsys=capsys)
    assert rc == 1 and out == ["lean: 0/0 accepted: no part to check, the step fails"]


def test_a_forgery_passes_only_when_lean_refuses_it(tmp_path, capsys):
    _part(tmp_path, "L0", 0, True)
    _part(tmp_path, "forged-sum-L3", 1, False, why="zerocheck")
    rc, out = _run(tmp_path, "L0", "forged-sum-L3", capsys=capsys)
    assert rc == 0 and out[-1] == "lean: 1/1 accepted, 1/1 forgeries refused"
    _part(tmp_path, "forged-sum-L3", 0, True)
    rc, out = _run(tmp_path, "L0", "forged-sum-L3", capsys=capsys)
    assert rc == 1 and out[-1] == "lean: 1/1 accepted, 0/1 forgeries refused: the step fails"
    _part(tmp_path, "forged-sum-L3", 2)
    assert _run(tmp_path, "L0", "forged-sum-L3", capsys=capsys)[0] == 1


def _statement(d: Path, rc: int, line: str) -> None:
    (d / "inner-statement.log").write_text(line + "\n")
    (d / "time-inner-statement.txt").write_text(f"\tExit status: {rc}\n")


def test_the_inner_statement_passes_only_with_its_units_derived_from_the_program(tmp_path, capsys):
    """`flock-verify statement`'s line is the part's verdict: units derived from the verifier's program pass; units taken as
    stated (no program) and a setup refusal (f05d's, a unit-class template) fail."""
    def stmt(units: str, note: str) -> str:
        return json.dumps({"statement_digest": "5b6f", "partition_check": {"rule": "verity/partition/v1", "units": units, "note": note}})

    _statement(tmp_path, 0, stmt("derived", "units derived: Q_word v1 over the verifier's program"))
    rc, out = _run(tmp_path, "inner-statement", capsys=capsys)
    assert rc == 0 and out == ["lean inner-statement: exit 0, accepted true, why \"units derived: Q_word v1 over the verifier's program\"",
                               "lean: 1/1 accepted"]
    _statement(tmp_path, 0, stmt("as stated", "units as stated"))
    assert _run(tmp_path, "inner-statement", capsys=capsys)[0] == 1
    _statement(tmp_path, 1, "setup: partition: unit 0 is a unit of GemmCoordinate_v2{K=4096}, not of the circuit's template unit-class/731c")
    rc, out = _run(tmp_path, "inner-statement", capsys=capsys)
    assert rc == 1 and "exit 1, accepted false, why \"setup: partition: unit 0 is a unit of" in out[0] and out[0].endswith("<- FAILS")
