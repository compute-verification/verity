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


def test_a_forgery_over_several_parts_needs_one_refusal_and_every_verdict(tmp_path, capsys):
    """V* accepts only when every part does: a forgery staged at several parts (``forged-KIND-PART``) is refused when one
    part refuses it, the parts it leaves honest accepting; it fails when none refuses or a part has no VERDICT."""
    for p in ("L0", "alg-p0", "alg-p1"):
        _part(tmp_path, p, 0, True)
    _part(tmp_path, "forged-comb-alg-p0", 1, False, why="rec-v row")
    _part(tmp_path, "forged-comb-alg-p1", 0, True)
    _part(tmp_path, "forged-inner-query-L0", 1, False, why="rec-dirs row")
    _part(tmp_path, "forged-inner-query-alg-p0", 0, True)
    parts = ("L0", "alg-p0", "alg-p1", "forged-comb-alg-p0", "forged-comb-alg-p1", "forged-inner-query-L0",
             "forged-inner-query-alg-p0")
    rc, out = _run(tmp_path, *parts, capsys=capsys)
    assert rc == 0 and out[-1] == "lean: 3/3 accepted, 2/2 forgeries refused"
    assert "lean forgery comb: refused by ['forged-comb-alg-p0'] of 2" in out
    assert "lean forgery inner-query: refused by ['forged-inner-query-L0'] of 2" in out
    _part(tmp_path, "forged-comb-alg-p0", 0, True)
    rc, out = _run(tmp_path, *parts, capsys=capsys)
    assert rc == 1 and "lean forgery comb: refused by no part of 2  <- FAILS" in out
    _part(tmp_path, "forged-comb-alg-p0", 1, False, why="rec-v row")
    (tmp_path / "forged-comb-alg-p1.log").unlink()
    assert _run(tmp_path, *parts, capsys=capsys)[0] == 1


def _live(d: Path, name: str, stopped: str | None) -> None:
    """``PART.live``: the prover's LIVE verdict as oprove keeps it, stopped at ``stopped``."""
    (d / f"{name}.live").write_text(json.dumps({"accepted": False, "why": stopped or "", "prover_stopped": stopped}) + "\n")


def test_a_forgery_the_firewall_held_back_is_not_exercised(tmp_path, capsys):
    """MODES=firewall: the firewall may hold a forgery's proofs back, so no session reaches Lean. The firewall is the
    prover's own check, which a dishonest prover doesn't run, so its FIREWALL-REFUSED stop leaves the verifier unexercised:
    the forgery fails the step as not exercised, never counts as refused. Any other stop, or none, leaves the part without
    a verdict, and a firewall refusal never passes an honest part."""
    for p in ("L0", "alg-p0"):
        _part(tmp_path, p, 0, True)
    _live(tmp_path, "forged-comb-alg-p0", "FIREWALL-REFUSED: alg-p0: the constraints [3] fail on a rep's pads")
    _part(tmp_path, "forged-comb-L0", 0, True)
    parts = ("L0", "alg-p0", "forged-comb-alg-p0", "forged-comb-L0")
    rc, out = _run(tmp_path, *parts, capsys=capsys)
    assert rc == 1 and out[-1] == "lean: 2/2 accepted, 0/1 forgeries refused, 1 not exercised (no proof produced): the step fails"
    assert out[2].startswith("lean forged-comb-alg-p0 (forgery comb: one of its parts must be refused): refused by the "
                             "firewall before Lean, no proof produced (not exercised), why \"FIREWALL-REFUSED: alg-p0: the "
                             "constraints [3]")
    assert ("lean forgery comb: no proof produced (not exercised) at ['forged-comb-alg-p0'] of 2, so the verifier never "
            "checked it  <- FAILS") in out
    _part(tmp_path, "forged-comb-alg-p0", 1, False, why="rec-v row")
    rc, out = _run(tmp_path, *parts, capsys=capsys)
    assert rc == 0 and "lean forgery comb: refused by ['forged-comb-alg-p0'] of 2" in out
    for f in ("forged-comb-alg-p0.log", "time-forged-comb-alg-p0.txt"):
        (tmp_path / f).unlink()
    _live(tmp_path, "forged-comb-alg-p0", "PROVER-STOPPED: the verifier hung up")
    assert _run(tmp_path, *parts, capsys=capsys)[0] == 1
    _live(tmp_path, "forged-comb-alg-p0", None)
    assert _run(tmp_path, *parts, capsys=capsys)[0] == 1
    (tmp_path / "alg-p0.log").unlink()
    (tmp_path / "time-alg-p0.txt").unlink()
    _live(tmp_path, "alg-p0", "FIREWALL-REFUSED: alg-p0's level-0 root")
    rc, out = _run(tmp_path, "L0", "alg-p0", capsys=capsys)
    assert rc == 1 and out[1].endswith("<- FAILS") and out[-1] == "lean: 1/2 accepted: the step fails"


def test_a_forgery_with_no_proof_is_not_exercised_not_refused(tmp_path, capsys):
    """ecd2's forged-coef-L0: the firewall's prover produced no proof (R7, a lone rep), so Lean refused the session for a
    missing file before any check. That is "no proof produced (not exercised)": it fails the step, never counts as refused,
    and a forgery another part of it refused is still refused."""
    missing = "session files: no such file or directory (error code: 2)\n  file: /v-sess/forged-coef-L0/circuit.rep0.bin"
    for p in ("L0", "alg-p0"):
        _part(tmp_path, p, 0, True)
    _part(tmp_path, "forged-coef-L0", 1, False, why=missing)
    _part(tmp_path, "forged-sum-L0", 1, False, why="zerocheck")
    parts = ("L0", "alg-p0", "forged-coef-L0", "forged-sum-L0")
    rc, out = _run(tmp_path, *parts, capsys=capsys)
    assert rc == 1
    assert out[2].startswith("lean forged-coef-L0 (forgery coef: one of its parts must be refused): exit 1, no proof produced "
                             "(not exercised), why 'session files: no such file")
    assert ("lean forgery coef: no proof produced (not exercised) at ['forged-coef-L0'] of 1, so the verifier never checked "
            "it  <- FAILS") in out
    assert out[-1] == "lean: 2/2 accepted, 1/2 forgeries refused, 1 not exercised (no proof produced): the step fails"
    _part(tmp_path, "forged-coef-alg-p0", 1, False, why="rec-v row")
    rc, out = _run(tmp_path, *parts, "forged-coef-alg-p0", capsys=capsys)
    assert rc == 0 and "lean forgery coef: refused by ['forged-coef-alg-p0'] of 2" in out
    _part(tmp_path, "L0", 1, False, why=missing)
    rc, out = _run(tmp_path, "L0", capsys=capsys)
    assert rc == 1 and "no proof produced (not exercised)" in out[0] and out[0].endswith("<- FAILS")


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


def test_a_forged_inner_coin_no_part_run_reads_is_not_covered(tmp_path, capsys):
    """A forged inner coin is refused only by the parts that read it (the challenge by the algebra, the query by L0): a
    subset run without one (LEVELS=L0, L3, cap) can't refuse it, so its forgery is "not covered by the parts run" and fails
    the step, never "refused by no part" as if Lean had accepted it; with a covering part that accepts it, the step fails as
    a real acceptance."""
    _part(tmp_path, "L0", 0, True)
    _part(tmp_path, "forged-inner-challenge-L0", 0, True)
    _part(tmp_path, "forged-inner-query-L0", 1, False, why="rec-dirs row")
    parts = ("L0", "forged-inner-challenge-L0", "forged-inner-query-L0")
    rc, out = _run(tmp_path, *parts, capsys=capsys)
    assert rc == 1
    assert ("lean forgery inner-challenge: not covered by the parts run (needs one of ['alg'], or its parts; ran ['L0'])  "
            "<- FAILS") in out
    assert not any("refused by no part" in line for line in out)
    assert out[-1] == "lean: 1/1 accepted, 1/2 forgeries refused, 1 not covered by the parts run: the step fails"
    for p in ("alg-p0", "alg-p1"):
        _part(tmp_path, p, 0, True)
        _part(tmp_path, f"forged-inner-challenge-{p}", 1, False, why="rec-v row")
    parts += ("alg-p0", "alg-p1", "forged-inner-challenge-alg-p0", "forged-inner-challenge-alg-p1")
    rc, out = _run(tmp_path, *parts, capsys=capsys)
    assert rc == 0 and out[-1] == "lean: 3/3 accepted, 2/2 forgeries refused"
    for p in ("alg-p0", "alg-p1"):
        _part(tmp_path, f"forged-inner-challenge-{p}", 0, True)
    rc, out = _run(tmp_path, *parts, capsys=capsys)
    assert rc == 1 and "lean forgery inner-challenge: refused by no part of 3  <- FAILS" in out


@pytest.mark.parametrize("kind, parts, rc", [
    ("inner-challenge", ["L0"], 1), ("inner-challenge", ["L3", "cap-L3"], 1), ("inner-challenge", ["L0", "alg-p0"], 0),
    ("inner-challenge", ["alg"], 0), ("inner-query", ["L1", "alg-p0"], 1), ("inner-query", ["L0"], 0),
    ("comb", ["L0"], 0)])
def test_cover_says_whether_a_part_run_reads_the_forged_coin(capsys, kind, parts, rc):
    """``--cover``, which 85's lean step runs before it forges an inner coin, refuses the forgery when no part reads it; a
    kind COVERS doesn't name (a forged statement, its own part) is not refused."""
    assert LV.main(["--cover", kind, *parts]) == rc
    out = capsys.readouterr().out
    assert ("not covered by the parts run" in out) == bool(rc)
