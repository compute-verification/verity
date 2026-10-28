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


# `flock-verify rec-compose`'s rule (`RecCompose.check`, whose #guards are `Flock/Test/RecCompose.lean`), for the wiring:
# one log per V* statement, every VERDICT accepted at the SHA-512 of the one rows file and of the one inner shape file.
COMPOSER = r'''
import hashlib, json, sys
a = sys.argv[1:]
assert a[0] == "rec-compose", a
flags = {a[i]: a[i + 1] for i in range(1, len(a) - 1) if a[i] in ("--rec-inner", "--rec-rows")}
logs = [x for i, x in enumerate(a[1:], 1) if x not in flags and x not in flags.values()]
digest = hashlib.sha512(open(flags["--rec-rows"], "rb").read()).hexdigest()
inner = hashlib.sha512(open(flags["--rec-inner"], "rb").read()).hexdigest()
n = json.load(open(flags["--rec-inner"]))["n"]
parts, ok = [], True
for f in logs:
    vs = [json.loads(ln[8:]) for ln in open(f).read().splitlines() if ln.startswith("VERDICT ")]
    ok = ok and bool(vs) and all(v.get("accepted") is True and v.get("rec_rows_sha512") == digest
                                 and v.get("rec_inner_sha512") == inner for v in vs)
    parts.append(vs[0].get("rec_part") if vs else None)
ok = ok and sorted(parts, key=str) == list(range(n))
print("REC-COMPOSE " + json.dumps({"accepted": ok, "parts": len(logs)}))
sys.exit(0 if ok else 1)
'''


def _composer(d: Path, n: int, rows: bytes = b'{"ports":{}}') -> tuple[str, str]:
    """A stand-in `rec-compose` binary and 85's files beside the logs: ``rec-inner.json``, ``rows.json`` and V*'s ``n``
    statements in ``rec-statements.txt``. The rows file's SHA-512; `_inner` is the inner shape file's."""
    import hashlib

    (d / "compose.py").write_text(COMPOSER)
    binary = d / "flock-verify"
    binary.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{d / "compose.py"}" "$@"\n')
    binary.chmod(0o755)
    (d / "rec-inner.json").write_text(json.dumps({"n": n}))
    (d / "rows.json").write_bytes(rows)
    statements = [[f"s{j}", 1] for j in range(n)]
    (d / "rec-statements.txt").write_text(f"RECVSTAR {json.dumps({'statements': statements})}\n")
    return str(binary), hashlib.sha512(rows).hexdigest()


def _inner(d: Path) -> str:
    """The SHA-512 of the inner shape file ``rec-inner.json`` that `_composer` wrote in ``d``."""
    import hashlib

    return hashlib.sha512((d / "rec-inner.json").read_bytes()).hexdigest()


def _vpart(d: Path, name: str, j: int, digest: str, rc: int = 0, accepted: bool = True, inner: str | None = None) -> None:
    """A V* part 85 gave V*'s inputs (``own-PART.txt``): its log's VERDICT names its part, the rows it checked and the inner
    shape it read (``d``'s ``rec-inner.json`` unless ``inner``)."""
    v = {"session": f"/s/{name}", "accepted": accepted, "why": "" if accepted else "x", "rec_part": j, "rec_rows_sha512": digest,
         "rec_inner_sha512": inner or _inner(d)}
    (d / f"{name}.log").write_text(f"VERDICT {json.dumps(v)}\n")
    (d / f"time-{name}.txt").write_text(f"\tExit status: {rc}\n")
    (d / f"own-{name}.txt").write_text("--rec-rows rows.json")


def test_honest_parts_compose(tmp_path, capsys):
    """Every V* part accepted at the one rows file composes, and the step passes on Lean's composition."""
    binary, h = _composer(tmp_path, 3)
    for j, p in enumerate(("L0", "alg-p0", "cap")):
        _vpart(tmp_path, p, j, h)
    rc = LV.main(["--compose", binary, str(tmp_path), "L0", "alg-p0", "cap"])
    out = capsys.readouterr().out.splitlines()
    assert rc == 0 and out[-2] == 'lean compose: REC-COMPOSE {"accepted": true, "parts": 3}'


def test_a_part_checked_against_other_rows_is_refused(tmp_path, capsys):
    """Each part accepting alone is not enough: a part verified against another rows file (the prover's per-part rows) has
    another digest, and the composition refuses, so the step fails."""
    binary, h = _composer(tmp_path, 3)
    _vpart(tmp_path, "L0", 0, h)
    _vpart(tmp_path, "alg-p0", 1, h)
    _vpart(tmp_path, "cap", 2, "ab" * 64)
    rc = LV.main(["--compose", binary, str(tmp_path), "L0", "alg-p0", "cap"])
    out = capsys.readouterr().out.splitlines()
    assert rc == 1 and out[-2].endswith("<- FAILS") and out[-2].startswith("lean compose: REC-COMPOSE")
    assert out[-1] == "lean: 3/3 accepted: the step fails"


def test_a_part_under_another_inner_shape_is_refused(tmp_path, capsys):
    """A part verified under another inner shape file (`--rec-inner`) names another `rec_inner_sha512`, and the
    composition refuses, as `Flock/Test/RecCompose.lean`'s #guard does, so the step fails."""
    binary, h = _composer(tmp_path, 3)
    _vpart(tmp_path, "L0", 0, h)
    _vpart(tmp_path, "alg-p0", 1, h, inner="cd" * 64)
    _vpart(tmp_path, "cap", 2, h)
    rc = LV.main(["--compose", binary, str(tmp_path), "L0", "alg-p0", "cap"])
    out = capsys.readouterr().out.splitlines()
    assert rc == 1 and out[-2].endswith("<- FAILS") and out[-2].startswith("lean compose: REC-COMPOSE")


def test_a_staged_rows_file_with_other_rows_is_refused(tmp_path, capsys):
    """Every part checked against one file, but not the one staged: the composition reads the staged file's digest."""
    binary, h = _composer(tmp_path, 2)
    for j, p in enumerate(("L0", "cap")):
        _vpart(tmp_path, p, j, h)
    (tmp_path / "rows.json").write_bytes(b'{"ports":{"acc0":{}}}')
    assert LV.main(["--compose", binary, str(tmp_path), "L0", "cap"]) == 1


def test_a_forgery_in_its_parts_place_must_not_compose(tmp_path, capsys):
    """A forgery of a V* part, put in that part's place, is refused by the composition; one the composition would accept
    fails the step."""
    binary, h = _composer(tmp_path, 2)
    _vpart(tmp_path, "L0", 0, h)
    _vpart(tmp_path, "cap", 1, h)
    _vpart(tmp_path, "forged-sum-L0", 0, h, rc=1, accepted=False)
    rc = LV.main(["--compose", binary, str(tmp_path), "L0", "cap", "forged-sum-L0"])
    out = capsys.readouterr().out.splitlines()
    assert rc == 0 and any(ln.startswith("lean compose forged-sum-L0: REC-COMPOSE") and "FAILS" not in ln for ln in out)


def test_a_forged_inner_composes_as_one_forgery(tmp_path, capsys):
    """A forged inner coin (85's NEG_INNER) is every V* part verified against it: the parts that read the coin refuse it,
    the others accept, as they should. Its parts are put in place together and that composition is refused, so the step
    passes; putting in only a part that doesn't read the coin would compose and fail it."""
    binary, h = _composer(tmp_path, 3)
    for j, p in enumerate(("L0", "alg-p0", "cap")):
        _vpart(tmp_path, p, j, h)
    _vpart(tmp_path, "forged-inner-challenge-L0", 0, h)
    _vpart(tmp_path, "forged-inner-challenge-alg-p0", 1, h, rc=1, accepted=False)
    _vpart(tmp_path, "forged-inner-challenge-cap", 2, h)
    parts = ["L0", "alg-p0", "cap", *(f"forged-inner-challenge-{p}" for p in ("L0", "alg-p0", "cap"))]
    rc = LV.main(["--compose", binary, str(tmp_path), *parts])
    out = capsys.readouterr().out.splitlines()
    assert 'lean compose forged-inner-challenge (3 parts): REC-COMPOSE {"accepted": false, "parts": 3}' in out
    assert rc == 0 and out[-1] == "lean: 3/3 accepted, 1/1 forgeries refused"


def test_a_forgery_the_composition_accepts_fails(tmp_path, capsys):
    """A forgery whose every part Lean accepts composes, and the step fails on it."""
    binary, h = _composer(tmp_path, 2)
    for j, p in enumerate(("L0", "alg-p0")):
        _vpart(tmp_path, p, j, h)
        _vpart(tmp_path, f"forged-inner-challenge-{p}", j, h)
    rc = LV.main(["--compose", binary, str(tmp_path), "L0", "alg-p0", "forged-inner-challenge-L0",
                  "forged-inner-challenge-alg-p0"])
    out = capsys.readouterr().out.splitlines()
    assert rc == 1 and any(ln.startswith("lean compose forged-inner-challenge (2 parts)") and ln.endswith("<- FAILS")
                           for ln in out)


def test_a_run_of_some_parts_is_not_composed(tmp_path, capsys):
    """``LEVELS`` picks some of V*'s parts: there is no recursive statement to compose, and the step says so."""
    binary, h = _composer(tmp_path, 3)
    _vpart(tmp_path, "L3", 0, h)
    rc = LV.main(["--compose", binary, str(tmp_path), "L3"])
    out = capsys.readouterr().out.splitlines()
    assert rc == 0 and "lean compose: 1 of V*'s 3 parts run, so the recursive statement is not composed" in out
