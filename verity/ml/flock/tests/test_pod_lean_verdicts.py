"""The pod jobs' Lean step fails closed (`verity/ml/flock/pod/lean_verdicts.py`, which 85-rec-reprice.sh and 86-zk-cpu-steps.sh
run after their parallel `verify` lines): a part passes on its exit and its log's VERDICT together, a forgery only when Lean
refuses it, and a part that never ran, or no part at all, fails the step."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from verity.core.protocols.sampled_proofs import challenger as C
from verity_flock import rec_compose as RC

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
    """MODES=ZK server: the ZK server may hold a forgery's proofs back, so no session reaches Lean. The ZK server is the
    prover's own check, which a dishonest prover doesn't run, so its FIREWALL-REFUSED stop leaves the verifier unexercised:
    the forgery fails the step as not exercised, never counts as refused. Any other stop, or none, leaves the part without
    a verdict, and a ZK server refusal never passes an honest part."""
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
    """ecd2's forged-coef-L0: the ZK server's prover produced no proof (R7, a lone rep), so Lean refused the session for a
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
# its mirror, `verity_flock.rec_compose`, which `test_lean_rec_compose.py` holds to the binary.
COMPOSER = r'''
import json, sys
from verity_flock import rec_compose as RC
a = sys.argv[1:]
assert a[0] == "rec-compose", a
flags = {a[i]: a[i + 1] for i in range(1, len(a) - 1) if a[i] in ("--rec-inner", "--rec-rows", "--rec-run", "--rec-links")}
logs = [x for i, x in enumerate(a[1:], 1) if x not in flags and x not in flags.values()]
inner = open(flags["--rec-inner"], "rb").read()
links = flags.get("--rec-links")
links = None if links is None else (bytes.fromhex(links.split(":")[0]), int(links.split(":")[1]))
ok, why = RC.check(json.loads(inner)["n"], RC.digest(open(flags["--rec-rows"], "rb").read()), RC.digest(inner),
                   open(flags["--rec-run"]).read(), [(f, open(f).read()) for f in logs], links)
print("REC-COMPOSE " + json.dumps({"accepted": ok, "why": why, "parts": len(logs)}))
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


def _key(name: str) -> str:
    """A part's session's coin key, 256 bytes in lowercase hex."""
    return hashlib.sha512(name.encode()).hexdigest() * 4


def _record(d: Path, digest: str, parts: list[tuple[str, int]], session: str = "s", transcript: str = "t") -> None:
    """The challenger's interactive-run record ``rec-run.json`` (`challenger.RunRecord`): the staging of the rows file
    ``digest`` and the shared set `_vpart` names, one session opened for each (name, part) of ``parts``, the close."""
    r = C.RunRecord()
    r.stage(digest, _shared(d, None, session, transcript))
    for name, j in parts:
        r.open(name, j, _key(name))
    r.close()
    r.write(d / "rec-run.json")


def _shared(d: Path, inner: str | None, session: str, transcript: str, links: str | None = None) -> dict:
    """The shared set a part reads; its link table ``links`` (``ROOT:N``), else DIR's ``rec-links.txt`` when it has one."""
    rec_inner = inner.encode() if inner is not None else (d / "rec-inner.json").read_bytes()
    if links is None and (d / "rec-links.txt").exists():
        links = (d / "rec-links.txt").read_text().strip()
    root, _, rows = (links or "").partition(":")
    return RC.shared(rec_inner=rec_inner, inner_circuit=b"c", inner_public=b"p", inner_partition=b"{}",
                     inner_session=session.encode(), inner_transcript=transcript.encode(), inner_statement="i",
                     inner_statement_coins="os", inner_tables=[],
                     rec_links=(bytes.fromhex(root), int(rows)) if links else None)


def _vpart(d: Path, name: str, j: int, digest: str, rc: int = 0, accepted: bool = True, inner: str | None = None,
           session: str = "s", transcript: str = "t", key: str | None = None, links: str | None = None) -> None:
    """A V* part 85 gave V*'s inputs (``own-PART.txt``): its log's VERDICT names its part, the rows it checked, the
    inputs every part shares (at DIR's ``rec-inner.json`` unless ``inner`` is another inner shape file's bytes, at
    the inner session record ``session`` and transcript ``transcript``, and at the link table ``links``, else DIR's),
    ``--zk`` and its coin key (``key``'s, else its own name's)."""
    v = {"session": f"/s/{name}", "accepted": accepted, "why": "" if accepted else "x", "rec_part": j,
         "rec_rows_sha512": digest, "rec_shared": _shared(d, inner, session, transcript, links), "zk": True,
         "coin_key": _key(key or name)}
    (d / f"{name}.log").write_text(f"VERDICT {json.dumps(v)}\n")
    (d / f"time-{name}.txt").write_text(f"\tExit status: {rc}\n")
    (d / f"own-{name}.txt").write_text(f"--rec-rows rows.json --rec-part {j}")


def test_honest_parts_compose(tmp_path, capsys):
    """Every V* part accepted at the one rows file composes, and the step passes on Lean's composition."""
    binary, h = _composer(tmp_path, 3)
    for j, p in enumerate(("L0", "alg-p0", "cap")):
        _vpart(tmp_path, p, j, h)
    _record(tmp_path, h, [("L0", 0), ("alg-p0", 1), ("cap", 2)])
    rc = LV.main(["--compose", binary, str(tmp_path), "L0", "alg-p0", "cap"])
    out = capsys.readouterr().out.splitlines()
    assert rc == 0 and out[-2] == 'lean compose: REC-COMPOSE {"accepted": true, "why": "", "parts": 3}'


def test_a_retried_part_does_not_compose(tmp_path, capsys):
    """Every part accepted, but the challenger's record opened two sessions for one of them (a retry, the accepted one
    handed over): the composition refuses, so the step fails."""
    binary, h = _composer(tmp_path, 3)
    for j, p in enumerate(("L0", "alg-p0", "cap")):
        _vpart(tmp_path, p, j, h)
    _record(tmp_path, h, [("L0", 0), ("alg-p0-first", 1), ("alg-p0", 1), ("cap", 2)])
    rc = LV.main(["--compose", binary, str(tmp_path), "L0", "alg-p0", "cap"])
    out = capsys.readouterr().out.splitlines()
    assert rc == 1 and out[-2].endswith("<- FAILS") and out[-2].startswith("lean compose: REC-COMPOSE")


def test_a_run_without_its_record_is_not_composed(tmp_path, capsys):
    """Without the challenger's interactive-run record the parts are not shown to be one run: no composition, and the
    step says so."""
    binary, h = _composer(tmp_path, 2)
    for j, p in enumerate(("L0", "cap")):
        _vpart(tmp_path, p, j, h)
    rc = LV.main(["--compose", binary, str(tmp_path), "L0", "cap"])
    out = capsys.readouterr().out.splitlines()
    assert rc == 0 and out[-2].startswith("lean compose: no interactive-run record (rec-run.json, or a firewalled oprove")


def _challenger(d: Path, opens: list[tuple[str, int]], session: str = "s", close: bool = True,
                name: str = "rec-run.jsonl") -> list[dict]:
    """The record 85's firewalled oprove keeps, one event a line (``rec-run.jsonl``): its staging of the rows file and
    the shared set the parts were given (`_shared`'s, at the inner session record ``session``), then one open per
    (key name, part) of ``opens`` as ``flock-circuit serve --rec-run`` logs it at each Hello, then (``close``) the close;
    under another ``name`` (a forgery's), its opens alone. Its events."""
    stage = {"event": "stage", "rec_rows_sha512": RC.digest((d / "rows.json").read_bytes()),
             "rec_shared": _shared(d, None, session, "t")}
    evs = [*([stage] if name == "rec-run.jsonl" else []),
           *({"event": "open", "session": f"l{i:04}-{k}", "rec_part": j, "coin_key": _key(k)} for i, (k, j) in enumerate(opens)),
           *([{"event": "close"}] if close and name == "rec-run.jsonl" else [])]
    (d / name).write_text("".join(json.dumps(e, separators=(",", ":")) + "\n" for e in evs))
    return evs


def test_a_firewalled_oprove_composes_over_the_record_its_coin_servers_logged(tmp_path, capsys):
    """The run's record is the challenger's ``rec-run.jsonl``, event for event: oprove's staging, each part's open as its
    coin server logged it at the Hello, the close; the parts compose over it."""
    binary, h = _composer(tmp_path, 3)
    parts = {"L0": 0, "alg-p0": 1, "cap": 2}
    for p, j in parts.items():
        _vpart(tmp_path, p, j, h)
    evs = _challenger(tmp_path, list(parts.items()))
    rc = LV.main(["--compose", binary, str(tmp_path), *parts])
    out = capsys.readouterr().out.splitlines()
    assert rc == 0 and out[-2] == 'lean compose: REC-COMPOSE {"accepted": true, "why": "", "parts": 3}', out
    assert json.loads((tmp_path / "rec-run.json").read_text()) == {"events": evs, "kind": C.RUN_KIND}


def test_a_second_session_logged_for_a_part_does_not_compose(tmp_path, capsys):
    """A part's coin server that answered two Hellos logged both, and the composition refuses the retry."""
    binary, h = _composer(tmp_path, 2)
    parts = {"L0": 0, "cap": 1}
    for p, j in parts.items():
        _vpart(tmp_path, p, j, h)
    _challenger(tmp_path, [("L0", 0), ("L0-again", 0), ("cap", 1)])
    rc = LV.main(["--compose", binary, str(tmp_path), *parts])
    out = capsys.readouterr().out.splitlines()
    assert rc == 1 and out[-2].endswith("<- FAILS") and "part 0: 2 sessions opened in the run record" in out[-2]


@pytest.mark.parametrize("retried, opens, passes", [
    ("L0", [("L0", 0), ("L0-retry", 0), ("cap", 1)], True),
    ("L0", [("L0", 0), ("cap", 1)], False),
    ("cap", [("L0", 0), ("L0-retry", 0), ("cap", 1)], False),
])
def test_a_forced_retry_passes_only_when_its_composition_is_refused_as_retried(tmp_path, capsys, retried, opens, passes):
    """85's RETRY gives a statement a second session, logged as any other: the step passes only when the composition is
    refused for that statement's part being retried, and fails when it composes (the retry not logged) or is refused for
    another part."""
    binary, h = _composer(tmp_path, 2)
    parts = {"L0": 0, "cap": 1}
    for p, j in parts.items():
        _vpart(tmp_path, p, j, h)
    _challenger(tmp_path, opens)
    (tmp_path / "retried.txt").write_text(retried + "\n")
    rc = LV.main(["--compose", binary, str(tmp_path), *parts])
    out = capsys.readouterr().out.splitlines()
    assert out[-2].startswith(f"lean compose (RETRY={retried}, part {parts[retried]}: refused as retried")
    assert (rc, out[-2].endswith("<- FAILS")) == ((0, False) if passes else (1, True)), out


@pytest.mark.parametrize("close, torn, why", [
    (False, False, "not closed"),
    (True, True, "no interactive-run record from rec-run.jsonl (rec-run.jsonl: its line 2 is not JSON)"),
])
def test_an_unclosed_or_torn_record_does_not_compose(tmp_path, capsys, close, torn, why):
    """An oprove that never closed its record (it stopped before the last session ended), or a line not written whole, is
    no run's record: the composition refuses."""
    binary, h = _composer(tmp_path, 2)
    parts = {"L0": 0, "cap": 1}
    for p, j in parts.items():
        _vpart(tmp_path, p, j, h)
    _challenger(tmp_path, list(parts.items()), close=close)
    if torn:
        lines = (tmp_path / "rec-run.jsonl").read_text().splitlines(keepends=True)
        (tmp_path / "rec-run.jsonl").write_text(lines[0] + lines[1][:40] + "\n" + "".join(lines[2:]))
    rc = LV.main(["--compose", binary, str(tmp_path), *parts])
    out = capsys.readouterr().out.splitlines()
    assert rc == 1 and why in out[-2] and out[-2].endswith("<- FAILS"), out


def test_a_part_given_other_shared_files_than_the_staged_ones_does_not_compose(tmp_path, capsys):
    """oprove stages the shared set of the files it gives every part, so parts whose VERDICTs name another inner
    session's record are refused, though they agree with each other."""
    binary, h = _composer(tmp_path, 2)
    parts = {"L0": 0, "cap": 1}
    for p, j in parts.items():
        _vpart(tmp_path, p, j, h, session="other")
    _challenger(tmp_path, list(parts.items()))
    rc = LV.main(["--compose", binary, str(tmp_path), *parts])
    assert rc == 1 and capsys.readouterr().out.splitlines()[-2].endswith("<- FAILS")


def test_a_forgery_is_composed_over_its_own_coin_servers_opens_in_its_parts_place(tmp_path, capsys):
    """A forgery's coin server logs into a record of its own, so the honest run's stays one session a part; its
    composition is over the run's record with its part's opens replaced by the forgery's, so it is refused by the
    forgery's own verdict, not by a session the record lacks."""
    binary, h = _composer(tmp_path, 2)
    parts = {"L0": 0, "cap": 1}
    for p, j in parts.items():
        _vpart(tmp_path, p, j, h)
    _vpart(tmp_path, "forged-sum-L0", 0, h, rc=1, accepted=False)
    stage, _, cap, close = _challenger(tmp_path, list(parts.items()))
    (forged,) = _challenger(tmp_path, [("forged-sum-L0", 0)], name="rec-run-forged-sum-L0.jsonl")
    rc = LV.main(["--compose", binary, str(tmp_path), *parts, "forged-sum-L0"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert json.loads((tmp_path / "rec-run-forged-sum-L0.json").read_text())["events"] == [stage, cap, forged, close]
    assert "lean compose: REC-COMPOSE {\"accepted\": true" in out
    forged_line = next(ln for ln in out.splitlines() if ln.startswith("lean compose forged-sum-L0: REC-COMPOSE "))
    assert "forged-sum-L0.log: a session is not accepted" in LV.why_of(forged_line.split(": ", 1)[1])


def test_a_wired_run_composes_at_its_link_table_and_its_forged_table_row_is_refused(tmp_path, capsys):
    """85's REC_LINKS=1: every part, the link table's included, read the run's ``--rec-links`` (``rec-links.txt``), which
    the composition is given as its own, so the parts compose. The forgery ``forged-link-links`` (a table row naming a writer
    whose output is not its reader's row) was verified at a table of its own, which its part refuses, and its composition
    in the link part's place is refused."""
    binary, h = _composer(tmp_path, 3)
    table, forged_table = "ab" * 64 + ":2", "cd" * 64 + ":2"
    (tmp_path / "rec-links.txt").write_text(table + "\n")
    parts = {"L0": 0, "links": 1, "cap": 2}
    for p, j in parts.items():
        _vpart(tmp_path, p, j, h)
    _vpart(tmp_path, "forged-link-links", 1, h, rc=1, accepted=False, links=forged_table)
    _challenger(tmp_path, list(parts.items()))
    _challenger(tmp_path, [("forged-link-links", 1)], name="rec-run-forged-link-links.jsonl")
    rc = LV.main(["--compose", binary, str(tmp_path), *parts, "forged-link-links"])
    out = capsys.readouterr().out.splitlines()
    assert rc == 0, out
    assert 'lean compose: REC-COMPOSE {"accepted": true, "why": "", "parts": 3}' in out
    line = next(ln for ln in out if ln.startswith("lean compose forged-link-links: REC-COMPOSE "))
    assert not line.endswith("<- FAILS") and '"accepted": false' in line, out
    assert any(ln.startswith("lean forged-link-links (forgery link:") for ln in out), out


def test_a_wired_run_without_its_link_table_does_not_compose(tmp_path, capsys):
    """Parts verified at a link table compose only at that table: without ``rec-links.txt`` the composition is given
    none, and refuses them."""
    binary, h = _composer(tmp_path, 2)
    parts = {"L0": 0, "links": 1}
    for p, j in parts.items():
        _vpart(tmp_path, p, j, h, links="ab" * 64 + ":2")
    _challenger(tmp_path, list(parts.items()))
    rc = LV.main(["--compose", binary, str(tmp_path), *parts])
    out = capsys.readouterr().out.splitlines()
    assert rc == 1 and out[-2].endswith("<- FAILS") and "not the composition's own --rec-links none" in out[-2], out


def test_a_forged_inner_keeps_its_parts_honest_opens(tmp_path, capsys):
    """A forged inner (85's NEG_INNER) is verified over each part's honest session, with no coin server of its own: its
    composition is over the challenger's record as logged, and Lean refuses it for the part that reads the coin."""
    binary, h = _composer(tmp_path, 2)
    parts = {"L0": 0, "alg-p0": 1}
    for p, j in parts.items():
        _vpart(tmp_path, p, j, h)
    _vpart(tmp_path, "forged-inner-challenge-L0", 0, h, key="L0")
    _vpart(tmp_path, "forged-inner-challenge-alg-p0", 1, h, rc=1, accepted=False, key="alg-p0")
    for p in parts:
        (tmp_path / f"own-forged-inner-challenge-{p}.txt").unlink()
    evs = _challenger(tmp_path, list(parts.items()))
    rc = LV.main(["--compose", binary, str(tmp_path), *parts, *(f"forged-inner-challenge-{p}" for p in parts)])
    out = capsys.readouterr().out.splitlines()
    line = next(ln for ln in out if ln.startswith("lean compose forged-inner-challenge (2 parts): REC-COMPOSE "))
    assert rc == 0 and not line.endswith("<- FAILS"), out
    assert "forged-inner-challenge-alg-p0.log: a session is not accepted" in LV.why_of(line.split(": ", 1)[1])
    assert json.loads((tmp_path / "rec-run.json").read_text())["events"] == evs


def test_a_staged_forgery_without_its_own_record_fails(tmp_path, capsys):
    """A staged forgery proved on its own coin server must have that server's record: without it there is nothing to
    compose it over, which is no refusal by Lean, and the step fails."""
    binary, h = _composer(tmp_path, 2)
    parts = {"L0": 0, "cap": 1}
    for p, j in parts.items():
        _vpart(tmp_path, p, j, h)
    _vpart(tmp_path, "forged-sum-L0", 0, h, rc=1, accepted=False)
    _challenger(tmp_path, list(parts.items()))
    rc = LV.main(["--compose", binary, str(tmp_path), *parts, "forged-sum-L0"])
    out = capsys.readouterr().out.splitlines()
    line = next(ln for ln in out if ln.startswith("lean compose forged-sum-L0: "))
    assert rc == 1 and "no interactive-run record from rec-run.jsonl" in line and line.endswith("<- FAILS"), out


def test_a_part_checked_against_other_rows_is_refused(tmp_path, capsys):
    """Each part accepting alone is not enough: a part verified against another rows file (the prover's per-part rows) has
    another digest, and the composition refuses, so the step fails."""
    binary, h = _composer(tmp_path, 3)
    _vpart(tmp_path, "L0", 0, h)
    _vpart(tmp_path, "alg-p0", 1, h)
    _vpart(tmp_path, "cap", 2, "ab" * 64)
    _record(tmp_path, h, [("L0", 0), ("alg-p0", 1), ("cap", 2)])
    rc = LV.main(["--compose", binary, str(tmp_path), "L0", "alg-p0", "cap"])
    out = capsys.readouterr().out.splitlines()
    assert rc == 1 and out[-2].endswith("<- FAILS") and out[-2].startswith("lean compose: REC-COMPOSE")
    assert out[-1] == "lean: 3/3 accepted: the step fails"


def test_a_part_under_another_inner_shape_is_refused(tmp_path, capsys):
    """A part verified under another inner shape file (`--rec-inner`) names another in its `rec_shared`, and the
    composition refuses, as `Flock/Test/RecCompose.lean`'s #guards do, so the step fails."""
    binary, h = _composer(tmp_path, 3)
    _vpart(tmp_path, "L0", 0, h)
    _vpart(tmp_path, "alg-p0", 1, h, inner="cd" * 64)
    _vpart(tmp_path, "cap", 2, h)
    _record(tmp_path, h, [("L0", 0), ("alg-p0", 1), ("cap", 2)])
    rc = LV.main(["--compose", binary, str(tmp_path), "L0", "alg-p0", "cap"])
    out = capsys.readouterr().out.splitlines()
    assert rc == 1 and out[-2].endswith("<- FAILS") and out[-2].startswith("lean compose: REC-COMPOSE")


@pytest.mark.parametrize("other", [{"session": "ff" * 64}, {"transcript": "22" * 64}])
def test_parts_from_different_inner_sessions_are_refused(tmp_path, capsys, other):
    """Two parts verified over different inner sessions (85's `$REC_INNER`, `--rec-inner-session`) or transcripts of one
    (`--rec-inner-transcript`) name different ones in their `rec_shared`, each accepting alone, and the composition refuses, as `Flock/Test/RecCompose.lean`'s #guards do, so the step fails."""
    binary, h = _composer(tmp_path, 3)
    _vpart(tmp_path, "L0", 0, h)
    _vpart(tmp_path, "alg-p0", 1, h)
    _vpart(tmp_path, "cap", 2, h, **other)
    _record(tmp_path, h, [("L0", 0), ("alg-p0", 1), ("cap", 2)])
    rc = LV.main(["--compose", binary, str(tmp_path), "L0", "alg-p0", "cap"])
    out = capsys.readouterr().out.splitlines()
    assert rc == 1 and out[-2].endswith("<- FAILS") and out[-2].startswith("lean compose: REC-COMPOSE")
    assert out[-1] == "lean: 3/3 accepted: the step fails"


def test_a_staged_rows_file_with_other_rows_is_refused(tmp_path, capsys):
    """Every part checked against one file, but not the one staged: the composition reads the staged file's digest."""
    binary, h = _composer(tmp_path, 2)
    for j, p in enumerate(("L0", "cap")):
        _vpart(tmp_path, p, j, h)
    _record(tmp_path, h, [("L0", 0), ("cap", 1)])
    (tmp_path / "rows.json").write_bytes(b'{"ports":{"acc0":{}}}')
    assert LV.main(["--compose", binary, str(tmp_path), "L0", "cap"]) == 1


def test_a_forgery_in_its_parts_place_must_not_compose(tmp_path, capsys):
    """A forgery of a V* part, put in that part's place, is refused by the composition; one the composition would accept
    fails the step."""
    binary, h = _composer(tmp_path, 2)
    _vpart(tmp_path, "L0", 0, h)
    _vpart(tmp_path, "cap", 1, h)
    _vpart(tmp_path, "forged-sum-L0", 0, h, rc=1, accepted=False)
    _record(tmp_path, h, [("L0", 0), ("cap", 1)])
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
    _run_of_logs(tmp_path, ["L0", "alg-p0", "cap"])
    rc = LV.main(["--compose", binary, str(tmp_path), *parts])
    out = capsys.readouterr().out.splitlines()
    assert any(ln.startswith('lean compose forged-inner-challenge (3 parts): REC-COMPOSE {"accepted": false, "why": ')
               and ln.endswith('"parts": 3}') for ln in out), out
    assert rc == 0 and out[-1] == "lean: 3/3 accepted, 1/1 forgeries refused"


def _run_of_logs(d: Path, names: list[str]) -> None:
    """``rec-run.json`` (`challenger.RunRecord`'s form) opening the sessions the logs of ``names`` name: staged at the first
    one's rows and shared set, each its VERDICT's part and coin key, closed. A composition reads it where the challenger's
    record of the run is required; one that requires none never reads it."""
    vs = [json.loads((d / f"{n}.log").read_text().splitlines()[0][8:]) for n in names]
    events = [{"event": "stage", "rec_rows_sha512": vs[0]["rec_rows_sha512"], "rec_shared": vs[0].get("rec_shared")},
              *({"event": "open", "session": n, "rec_part": v["rec_part"], "coin_key": v.get("coin_key")}
                for n, v in zip(names, vs)), {"event": "close"}]
    (d / "rec-run.json").write_text(json.dumps({"events": events, "kind": "verity/rec-run/v1"}))


def test_a_forgery_the_composition_accepts_fails(tmp_path, capsys):
    """A forgery whose every part Lean accepts composes, and the step fails on it."""
    binary, h = _composer(tmp_path, 2)
    for j, p in enumerate(("L0", "alg-p0")):
        _vpart(tmp_path, p, j, h)
        _vpart(tmp_path, f"forged-inner-challenge-{p}", j, h)
    # the record opens the forgery's sessions, so only the forgery's composition can be one run
    _run_of_logs(tmp_path, ["forged-inner-challenge-L0", "forged-inner-challenge-alg-p0"])
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
