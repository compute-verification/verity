"""`verify-members` (`Flock.VerifyMembers`, `verifyMember`): the Lean verifier's `verify` for each member of a run in one
process, the run's registered program read once and kept, and each member's circuit file kept for the member after it.
What it holds, each on real sessions:

- a member's output is `verify`'s for that member alone, byte for byte, with its exit status on the `MEMBER` line after
  it, whether the member is accepted, refused, names a program that is other bytes than the kept one (the same
  descriptor, and another), cannot be read or is no argument list;
- nothing but the kept program's digest and readings, and the SHA-512 and the parse of the circuit file of the member
  before, reaches a later member: a member with another program is verified against its own, never the kept one; a
  member whose circuit file is other bytes, or the same bytes under another statement's name, is verified against its
  own file under its own name; and a session's coin key is fresh for the next member as it is for the next `verify`,
  while inside one member it is used once;
- a member that stops, on its usage, an exception or a line that is no argument list, stops alone, and the members after
  it get their verdicts;
- each member is answered before the next line is read: the tests write a line only once the one before it is answered.

`Flock/VerifyMembers.lean`'s `#guard`s hold the line formats; these hold the command to `verify` on sessions. Each verify
sets up a table (m = 22 for the plain session, m = 25 for the `--zk` one), so the module is slow. No cheaper test reaches
it: the equation `verifyMember_eq` is about the two actions, and only a run shows the process prints what `verify` prints.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
# the verifier's package and executable, and the plain fixture: ``stage/`` (circuit.txt, pub-32.bin, program.json,
# partition.json) and ``session/``, a plain session on the keyed coin tree that `verify --coins tree` accepts with the
# stage's program
from test_lean_plain_tree import BIN, FIXTURE as PLAIN, LEAN

VERIFIER = Path(__file__).resolve().parents[2] / "agreement"
NOT_A_LIST = "a member's line is a JSON array of strings"

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


@pytest.fixture(scope="module")
def built():
    subprocess.run(["lake", "build", BIN.name], cwd=LEAN, check=True, capture_output=True)


def _alone(args: list[str]) -> tuple[int, bytes]:
    """`verify` on one member's arguments, a process of its own: its exit status and what it printed."""
    p = subprocess.run([str(BIN), "verify", *args], capture_output=True, check=False)
    return p.returncode, p.stdout


def _members(kept: list[str], lines: list[bytes], log: Path) -> tuple[int, list[list[bytes]], bytes]:
    """`verify-members` with the kept program's flags ``kept`` on ``lines``, each written once the line before it is
    answered: its exit status, each line's answer (what it printed up to that member's `MEMBER` line) and what it printed
    after the last. Its standard error goes to ``log``."""
    with log.open("wb") as err, subprocess.Popen([str(BIN), "verify-members", *kept], stdin=subprocess.PIPE,
                                                 stdout=subprocess.PIPE, stderr=err) as p:
        answers = []
        for line in lines:
            p.stdin.write(line + b"\n")
            p.stdin.flush()
            answer = []
            for printed in p.stdout:
                answer.append(printed)
                if printed.startswith(b"MEMBER "):
                    break
            answers.append(answer)
        p.stdin.close()
        rest = p.stdout.read()
        return p.wait(), answers, rest


def _line(args: list[str]) -> bytes:
    return json.dumps(args).encode()


def _verdicts(printed: bytes) -> list[dict]:
    return [json.loads(x.split(b" ", 1)[1]) for x in printed.splitlines() if x.startswith(b"VERDICT ")]


def _as_alone(index: int, args: list[str], answer: list[bytes], alone: dict[tuple[str, ...], tuple[int, bytes]]) -> tuple[int, bytes]:
    """Line ``index``'s answer is what `verify` prints for ``args`` alone, then its exit status on the `MEMBER` line:
    `verify`'s (status, output), run once for each distinct argument list (``alone``)."""
    if tuple(args) not in alone:
        alone[tuple(args)] = _alone(args)
    rc, printed = alone[tuple(args)]
    assert b"".join(answer[:-1]) == printed, (index, answer, printed)
    assert answer[-1:] == [b'MEMBER {"exit":%d,"index":%d}\n' % (rc, index)], (index, answer, rc)
    return rc, printed


def test_each_member_gets_the_verdict_verify_gives_it_alone_and_nothing_of_another_member(built, tmp_path):
    """Nine lines in one process, the stage's program kept: an accepted member, a refused one, two whose program is other
    bytes (the same descriptor, and another), a line that is no argument list, a member whose circuit file is missing
    (an exception), one stopped at its usage, the accepted one again and one of two sessions with one coin key. Each
    is answered as `verify` answers it alone, and the kept program changes no verdict."""
    assert (PLAIN / "live.json").is_file(), (
        f"{PLAIN} is not fetched: `uv run research data fetch-fixtures verity/protocols/flock/prover/tests/fixtures/plain-tree`")
    st = PLAIN / "stage"
    program = st / "program.json"

    def member(*sessions: Path, program: Path = program, circuit: Path = st / "circuit.txt", coins: str = "tree") -> list[str]:
        return ["--statement", "verity/flock-circuit", "--coins", coins, "--circuit", str(circuit), "--public",
                str(st / "pub-32.bin"), "--partition", str(st / "partition.json"), "--program", str(program), "--session",
                *map(str, sessions)]

    flipped = shutil.copytree(PLAIN / "session", tmp_path / "flipped")
    rep = sorted(flipped.glob("*.bin"))[0]
    b = bytearray(rep.read_bytes())
    b[len(b) // 2] ^= 1
    rep.write_bytes(bytes(b))
    again = shutil.copytree(PLAIN / "session", tmp_path / "again")
    # the kept program's descriptor in other bytes: JSON text may end in whitespace, so its digest and readings are the same
    spaced = tmp_path / "spaced.json"
    spaced.write_bytes(program.read_bytes() + b"\n")
    # another descriptor: one more key, so another digest, which the stage's partition does not name
    other = tmp_path / "other.json"
    other.write_text(json.dumps({**json.loads(program.read_text()), "not-the-registered-program": 1}))
    members: dict[str, list[str] | bytes] = {
        "accepted": member(PLAIN / "session"),
        "refused": member(flipped),
        "spaced": member(PLAIN / "session", program=spaced),
        "other": member(PLAIN / "session", program=other),
        "not a list": b'{"args": []}',
        "no circuit": member(PLAIN / "session", circuit=tmp_path / "no-circuit.txt"),
        "usage": member(PLAIN / "session", coins="os"),
        "accepted again": member(PLAIN / "session"),
        "key twice": member(PLAIN / "session", again),
    }
    rc, answers, rest = _members(["--program", str(program)], [x if isinstance(x, bytes) else _line(x) for x in members.values()],
                                 tmp_path / "members.log")
    assert (rc, rest, len(answers)) == (1, b"", len(members)), (rc, rest, answers)
    runs: dict[tuple[str, ...], tuple[int, bytes]] = {}
    alone = {}
    for i, ((name, x), answer) in enumerate(zip(members.items(), answers)):
        if isinstance(x, bytes):
            # a line that is no argument list: no verdict and its own status, and the members after it are answered
            assert len(answer) == 1 and answer[0].startswith(b"MEMBER "), answer
            assert json.loads(answer[0].split(b" ", 1)[1]) == {"error": NOT_A_LIST, "exit": 2, "index": i}, answer
        else:
            alone[name] = _as_alone(i, x, answer, runs)

    # and `verify` alone says of each member what it was made to show
    def only(name: str) -> dict:
        [v] = _verdicts(alone[name][1])
        return v

    assert (alone["accepted"][0], only("accepted")["accepted"], only("accepted")["why"]) == (0, True, "")
    assert (alone["refused"][0], only("refused")["accepted"]) == (1, False)
    # other bytes of the same descriptor are read anew and give the kept program's digest
    assert alone["spaced"][0] == 0 and only("spaced") == only("accepted")
    # another program is the member's own, never the kept one: refused, and the digest named is not the kept program's
    assert (alone["other"][0], only("other")["accepted"]) == (1, False), only("other")
    assert "the verifier's program is not the one the partition object names" in only("other")["why"], only("other")
    assert only("other").get("program_sha512") != only("accepted")["program_sha512"]
    # an exception and a usage error print no verdict; their statuses are `verify`'s
    assert alone["no circuit"] == (1, b"") and alone["usage"] == (2, b"")
    # a session's coin key is used once inside a member and is fresh for the next member
    assert alone["accepted again"] == alone["accepted"]
    first, second = _verdicts(alone["key twice"][1])
    assert (alone["key twice"][0], first["accepted"], second["accepted"]) == (1, True, False)
    assert second["why"] == "S2: coin_tree.key is an earlier session's; a session key is never reused", second
    log = (tmp_path / "members.log").read_text()
    assert "uncaught exception: " in log and f"verify-members: member {list(members).index('not a list')}: " in log, log

    # every member accepted: exit 0
    rc, [answer], rest = _members(["--program", str(program)], [_line(members["accepted"])], tmp_path / "one.log")
    assert (rc, rest, b"".join(answer[:-1]), answer[-1]) == (0, b"", alone["accepted"][1], b'MEMBER {"exit":0,"index":0}\n')


def test_a_member_reads_the_kept_circuit_file_only_where_its_own_is_the_same(built, tmp_path):
    """Seven lines in one process, each after a member that kept a circuit file: the accepted member; its circuit file
    at another path (the same bytes, so the kept parse); the file with one more byte (other bytes: its own file, not
    accepted); the accepted member again (its own file again, the kept one being the longer); the same file under
    another statement's name (the kept bytes under other tags: its own parse, not accepted); a file that is no circuit,
    and that file again (a kept refusal). Each is answered as `verify` answers it alone. Were a member answered from
    the kept file where its own differs, the longer file's member or the other statement's would be accepted here. Each
    line is a verify of its own, so the test is slow, and no cheaper one reaches it: which file a member is answered
    from shows only in a run."""
    assert (PLAIN / "live.json").is_file(), (
        f"{PLAIN} is not fetched: `uv run research data fetch-fixtures verity/protocols/flock/prover/tests/fixtures/plain-tree`")
    st = PLAIN / "stage"

    def member(circuit: Path, statement: str = "verity/flock-circuit") -> list[str]:
        return ["--statement", statement, "--coins", "tree", "--circuit", str(circuit), "--public", str(st / "pub-32.bin"),
                "--partition", str(st / "partition.json"), "--program", str(st / "program.json"), "--session",
                str(PLAIN / "session")]

    same = tmp_path / "same.txt"
    same.write_bytes((st / "circuit.txt").read_bytes())
    longer = tmp_path / "longer.txt"
    longer.write_bytes((st / "circuit.txt").read_bytes() + b"\n")
    nothing = tmp_path / "not-a-circuit.txt"
    nothing.write_text("not a circuit\n")
    members = {
        "accepted": member(st / "circuit.txt"),
        "same bytes": member(same),
        "longer": member(longer),
        "accepted again": member(st / "circuit.txt"),
        "other statement": member(st / "circuit.txt", "verity/flock-circuit+plain-leaves"),
        "not a circuit": member(nothing),
        "not a circuit again": member(nothing),
    }
    rc, answers, rest = _members(["--program", str(st / "program.json")], [_line(x) for x in members.values()],
                                 tmp_path / "members.log")
    assert (rc, rest, len(answers)) == (1, b"", len(members)), (rc, rest, answers)
    runs: dict[tuple[str, ...], tuple[int, bytes]] = {}
    alone = {name: _as_alone(i, x, answer, runs) for i, ((name, x), answer) in enumerate(zip(members.items(), answers))}
    [accepted] = _verdicts(alone["accepted"][1])
    assert (alone["accepted"][0], accepted["accepted"], accepted["why"]) == (0, True, "")
    assert alone["same bytes"][0] == 0, alone["same bytes"]
    for name in ("longer", "other statement", "not a circuit"):
        assert alone[name][0] != 0 and not any(v["accepted"] for v in _verdicts(alone[name][1])), (name, alone[name])


def test_it_stops_before_any_member_without_a_program_it_can_read(built, tmp_path):
    """Its own `--program` missing or unreadable: exit 2, nothing printed, whatever members it is handed. With one and no
    member, exit 0 and nothing printed."""
    line = _line(["--circuit", "c", "--public", "p", "--session", "s"]) + b"\n"
    for kept in ([], ["--program", str(tmp_path / "none.json")]):
        p = subprocess.run([str(BIN), "verify-members", *kept], input=line, capture_output=True, check=False)
        assert (p.returncode, p.stdout) == (2, b"") and p.stderr.strip(), (kept, p)
    (tmp_path / "program.json").write_text("{}")
    p = subprocess.run([str(BIN), "verify-members", "--program", str(tmp_path / "program.json")], input=b"", capture_output=True,
                       check=False)
    assert (p.returncode, p.stdout) == (0, b""), p


def test_a_zk_member_on_registered_reads_gets_verifys_verdict_whichever_bytes_name_its_program(built, tmp_path, monkeypatch):
    """The served form: a `--zk` session of one table on registered reads (`test_lean_served_registered.py`'s `row/v1`
    fixture), accepted. Its setup reads the program through all three kept readings (the registered-input check, the
    template's units and the cover of the registered reads), which the plain fixture's, with no registered read, does not.
    As a member whose program is the kept bytes, and as one whose program is the same descriptor in other bytes, it gets
    the line `verify` prints for it alone."""
    monkeypatch.syspath_prepend(str(VERIFIER))
    import ci
    import served_registered as SR
    from test_lean_served_registered import FIXTURES, STATEMENT

    from verity.protocols.spot.verifier.one_stage import registered as G

    (tmp_path / "store").mkdir()
    try:
        fx = ci.fetch_sessions(FIXTURES[G.SCHEMA], tmp_path / "store")
    except (subprocess.CalledProcessError, OSError) as e:
        pytest.skip(f"the evidence store is not reachable: {e}")
    st = fx / "stage"
    program, query = SR.program()
    reads = SR.write_registered(SR.registered_file(program, query, *SR.load(st)), tmp_path / "registered.json")
    spaced = tmp_path / "spaced.json"
    spaced.write_bytes((st / "program.json").read_bytes() + b"\n")
    members = [["--zk", "--statement", STATEMENT, "--circuit", str(st / "circuit.txt"), "--public", str(st / "pub-32.bin"),
                "--partition", str(st / "partition.json"), "--program", str(p), "--registered", str(reads), "--session",
                str(fx / "session")] for p in (st / "program.json", spaced)]
    rc, answers, rest = _members(["--program", str(st / "program.json")], [_line(m) for m in members], tmp_path / "members.log")
    runs: dict[tuple[str, ...], tuple[int, bytes]] = {}
    (kept_rc, kept), (spaced_rc, anew) = (_as_alone(i, m, a, runs) for i, (m, a) in enumerate(zip(members, answers, strict=True)))
    assert (rc, rest, kept_rc, spaced_rc) == (0, b"", 0, 0)
    [v] = _verdicts(kept)
    assert (v["accepted"], v["why"]) == (True, "") and kept == anew, (kept, anew)
