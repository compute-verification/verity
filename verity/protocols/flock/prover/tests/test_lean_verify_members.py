"""`verify-members` (`Flock.VerifyMembers`, `verifyMember`): the Lean verifier's `verify` for each member of a run in one
process, the run's registered program read once and kept. What it holds, each on real sessions:

- a member's output is `verify`'s for that member alone, byte for byte, with its exit status on the `MEMBER` line after
  it, whether the member is accepted, refused, names a program that is other bytes than the kept one (the same
  descriptor, and another), cannot be read or is no argument list;
- nothing but the kept program's digest and readings reaches a later member: a member with another program is verified
  against its own, never the kept one, and a session's coin key is fresh for the next member as it is for the next
  `verify`, while inside one member it is used once;
- a member that stops, on its usage, an exception or a line that is no argument list, stops alone, and the members after
  it get their verdicts;
- each member is answered before the next line is read: the tests write a line only once the one before it is answered;
- a member's lines on standard error are there when its `MEMBER` line arrives, and are `verify`'s for it alone but for
  the milliseconds they report;
- the same holds in the form a caller that keeps its files by content hands: `--archive` and a key for every file, each
  member with an archive of its own;
- the process opens no socket from its start to its end, so one that is started before a session is opened and kept
  while later sessions run reaches no other host: the order of the challenger's release log is not its to change.

`Flock/VerifyMembers.lean`'s `#guard`s hold the line formats; these hold the command to `verify` on sessions. Each verify
sets up a table (m = 22 for the plain session, m = 25 for the `--zk` one), so the module is slow. No cheaper test reaches
it: the equation `verifyMember_eq` is about the two actions, and only a run shows the process prints what `verify` prints.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest
# the verifier's package and executable, and the plain fixture: ``stage/`` (circuit.txt, pub-32.bin, program.json,
# partition.json) and ``session/``, a plain session on the keyed coin tree that `verify --coins tree` accepts with the
# stage's program
from test_lean_plain_tree import BIN, FIXTURE as PLAIN, LEAN

VERIFIER = Path(__file__).resolve().parents[2] / "agreement"
NOT_A_LIST = "a member's line is a JSON array of strings"

#: A command prefix: the command after it runs under the kernel's system-call filter (seccomp), which kills the process, and
#: any it starts, at its first `socket` call (SIGSYS). The filter loads the call's number, kills on `socket` and allows every
#: other call. Exit status 97 where the kernel installs no filter.
NO_SOCKET = (sys.executable, "-c", """
import ctypes, os, platform, sys
nr = {"x86_64": 41, "aarch64": 198}.get(platform.machine())
class Filter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint)]
class Program(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(Filter))]
steps = (Filter * 4)(Filter(0x20, 0, 0, 0), Filter(0x15, 0, 1, nr or 0), Filter(0x06, 0, 0, 0x80000000),
                     Filter(0x06, 0, 0, 0x7FFF0000))
libc = ctypes.CDLL(None, use_errno=True)
PR_SET_NO_NEW_PRIVS, PR_SET_SECCOMP, SECCOMP_MODE_FILTER = 38, 22, 2
if nr is None or libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) or libc.prctl(
        PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(Program(len(steps), steps)), 0, 0):
    sys.exit(97)
os.execv(sys.argv[1], sys.argv[1:])
""")

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


@pytest.fixture(scope="module")
def built():
    subprocess.run(["lake", "build", BIN.name], cwd=LEAN, check=True, capture_output=True)


def _alone(args: list[str]) -> tuple[int, bytes, bytes]:
    """`verify` on one member's arguments, a process of its own: its exit status and what it printed and wrote to
    standard error."""
    p = subprocess.run([str(BIN), "verify", *args], capture_output=True, check=False)
    return p.returncode, p.stdout, p.stderr


def _members(kept: list[str], lines: list[bytes], log: Path,
             under: tuple[str, ...] = ()) -> tuple[int, list[list[bytes]], bytes, list[bytes]]:
    """`verify-members` with the kept program's flags ``kept`` on ``lines``, each written once the line before it is
    answered, run under the command prefix ``under``: its exit status, each line's answer (what it printed up to that
    member's `MEMBER` line), what it printed after the last, and its standard error, the file ``log``, cut where each
    `MEMBER` line arrived: a piece a line, then what it wrote after the last."""
    with log.open("wb") as err, subprocess.Popen([*under, str(BIN), "verify-members", *kept], stdin=subprocess.PIPE,
                                                 stdout=subprocess.PIPE, stderr=err) as p:
        answers, marks = [], [0]
        for line in lines:
            p.stdin.write(line + b"\n")
            p.stdin.flush()
            answer = []
            for printed in p.stdout:
                answer.append(printed)
                if printed.startswith(b"MEMBER "):
                    break
            answers.append(answer)
            marks.append(log.stat().st_size)
        p.stdin.close()
        rest = p.stdout.read()
        rc = p.wait()
    written = log.read_bytes()
    return rc, answers, rest, [written[a:b] for a, b in zip(marks, [*marks[1:], len(written)])]


def _line(args: list[str]) -> bytes:
    return json.dumps(args).encode()


def _verdicts(printed: bytes) -> list[dict]:
    return [json.loads(x.split(b" ", 1)[1]) for x in printed.splitlines() if x.startswith(b"VERDICT ")]


def _stages(err: bytes) -> bytes:
    """Standard error but for the milliseconds its stage lines report."""
    return re.sub(rb"\d+ ms\b", b"N ms", err)


def _as_alone(index: int, args: list[str], answer: list[bytes], err: bytes,
              alone: dict[tuple[str, ...], tuple[int, bytes, bytes]]) -> tuple[int, bytes]:
    """Line ``index``'s answer is what `verify` prints for ``args`` alone, then its exit status on the `MEMBER` line, and
    ``err``, what standard error gained by the time that line arrived, is what `verify` writes there for ``args`` alone
    but for its milliseconds: `verify`'s (status, output), run once for each distinct argument list (``alone``)."""
    if tuple(args) not in alone:
        alone[tuple(args)] = _alone(args)
    rc, printed, wrote = alone[tuple(args)]
    assert b"".join(answer[:-1]) == printed, (index, answer, printed)
    assert answer[-1:] == [b'MEMBER {"exit":%d,"index":%d}\n' % (rc, index)], (index, answer, rc)
    assert _stages(err) == _stages(wrote), (index, err, wrote)
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
    rc, answers, rest, errs = _members(["--program", str(program)],
                                       [x if isinstance(x, bytes) else _line(x) for x in members.values()], tmp_path / "members.log")
    assert (rc, rest, len(answers), errs[-1]) == (1, b"", len(members), b""), (rc, rest, answers, errs)
    runs: dict[tuple[str, ...], tuple[int, bytes, bytes]] = {}
    alone = {}
    for i, ((name, x), answer, err) in enumerate(zip(members.items(), answers, errs)):
        if isinstance(x, bytes):
            # a line that is no argument list: no verdict and its own status, and the members after it are answered
            assert len(answer) == 1 and answer[0].startswith(b"MEMBER "), answer
            assert json.loads(answer[0].split(b" ", 1)[1]) == {"error": NOT_A_LIST, "exit": 2, "index": i}, answer
            assert err == f"verify-members: member {i}: {NOT_A_LIST}\n".encode(), err
        else:
            alone[name] = _as_alone(i, x, answer, err, runs)

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
    # an accepted member's stage lines, its session's last, were there when its `MEMBER` line arrived, and an exception's
    # line is in its own member's piece
    assert re.search(rb"^inputs: \d+ ms$", errs[0], re.M), errs[0]
    assert re.search(rb"^%s: \d+ ms\n\Z" % re.escape(str(PLAIN / "session").encode()), errs[0], re.M), errs[0]
    assert errs[list(members).index("no circuit")].startswith(b"uncaught exception: "), errs

    # every member accepted: exit 0
    rc, [answer], rest, _ = _members(["--program", str(program)], [_line(members["accepted"])], tmp_path / "one.log")
    assert (rc, rest, b"".join(answer[:-1]), answer[-1]) == (0, b"", alone["accepted"][1], b'MEMBER {"exit":0,"index":0}\n')


def test_members_whose_files_are_held_by_content_each_in_an_archive_of_its_own(built, tmp_path):
    """The form a caller that keeps its files by content hands: `--archive DIR` and the kept program's key on the command
    line, and in every line the member's own `--archive` and a SHA-512 key for each of its files. Five lines: an accepted
    member; a member with an archive of its own holding the same objects; one whose archive holds other bytes under the
    circuit's key (stopped before any session, as `verify` stops it); one whose archive holds no program; the first
    again. Each is answered as `verify` answers it alone, on both streams."""
    assert (PLAIN / "live.json").is_file(), (
        f"{PLAIN} is not fetched: `uv run research data fetch-fixtures verity/protocols/flock/prover/tests/fixtures/plain-tree`")
    st = PLAIN / "stage"
    files = {name: (st / file).read_bytes() for name, file in (("circuit", "circuit.txt"), ("public", "pub-32.bin"),
                                                               ("partition", "partition.json"), ("program", "program.json"))}
    keys = {name: hashlib.sha512(b).hexdigest() for name, b in files.items()}

    def archive(name: str, **other: bytes | None) -> Path:
        """An archive of the stage's four files under their keys; ``other``: a file's object as other bytes, or not held."""
        d = tmp_path / name / "sha512"
        d.mkdir(parents=True)
        for file, b in {**files, **other}.items():
            if b is not None:
                (d / keys[file]).write_bytes(b)
        return d.parent

    def member(held: Path) -> list[str]:
        return ["--statement", "verity/flock-circuit", "--archive", str(held), "--coins", "tree", "--circuit", keys["circuit"],
                "--public", keys["public"], "--partition", keys["partition"], "--program", keys["program"], "--session",
                str(PLAIN / "session")]

    first = archive("first")
    members = {
        "accepted": member(first),
        "its own archive": member(archive("second")),
        "other bytes under the circuit's key": member(archive("third", circuit=files["circuit"] + b"\n")),
        "no program held": member(archive("fourth", program=None)),
        "accepted again": member(first),
    }
    rc, answers, rest, errs = _members(["--archive", str(first), "--program", keys["program"]],
                                       [_line(x) for x in members.values()], tmp_path / "members.log")
    assert (rc, rest, len(answers), errs[-1]) == (1, b"", len(members), b""), (rc, rest, answers, errs)
    runs: dict[tuple[str, ...], tuple[int, bytes, bytes]] = {}
    alone = {name: _as_alone(i, x, answer, err, runs)
             for i, ((name, x), answer, err) in enumerate(zip(members.items(), answers, errs))}
    for name in ("accepted", "its own archive"):
        [v] = _verdicts(alone[name][1])
        assert (alone[name][0], v["accepted"], v["why"]) == (0, True, ""), (name, alone[name])
    # an object that does not hash to its key, and one that is not held, stop the member before any session
    for name in ("other bytes under the circuit's key", "no program held"):
        assert alone[name][0] != 0 and not any(v["accepted"] for v in _verdicts(alone[name][1])), (name, alone[name])
    # the kept process stops before any member without its own program in its own archive
    p = subprocess.run([str(BIN), "verify-members", "--archive", str(tmp_path / "fourth"), "--program", keys["program"]],
                       input=_line(members["accepted"]) + b"\n", capture_output=True, check=False)
    assert (p.returncode, p.stdout) == (2, b"") and p.stderr.strip(), p


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
    rc, answers, rest, errs = _members(["--program", str(st / "program.json")], [_line(m) for m in members], tmp_path / "members.log")
    runs: dict[tuple[str, ...], tuple[int, bytes, bytes]] = {}
    (kept_rc, kept), (spaced_rc, anew) = (_as_alone(i, m, a, e, runs) for i, (m, a, e) in enumerate(zip(members, answers, errs)))
    assert (rc, rest, kept_rc, spaced_rc) == (0, b"", 0, 0)
    [v] = _verdicts(kept)
    assert (v["accepted"], v["why"]) == (True, "") and kept == anew, (kept, anew)


def test_the_kept_process_opens_no_socket_from_its_start_to_its_end(built, tmp_path):
    """The kept process under a system-call filter that kills it at its first `socket` call: it starts, reads its
    program, answers an accepted member, a refused one and the accepted one again, each as `verify` answers it alone,
    and ends with its own exit status. So before, during and between its members it opens no connection: it reads the
    files its lines name and writes its two streams, and nothing it does reaches another host. That is what lets it be
    started before a session is opened and kept while later sessions run: what the challenger's release log holds, and
    in what order, is decided by who talks to the challenger, and this process does not. The control: under the same
    filter a process that opens a socket is killed. Skipped where the kernel installs no filter."""
    opens = subprocess.run([*NO_SOCKET, sys.executable, "-c", "import socket; socket.socket()"], check=False)
    if opens.returncode == 97:
        pytest.skip("this kernel installs no system-call filter for the process")
    assert opens.returncode == -signal.SIGSYS, opens.returncode
    st = PLAIN / "stage"
    flipped = shutil.copytree(PLAIN / "session", tmp_path / "flipped")
    rep = sorted(flipped.glob("*.bin"))[0]
    b = bytearray(rep.read_bytes())
    b[len(b) // 2] ^= 1
    rep.write_bytes(bytes(b))
    members = [["--statement", "verity/flock-circuit", "--coins", "tree", "--circuit", str(st / "circuit.txt"), "--public",
                str(st / "pub-32.bin"), "--partition", str(st / "partition.json"), "--program", str(st / "program.json"),
                "--session", str(session)] for session in (PLAIN / "session", flipped, PLAIN / "session")]
    rc, answers, rest, errs = _members(["--program", str(st / "program.json")], [_line(m) for m in members],
                                       tmp_path / "members.log", under=NO_SOCKET)
    assert (rc, rest, len(answers), errs[-1]) == (1, b"", len(members), b""), (rc, rest, answers, errs)
    runs: dict[tuple[str, ...], tuple[int, bytes, bytes]] = {}
    statuses = [_as_alone(i, m, answer, err, runs)[0] for i, (m, answer, err) in enumerate(zip(members, answers, errs))]
    assert statuses == [0, 1, 0], statuses
