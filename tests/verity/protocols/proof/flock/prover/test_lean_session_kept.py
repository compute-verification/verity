"""The ZK server's kept session loop (the Lean verifier's `Session.kept`, `session --kept`): one loop
for every statement of a window, which reads the window's partition and program once and takes each member's statement
on its standard input (`member LEN`, then the member's flags as a JSON array). What it must keep of a loop of one
statement (`session`) is what an accepted session means: each session is of one member's statement from its start to its end, the one
current when it started, whatever `member` arrives while it is open; its `Hello` is the one a loop of that member's own
sends (`test_lean_hello.py`'s recorded `Hello` at the loop's nonce), and its plan is `session --schedule`'s on the
member's flags. A session while it has no member is refused without reaching the auditor, and a member's flags never
name the window's files or the loop's own flags. Between two members it holds only the window as read: the next member
starts on no frame, level 0 or setup of the one before, and each member's sessions take their own coins after their own
statement leaves."""
from __future__ import annotations

import json
import re
import socket
import subprocess

import pytest
from test_lean_hello import BIN, NONCE, _args, _record, built  # noqa: F401
from test_lean_session import (COMMIT, REGISTER, STOP, Auditor, _answer, _bytes, _name, _statement_m, _streams, _zeros, call,
                               coins, commit, hello, level0, open_, plain, reason, zk)  # noqa: F401


class Kept:
    """A running kept loop (`session --kept`) on the window's flags `window`."""

    def __init__(self, window: list[str] | None = None):
        self.p = subprocess.Popen([str(BIN), "session", "--kept", *(window or [])], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        line = self.p.stdout.readline().decode()
        assert line.startswith("LISTENING "), (line, self.p.stderr.read().decode() if self.p.poll() is not None else "")
        host, port = line.split()[1].split(":")
        self.addr = (host, int(port))
        line = self.p.stdout.readline().decode()
        assert line.startswith("FRAMES "), line
        host, port = line.split()[1].split(":")
        self.frames = (host, int(port))

    def send(self, verb: str, body: bytes) -> None:
        self.p.stdin.write(f"{verb} {len(body)}\n".encode() + body)
        self.p.stdin.flush()

    def line(self) -> str:
        return self.p.stdout.readline().decode().rstrip("\n")

    def member(self, args: list[str]) -> str:
        self.send("member", json.dumps(args).encode())
        return self.line()

    def schedule(self, sample: str) -> list[str]:
        self.send("schedule", sample.encode())
        first = self.line()
        return [first, *(self.line() for _ in range(int(first.split()[1]) if first.startswith("ok ") else 0))]

    def state(self) -> dict:
        """What the loop holds (`keptState`)."""
        self.send("state", b"")
        line = self.line()
        assert line.startswith("state "), line
        return json.loads(line.removeprefix("state "))

    def said(self, prefix: str | tuple[str, ...]) -> str:
        """The loop's next line on its standard error that starts with `prefix` (or one of them)."""
        while not (line := self.p.stderr.readline().decode()).startswith(prefix):
            assert line, "the loop's standard error closed"
        return line.rstrip("\n")

    def connect(self) -> socket.socket:
        return socket.create_connection(self.addr)

    def close(self):
        self.p.stdin.close()
        assert self.p.wait() == 0


@pytest.fixture
def junk(tmp_path):
    (tmp_path / "c").write_bytes(b"x")
    (tmp_path / "p").write_bytes(b"x")
    return ["--circuit", str(tmp_path / "c"), "--public", str(tmp_path / "p")]


@pytest.mark.parametrize("flags", [["--circuit", "c"], ["--verifier", "127.0.0.1:9"], ["--schedule"]])
def test_its_start_takes_the_windows_flags_alone(flags):
    p = subprocess.run([str(BIN), "session", "--kept", *flags], input=b"", capture_output=True, check=False)
    assert p.returncode == 2 and p.stderr.decode().startswith("--kept: the window's flags alone"), p.stderr


def test_a_session_before_the_first_member_is_refused():
    k = Kept()
    s = k.connect()
    assert reason(call(s, REGISTER)) == STOP + "a session while the kept loop has no member"
    s.close()
    k.close()


@pytest.mark.parametrize("extra, why", [
    (["--partition", "f"], "a member's flags name --partition, which is the kept loop's"),
    (["--program", "f"], "a member's flags name --program, which is the kept loop's"),
    (["--salt-key", "f"], "a member's flags name --salt-key, which is the kept loop's"),
])
def test_a_member_never_names_the_windows_files(junk, extra, why):
    k = Kept()
    assert k.member([*junk, "--verifier", "127.0.0.1:9", *extra]) == f"refused {why}"
    k.close()


def test_a_member_names_its_auditor(junk):
    k = Kept()
    assert k.member(junk) == "refused a member's flags name no --verifier HOST:PORT"
    k.send("member", b"{}")
    assert k.line() == "refused a member's flags that are not a JSON array"
    k.close()


def test_a_request_it_cannot_read_ends_the_loop():
    k = Kept()
    k.p.stdin.write(b"no such request\n")
    k.p.stdin.flush()
    assert k.line() == "err a request `no such request`"
    assert k.p.wait() == 1


def test_a_members_session_is_set_up_on_its_files(junk):
    """A member of files no statement sets up from: `ok`, and its session's `Hello` is refused at the setup, as a loop of
    its own refuses it, without reaching the auditor."""
    auditor = Auditor(plain)
    k = Kept()
    assert k.member([*junk, "--verifier", auditor.addr]) == "ok"
    s = k.connect()
    assert call(s, REGISTER) == plain(REGISTER)
    assert reason(call(s, hello(""))).startswith(STOP + "the Lean Hello: setup: ")
    s.close()
    k.close()
    assert [[f[0] for f in c] for c in auditor.got] == [[10]]


def _sent_hello(auditor: Auditor, i: int) -> str:
    """The `Hello` that left session `i` for the auditor."""
    sent = auditor.got[i][1]
    assert sent[:1] == b"\x01", sent[:40]
    return sent[5:].decode()


@pytest.mark.slow
def test_each_session_is_of_the_member_current_at_its_start(zk):
    """Two members, the fixture's statement over its 16-unit and its 32-unit population, through one kept loop. A session
    of the first sends the first's recorded `Hello` at the loop's nonce. A second member sent while a session is open is
    answered only once that session ends, and that session's `Hello` is still the first member's; the session after it
    sends the second's. At each member, the plan is `session --schedule`'s on the member's flags."""
    first, n16, null = _record(zk, "selftest/honest")
    second, n32, _ = _record(zk, "m26")
    answers = {10: plain(REGISTER), 1: _answer(), 3: b"\x01", 7: coins([])}
    auditor = Auditor(lambda f: answers[f[0]])
    members = [_args(zk, "pub-16.bin"), _args(zk, "pub-32.bin")]
    fresh = [subprocess.run([str(BIN), "session", "--schedule", *args], input=null.encode(), capture_output=True,
                            check=True).stdout.decode().splitlines() for args in members]
    k = Kept()
    assert k.member([*members[0], "--verifier", auditor.addr]) == "ok"
    assert k.schedule(null) == fresh[0]
    s = k.connect()
    call(s, REGISTER)
    call(s, hello(""))
    s.close()
    s = k.connect()
    call(s, REGISTER)
    k.send("member", json.dumps([*members[1], "--verifier", auditor.addr]).encode())
    call(s, hello(""))
    s.close()
    assert k.line() == "ok"
    s = k.connect()
    call(s, REGISTER)
    call(s, hello(""))
    s.close()
    assert k.schedule(null) == fresh[1]
    k.close()
    for i, (want, nonce) in enumerate([(first, n16), (first, n16), (second, n32)]):
        got = _sent_hello(auditor, i)
        assert got == want.replace(nonce, NONCE.search(got)[1]), i



EMPTY = {"frames": {"sessions": 0, "reps": 0, "salts": 0, "files": 0, "inner": 0},
         "level0": {"slots": 0, "sessions": 0, "tables": 0}}


def _flags(args: list[str]) -> str:
    return json.dumps(args, separators=(",", ":"))


def _live(k: Kept, args: list[str]) -> tuple[socket.socket, bytes]:
    """A session of the member at `args`, past its level 0 and its `Commit`, still open, and what its `Commit` carried
    past its Σ. A session before it, refused at its first `Commit`, names the statement's table."""
    first = _streams(args)[0]
    s = k.connect()
    call(s, REGISTER)
    call(s, hello(""))
    assert call(s, open_(first)) == b"\x01"
    [table] = re.search(r"tables are \[(.*)\]$", reason(call(s, COMMIT)))[1].split(", ")
    s.close()
    s = k.connect()
    call(s, REGISTER)
    name = _name(call(s, hello("")))
    assert call(s, open_(first)) == b"\x01"
    m = _statement_m(k, name)
    [(_, _, r)] = level0(k, [(m, _zeros(m))], name)
    live = k.state()
    assert live["member"]["setups"] == 1 and live["level0"]["slots"] == 1 and live["open"] >= 1, live
    assert live["frames"]["sessions"] >= 1 and live["level0"]["sessions"] >= 1, live
    call(s, commit([(table, r)]))
    return s, commit([(table, r)])[5:]


@pytest.mark.slow
def test_between_two_members_it_holds_only_the_window_and_each_takes_its_own_coins(zk):
    """Two members, the fixture's statement over its 16-unit and its 32-unit population, each with a session that has its
    level 0 sampled and its `Commit` sent; the second member is asked for while the first's session is still open, so the
    change waits for it. Its end, with its level 0 freed, is on standard error before the change, and the change finds
    nothing left to empty. When the second member is answered `ok` the loop holds the window as read and the second
    member's flags, and nothing else: no setup at a sample, no frame, salt, proof file or inner proof, no level 0, no
    session's name, nothing open. While a session was live the same report held its level 0 and its session, so the
    report sees them. Each member's sessions reach the auditor on connections of their own: the member's recorded
    `Hello`, the statement, first, and only then its `Open` and its `Commit`, the session's first coin request, carrying
    that `Hello`'s Σ. A loop whose count of open sessions drops before a session's entries are freed fails here whenever
    its change of member overtakes the freeing, which such a loop allows and this test cannot force. No other test here
    samples a level 0 under a kept loop, so none would see one outlive its member."""
    records = [_record(zk, "selftest/honest"), _record(zk, "m26")]
    roots = iter(range(1, 256))
    answers = {10: plain(REGISTER), 3: b"\x01", 7: coins([])}
    auditor = Auditor(lambda f: _answer(root=next(roots)) if f[0] == 1 else answers[f[0]])
    members = [_args(zk, "pub-16.bin"), _args(zk, "pub-32.bin")]
    flags = [[*args, "--verifier", auditor.addr] for args in members]
    k = Kept()
    assert k.member(flags[0]) == "ok"
    assert k.said("kept: member 0 ok, ").endswith(f", 0 entries left emptied: {_flags(flags[0])}")
    assert k.state() == {"window": {"partition": False, "program": 0}, "member": {"flags": flags[0], "setups": 0},
                         "open": 0, **EMPTY}
    s, left0 = _live(k, members[0])
    k.send("member", json.dumps(flags[1]).encode())
    s.close()
    assert k.line() == "ok"
    assert k.said(("level 0: session ", "kept: member ")).startswith("level 0: session 1 ended; ")
    assert k.said("kept: member 1 ok, ").endswith(f", 0 entries left emptied: {_flags(flags[1])}")
    assert k.state() == {"window": {"partition": False, "program": 0}, "member": {"flags": flags[1], "setups": 0},
                         "open": 0, **EMPTY}
    s, left1 = _live(k, members[1])
    s.close()
    k.close()
    assert [[f[0] for f in c] for c in auditor.got] == [[10, 1, 3], [10, 1, 3, 7]] * 2
    for i, ((want, nonce, _), left) in enumerate(zip(records, [left0, left1])):
        for c in auditor.got[2 * i:2 * i + 2]:
            got = c[1][5:].decode()
            assert got == want.replace(nonce, NONCE.search(got)[1]), i
        _, sent, _, commit_left = auditor.got[2 * i + 1]
        sigma = bytes.fromhex(json.loads(sent[5:])["link"]["sigma"])
        assert commit_left == b"\x07" + _bytes(sigma) + left, i
