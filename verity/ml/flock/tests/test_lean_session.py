"""The firewall's session loop in Lean (`Flock/Firewall/Session.lean`, `flock-firewall session`), which owns the auditor's
connection: a prover connects to it as it would to the auditor, and each connection is one session it relays on its own
connection to the auditor. Here a scripted auditor stands on the far side. What crosses is held to the loop's rules:
`Register` and other frames relayed as they are, the `Hello` that leaves the Lean one at the auditor's draw (at the
fixture's statement, `test_lean_hello.py`'s recorded `Hello` at the loop's nonce) and never the prover's, `Hello`'s
answer and every coin checked, and every coin after a refusal refused. The loop serves the reps' frames on a port of its
own, where a frame takes only the coins a session's opening fed it. `flock-circuit`'s firewalled sessions through it
are the end-to-end check (live PROTOCOL.md §10.4)."""
from __future__ import annotations

import json
import re
import socket
import struct
import subprocess
import sys
import threading

import pytest
from test_firewall_frame import ALG, LAY, rep_rounds, wire
from test_lean_hello import (  # noqa: F401
    BIN,
    FIXTURE,
    NONCE,
    STATEMENT,
    VERIFIER,
    _args,
    _record,
    built,
)

STOP = "PROVER-STOPPED: "


def _bytes(b: bytes) -> bytes:
    """`u32le(len) ‖ b`: a field's bytes, and a frame on the wire (`write_frame`)."""
    return struct.pack("<I", len(b)) + b


def _recv(s: socket.socket, n: int) -> bytes | None:
    out = b""
    while len(out) < n:
        b = s.recv(n - len(out))
        if not b:
            return None
        out += b
    return out


def _read(s: socket.socket) -> bytes | None:
    h = _recv(s, 4)
    return None if h is None else _recv(s, struct.unpack("<I", h)[0])


REGISTER = b"\x0a"
COMMIT = b"\x07" + _bytes(b"") + struct.pack("<I", 0) + struct.pack("<I", 0)


def hello(text: str) -> bytes:
    return b"\x01" + _bytes(text.encode())


def ping(b: bytes) -> bytes:
    return b"\x09" + _bytes(b)


def round_(stream: str, content: bytes, n: int) -> bytes:
    return b"\x04" + _bytes(stream.encode()) + _bytes(content) + struct.pack("<I", n)


def draw(text: str) -> bytes:
    return b"\x06" + _bytes(text.encode())


def coins(words: list[bytes]) -> bytes:
    return b"\x02" + struct.pack("<I", len(words)) + b"".join(words)


def reason(resp: bytes) -> str:
    assert resp[:1] == b"\x03", resp[:40]
    return resp[5:].decode()


class Auditor:
    """A scripted auditor: each connection's frames, answered in turn by `answer(frame)`."""

    def __init__(self, answer):
        self.answer, self.got, self.lock = answer, [], threading.Lock()
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(8)
        self.addr = f"127.0.0.1:{self.srv.getsockname()[1]}"
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                c, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self._conn, args=(c,), daemon=True).start()

    def _conn(self, c):
        mine = []
        with self.lock:
            self.got.append(mine)
        while (f := _read(c)) is not None:
            mine.append(f)
            c.sendall(_bytes(self.answer(f)))
        c.close()


class Loop:
    """A running `flock-firewall session` against `auditor`, on the statement flags `args`."""

    def __init__(self, auditor: Auditor, args: list[str]):
        self.p = subprocess.Popen([str(BIN), "session", "--verifier", auditor.addr, *args], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        line = self.p.stdout.readline().decode()
        assert line.startswith("LISTENING "), (line, self.p.stderr.read().decode() if self.p.poll() is not None else "")
        host, port = line.split()[1].split(":")
        self.addr = (host, int(port))
        line = self.p.stdout.readline().decode()
        assert line.startswith("FRAMES "), line
        host, port = line.split()[1].split(":")
        self.frames = (host, int(port))

    def connect(self) -> socket.socket:
        return socket.create_connection(self.addr)

    def close(self):
        self.p.stdin.close()
        assert self.p.wait() == 0


def call(s: socket.socket, f: bytes) -> bytes | None:
    s.sendall(_bytes(f))
    return _read(s)


def plain(f: bytes) -> bytes:
    """The auditor's answers: Register a draw of none, Ping `Ok`, anything else 4 zero words."""
    return {10: draw("null"), 9: b"\x01"}.get(f[0], coins([b"\0" * 16] * 4))


@pytest.fixture
def junk(tmp_path):
    """Files no statement sets up from: the loop starts, and a session's Hello is refused at the setup."""
    (tmp_path / "c").write_bytes(b"x")
    (tmp_path / "p").write_bytes(b"x")
    return ["--circuit", str(tmp_path / "c"), "--public", str(tmp_path / "p")]


@pytest.fixture
def loop(junk):
    auditor = Auditor(plain)
    lp = Loop(auditor, junk)
    yield auditor, lp
    lp.close()


def test_usage(junk):
    p = subprocess.run([str(BIN), "session", *junk], input=b"", capture_output=True, check=False)
    assert p.returncode == 2 and p.stderr.decode().startswith("usage: flock-firewall session")


def test_register_and_other_frames_are_relayed(loop):
    auditor, lp = loop
    s = lp.connect()
    assert call(s, REGISTER) == draw("null")
    assert call(s, ping(b"abc")) == b"\x01"
    s.close()
    assert auditor.got[0] == [REGISTER, ping(b"abc")]


def test_the_provers_hello_never_leaves(loop):
    """A statement the setup refuses: the prover is refused at its `Hello`, which does not reach the auditor, and every coin
    after is refused for the same reason, without asking the auditor."""
    auditor, lp = loop
    s = lp.connect()
    call(s, REGISTER)
    why = reason(call(s, hello('{"the": "prover\'s"}')))
    assert why.startswith(STOP + "the Lean Hello: setup: "), why
    assert reason(call(s, round_("rep0", b"r", 2))) == why
    assert reason(call(s, COMMIT)) == why
    assert call(s, ping(b"p")) == b"\x01"
    s.close()
    assert auditor.got[0] == [REGISTER, ping(b"p")]


def test_coins_before_the_commitment(loop):
    _, lp = loop
    s = lp.connect()
    assert reason(call(s, round_("rep0", b"r", 2))) == STOP + "coins before the step-0 coin commitment"


def test_a_finish_before_the_commitment_never_leaves(loop):
    auditor, lp = loop
    s = lp.connect()
    call(s, REGISTER)
    assert reason(call(s, b"\x06")) == STOP + "a Finish before the step-0 coin commitment"
    s.close()
    assert auditor.got[0] == [REGISTER]


def test_a_draw_that_is_not_json(junk):
    lp = Loop(Auditor(lambda f: draw("{")), junk)
    s = lp.connect()
    assert reason(call(s, REGISTER)).startswith(STOP + "the draw is not JSON")
    lp.close()


def test_a_request_it_cannot_read_ends_the_session(loop):
    _, lp = loop
    s = lp.connect()
    assert reason(call(s, b"")) == STOP + "an empty request frame"
    assert _read(s) is None
    assert reason(call(lp.connect(), round_("rep0", b"", 0))) == STOP + "coins before the step-0 coin commitment"


def test_sessions_at_once(loop):
    """Each connection is a session of its own, on its own connection to the auditor, served at once."""
    auditor, lp = loop
    a, b = lp.connect(), lp.connect()
    assert call(b, REGISTER) == draw("null") and call(a, REGISTER) == draw("null")
    assert call(a, ping(b"a")) == b"\x01" and call(b, ping(b"b")) == b"\x01"
    a.close()
    b.close()
    assert sorted(auditor.got, key=len) == sorted([[REGISTER, ping(b"a")], [REGISTER, ping(b"b")]], key=len)


class TcpFrame:
    """A rep's frame on a connection to the loop's frames port, asked as `test_firewall_frame.Frame` asks its own."""

    def __init__(self, lp: Loop):
        self.s = socket.create_connection(lp.frames)
        self.r = self.s.makefile("rb")

    def ask(self, line: str, body: bytes = b""):
        self.s.sendall(line.encode() + b"\n" + body)
        r = self.r.readline().decode().rstrip("\n")
        return (r, self.r.read(int(r.split(" ")[1]))) if r.startswith("ok ") else r

    def close(self):
        self.r.close()
        self.s.close()


def test_a_frame_on_the_loops_port_draws_and_releases_but_takes_no_coins_the_loop_did_not_open(loop):
    """The loop serves the reps' frames itself: a frame on its frames port starts and releases as `flock-firewall frame`
    does, but its `coins` are the ones a session's opening fed it, and with no session's `Round` carrying its release
    there are none, so the coins the request brings are refused and end the rep."""
    _, lp = loop
    f = TcpFrame(lp)
    start = f.ask(json.dumps({"layout": LAY}))
    assert isinstance(start, tuple) and len(start[1]) == int(start[0].split()[1]), start
    r0, n = rep_rounds(ALG)[0]
    out = f.ask(f"round {n} {len(r0)} {len(r0)}", r0 + r0)
    assert isinstance(out, tuple) and out[0].endswith(" pass"), out
    assert f.ask(f"coins {n}", wire([2] * n)) == "err coins for a round whose coins the session loop has not opened"
    assert f.r.readline() == b""
    f.close()


def test_each_frame_draws_its_own(loop):
    _, lp = loop
    a, b = TcpFrame(lp), TcpFrame(lp)
    da, db = a.ask(json.dumps({"layout": LAY})), b.ask(json.dumps({"layout": LAY}))
    assert da[0] == db[0] and da[1] != db[1]
    a.close()
    b.close()


@pytest.fixture(scope="module")
def zk(tmp_path_factory):
    sys.path.insert(0, str(VERIFIER))
    import ci

    try:
        return ci.fetch(FIXTURE, tmp_path_factory.mktemp("store"))
    except (subprocess.CalledProcessError, OSError) as e:
        pytest.skip(f"the evidence store is not reachable: {e}")


def _answer(key_top: int = 0) -> bytes:
    """`Hello`'s answer: a root and a session key, 20 words, the key's last byte `key_top`."""
    return coins([b"\x11" * 16] * 4 + [b"\x22" * 16] * 15 + [b"\x22" * 15 + bytes([key_top])])


@pytest.mark.slow
@pytest.mark.parametrize("key_top, why", [(0x00, None), (0x80, STOP + "the verifier's session key has bit 2047 set")])
def test_the_lean_hello_leaves_and_its_answer_is_checked(zk, key_top, why):
    """At the fixture's statement and a draw of none, the `Hello` that leaves is the recorded one at the loop's own nonce,
    whatever the prover sent; its answer passes through when it is a root and a key, else is refused. After it, a
    `Commit` whose roots no Lean level 0 drew is refused without reaching the auditor (`test_a_commit_leaves_only_with_
    the_loops_level_0` takes one that does)."""
    recorded, nonce, _ = _record(zk, "selftest/honest")
    answers = {10: draw("null"), 1: _answer(key_top), 7: coins([])}
    auditor = Auditor(lambda f: answers[f[0]])
    lp = Loop(auditor, _args(zk, "pub-16.bin"))
    s = lp.connect()
    call(s, REGISTER)
    r = call(s, hello("anything"))
    sent = auditor.got[0][1]
    assert sent[:1] == b"\x01"
    text = sent[5:].decode()
    got = NONCE.search(text)[1]
    assert text == recorded.replace(nonce, got)
    if why is None:
        assert r == _answer(key_top)
        assert reason(call(s, COMMIT)).startswith(STOP + "a Commit whose roots are of [], where the session's tables are [")
    else:
        assert reason(r) == why
        assert reason(call(s, COMMIT)) == why
    s.close()
    lp.close()
    assert [f[0] for f in auditor.got[0]] == [10, 1]


def commit(roots: list[tuple[str, bytes]]) -> bytes:
    return (b"\x07" + _bytes(b"") + struct.pack("<I", len(roots)) + b"".join(_bytes(t.encode()) + _bytes(r) for t, r in roots)
            + struct.pack("<I", 0))


def level0(lp: Loop, m: int, z: bytes):
    """The loop's answer to `level0 M LEN` and the witness `z` on its frames port: `(line, draws, root)`, or the line."""
    with socket.create_connection(lp.frames) as s, s.makefile("rb") as r:
        s.sendall(f"level0 {m} {len(z)}\n".encode() + z)
        line = r.readline().decode().rstrip("\n")
        if not line.startswith("ok "):
            return line
        n, k = map(int, line.split()[1:])
        b = r.read(n)
        return line, b[:n - k], b[n - k:]


@pytest.mark.parametrize("m, z, why", [(3, b"\0" * 16, "err m = 3: "), (25, b"\0" * 3, "err a 3-byte witness, not whole words"),
                                       (25, b"\0" * 16, "err level 0: a 1-word witness where m = 25 takes ")])
def test_a_level_0_the_loop_cannot_draw(loop, m, z, why):
    _, lp = loop
    got = level0(lp, m, z)
    assert isinstance(got, str) and got.startswith(why), got


@pytest.mark.slow
def test_a_commit_leaves_only_with_the_loops_level_0(zk):
    """The loop draws a table's level-0 hiding and salts itself, fresh for each table: two draws over one witness differ,
    in the hiding and the root. A `Commit` leaves only when each root it carries is one the loop drew at its table's `m`,
    which the session then owns: a root drawn at another `m` is refused, never reaching the auditor, and one at the
    statement's leaves, its opening then checked (zero words refused)."""
    auditor = Auditor(lambda f: {10: draw("null"), 1: _answer(), 7: coins([])}[f[0]])
    lp = Loop(auditor, _args(zk, "pub-16.bin"))

    def session():
        s = lp.connect()
        call(s, REGISTER)
        assert call(s, hello("")) == _answer()
        return s

    s = session()
    why = reason(call(s, COMMIT))
    [table] = re.search(r"tables are \[(.*)\]$", why)[1].split(", ")
    s.close()
    other = 25
    _, d0, r0 = level0(lp, other, b"\0" * 16 * 2 ** (other - 7))
    _, d1, r1 = level0(lp, other, b"\0" * 16 * 2 ** (other - 7))
    assert len(d0) == len(d1) and d0 != d1 and r0 != r1
    s = session()
    why = reason(call(s, commit([(table, r0)])))
    s.close()
    if "drawn at m" in why:
        m = int(re.search(r"its statement's is (\d+)$", why)[1])
        assert why == STOP + f"a Commit whose root of {table} was drawn at m = {other}, where its statement's is {m}"
        _, _, r = level0(lp, m, b"\0" * 16 * 2 ** (m - 7))
        s = session()
        why = reason(call(s, commit([(table, r)])))
        s.close()
    assert why.startswith(STOP + "coin opening of session round 0: 0 words"), why
    lp.close()
    assert [[f[0] for f in c] for c in auditor.got][-1] == [10, 1, 7]
    assert all([f[0] for f in c] == [10, 1] for c in auditor.got[:-1])


@pytest.mark.slow
def test_a_round_before_the_commitment_never_leaves(zk):
    """Past `Hello`'s answer the first opening is the session stream's, at `Commit`: a `Round` in its place is refused
    without reaching the auditor, and so is every coin after it."""
    auditor = Auditor(lambda f: {10: draw("null"), 1: _answer()}[f[0]])
    lp = Loop(auditor, _args(zk, "pub-16.bin"))
    s = lp.connect()
    call(s, REGISTER)
    assert call(s, hello("")) == _answer()
    why = reason(call(s, round_("t/rep0", b"r", 6)))
    assert why == STOP + "a round of t/rep0 before the step-0 coin commitment", why
    assert reason(call(s, COMMIT)) == why
    s.close()
    lp.close()
    assert [f[0] for f in auditor.got[0]] == [10, 1]


@pytest.mark.slow
def test_a_finish_before_the_last_opening_never_leaves(zk):
    """`Finish` leaves only after the schedule's last opening: one right after `Hello`'s answer is refused without reaching
    the auditor, and so is every coin after it."""
    auditor = Auditor(lambda f: {10: draw("null"), 1: _answer()}[f[0]])
    lp = Loop(auditor, _args(zk, "pub-16.bin"))
    s = lp.connect()
    call(s, REGISTER)
    assert call(s, hello("")) == _answer()
    why = reason(call(s, b"\x06"))
    assert why.startswith(STOP + "a Finish after 0 openings, where the schedule's are "), why
    assert int(why.rsplit(" ", 1)[1]) == len(_schedule(_args(zk, "pub-16.bin"), "null"))
    assert reason(call(s, COMMIT)) == why
    s.close()
    lp.close()
    assert [f[0] for f in auditor.got[0]] == [10, 1]


def _schedule(args: list[str], draw: str) -> list[str]:
    p = subprocess.run([str(BIN), "session", "--schedule", *args], input=draw.encode(), capture_output=True, check=True)
    return p.stdout.decode().splitlines()


def test_a_schedule_the_setup_refuses(junk):
    [line] = _schedule(junk, "null")
    assert line.startswith("refused "), line


@pytest.mark.slow
@pytest.mark.parametrize("name, public, tables", [("selftest/honest", "pub-16.bin", 1), ("m26", "pub-32.bin", 1),
                                                   ("j2", "pub-32.bin", 2), ("drawn", "pub-32.bin", 1)])
def test_the_schedule_is_the_recorded_rounds(zk, name, public, tables):
    """At a recorded session's statement and draw, the openings the loop asks for after the session stream's are the record's
    rounds: each rep stream's coin counts in order, the streams one after another in table order, rep 0 before rep 1."""
    _, _, draw_ = _record(zk, name)
    d = zk / "sessions" / name
    if "/" not in name:
        [d] = [p for p in d.iterdir() if p.is_dir()]
    rec = json.loads((d / "session.json").read_text())
    lines = _schedule(_args(zk, public, tables), draw_)
    assert lines[0] == f"ok {len(lines) - 1}", lines[:2]
    plan = [(s, int(n)) for s, n in (line.rsplit(" ", 1) for line in lines[1:])]
    order = [s for i, (s, _) in enumerate(plan) if i == 0 or plan[i - 1][0] != s]
    tabs = list(dict.fromkeys(s.rsplit("/", 1)[0] for s in order))
    assert len(tabs) == tables and order == [f"{t}/rep{r}" for t in tabs for r in (0, 1)], order
    got = {s: [n for t, n in plan if t == s] for s in order}
    assert got == {s["stream"]: [r["n"] for r in s["rounds"]] for s in rec["streams"]}


@pytest.mark.slow
def test_each_session_its_own_nonce(zk):
    recorded, nonce, _ = _record(zk, "selftest/honest")
    auditor = Auditor(lambda f: {10: draw("null"), 1: _answer()}[f[0]])
    lp = Loop(auditor, _args(zk, "pub-16.bin"))
    for _ in range(2):
        s = lp.connect()
        call(s, REGISTER)
        call(s, hello(""))
        s.close()
    texts = [c[1][5:].decode() for c in auditor.got]
    got = [NONCE.search(t)[1] for t in texts]
    assert got[0] != got[1] and texts == [recorded.replace(nonce, g) for g in got]
    lp.close()
