"""The firewall's session loop in Lean (`Flock/Firewall/Session.lean`, `flock-firewall session`), which owns the auditor's
connection: a prover connects to it as it would to the auditor, and each connection is one session it relays on its own
connection to the auditor. Here a scripted auditor stands on the far side. What crosses is held to the loop's rules:
every frame that leaves the loop's own encoding (a `Proof` dropped, a frame it does not serve refused), the `Hello`
that leaves the Lean one at the auditor's draw (at the fixture's statement, `test_lean_hello.py`'s recorded `Hello` at
the loop's nonce) and never the prover's, `Hello`'s answer and every coin checked, and every coin after a refusal
refused. The loop starts the reps' frames on a port of its
own and drives each itself, every coin a frame takes the loop's own opening. `flock-circuit`'s firewalled sessions through it
are the end-to-end check (live PROTOCOL.md §10.4)."""
from __future__ import annotations

import json
import re
import socket
import struct
import subprocess
import threading

import pytest
from test_lean_hello import (  # noqa: F401
    BIN,
    FIXTURE,
    NONCE,
    STATEMENT,
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


PING = ping(b"\0" * 64)


def test_every_frame_leaves_as_the_loops_own(loop):
    """`Register` and `Ping` leave as the loop's own frames (the prover's probe, 64 zero bytes), whatever bytes the prover
    put in or after them."""
    auditor, lp = loop
    s = lp.connect()
    assert call(s, REGISTER + b"trailing") == draw("null")
    assert call(s, ping(b"abc") + b"trailing") == b"\x01"
    s.close()
    assert auditor.got[0] == [REGISTER, PING]


def test_a_provers_proof_never_leaves(loop):
    """A `Proof` the prover sends is answered `Ok` and dropped: the proofs that leave are the loop's finish's."""
    auditor, lp = loop
    s = lp.connect()
    call(s, REGISTER)
    proof = b"\x05" + _bytes(b"t") + struct.pack("<I", 0) + _bytes(b"junk")
    assert call(s, proof) == b"\x01"
    s.close()
    assert auditor.got[0] == [REGISTER]


@pytest.mark.parametrize("frame, why", [
    (b"\x63\x01\x02", "a request the session loop does not serve"),
    (b"\x08" + _bytes(b"state") + struct.pack("<I", 2), "a request the session loop does not serve"),
    (b"\x02" + _bytes(b"\0" * 32), "a Link before the session's Hello"),
    (b"\x03" + _bytes(b"rep0") + _bytes(b"") + _bytes(b"label") + b"\0" * 8,
     "an Open of rep0, which is none of the session's rep streams"),
    (b"\x03", "an Open frame without its stream"),
])
def test_a_frame_the_loop_does_not_serve_never_leaves(loop, frame, why):
    """An unknown tag, `Prime`, a `Link` before the `Hello` and an `Open` of a stream that is none of the session's rep
    streams are refused, and every coin after: none reaches the auditor."""
    auditor, lp = loop
    s = lp.connect()
    call(s, REGISTER)
    got = reason(call(s, frame))
    assert got == STOP + why, got
    if frame != b"\x03":
        assert reason(call(s, round_("rep0", b"r", 2))) == got
    s.close()
    assert auditor.got[0] == [REGISTER]


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
    assert auditor.got[0] == [REGISTER, PING]


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


@pytest.mark.parametrize("line, why", [
    (json.dumps({"layout": {"n_mlv": 1, "n_lc": 1, "n_ood0": 1, "k0": 1}}), "err a request `{line}`"),
    ("rep " + "ab" * 64 + " 0 0", "err a rep for a session the loop is not serving (coin tree root " + "ab" * 64 + ")"),
    ("rep " + "ab" * 64 + " 0 2", "err rep 2: a table has reps 0 and 1"),
    ("rep x 0", "err a request `rep x 0`"),
    ("place " + "ab" * 64 + " 0 0", "err a place for a session the loop is not serving (coin tree root " + "ab" * 64 + ")"),
    ("vals " + "ab" * 64 + " 0 1", "err a vals for a session the loop is not serving (coin tree root " + "ab" * 64 + ")"),
    ("places " + "ab" * 64 + " 0 0", "err a request `{line}`"),
    ("vals x 0", "err a request `vals x 0`"),
])
def test_a_frame_starts_only_as_a_live_sessions_rep(loop, line, why):
    """The loop drives the reps' frames itself: a frame starts on its frames port only as `rep SESSION TABLE REP`, for a
    session it is serving, at the statement's start. A start the prover brings (`flock-firewall frame`'s layout line) is
    refused, as is a rep of a session the loop isn't serving, and so is a rep's place or values (`place`, `vals`)."""
    _, lp = loop
    f = TcpFrame(lp)
    assert f.ask(line) == why.format(line=line)
    f.close()


@pytest.mark.slow
def test_each_reps_frame_draws_its_own(zk):
    """Each rep's frame the loop starts for a live session draws its own randomness at the statement's start, and starts
    once: the two reps' draws have one size and differ, a rep started again is refused, and so is a table the session
    doesn't have."""
    S = Sessions(_args(zk, "pub-16.bin"))
    s, name = S.session()
    a, b = TcpFrame(S.lp), TcpFrame(S.lp)
    da, db = a.ask(f"rep {name} 0 0"), b.ask(f"rep {name} 0 1")
    assert isinstance(da, tuple) and isinstance(db, tuple), (da, db)
    assert da[0] == db[0] and da[1] != db[1]
    c = TcpFrame(S.lp)
    assert c.ask(f"rep {name} 0 0") == "err table 0 rep 0's frame started already"
    c.close()
    c = TcpFrame(S.lp)
    assert c.ask(f"rep {name} 9 0") == "err table 9 of the session's 1"
    c.close()
    c = TcpFrame(S.lp)
    assert c.ask(f"vals {name} 0 1") == ("ok 0", b"")
    c.close()
    c = TcpFrame(S.lp)
    got = c.ask(f"place {name} 0 0")
    assert got.startswith("err a place of ") and got.endswith(" before the session's Commit"), got
    c.close()
    c = TcpFrame(S.lp)
    assert c.ask(f"place {name} 0 7") == "err a place of table 0 rep 7, whose frame the loop has not started"
    c.close()
    a.close()
    b.close()
    s.close()
    S.lp.close()


@pytest.fixture(scope="module")
def zk():
    assert (FIXTURE / "stage" / "circuit.txt").is_file(), (
        f"{FIXTURE} is not fetched: `uv run research data fetch-fixtures verity/ml/flock/tests/fixtures/zk-sessions`")
    return FIXTURE


def _answer(key_top: int = 0, root: int = 0x11) -> bytes:
    """`Hello`'s answer: a root (every byte `root`) and a session key, 20 words, the key's last byte `key_top`."""
    return coins([bytes([root]) * 16] * 4 + [b"\x22" * 16] * 15 + [b"\x22" * 15 + bytes([key_top])])


def _name(answer: bytes) -> str:
    """The session's name on the loop's frames port (`Level0.nameOf`): its coin tree root's hex."""
    return answer[5:69].hex()


class Sessions:
    """A scripted auditor whose every `Hello` is answered with a coin tree root of its own, on a loop at `args`."""

    def __init__(self, args: list[str]):
        roots = iter(range(1, 256))
        answers = {10: draw("null"), 7: coins([])}
        self.auditor = Auditor(lambda f: _answer(root=next(roots)) if f[0] == 1 else answers[f[0]])
        self.lp = Loop(self.auditor, args)

    def session(self) -> tuple[socket.socket, str]:
        """A session past its `Hello`'s answer, and its name."""
        s = self.lp.connect()
        call(s, REGISTER)
        a = call(s, hello(""))
        assert a[:1] == b"\x02" and len(a) == 5 + 16 * 20, a[:40]
        return s, _name(a)

    def ended(self) -> str:
        """The loop's next line on its standard error that a session ended and its level 0 were freed."""
        while not (line := self.lp.p.stderr.readline().decode()).startswith("level 0: session "):
            assert line, "the loop's standard error closed"
        return line.rstrip("\n")


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


def level0(lp: Loop, m: int, z: bytes, name: str = "00" * 64, table: int = 0):
    """The loop's answer to `level0 SESSION TABLE M LEN` and the witness `z` on its frames port, for table `table` of the
    session named `name`: `(line, draws, root)`, or the line."""
    with socket.create_connection(lp.frames) as s, s.makefile("rb") as r:
        s.sendall(f"level0 {name} {table} {m} {len(z)}\n".encode() + z)
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


def _zeros(m: int) -> bytes:
    """A zero packed witness at `m`."""
    return b"\0" * 16 * 2 ** (m - 7)


@pytest.mark.slow
def test_a_commit_leaves_only_with_the_loops_level_0(zk):
    """The loop draws a table's level-0 hiding and salts itself, fresh for each table: two draws over one witness differ,
    in the hiding and the root. It draws a table's once a session: a second request for the table, or one for a table the
    session lacks, is refused, so the prover can't draw until a root suits it (NIGHTREV's grinding). A `Commit` leaves only
    when each root it carries is the one the loop drew for its own session's table at its `m`: a root drawn for another
    session, or at another `m`, is refused, never reaching the auditor, and one at the statement's leaves, its opening
    then checked (zero words refused). The `Commit` that leaves is the loop's own, whatever the prover's carried beside
    its roots: `root_F` the session's Σ, no public words, nothing after."""
    S = Sessions(_args(zk, "pub-16.bin"))
    s, name = S.session()
    why = reason(call(s, COMMIT))
    [table] = re.search(r"tables are \[(.*)\]$", why)[1].split(", ")
    other = 25
    _, d0, r0 = level0(S.lp, other, _zeros(other), name)
    assert level0(S.lp, other, _zeros(other), name) == (
        "err a second level-0 request for table 0 of this session: the loop draws a table's level 0 once")
    assert level0(S.lp, other, _zeros(other), name, table=1) == "err a level-0 request for table 1 of a session with 1 tables"
    s2, name2 = S.session()
    _, d1, r1 = level0(S.lp, other, _zeros(other), name2)
    assert len(d0) == len(d1) and d0 != d1 and r0 != r1
    assert reason(call(s2, commit([(table, r0)]))) == (
        STOP + f"a Commit whose root of {table} is not the level 0 the loop drew for this session's table 0")
    s.close()
    s2.close()
    s, name = S.session()
    _, _, r = level0(S.lp, other, _zeros(other), name)
    why = reason(call(s, doctored([(table, r)])))
    s.close()
    if "drawn at m" in why:
        m = int(re.search(r"its statement's is (\d+)$", why)[1])
        assert why == STOP + f"a Commit whose root of {table} was drawn at m = {other}, where its statement's is {m}"
        s, name = S.session()
        _, _, r = level0(S.lp, m, _zeros(m), name)
        why = reason(call(s, doctored([(table, r)])))
        s.close()
    assert why.startswith(STOP + "coin opening of session round 0: 0 words"), why
    S.lp.close()
    assert [[f[0] for f in c] for c in S.auditor.got][-1] == [10, 1, 7]
    assert all([f[0] for f in c] == [10, 1] for c in S.auditor.got[:-1])
    _, sent, left = S.auditor.got[-1]
    sigma = bytes.fromhex(json.loads(sent[5:])["link"]["sigma"])
    assert left == b"\x07" + _bytes(sigma) + commit([(table, r)])[5:]


def doctored(roots: list[tuple[str, bytes]]) -> bytes:
    """A `Commit` of `roots` with a `root_F` that is not Σ, a public word, and bytes after it."""
    return (b"\x07" + _bytes(b"\xab" * 63) + commit(roots)[5:-4] + struct.pack("<I", 1) + _bytes(b"x") + _bytes(b"\1" * 16)
            + b"trailing")


@pytest.mark.slow
def test_a_sessions_level_0_lives_only_as_long_as_it(zk):
    """A level 0 is drawn only for a session the loop is serving, named by its coin tree root, and every one drawn for a
    session, committed or not, is freed when that session ends, whatever the others hold: a prover that draws
    and never commits leaves nothing behind."""
    S = Sessions(_args(zk, "pub-16.bin"))
    got = level0(S.lp, 25, _zeros(25), "ab" * 64)
    assert got == f"err a level-0 request for a session the loop is not serving (coin tree root {'ab' * 64})", got
    s, name = S.session()
    assert level0(S.lp, 25, _zeros(25), name)[0].startswith("ok ")
    assert level0(S.lp, 25, _zeros(25), name).startswith("err a second level-0 request for table 0 of this session")
    s.close()
    assert S.ended() == "level 0: session 0 ended; 1 drawn for it freed, 0 kept"
    (s1, n1), (s2, n2) = S.session(), S.session()
    for n in (n1, n2):
        assert level0(S.lp, 25, _zeros(25), n)[0].startswith("ok ")
    s1.close()
    assert S.ended() == "level 0: session 1 ended; 1 drawn for it freed, 1 kept"
    s2.close()
    assert S.ended() == "level 0: session 2 ended; 1 drawn for it freed, 0 kept"
    S.lp.close()


def test_the_loops_finish_needs_a_statement(loop):
    """A finish on the loop's frames port sets the session's statement up at its draw first: one the setup refuses ends
    it."""
    _, lp = loop
    f = TcpFrame(lp)
    got = f.ask("finish 4", b"null")
    assert isinstance(got, str) and got.startswith("err setup: "), got
    f.close()


@pytest.mark.slow
def test_the_loops_finish_reads_its_level_0_and_draws_each_reps_salts(zk):
    """The loop's finish (`Finish.serveLoop`) takes a table's level 0 from the loop, by its root, and only one the loop drew
    at the table's `m`, for a session it still serves; its answer is that root. Each rep's salts for levels ≥ 1 it draws
    itself, once: whole 192-byte leaves a level, fresh for each rep, and a rep asked again, or a rep's request before its
    salts, is refused. Once the session that drew the level 0 ends, the loop has freed it, and finish refuses its root."""
    S = Sessions(_args(zk, "pub-16.bin"))
    s, name = S.session()
    lp = S.lp

    def finish():
        f = TcpFrame(lp)
        assert f.ask("finish 4", b"null") == ("ok 0", b"")
        return f

    f = finish()
    assert f.ask("level0 0 64", b"\x01" * 64) == "err finish: table 0's root is no level 0's the session loop drew"
    f.close()
    other = 25
    _, _, r0 = level0(lp, other, _zeros(other), name)
    f = finish()
    got = f.ask(f"level0 0 {len(r0)}", r0)
    f.close()
    if isinstance(got, str):
        m = int(re.fullmatch(rf"err finish: a level 0 drawn at m = {other} for a table at m = (\d+)", got)[1])
        s.close()
        assert S.ended() == "level 0: session 0 ended; 1 drawn for it freed, 0 kept"
        s, name = S.session()
        _, _, r0 = level0(lp, m, _zeros(m), name)
    f = finish()
    assert f.ask(f"level0 0 {len(r0)}", r0) == (f"ok {len(r0)}", r0)
    assert f.ask("rep 0 0 0") == "err table 0 rep 0 before its salts for levels ≥ 1"
    f.close()
    f = finish()
    f.ask(f"level0 0 {len(r0)}", r0)
    reps = []
    for r in (0, 1):
        line, b = f.ask(f"salts 0 {r} 0")
        (n,) = struct.unpack_from("<Q", b)
        levels, o = [], 8
        for _ in range(n):
            (k,) = struct.unpack_from("<Q", b, o)
            levels.append(b[o + 8:o + 8 + k])
            o += 8 + k
        assert o == len(b) and n >= 1 and all(len(x) and len(x) % 192 == 0 for x in levels), (line, n)
        reps.append(levels)
    assert [len(x) for x in reps[0]] == [len(x) for x in reps[1]] and reps[0] != reps[1]
    assert f.ask("salts 0 0 0") == "err table 0 rep 0's salts for levels ≥ 1 were drawn already"
    f.close()
    s.close()
    assert re.match(r"level 0: session \d+ ended; ", S.ended())
    f = finish()
    assert f.ask(f"level0 0 {len(r0)}", r0) == "err finish: table 0's root is no level 0's the session loop drew"
    f.close()
    lp.close()


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
