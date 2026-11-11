"""The ZK server's session loop in Lean (`Flock/Firewall/Session.lean`, `flock-firewall session`), which owns the auditor's
connection: a prover connects to it as it would to the auditor, and each connection is one session it relays on its own
connection to the auditor. Here a scripted auditor stands on the far side. What crosses is held to the loop's rules:
every frame that leaves the loop's own encoding (a `Proof` dropped, a frame it does not serve refused), the `Hello`
that leaves the Lean one at the auditor's sample (at the fixture's statement, `test_lean_hello.py`'s recorded `Hello` at
the loop's nonce) and never the prover's, `Hello`'s answer and every coin checked, each request leaving once at its place
in the honest prover's order, a `Commit` answered without coins ending the session, a `Ping` answered by the loop and
never leaving, every request after a refusal refused
without reaching the auditor, and nothing of a session read after its `Finish`. The loop starts the reps' frames on a
port of its own and drives each itself, every coin a frame takes the loop's own opening, and a coin a frame refuses (a
zerocheck outer coin `1`) stops the session at that round. `flock-circuit`'s ZK server sessions through it are the
end-to-end check (live PROTOCOL.md §10.4)."""
from __future__ import annotations

import hashlib
import json
import os
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

from verity.commitments import hm96

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
FINISH = b"\x06"
COMMIT = b"\x07" + _bytes(b"") + struct.pack("<I", 0) + struct.pack("<I", 0)


def hello(text: str) -> bytes:
    return b"\x01" + _bytes(text.encode())


def ping(b: bytes) -> bytes:
    return b"\x09" + _bytes(b)


def round_(stream: str, content: bytes, n: int) -> bytes:
    return b"\x04" + _bytes(stream.encode()) + _bytes(content) + struct.pack("<I", n)


def open_(stream: str) -> bytes:
    return b"\x03" + _bytes(stream.encode()) + _bytes(b"") + _bytes(b"label") + b"\0" * 8


LINK = b"\x02" + _bytes(b"\0" * 32)


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


def answer(s: socket.socket, f: bytes) -> bytes | None | OSError:
    """`f`'s answer on `s`: its bytes, `None` when the connection closed after reading it, or the error a send to a
    connection the loop already closed raises."""
    try:
        return call(s, f)
    except (BrokenPipeError, ConnectionResetError) as e:
        return e


def closed(s: socket.socket, f: bytes) -> bool:
    """Whether the loop has ended the session on `s`: `f` gets no answer, the connection closed."""
    a = answer(s, f)
    return a is None or isinstance(a, (BrokenPipeError, ConnectionResetError))


def plain(f: bytes) -> bytes:
    """The auditor's answers: Register a sample of none, anything else 4 zero words."""
    return {10: draw("null")}.get(f[0], coins([b"\0" * 16] * 4))


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


def test_every_frame_leaves_as_the_loops_own(loop):
    """`Register` leaves as the loop's own frame, whatever bytes the prover put after it, and a `Ping` never leaves: the loop
    answers it `Ok` itself."""
    auditor, lp = loop
    s = lp.connect()
    assert call(s, REGISTER + b"trailing") == draw("null")
    assert call(s, ping(b"abc") + b"trailing") == b"\x01"
    s.close()
    assert auditor.got[0] == [REGISTER]


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
    (LINK, "a Link before the session's Hello"),
    (open_("rep0"), "an Open of rep0, which is none of the session's rep streams"),
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
    assert auditor.got[0] == [REGISTER]


def test_coins_before_the_commitment(loop):
    _, lp = loop
    s = lp.connect()
    assert reason(call(s, round_("rep0", b"r", 2))) == STOP + "coins before the step-0 coin commitment"


def test_a_finish_before_the_commitment_never_leaves(loop):
    auditor, lp = loop
    s = lp.connect()
    call(s, REGISTER)
    assert reason(call(s, FINISH)) == STOP + "a Finish before the step-0 coin commitment"
    s.close()
    assert auditor.got[0] == [REGISTER]


def test_a_session_ends_at_its_finish(loop):
    """A session ends at its `Finish`, whatever the loop decided of it (here a refusal): the loop reads nothing more of
    it, so a second `Finish`, a `Ping` and a `Register` after it get no answer, and nothing more of the session reaches
    the auditor. The loop serves the next session as before."""
    auditor, lp = loop
    s = lp.connect()
    call(s, REGISTER)
    assert reason(call(s, FINISH)) == STOP + "a Finish before the step-0 coin commitment"
    assert closed(s, FINISH) and closed(s, ping(b"p")) and closed(s, REGISTER)
    s.close()
    t = lp.connect()
    assert call(t, REGISTER) == draw("null")
    t.close()
    assert auditor.got == [[REGISTER], [REGISTER]]


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
    ("rep " + "ab" * 64 + " 0 0", "err a rep for a session the loop is not serving (id " + "ab" * 64 + ")"),
    ("rep " + "ab" * 64 + " 0 2", "err rep 2: a table has reps 0 and 1"),
    ("rep x 0", "err a request `rep x 0`"),
    ("place " + "ab" * 64 + " 0 0", "err a place for a session the loop is not serving (id " + "ab" * 64 + ")"),
    ("vals " + "ab" * 64 + " 0 1", "err a vals for a session the loop is not serving (id " + "ab" * 64 + ")"),
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
    """Each rep's frame the loop starts for a live session samples its own randomness at the statement's start, and starts
    once, and only once the session's level 0 is sampled: a rep before it is refused, so the pads' root, a sample of the
    loop's, reaches the prover only after every table's witness is in. Each answer is the pads' root alone (the samples never
    leave the loop) and the two reps' roots differ, a rep started again is refused, and so is a table the session doesn't
    have."""
    S = Sessions(_args(zk, "pub-16.bin"))
    s, name = S.session()
    c = TcpFrame(S.lp)
    assert c.ask(f"rep {name} 0 0") == "err " + EARLY.format(i=0, r=0)
    c.close()
    m = _statement_m(S.lp, name)
    assert level0(S.lp, [(m, _zeros(m))], name)[0][0].startswith("ok ")
    a, b = TcpFrame(S.lp), TcpFrame(S.lp)
    da, db = a.ask(f"rep {name} 0 0"), b.ask(f"rep {name} 0 1")
    assert isinstance(da, tuple) and isinstance(db, tuple), (da, db)
    assert da[0] == db[0] == "ok 64" and da[1] != db[1]
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
        f"{FIXTURE} is not fetched: `uv run research data fetch-fixtures verity/protocols/flock/prover/tests/fixtures/zk-sessions`")
    return FIXTURE


def _answer(key_top: int = 0, root: int = 0x11) -> bytes:
    """`Hello`'s answer: a root (every byte `root`) and a session key, 20 words, the key's last byte `key_top`."""
    return coins([bytes([root]) * 16] * 4 + [b"\x22" * 16] * 15 + [b"\x22" * 15 + bytes([key_top])])


def _name(a: bytes) -> str:
    """The session's name on the loop's frames port (`Level0.nameOf`): the hex of the loop's id for it, four words, which
    is what the prover receives for `Hello`'s answer (`Session.idFrame`)."""
    assert a[:5] == b"\x02" + struct.pack("<I", 4) and len(a) == 5 + 64, a[:40]
    return a[5:].hex()


class Sessions:
    """A scripted auditor whose every `Hello` is answered with a coin tree root of its own, and every `Commit` with
    `at_commit`, on a loop at `args`."""

    def __init__(self, args: list[str], at_commit: bytes = coins([])):
        roots = iter(range(1, 256))
        answers = {10: draw("null"), 3: b"\x01", 7: at_commit}
        self.auditor = Auditor(lambda f: _answer(root=next(roots)) if f[0] == 1 else answers[f[0]])
        self.lp = Loop(self.auditor, args)

    def session(self, opened: str | None = None) -> tuple[socket.socket, str]:
        """A session past its `Hello`'s answer, and its name; past the `Open` of its first rep stream `opened` too, if
        named: the `Commit`'s place."""
        s = self.lp.connect()
        call(s, REGISTER)
        name = _name(call(s, hello("")))
        if opened is not None:
            assert call(s, open_(opened)) == b"\x01"
        return s, name

    def ended(self) -> str:
        """The loop's next line on its standard error that a session ended and its level 0 were freed."""
        while not (line := self.lp.p.stderr.readline().decode()).startswith("level 0: session "):
            assert line, "the loop's standard error closed"
        return line.rstrip("\n")


@pytest.mark.slow
@pytest.mark.parametrize("key_top, why", [(0x00, None), (0x80, STOP + "the verifier's session key has bit 2047 set")])
def test_the_lean_hello_leaves_and_its_answer_is_checked(zk, key_top, why):
    """At the fixture's statement and a sample of none, the `Hello` that leaves is the recorded one at the loop's own nonce,
    whatever the prover sent; its answer is checked to be a root and a key, else refused, and never reaches the prover,
    which receives the loop's id for the session in its place: no byte of the coin tree's root or the session key. After
    it and the first rep stream's `Open`, a `Commit` whose roots no Lean level 0 drew is refused without reaching the
    auditor (`test_a_commit_leaves_only_with_the_loops_level_0` takes one that does)."""
    recorded, nonce, _ = _record(zk, "selftest/honest")
    answers = {10: draw("null"), 1: _answer(key_top), 3: b"\x01", 7: coins([])}
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
        words = [r[5 + 16 * i:21 + 16 * i] for i in range(4)]
        assert _name(r) and not set(words) & {b"\x11" * 16, b"\x22" * 16, b"\x22" * 15 + bytes([key_top])}, r.hex()
        assert call(s, open_(_streams(_args(zk, "pub-16.bin"))[0])) == b"\x01"
        assert reason(call(s, COMMIT)).startswith(STOP + "a Commit whose roots are of [], where the session's tables are [")
    else:
        assert reason(r) == why
        assert reason(call(s, COMMIT)) == why
    s.close()
    lp.close()
    assert [f[0] for f in auditor.got[0]] == ([10, 1, 3] if why is None else [10, 1])


def commit(roots: list[tuple[str, bytes]]) -> bytes:
    return (b"\x07" + _bytes(b"") + struct.pack("<I", len(roots)) + b"".join(_bytes(t.encode()) + _bytes(r) for t, r in roots)
            + struct.pack("<I", 0))


SECOND = "a second level-0 request for this session: the loop draws its tables' level 0 once, after every table's witness"
COUNT = ("a level-0 request of {k} tables' witnesses for a session with {n} tables: the loop draws level 0 only on every"
         " table's witness")
EARLY = "table {i} rep {r} before the session's level 0: a rep's frame starts once every table's witness is in"


def level0(lp: Loop, tables: list[tuple[int, bytes]], name: str = "00" * 64):
    """The loop's answer to `level0 SESSION M₀ LEN₀ …` and each table's witness after it on its frames port, for the
    session named `name`, a `(m, z)` for each table in table order: a `(line, draws, root)` for each table, or the line."""
    with socket.create_connection(lp.frames) as s, s.makefile("rb") as r:
        sizes = "".join(f" {m} {len(z)}" for m, z in tables)
        s.sendall(f"level0 {name}{sizes}\n".encode() + b"".join(z for _, z in tables))
        got = []
        for _ in tables or [None]:
            line = r.readline().decode().rstrip("\n")
            if not line.startswith("ok "):
                assert not got, (got, line)
                return line
            n, k = map(int, line.split()[1:])
            b = r.read(n)
            got.append((line, b[:n - k], b[n - k:]))
        return got


@pytest.mark.parametrize("tables, why", [
    ([(25, b"\0" * 3)], "err a 3-byte witness, not whole words"),
    ([(25, b"\0" * 16)], f"err a level-0 request for a session the loop is not serving (id {'00' * 64})"),
    ([], f"err a request `level0 {'00' * 64}`"),
])
def test_a_level_0_the_loop_cannot_draw(loop, tables, why):
    """A level-0 request is a witness of whole words for each of a live session's tables; anything else is refused before
    the loop samples (`test_the_loop_draws_level_0_at_the_statement_with_its_own_mask_words` takes a live session's)."""
    _, lp = loop
    assert level0(lp, tables) == why


def _zeros(m: int) -> bytes:
    """A zero packed witness at `m`."""
    return b"\0" * 16 * 2 ** (m - 7)


def _statement_m(lp: Loop, name: str, tables: int = 1) -> int:
    """Table 0's statement's `m` in the session named `name` of `tables` tables, as the loop states it refusing a level 0
    at another."""
    got = level0(lp, [(3, b"\0" * 16)] * tables, name)
    return int(re.fullmatch(r"err a level-0 request at m = 3, where table 0's statement is at m = (\d+)", got)[1])


def _mask_words(m: int) -> int:
    """The fixture statement's mask-slot words at `m`: `2^(mask_log − 7)` a block, `2^(m − k_log)` blocks."""
    meta = json.loads((FIXTURE / "stage" / "circuit.txt").read_text().split("\n")[1].removeprefix("META "))
    [r] = [r for r in meta["ranges"] if r["type"] == "mask"]
    return 2 ** (r["slot_log"] - 7) * 2 ** (m - meta["k_log"])


@pytest.mark.slow
def test_the_loop_draws_level_0_at_the_statement_with_its_own_mask_words(zk):
    """The loop samples a table's level 0 only at its statement's `m`, on a witness of that `m`'s words, and writes the mask
    slot's words itself: its answer leads with them, a word for each of the slot's, fresh for each sample and never the
    witness's, whatever the prover's witness held there. So the prover has no say in the words that hide the claims'
    ring-switch values (`flock-circuit`'s `zk_worker_level0` holds a worker that keeps its own to the session's stop)."""
    S = Sessions(_args(zk, "pub-16.bin"))
    s, name = S.session()
    m = _statement_m(S.lp, name)
    n = _mask_words(m)
    assert level0(S.lp, [(m, b"\0" * 16)], name).startswith("err the mask slot: its word ")
    big = _zeros(m) + b"\0" * 16
    assert level0(S.lp, [(m, big)], name).startswith(f"err level 0: a {len(big) // 16}-word witness where m = {m} takes ")
    mine = b"\xab" * 16
    [(_, d0, _)] = level0(S.lp, [(m, mine * 2 ** (m - 7))], name)
    s2, name2 = S.session()
    [(_, d1, _)] = level0(S.lp, [(m, mine * 2 ** (m - 7))], name2)
    w0, w1 = ([d[16 * i:16 * i + 16] for i in range(n)] for d in (d0, d1))
    assert len(d0) == len(d1) > 16 * n and mine not in w0 + w1 and len(set(w0 + w1)) == 2 * n
    s.close()
    s2.close()
    S.lp.close()


@pytest.mark.slow
def test_a_commit_leaves_only_with_the_loops_level_0(zk):
    """The loop samples a table's level-0 hiding and salts itself, fresh for each table: two samples over one witness differ,
    in the hiding and the root. It samples a session's once: a second request, or one whose witnesses are not one for each
    of the session's tables, is refused, so the prover can't sample until a root suits it. A `Commit` leaves only
    when each root it carries is the one the loop sampled for its own session's table, which it samples only at the
    table's `m`: a root sampled for another session is refused, never reaching the auditor, and the session's own leaves,
    its opening then checked (zero words refused). The `Commit` that leaves is the loop's own, whatever the prover's
    carried beside its roots: `root_F` the session's Σ, no public words, nothing after."""
    args = _args(zk, "pub-16.bin")
    first = _streams(args)[0]
    S = Sessions(args)
    s, name = S.session(first)
    why = reason(call(s, COMMIT))
    [table] = re.search(r"tables are \[(.*)\]$", why)[1].split(", ")
    m = _statement_m(S.lp, name)
    assert level0(S.lp, [(m, _zeros(m))] * 2, name) == "err " + COUNT.format(k=2, n=1)
    [(_, d0, r0)] = level0(S.lp, [(m, _zeros(m))], name)
    assert level0(S.lp, [(m, _zeros(m))], name) == "err " + SECOND
    s2, name2 = S.session(first)
    [(_, d1, r1)] = level0(S.lp, [(m, _zeros(m))], name2)
    assert len(d0) == len(d1) and d0 != d1 and r0 != r1
    assert reason(call(s2, commit([(table, r0)]))) == (
        STOP + f"a Commit whose root of {table} is not the level 0 the loop drew for this session's table 0")
    s.close()
    s2.close()
    s, name = S.session(first)
    [(_, _, r)] = level0(S.lp, [(m, _zeros(m))], name)
    why = reason(call(s, doctored([(table, r)])))
    s.close()
    assert why.startswith(STOP + "coin opening of session round 0: 0 words"), why
    S.lp.close()
    assert [[f[0] for f in c] for c in S.auditor.got][-1] == [10, 1, 3, 7]
    assert all([f[0] for f in c] == [10, 1, 3] for c in S.auditor.got[:-1])
    _, sent, _, left = S.auditor.got[-1]
    sigma = bytes.fromhex(json.loads(sent[5:])["link"]["sigma"])
    assert left == b"\x07" + _bytes(sigma) + commit([(table, r)])[5:]


@pytest.mark.slow
@pytest.mark.parametrize("answer, said", [(b"\x01", ""), (b"\x03" + _bytes(b"no coins here"), ": no coins here")])
def test_a_commit_answered_without_coins_ends_the_session(zk, answer, said):
    """A `Commit` the auditor answers with anything but coins (`Ok`, a refusal) ends the session, refused, with the
    auditor's reason where it gave one and nothing else of its answer: the `Commit` sent again, and every request after
    it, is refused for that reason without reaching the auditor, so a `Commit` leaves once however often the prover
    sends it (`Flock.Test.SessionPlace` holds a `Round`'s answer to the same)."""
    args = _args(zk, "pub-16.bin")
    first = _streams(args)[0]
    S = Sessions(args, answer)
    s, _ = S.session(first)
    [table] = re.search(r"tables are \[(.*)\]$", reason(call(s, COMMIT)))[1].split(", ")
    s.close()
    s, name = S.session(first)
    m = _statement_m(S.lp, name)
    [(_, _, r)] = level0(S.lp, [(m, _zeros(m))], name)
    why = reason(call(s, commit([(table, r)])))
    assert why == STOP + "the auditor answered the step-0 coin commitment without coins" + said, why
    assert [reason(call(s, f)) for f in (commit([(table, r)]), LINK, round_(first, b"r", 2))] == [why] * 3
    assert call(s, ping(b"p")) == b"\x01"
    s.close()
    S.lp.close()
    assert [f[0] for f in S.auditor.got[-1]] == [10, 1, 3, 7]


@pytest.mark.slow
def test_the_loop_draws_level_0_only_once_every_tables_witness_is_in(zk):
    """In a session of two tables the loop samples neither table's level 0 until it holds both witnesses: a request carrying
    one table's witness, or three, is refused with nothing sampled, and so is a rep's start, so none of the session's samples
    (level 0's mask words, hiding and salts, a rep's pads) reaches the prover before both witnesses are its. On both it
    answers both tables, each its own sample. A witness after that sample, alone or beside the other table's, is refused: each
    table's witness is fixed before any of the session's samples, as the model (`FlockSoundness.ZK.Table.view`) fixes it
    before the tape, and the worker selftest's `zk_worker_level0` holds `flock-circuit` to the same refusals."""
    S = Sessions(_args(zk, "pub-16.bin", 2))
    s, name = S.session()
    m = _statement_m(S.lp, name, 2)
    z = _zeros(m)
    for k in (1, 3):
        assert level0(S.lp, [(m, z)] * k, name) == "err " + COUNT.format(k=k, n=2)
    for i in (0, 1):
        f = TcpFrame(S.lp)
        assert f.ask(f"rep {name} {i} 0") == "err " + EARLY.format(i=i, r=0)
        f.close()
    got = level0(S.lp, [(m, z)] * 2, name)
    assert isinstance(got, list) and len(got) == 2, got
    (l0, d0, r0), (l1, d1, r1) = got
    assert l0 == l1 and l0.startswith("ok ") and d0 != d1 and r0 != r1
    for late in ([(m, z)], [(m, z)] * 2, [(m, b"\1" * len(z))] * 2):
        assert level0(S.lp, late, name) == "err " + SECOND
    f = TcpFrame(S.lp)
    got = f.ask(f"rep {name} 1 0")
    assert isinstance(got, tuple) and got[0] == "ok 64", got
    f.close()
    s.close()
    assert S.ended() == "level 0: session 0 ended; 2 drawn for it freed, 0 kept"
    S.lp.close()


def doctored(roots: list[tuple[str, bytes]]) -> bytes:
    """A `Commit` of `roots` with a `root_F` that is not Σ, a public word, and bytes after it."""
    return (b"\x07" + _bytes(b"\xab" * 63) + commit(roots)[5:-4] + struct.pack("<I", 1) + _bytes(b"x") + _bytes(b"\1" * 16)
            + b"trailing")


@pytest.mark.slow
def test_a_sessions_level_0_lives_only_as_long_as_it(zk):
    """A level 0 is sampled only for a session the loop is serving, named by the loop's id for it, and every one sampled for a
    session, committed or not, is freed when that session ends, whatever the others hold: a prover that samples
    and never commits leaves nothing behind."""
    S = Sessions(_args(zk, "pub-16.bin"))
    got = level0(S.lp, [(25, _zeros(25))], "ab" * 64)
    assert got == f"err a level-0 request for a session the loop is not serving (id {'ab' * 64})", got
    s, name = S.session()
    m = _statement_m(S.lp, name)
    assert level0(S.lp, [(m, _zeros(m))], name)[0][0].startswith("ok ")
    assert level0(S.lp, [(m, _zeros(m))], name) == "err " + SECOND
    s.close()
    assert S.ended() == "level 0: session 0 ended; 1 drawn for it freed, 0 kept"
    (s1, n1), (s2, n2) = S.session(), S.session()
    for n in (n1, n2):
        assert level0(S.lp, [(m, _zeros(m))], n)[0][0].startswith("ok ")
    s1.close()
    assert S.ended() == "level 0: session 1 ended; 1 drawn for it freed, 1 kept"
    s2.close()
    assert S.ended() == "level 0: session 2 ended; 1 drawn for it freed, 0 kept"
    S.lp.close()


def test_the_loops_finish_needs_a_statement(loop):
    """A finish on the loop's frames port sets the session's statement up at its sample first: one the setup refuses ends
    it."""
    _, lp = loop
    f = TcpFrame(lp)
    got = f.ask("finish 4", b"null")
    assert isinstance(got, str) and got.startswith("err setup: "), got
    f.close()


@pytest.mark.slow
def test_the_loops_finish_reads_its_level_0_and_draws_each_reps_salts(zk):
    """The loop's finish (`Finish.serveLoop`) takes a table's level 0 from the loop, by its root, and only one the loop sampled
    at the table's `m`, for a session it still serves; its answer is that root. A rep's request before its salts for levels
    ≥ 1 is refused, and so are the salts of a rep whose frame has not taken its algebraic phase's last coins, here one whose
    level 0 no `Commit` carries (the selftest's ZK server sessions sample them past it, and `zk_worker_finish`'s
    `salts-early` and `salts-again` hold a worker asking before it, or a second time, to the refusal). Once the session
    that sampled the level 0 ends, the loop has freed it, and finish refuses its root."""
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
    m = _statement_m(lp, name)
    [(_, _, r0)] = level0(lp, [(m, _zeros(m))], name)
    f = finish()
    assert f.ask(f"level0 0 {len(r0)}", r0) == (f"ok {len(r0)}", r0)
    assert f.ask("rep 0 0 0") == "err table 0 rep 0 before its salts for levels ≥ 1"
    f.close()
    for r in (0, 1):
        f = finish()
        f.ask(f"level0 0 {len(r0)}", r0)
        assert f.ask(f"salts 0 {r} 0") == "err finish: table 0: a level 0 no session's Commit carries"
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
    _name(call(s, hello("")))
    why = reason(call(s, round_("t/rep0", b"r", 6)))
    assert why == STOP + "a round of t/rep0 before the step-0 coin commitment", why
    assert reason(call(s, COMMIT)) == why
    s.close()
    lp.close()
    assert [f[0] for f in auditor.got[0]] == [10, 1]


@pytest.mark.slow
def test_a_finish_before_the_last_opening_never_leaves(zk):
    """`Finish` leaves only after the schedule's last opening: one right after `Hello`'s answer is refused without reaching
    the auditor, and ends the session, so a second `Finish` and a `Commit` after it get no answer and nothing more
    reaches the auditor."""
    auditor = Auditor(lambda f: {10: draw("null"), 1: _answer()}[f[0]])
    lp = Loop(auditor, _args(zk, "pub-16.bin"))
    s = lp.connect()
    call(s, REGISTER)
    _name(call(s, hello("")))
    why = reason(call(s, FINISH))
    assert why.startswith(STOP + "a Finish after 0 openings, where the schedule's are "), why
    assert int(why.rsplit(" ", 1)[1]) == len(_schedule(_args(zk, "pub-16.bin"), "null"))
    assert closed(s, FINISH) and closed(s, COMMIT)
    s.close()
    lp.close()
    assert [f[0] for f in auditor.got[0]] == [10, 1]


def _schedule(args: list[str], draw: str) -> list[str]:
    p = subprocess.run([str(BIN), "session", "--schedule", *args], input=draw.encode(), capture_output=True, check=True)
    return p.stdout.decode().splitlines()


def _streams(args: list[str]) -> list[str]:
    """The session's rep streams at a sample of none, in the order they open: the schedule's."""
    return list(dict.fromkeys(line.rsplit(" ", 1)[0] for line in _schedule(args, "null")[1:]))


@pytest.mark.slow
def test_each_request_leaves_once_at_its_place(zk):
    """Past `Register`, the loop sends `Hello`, the first rep stream's `Open` and `Commit` only at their places in the honest
    prover's order, each once: a second `Register`, `Hello` or `Open`, a later rep stream's `Open` before the first's, a
    `Commit` before the first `Open` and a `Link` before `Commit`'s coins are refused without reaching the auditor, and so
    is every request after them, a `Ping` alone answered by the loop. The `Open` that leaves is the loop's own, at its
    stream's domain, whatever the prover's carried. The places past `Commit`, whose coins only a coin tree opens, are
    `Flock.Test.SessionPlace`'s, and the honest session end to end `flock-circuit`'s firewalled selftests'."""
    args = _args(zk, "pub-16.bin")
    a, b = _streams(args)[:2]
    S = Sessions(args)
    cases = {
        "a second Register": ([REGISTER, REGISTER], [10]),
        "a second Hello": ([REGISTER, hello(""), hello("")], [10, 1]),
        f"an Open of {b} before the Open of {a}": ([REGISTER, hello(""), open_(b)], [10, 1]),
        f"a second Open of {a}": ([REGISTER, hello(""), open_(a), open_(a)], [10, 1, 3]),
        f"a Commit before the Open of {a}": ([REGISTER, hello(""), COMMIT], [10, 1]),
        "a Link at opening 0, where its place is just after the step-0 coin commitment":
            ([REGISTER, hello(""), open_(a), LINK], [10, 1, 3]),
    }
    for why, (frames, left) in cases.items():
        s = S.lp.connect()
        got = [call(s, f) for f in frames]
        assert reason(got[-1]) == STOP + why, (why, got[-1][:80])
        assert all(r[:1] != b"\x03" for r in got[:-1]), (why, got)
        after = [call(s, f) for f in (ping(b"p"), REGISTER, hello(""), open_(a), LINK, COMMIT, ping(b"p"))]
        assert after[0] == after[-1] == b"\x01" and all(reason(r) == STOP + why for r in after[1:-1]), (why, after)
        s.close()
        sent = S.auditor.got[-1]
        assert [f[0] for f in sent] == left, (why, [f[0] for f in sent])
        if 3 in left:
            assert sent[2][:5 + len(a) + 4] == b"\x03" + _bytes(a.encode()) + _bytes(b"") and sent[2] != open_(a)
    S.lp.close()


def test_a_schedule_the_setup_refuses(junk):
    [line] = _schedule(junk, "null")
    assert line.startswith("refused "), line


@pytest.mark.slow
@pytest.mark.parametrize("name, public, tables", [("selftest/honest", "pub-16.bin", 1), ("m26", "pub-32.bin", 1),
                                                   ("j2", "pub-32.bin", 2), ("drawn", "pub-32.bin", 1)])
def test_the_schedule_is_the_recorded_rounds(zk, name, public, tables):
    """At a recorded session's statement and sample, the openings the loop asks for after the session stream's are the record's
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
def test_each_session_its_own_nonce_and_id(zk):
    """Each session's `Hello` leaves at a nonce of its own, and the prover receives an id of the loop's own for each in
    place of the auditor's answer, though the auditor answers both alike."""
    recorded, nonce, _ = _record(zk, "selftest/honest")
    auditor = Auditor(lambda f: {10: draw("null"), 1: _answer()}[f[0]])
    lp = Loop(auditor, _args(zk, "pub-16.bin"))
    names = []
    for _ in range(2):
        s = lp.connect()
        call(s, REGISTER)
        names.append(_name(call(s, hello(""))))
        s.close()
    texts = [c[1][5:].decode() for c in auditor.got]
    got = [NONCE.search(t)[1] for t in texts]
    assert got[0] != got[1] and texts == [recorded.replace(nonce, g) for g in got]
    assert names[0] != names[1] and _answer()[5:69].hex() not in names, names
    lp.close()


HM = hm96.SHA512


class CoinTree:
    """The verifier's coin tree, `coin-tree/hm96-sha512/v2` (`live/src/coin_tree.rs`'s `CoinTree`, checked by
    `Flock/Firewall/Coins.lean`): sampled at `Hello` under the prover's nonce `nonce` and a fresh session key, per stream and
    round a block of `spec["block"]` coins, `edit(stream, round, block)` choosing some first (`CoinTree::draw_with`). Every
    SHA-512 input is prefixed by the nonce; a block commits through its chain `T_i = H(CHAIN ‖ c_i ‖ T_{i+1})` from
    `T_block = H(TAIL)` into a keyed hm96 leaf, each stream's rounds are a subtree, and the streams' roots, padded to a power
    of two, are one more."""

    def __init__(self, spec: dict, nonce: bytes, edit):
        self.spec, self.nonce, self.key = spec, nonce, HM.fresh_key()
        self.rounds, streams = 1 << spec["rounds_log"], spec["streams"]
        self.blocks, self.tails, self.salts, self.trees = [], [], [], []
        for s, name in enumerate(streams):
            bs, ts, ss = [], [], []
            for k in range(self.rounds):
                b = [int.from_bytes(os.urandom(16), "little") for _ in range(spec["block"])]
                edit(name, k, b)
                t = [self.h(self.tag("tail"))]
                for c in reversed(b):
                    t.insert(0, self.h(self.tag("chain"), c.to_bytes(16, "little"), t[0]))
                bs.append(b)
                ts.append(t)
                ss.append(HM.fresh_salt())
            self.blocks.append(bs)
            self.tails.append(ts)
            self.salts.append(ss)
            self.trees.append(self.flat([self.leaf(s, k, ts[k][0], ss[k]) for k in range(self.rounds)]))
        self.width = 1 << (len(streams) - 1).bit_length()
        tops = [t[-1] for t in self.trees]
        self.top = self.flat(tops + [self.h(self.tag("pad"))] * (self.width - len(tops)))

    @staticmethod
    def tag(name: str) -> bytes:
        return f"verity/coin-tree/{name}/v2\0".encode()

    def h(self, *parts: bytes) -> bytes:
        return hashlib.sha512(self.nonce + b"".join(parts)).digest()

    def leaf(self, s: int, k: int, t0: bytes, salt: bytes) -> bytes:
        x = self.h(self.tag("round"), s.to_bytes(4, "big"), k.to_bytes(4, "big"), t0)
        b = bytes(p ^ q for p, q in zip(x, HM.mask(self.key, salt), strict=True))
        return self.h(HM.leaf_prefix(self.key), b, self.h(HM.salt_prefix, salt))

    def flat(self, leaves: list[bytes]) -> list[bytes]:
        """A tree's nodes, the leaves first and the root last."""
        t, at, n = list(leaves), 0, len(leaves)
        while n > 1:
            t += [self.h(t[at + 2 * i], t[at + 2 * i + 1]) for i in range(n // 2)]
            at, n = at + n, n // 2
        return t

    @staticmethod
    def path(t: list[bytes], n: int, j: int) -> list[bytes]:
        out, at = [], 0
        while n > 1:
            out.append(t[at + (j ^ 1)])
            at, n, j = at + n, n // 2, j // 2
        return out

    def answer(self) -> bytes:
        """`Hello`'s answer: the root, then the session key."""
        return _words(self.top[-1] + self.key)

    def opening(self, stream: str, k: int, n: int) -> bytes:
        """Round `k` of `stream`'s first `n` coins, `T_n`, the salt and the path: the stream's subtree, then the streams'."""
        s = self.spec["streams"].index(stream)
        path = self.path(self.trees[s], self.rounds, k) + self.path(self.top, self.width, s)
        return _words(b"".join(c.to_bytes(16, "little") for c in self.blocks[s][k][:n]) + self.tails[s][k][n]
                      + self.salts[s][k] + b"".join(path))


def _words(b: bytes) -> bytes:
    """`Resp::Coins` of the words of `b`."""
    return coins([b[i:i + 16] for i in range(0, len(b), 16)])


def _field(f: bytes, o: int) -> tuple[bytes, int]:
    (n,) = struct.unpack_from("<I", f, o)
    return f[o + 4:o + 4 + n], o + 4 + n


class TreeAuditor:
    """A scripted auditor that commits to its coins at `Hello` (`CoinTree`, `edit` choosing some) and answers the `Commit`
    and each `Round` with its stream's next opening."""

    def __init__(self, edit):
        self.edit, self.next, self.tree = edit, {}, None

    def opening(self, stream: str, n: int) -> bytes:
        k = self.next.get(stream, 0)
        self.next[stream] = k + 1
        return self.tree.opening(stream, k, n)

    def __call__(self, f: bytes) -> bytes:
        if f[0] == 1:
            h = json.loads(f[5:])
            self.tree = CoinTree(h["coins"], bytes.fromhex(h["coins"]["nonce"]), self.edit)
            self.links = h["link"]["points"] * h["link"]["m"]
            return self.tree.answer()
        if f[0] == 7:
            return self.opening("session", self.links)
        if f[0] == 4:
            stream, o = _field(f, 1)
            _, o = _field(f, o)
            return self.opening(stream.decode(), struct.unpack_from("<I", f, o)[0])
        return {10: draw("null")}.get(f[0], b"\x01")


def _domain(d: str) -> bytes:
    """A stream's domain frame (`Frames.domainFrame`), which its first round begins with."""
    b = d.encode()
    return b"\x01" + bytes(7) + struct.pack("<Q", len(b)) + b + bytes(-len(b) % 16)


def _place(lp: Loop, name: str) -> int:
    """The coins rep 0's next scheduled round squeezes (`place`)."""
    f = TcpFrame(lp)
    line = f.ask(f"place {name} 0 0")
    f.close()
    assert line.startswith("alg "), line
    return int(line.split()[1])


@pytest.mark.slow
@pytest.mark.parametrize("at, why", [((1, 0), STOP + "degenerate coin: zerocheck outer coin r_0 = 1"),
                                     ((1, 11), STOP + "degenerate coin: zerocheck outer coin r_11 = 1"),
                                     ((0, 5), None), ((2, 0), None)])
def test_an_outer_coin_of_one_stops_the_session_as_its_round_opens(zk, at, why):
    """A verifier that commits to a zerocheck outer coin `r_j = 1` (at the fixture's m = 25, rep 0's round 1 opens all twelve)
    stops the session where that round's coins open: they open the committed tree, the rep's Lean frame refuses them
    (`Frame.Masker.answer`), and the prover receives `Refused` in their place. Every request after it, a `Round`, a `Commit`
    or `Finish`, is refused for the same reason without reaching the auditor, which saw the schedule's rounds up to that
    one and nothing after. A coin `1` anywhere else (a skip coin of round 0, round 2's) is a coin like any other, which the
    prover receives once the frame took it. The prover is one whose device stopped, each round sent empty (the first its
    domain frame alone), so every round that leaves is the frame's scheduled one; its `Open` and `Link` are at the honest
    prover's places."""
    stream, (k1, j) = "circuit/rep0", at
    tree = TreeAuditor(lambda s, k, b: b.__setitem__(j, 1) if (s, k) == (stream, k1) else None)
    auditor = Auditor(tree)
    lp = Loop(auditor, _args(zk, "pub-16.bin"))
    s = lp.connect()
    call(s, REGISTER)
    name = _name(call(s, hello("")))
    assert call(s, open_(stream)) == b"\x01"
    [(_, _, root)] = level0(lp, [(25, _zeros(25))], name)
    f = TcpFrame(lp)
    assert isinstance(f.ask(f"rep {name} 0 0"), tuple)
    assert call(s, commit([("circuit", root)]))[:1] == b"\x02"
    assert call(s, LINK) == b"\x01"
    got = []
    for k in range(3):
        n = _place(lp, name)
        got.append((n, call(s, round_(stream, _domain("flock-circuit/fast100x2/rep0") if k == 0 else b"", n))))
    block = tree.tree.blocks[tree.tree.spec["streams"].index(stream)]
    stop = k1 if why else None
    for k, (n, r) in enumerate(got):
        if stop is None or k < stop:
            assert r == _words(b"".join(c.to_bytes(16, "little") for c in block[k][:n])), k
        else:
            assert reason(r) == why, k
    if why:
        assert reason(call(s, commit([("circuit", root)]))) == why
        assert reason(call(s, b"\x06")) == why
    s.close()
    f.close()
    lp.close()
    assert [x[0] for x in auditor.got[0]] == [10, 1, 3, 7, 2] + [4] * (3 if stop is None else stop + 1)
