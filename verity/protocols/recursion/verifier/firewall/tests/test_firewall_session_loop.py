"""`core-firewall session` (`Core.Firewall.Loop`): the release side around Flock's session loop, on the ZK server's state
directory, with a scripted auditor on the far side and files no statement sets up from (the loop starts; a `Hello` would
be refused at its setup). A session's `Ping`s are the loop's to answer until its first other request, which is admitted
or refused on the record before anything of the session reaches the auditor; a refused session reaches nothing of it.
An admitted session's proofs are held, and a stop at its `Finish`, or a cut, is recorded. Each request of an admitted
session leaves at most once, at its place, and once the loop refuses one nothing more of the session reaches the auditor,
its `Ping`s the loop's to answer to the end. Each record is the one
`verity_flock.firewall` writes for the same ops, byte for byte, and the slot orders two sessions. A session that goes,
released at its time through the whole loop, is the end-to-end check (`flock-circuit prove --firewall-state`)."""

from __future__ import annotations

import fcntl
import json
import shutil
import socket
import struct
import subprocess
import threading
from pathlib import Path

import pytest

import _release as R
import verity_difftest as D
from verity_flock import firewall as FW

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

STOP = "PROVER-STOPPED: "
REGISTER, FINISH, OK = b"\x0a", b"\x06", b"\x01"


def _bytes(b: bytes) -> bytes:
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


def call(s: socket.socket, f: bytes) -> bytes | None:
    s.sendall(_bytes(f))
    return _read(s)


def ping(b: bytes) -> bytes:
    return b"\x09" + _bytes(b)


PROOF = b"\x05" + _bytes(b"t") + struct.pack("<I", 0) + _bytes(b"proof")


def refusal(resp: bytes) -> str:
    assert resp[:1] == b"\x03", resp[:40]
    return resp[5:].decode()


def verdict(resp: bytes) -> dict:
    assert resp[:1] == b"\x04", resp[:40]
    return json.loads(resp[5:].decode())


class Auditor:
    """A scripted auditor: each connection's frames, `Register` answered with a sample of none, anything else `Ok`."""

    def __init__(self):
        self.got: list[list[bytes]] = []
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(8)
        self.addr = f"127.0.0.1:{self.srv.getsockname()[1]}"
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            c, _ = self.srv.accept()
            mine: list[bytes] = []
            self.got.append(mine)
            threading.Thread(target=self._conn, args=(c, mine), daemon=True).start()

    def _conn(self, c, mine):
        while (f := _read(c)) is not None:
            mine.append(f)
            c.sendall(_bytes(b"\x06" + _bytes(b"null") if f[:1] == REGISTER else OK))
        c.close()


@pytest.fixture(scope="module")
def exe():
    return D.build(R.PKG, "core-firewall")


class Loop:
    """A running `core-firewall session` for outer sessions of `session` at `part`, on the state directory `state`, with
    the public file `public` (by default junk) and the further `flags`."""

    def __init__(self, exe: Path, state: Path, session: str, part: str, auditor: Auditor, junk: Path,
                 public: Path | None = None, flags: tuple[str, ...] = ()):
        (junk / "c").write_bytes(b"x")
        if public is None:
            public = junk / "p"
            public.write_bytes(b"x")
        self.p = subprocess.Popen([str(exe), "session", "--state", str(state), "--session", session, "--part", part,
                                   "--verifier", auditor.addr, "--circuit", str(junk / "c"), "--public", str(public),
                                   *flags],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        line = self.p.stdout.readline().decode()
        assert line.startswith("LISTENING "), (line, self.p.stderr.read().decode() if self.p.poll() is not None else "")
        host, port = line.split()[1].split(":")
        self.addr = (host, int(port))
        assert self.p.stdout.readline().decode().startswith("FRAMES ")

    def connect(self) -> socket.socket:
        return socket.create_connection(self.addr)

    def close(self):
        self.p.stdin.close()
        assert self.p.wait() == 0


def slot_left(state: Path) -> None:
    """Returns once no session holds the slot: what the loop records of a session comes before it leaves the slot."""
    with open(state / "slot.lock", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        fcntl.flock(f, fcntl.LOCK_UN)


def record(state: Path) -> bytes:
    return (state / "record.jsonl").read_bytes()


def test_usage(exe):
    p = subprocess.run([str(exe), "session", "--verifier", "127.0.0.1:1"], input=b"", capture_output=True, check=False)
    assert p.returncode == 2 and p.stderr.decode().startswith("usage: core-firewall session --state DIR")


def test_a_refused_session_reaches_nothing_of_the_auditor(exe, tmp_path):
    """Its inner session has not finished: its `Ping` is the loop's to answer, its first other request is refused with the
    ZK server's why, its `Finish` answered with a verdict that does not accept, and the auditor is never connected to. The
    record is Python's for the same ops: open, slot, refused."""
    fw = FW.Firewall(tmp_path / "lean")
    s = fw.open("s", ["L0"])
    auditor = Auditor()
    lp = Loop(exe, tmp_path / "lean", s, "L0", auditor, tmp_path)
    c = lp.connect()
    assert call(c, ping(b"p")) == OK
    why = f"FIREWALL-REFUSED: recursive session {s}: its inner session has not finished"
    assert refusal(call(c, REGISTER)) == why
    assert verdict(call(c, FINISH)) == {"accepted": False, "why": why}
    c.close()
    lp.close()
    assert auditor.got == []
    py = FW.Firewall(tmp_path / "py")
    py.open("s", ["L0"])
    with py.slot(s, "L0"), pytest.raises(FW.Refused):
        py.admit(s, "L0")
    assert record(tmp_path / "lean") == record(tmp_path / "py")


def test_a_stop_at_finish_holds_the_proofs_and_cuts_the_auditor(exe, tmp_path):
    """Admitted, its `Register` leaves, but its proof is dropped (answered `Ok` by the loop) and its `Finish`, which the
    loop's own check refuses (before the step-0 coin commitment), is the session's stop: neither reaches the auditor, the
    prover is told in a verdict that does not accept, and the record is Python's for open, finish, slot, admit, stop."""
    fw = FW.Firewall(tmp_path / "lean")
    s = fw.open("s", ["L0", "L1"])
    fw.finish(s)
    auditor = Auditor()
    lp = Loop(exe, tmp_path / "lean", s, "L0", auditor, tmp_path)
    c = lp.connect()
    assert call(c, REGISTER)[:1] == b"\x06"
    assert call(c, PROOF) == OK
    why = STOP + "a Finish before the step-0 coin commitment"
    assert verdict(call(c, FINISH)) == {"accepted": False,
                                        "why": f"FIREWALL-REFUSED: {why}: recursive session {s} stopped at L0"}
    assert _read(c) is None
    c.close()
    lp.close()
    assert auditor.got == [[REGISTER]]
    (times,) = [json.loads(ln.split(" ", 1)[1]) for ln in lp.p.stderr.read().decode().splitlines() if ln.startswith("FIREWALL-TIMES ")]
    assert (times.pop("ready_ms") > 0, times) == (True, {"session": s, "part": "L0", "release_ms": None, "outcome": "stop"})
    py = FW.Firewall(tmp_path / "py")
    py.finish(py.open("s", ["L0", "L1"]))
    with py.slot(s, "L0"):
        py.admit(s, "L0")
        py.stop(s, "L0", why)
    assert record(tmp_path / "lean") == record(tmp_path / "py")


def test_a_session_cut_before_its_finish_is_its_stop(exe, tmp_path):
    fw = FW.Firewall(tmp_path / "lean")
    s = fw.open("s", ["L0"])
    fw.finish(s)
    auditor = Auditor()
    lp = Loop(exe, tmp_path / "lean", s, "L0", auditor, tmp_path)
    c = lp.connect()
    assert call(c, REGISTER)[:1] == b"\x06"
    c.close()
    slot_left(tmp_path / "lean")
    lp.close()
    py = FW.Firewall(tmp_path / "py")
    py.finish(py.open("s", ["L0"]))
    with py.slot(s, "L0"):
        py.admit(s, "L0")
        py.stop(s, "L0", "the prover ended the session before its Finish")
    assert record(tmp_path / "lean") == record(tmp_path / "py")


def test_the_loop_admits_as_the_end_to_end_record(exe, tmp_path):
    """The end-to-end run's three sessions that ended at their admission (`fixtures/release-e2e.jsonl`, frozen from Rust's
    release side): r0000 at L0 again after its release, a stop at L1, never opened, and a refusal; r0000 at L1, refused;
    r0003, never finished, refused. Each starts on the run's record before it and leaves the run's next lines, byte for byte.
    The run's releases and its missed time need a prover's proofs (`test_firewall_release_agreement` runs them through the
    loop's stages)."""
    lines = R.e2e_record()
    for s, p, start, end in [("r0000", "L0", 11, 14), ("r0000", "L1", 14, 16), ("r0003", "L0", 16, 18)]:
        state = tmp_path / f"{s}-{p}"
        state.mkdir()
        (state / "record.jsonl").write_text("".join(lines[:start]))
        auditor = Auditor()
        lp = Loop(exe, state, s, p, auditor, tmp_path)
        c = lp.connect()
        assert refusal(call(c, REGISTER)).startswith("FIREWALL-REFUSED: ")
        c.close()
        slot_left(state)
        lp.close()
        assert auditor.got == []
        assert record(state).decode() == "".join(lines[:end]), (s, p)


def test_the_slot_orders_two_sessions(exe, tmp_path):
    """A second session asks while the first holds the slot: it records `wait` and waits, and once the first is cut (its
    stop) it takes the slot and is refused, since the recursive session has stopped. Python reads the record."""
    state = tmp_path / "lean"
    fw = FW.Firewall(state)
    s = fw.open("s", ["L0"])
    fw.finish(s)
    lp = Loop(exe, state, s, "L0", Auditor(), tmp_path)
    a = lp.connect()
    assert call(a, REGISTER)[:1] == b"\x06"
    b = lp.connect()
    b.sendall(_bytes(REGISTER))
    assert lp.p.stderr.readline().decode() == f"firewall: {s} at L0 waits for the outer slot\n"
    a.close()
    assert refusal(_read(b)) == f"FIREWALL-REFUSED: recursive session {s} stopped at L0: the firewall serves nothing more of it"
    b.close()
    lp.close()
    got = [{k: e[k] for k in ("event", "part", "why") if k in e} for e in FW.Firewall(state).record()]
    assert got == [{"event": "open"}, {"event": "finish", "part": "inner"}, {"event": "slot", "part": "L0"},
                   {"event": "wait", "part": "L0"}, {"event": "stop", "part": "L0", "why": "the prover ended the session before its Finish"},
                   {"event": "slot", "part": "L0"}, {"event": "refused", "part": "L0"}]
    assert FW.Firewall(state).stopped(s)["part"] == "L0" and FW.Firewall(state).failures("s") == 1


HELLO = b"\x01" + _bytes(b"{}")
COMMIT = b"\x07" + _bytes(b"") + struct.pack("<I", 0) + struct.pack("<I", 0)
EARLY_ROUND = b"\x04" + _bytes(b"st") + _bytes(b"c") + struct.pack("<I", 1)
LINK = b"\x02" + _bytes(b"\0" * 32)
OPEN = b"\x03" + _bytes(b"t/rep0") + _bytes(b"") + _bytes(b"label") + b"\0" * 8
#: every kind of request a prover may send after a stop, `Ping`s among them
AFTER = [ping(b"p"), ping(b"\0" * 64), REGISTER, HELLO, LINK, OPEN, COMMIT, EARLY_ROUND, PROOF, b"\x63", ping(b"p")]


def _stopped(exe, tmp_path, frames: list[bytes], after: list[bytes]):
    """An admitted session whose last request of `frames` the loop refuses, then `after` and its `Finish`: the refusal's
    reason, the answers to `after`, the verdict, every frame the auditor received, and the session's `s`."""
    fw = FW.Firewall(tmp_path / "lean")
    s = fw.open("s", ["L0", "L1"])
    fw.finish(s)
    auditor = Auditor()
    lp = Loop(exe, tmp_path / "lean", s, "L0", auditor, tmp_path)
    c = lp.connect()
    got = [call(c, f) for f in frames]
    assert all(r[:1] != b"\x03" for r in got[:-1]), got
    why = refusal(got[-1])
    answers = [call(c, f) for f in after]
    v = verdict(call(c, FINISH))
    assert _read(c) is None
    c.close()
    lp.close()
    return why, answers, v, [f for conn in auditor.got for f in conn], s


@pytest.mark.parametrize("frames, why", [
    ([REGISTER, EARLY_ROUND], STOP + "coins before the step-0 coin commitment"),
    ([REGISTER, HELLO], STOP + "the Lean Hello: setup: "),
])
def test_nothing_reaches_the_auditor_after_the_loops_refusal(exe, tmp_path, frames, why):
    """Once the loop refuses an admitted session (here a round before the step-0 coin commitment, or a `Hello` its setup
    refuses), every request after it is refused for the same reason without reaching the auditor, `Register`, `Hello`,
    `Link`, `Open` and a `Proof` among them, and each `Ping` is answered `Ok` by the loop: the auditor receives the
    session's `Register` and nothing more, though the prover knows why it stopped. Its `Finish` is the stop, recorded as
    Python records it."""
    got, answers, v, sent, s = _stopped(exe, tmp_path, frames, AFTER)
    assert got.startswith(why), got
    assert [a for f, a in zip(AFTER, answers) if f[:1] == b"\x09"] == [OK] * 3
    assert [refusal(a) for f, a in zip(AFTER, answers) if f[:1] != b"\x09"] == [got] * (len(AFTER) - 3)
    assert v == {"accepted": False, "why": f"FIREWALL-REFUSED: {got}: recursive session {s} stopped at L0"}
    assert sent == [REGISTER]
    py = FW.Firewall(tmp_path / "py")
    py.finish(py.open("s", ["L0", "L1"]))
    with py.slot(s, "L0"):
        py.admit(s, "L0")
        py.stop(s, "L0", got)
    assert record(tmp_path / "lean") == record(tmp_path / "py")


@pytest.mark.parametrize("frames, why, left", [
    ([REGISTER, REGISTER], "a second Register", [REGISTER]),
    ([ping(b"p"), REGISTER, ping(b"p"), ping(b"p"), REGISTER], "a second Register", [REGISTER]),
    ([HELLO], "a Hello before the Register", []),
    ([REGISTER, LINK], "a Link before the session's Hello", [REGISTER]),
    ([REGISTER, OPEN], "an Open of t/rep0, which is none of the session's rep streams", [REGISTER]),
    ([REGISTER, COMMIT], "coins before the step-0 coin commitment", [REGISTER]),
])
def test_a_request_out_of_its_place_never_leaves(exe, tmp_path, frames, why, left):
    """Before any stop each request leaves once, at its place in the honest prover's order, and a `Ping` never does: a
    second `Register`, a `Hello` before the `Register`, and a `Link` or `Open` before the `Hello` are refused before they
    reach the auditor, and stop the session. `test_lean_session.py` takes the places past `Hello` at a statement the loop
    sets up, `Flock.Test.SessionPlace` those past `Commit`, and `flock-circuit`'s firewalled selftests the honest session
    accepted."""
    got, answers, v, sent, _ = _stopped(exe, tmp_path, frames, AFTER)
    assert got == STOP + why, got
    assert all(a == OK if f[:1] == b"\x09" else refusal(a) == got for f, a in zip(AFTER, answers))
    assert not v["accepted"] and sent == left
