"""`core-firewall session` (`Core.Firewall.Loop`): the release side around C-Flock's session loop, on the ZK server's state
directory, with a scripted auditor on the far side and files no statement sets up from (the loop starts; a `Hello` would
be refused at its setup). A session's `Ping`s are the loop's to answer until its first other request, which is admitted
or refused on the record before anything of the session reaches the auditor; a refused session reaches nothing of it.
An admitted session's proofs are held, and a stop at its `Finish`, or a cut, is recorded. Each record is the one
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
    """A running `core-firewall session` for outer sessions of `session` at `part`, on the state directory `state`."""

    def __init__(self, exe: Path, state: Path, session: str, part: str, auditor: Auditor, junk: Path):
        (junk / "c").write_bytes(b"x")
        (junk / "p").write_bytes(b"x")
        self.p = subprocess.Popen([str(exe), "session", "--state", str(state), "--session", session, "--part", part,
                                   "--verifier", auditor.addr, "--circuit", str(junk / "c"), "--public", str(junk / "p")],
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
