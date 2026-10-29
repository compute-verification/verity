"""The challenger's host (`challenger_party.py`): the verifier's `a5_coins.Loopback` opens and finishes a session on it
over loopback TCP, and each refused op (a second open while one is open, a root opened twice, a finish with nothing
open, a malformed root, an unknown op) reaches the verifier as a `CoinError` naming its reason; `Host` serves it on a
thread from `start` until `stop`. The host is a stand-in for `a5_coins.Local` (no serve.py), so only the wire is
tested."""

from __future__ import annotations

import socket
import sys
import threading
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import a5_coins as AC  # noqa: E402
import challenger_party as CP  # noqa: E402

ROOT_A, ROOT_B = "ab" * 64, "cd" * 64


class FakeLocal:
    def __init__(self, root: Path, main: list[str]) -> None:
        self.root, self.pin, self.session, self.opened = root, main[0], None, []

    def open(self, root: str, layout: list, minutes: int, scheme: str) -> tuple[str, str, str]:
        self.opened.append((root, layout, minutes, scheme))
        self.session = f"s{len(self.opened)}"
        return "127.0.0.1:4242", "5e" * 32, "c0" * 64

    def finish(self, dest: Path) -> tuple[Path, Path, Path, AC.Ch.ChallengerKey]:
        return dest / "session", dest / "windows.jsonl", dest / "pins.jsonl", AC.Ch.ChallengerKey(4, 5, b"\x07" * 80)


@pytest.fixture
def party(tmp_path, monkeypatch):
    monkeypatch.setattr(CP.AC, "Local", FakeLocal)
    p = CP.Party(tmp_path / "host", ["f" * 40])
    srv = socket.create_server(("127.0.0.1", 0))
    address = "%s:%d" % srv.getsockname()[:2]
    t = threading.Thread(target=CP.serve, args=(p, srv))
    t.start()
    yield p, address
    AC.Loopback(address)._ask({"op": "stop"})
    t.join()


def test_the_verifier_opens_and_finishes_a_session_on_the_challenger_party(party, tmp_path):
    p, address = party
    lb = AC.Loopback(address)
    assert lb.open(ROOT_A, [["work", 3]]) == ("127.0.0.1:4242", "5e" * 32, "c0" * 64)
    assert lb.session == "s1" and p.local.opened == [(ROOT_A, [["work", 3]], AC.MINUTES, AC.SCHEME)]
    session, ledger, pins, key = lb.finish(tmp_path / "dest")
    assert (session, ledger, pins) == (tmp_path / "dest" / "session", tmp_path / "dest" / "windows.jsonl",
                                       tmp_path / "dest" / "pins.jsonl")
    assert (key.W, key.H, key.public) == (4, 5, b"\x07" * 80)
    lb.open(ROOT_B, [["work", 1]])
    assert p.local.session == "s2"


def test_each_refused_op_reaches_the_verifier_by_its_reason(party, tmp_path):
    _, address = party
    lb = AC.Loopback(address)
    with pytest.raises(AC.Ch.CoinError, match="no-session"):
        lb.finish(tmp_path / "dest")
    with pytest.raises(AC.Ch.CoinError, match="root: 128 lowercase hex"):
        lb.open("AB" * 64, [])
    lb.open(ROOT_A, [])
    with pytest.raises(AC.Ch.CoinError, match="session-open"):
        lb.open(ROOT_B, [])
    lb.finish(tmp_path / "dest")
    with pytest.raises(AC.Ch.CoinError, match="root-opened"):
        lb.open(ROOT_A, [])
    with pytest.raises(AC.Ch.CoinError, match="unknown-op: fetch"):
        lb._ask({"op": "fetch"})
    with pytest.raises(AC.Ch.CoinError, match="a request is a JSON object"):
        lb._ask({"op": 3})


def test_a_host_root_is_made_once(tmp_path, monkeypatch):
    monkeypatch.setattr(CP.AC, "Local", FakeLocal)
    (tmp_path / "host").mkdir()
    with pytest.raises(SystemExit, match="exists"):
        CP.Party(tmp_path / "host", ["f" * 40])


def test_the_host_serves_from_start_until_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(CP.AC, "Local", FakeLocal)
    host = CP.Host(tmp_path / "host", ["f" * 40])
    at = host.start()
    assert at["pin"] == "f" * 40 and at["address"].startswith("127.0.0.1:")
    assert AC.Loopback(at["address"]).open(ROOT_A, [["work", 1]])[2] == "c0" * 64
    host.stop()
    assert not host.thread.is_alive()
    ip, port = at["address"].rsplit(":", 1)
    with pytest.raises(OSError):
        socket.create_connection((ip, int(port)))
