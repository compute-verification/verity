"""The coin link on the challenger's host (`coin_link.py`) over loopback, in front of `serve.py` with a test key and its
coins stubbed (`stub_serve.py`): the real client (`RemoteChallenger`) takes its session through the link, whose record
exempts exactly the coins it opened and counts the commitment and the close; the tunnel goes straight to the peer's line
of the peers file (a fake `ssh` records its argv), one that names a jump host or a proxy is refused, and a tunnel that ends
first stops the link; and the gate's `run` starts the session's link once serve.py listens and stops it once serve.py
exits, its record sealed in the `end` entry."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from verity.core.protocols.sampled_proofs import remote as R

from test_serve import Auditor, finish, session, start, trusted

HERE = Path(__file__).resolve().parent
LINK = HERE.parent / "coin_link.py"
sys.path.insert(0, str(HERE.parent))
import auditor as A  # noqa: E402
import coin_link as L  # noqa: E402

#: a fast link for a test: 5 ms buckets, link-windows of 20, and a slack no loaded machine reaches, so a late tick never
#: faults one
FAST = ["--bucket-ms", "5", "--window", "20", "--slack-ms", "600000", "--cell-bytes", "1024", "--coin-cells", "2"]


def link(out: Path, port: int, *extra: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-I", str(LINK), "--out", str(out), "--port", str(port), *FAST, *extra],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_a_session_through_the_link_exempts_exactly_its_coins_and_counts_the_rest(tmp_path):
    """The client's session (hello, two coins of one family, close) through the link: serve.py ends it closed and the
    verifier takes its out/; the link's record exempts the two openings and nothing else, counts the commitment and
    the closed frame, every link-window complete, and the client's four frames counted as egress."""
    auditor = Auditor(tmp_path)
    proc, out, first = start(tmp_path, auditor=auditor)
    relay = link(out, int(first.split(":")[1]))
    listening = relay.stdout.readline()
    assert listening.startswith("listening 127.0.0.1:")
    session(listening.split()[1], roots=2)
    assert finish(proc)[0] == 0
    relay.send_signal(signal.SIGTERM)
    _, err = relay.communicate()
    assert relay.returncode == 0, err

    record = json.loads((out / L.SUMMARY).read_text())
    assert record["complete"] and record["tunnel"] is None and record["peer"] is None
    assert record["coins"]["frames"] == 2 and set(record["counted"]) == {"commitment", "closed"}
    assert record["requests"]["frames"] == 4 and [e["cut"] for e in record["connections"]] == [False]
    assert all(w["coins"] is None and w["coin-requests"] is None for w in record["windows"])
    commitment, roots = trusted(out, auditor)
    assert roots == {"positions": (bytes([0]) * 64, bytes([1]) * 64), "replay": ()}
    assert record["counted"]["commitment"]["bytes"] == len(R.HEADER.pack(0, 0)) + len(
        json.dumps({"type": "commitment", "commitment": commitment.to_json()}, sort_keys=True, separators=(",", ":")))


def test_the_tunnel_goes_straight_to_the_peer_and_its_end_stops_the_link(tmp_path, monkeypatch):
    """With `--peer`, the link listens first, then opens `ssh -N -R PORT:LISTEN` with no jump or proxy to the peer's
    line of the peers file; when that tunnel ends first, the link stops and says so. A peers line that names a jump host
    or a proxy, and an out/ with no challenger record, are refused before anything listens."""
    proc, out, first = start(tmp_path)
    port = int(first.split(":")[1])
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "ssh").write_text(f"#!{sys.executable}\nimport json, sys\n"
                              f"open({str(tmp_path / 'ssh.json')!r}, 'w').write(json.dumps(sys.argv[1:]))\nsys.exit(255)\n")
    (bin_ / "ssh").chmod(0o755)
    peers = tmp_path / "peers"
    peers.write_text("vy-nebius-1 research@10.0.0.2\nvy-jump -J bastion research@10.0.0.3\n"
                     "vy-proxy -o 'ProxyCommand nc %h %p' research@10.0.0.4\n")
    monkeypatch.setenv("PATH", str(bin_), prepend=os.pathsep)
    relay = link(out, port, "--peer", "vy-nebius-1", "--peers", str(peers))
    listen = int(relay.stdout.readline().split(":")[1])
    _, err = relay.communicate()
    argv = json.loads((tmp_path / "ssh.json").read_text())
    assert argv == [*L.TUNNEL[1:], "-R", f"127.0.0.1:{port}:127.0.0.1:{listen}", "research@10.0.0.2"]
    assert {"ProxyJump=none", "ProxyCommand=none"} <= set(argv)
    record = json.loads((out / L.SUMMARY).read_text())
    assert record["tunnel"] == {"rc": 255, "ended_first": True} and record["peer"] == "vy-nebius-1"
    proc.terminate()
    proc.wait()

    for peer in ("vy-jump", "vy-proxy"):
        relay = link(out, port, "--peer", peer, "--peers", str(peers))
        stdout, err = relay.communicate()
        assert relay.returncode == 2 and stdout == "" and "routes through another host" in err
    relay = link(tmp_path / "empty", port)
    stdout, err = relay.communicate()
    assert relay.returncode == 2 and stdout == "" and "no challenger record" in err
    assert L.jump(["-oProxyJump=b", "h"]) == "ProxyJump=b" and L.jump(["-W", "h:1"]) == "-W"
    assert L.jump(["-o", "BatchMode=yes", "research@h"]) is None


STUB_LINK = r"""
import json, signal, sys
from pathlib import Path
out = Path(sys.argv[1])
started = {"record": (out / "record.json").is_file(), "closed": (out / "challenger.json").is_file()}
signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
print("listening 127.0.0.1:1", flush=True)
signal.sigwait({signal.SIGTERM})
(out / "coin-link.json").write_text(json.dumps({"started": started, "stopped": {
    "closed": (out / "challenger.json").is_file()}}))
"""


@pytest.fixture
def host(tmp_path) -> A.Host:
    (tmp_path / "var" / "sessions").mkdir(parents=True)
    (tmp_path / "peers").write_text("vy-challenger-1 research@10.0.0.1\nvy-nebius-1 research@10.0.0.2\n")
    (tmp_path / "coin-ports").write_text("vy-nebius-1 7871-7880\n")
    (tmp_path / "pins.jsonl").write_text(A.pin_line("c" * 40, "d" * 40, "2026-10-20T09:00:00Z") + "\n")
    return A.Host(root=tmp_path / "var", peers=tmp_path / "peers", pins=tmp_path / "pins.jsonl")


def test_the_gates_run_starts_the_link_once_serve_listens_and_stops_it_once_serve_exits(host, tmp_path, monkeypatch):
    """`run` without an argv of its own serves the session through `coin_link_argv`'s link, from the open's port on its
    peer: the link starts after serve.py has signed its record and listens, gets SIGTERM only once serve.py's session
    is closed, and its record and log are sealed in the end entry's files. A link that refuses to start, or ends while
    serve.py still waits for its client (its tunnel ended), ends the session at once, saying so."""
    from test_auditor import P, SECRET, T1, open_window
    monkeypatch.setenv("COINS_LOG", str(tmp_path / "coins.log"))
    signer = A.key_signer(host, P, workers=1)
    session = json.loads(open_window(host, signer)[1])["session"]
    body = A.sessions_of(A.read_ledger(host.ledger))[session]["open"]["body"]
    out = host.sessions / session / "out"
    real = A.coin_link_argv(host, session, body)
    assert real[2:] == [str(LINK), "--out", str(out), "--port", str(body["port"]), "--peer", "vy-nebius-1",
                        "--peers", str(host.peers), "--minutes", "121"] and body["port"] == 7871

    real_serve = A.serve_argv

    def serve_argv(host_, session_, body_):
        argv = real_serve(host_, session_, body_)
        argv[argv.index("--port") + 1] = "0"
        return [sys.executable, str(HERE / "stub_serve.py"), *argv[2:], "--params", "4,4", "--keygen-workers", "1"]

    monkeypatch.setattr(A, "serve_argv", serve_argv)
    monkeypatch.setattr(A, "coin_link_argv", lambda h, s, b: [sys.executable, "-c", STUB_LINK, str(h.sessions / s / "out")])

    def client(proc, seconds):
        address = (out / "stdout.log").read_text().split()[1]
        R.RemoteChallenger.connect(address, SECRET).close()
        return proc.wait()

    end = A.run(host, session, signer=signer, pinned="c" * 40, now=lambda: T1, wait=client)
    assert end["body"]["rc"] == 0 and end["body"]["closed"]
    assert json.loads((out / "coin-link.json").read_text()) == {"started": {"record": True, "closed": False},
                                                                "stopped": {"closed": True}}
    files = end["body"]["files"]
    assert {"coin-link.json", "coin-link.log"} <= set(files)
    assert files["coin-link.json"] == hashlib.sha512((out / "coin-link.json").read_bytes()).hexdigest()
    assert (out / "coin-link.log").read_text() == "listening 127.0.0.1:1\n"

    refuses = "import sys; print('the coin link refused to start: no', file=sys.stderr); sys.exit(2)"
    monkeypatch.setattr(A, "coin_link_argv", lambda h, s, b: [sys.executable, "-c", refuses])
    session = json.loads(open_window(host, signer, token="cd34", commit="cd" * 64)[1])["session"]
    end = A.run(host, session, signer=signer, pinned="c" * 40, now=lambda: T1, wait=client)
    assert not end["body"]["closed"] and "the coin link didn't start (exit 2" in end["body"]["ended"]
    assert "refused to start: no" in (host.sessions / session / "out" / "coin-link.log").read_text()

    ends = "print('listening 127.0.0.1:1', flush=True)"
    monkeypatch.setattr(A, "coin_link_argv", lambda h, s, b: [sys.executable, "-c", ends])
    session = json.loads(open_window(host, signer, token="ef56", commit="ef" * 64)[1])["session"]
    end = A.run(host, session, signer=signer, pinned="c" * 40, now=lambda: T1)
    assert not end["body"]["closed"] and "the coin link ended first (exit 0" in end["body"]["ended"]


def test_a_coin_link_that_never_listens_is_killed_and_ends_the_session(host, tmp_path, monkeypatch):
    """A link that prints nothing, so never listens, holds the session no longer than serve.py's start may take
    (`START_SECONDS`): it is killed and the session ends, saying so, instead of waiting for the window's deadline. The
    link here waits for a signal it never gets, so the bound is 0, and only its expiry, not its length, is tested."""
    from test_auditor import P, T1, open_window
    monkeypatch.setenv("COINS_LOG", str(tmp_path / "coins.log"))
    signer = A.key_signer(host, P, workers=1)
    session = json.loads(open_window(host, signer)[1])["session"]
    real_serve = A.serve_argv

    def serve_argv(host_, session_, body_):
        argv = real_serve(host_, session_, body_)
        argv[argv.index("--port") + 1] = "0"
        return [sys.executable, str(HERE / "stub_serve.py"), *argv[2:], "--params", "4,4", "--keygen-workers", "1"]

    monkeypatch.setattr(A, "serve_argv", serve_argv)
    monkeypatch.setattr(A, "coin_link_argv", lambda h, s, b: [sys.executable, "-c", "import signal; signal.pause()"])
    monkeypatch.setattr(A, "START_SECONDS", 0)
    end = A.run(host, session, signer=signer, pinned="c" * 40, now=lambda: T1)
    assert not end["body"]["closed"] and end["body"]["rc"] != 0
    assert end["body"]["ended"].startswith("the coin link didn't listen within 0 s, so it was killed")
    assert (host.sessions / session / "out" / "coin-link.log").read_text() == ""
