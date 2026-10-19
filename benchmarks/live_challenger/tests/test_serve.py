"""The challenger's host process (`serve.py`) over loopback with a test key: it signs and writes its record before it
listens, serves one session and publishes nothing secret; the verifier checks the session from the process's `out/`
alone.  A second session signs at the key's next leaf, a key inside `out/` is refused, a broken sequence ends the
session with exit 1, and with `--peer` the tunnel opens once the challenger listens and a tunnel that ends first ends it."""

from __future__ import annotations

import itertools
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

from verity.core.protocols.sampled_proofs import challenger as C
from verity.core.protocols.sampled_proofs import remote as R
from verity.ml.crypto import lms

SERVE = Path(__file__).resolve().parents[1] / "serve.py"
FAMILIES = [["positions", 2], ["replay", 1]]
HOST = "vy-nebius-cpu-1"


def lms_verify(key: C.ChallengerKey, message: bytes, signature: bytes) -> bool:
    p = lms.Params(64, key.W, key.H)
    return lms.verify(p, key.public, message, lms.Signature.decode(p, signature))


def start(tmp_path: Path, out: str = "out", key: Path | None = None, *extra: str, env=None) -> tuple:
    out_dir = tmp_path / out
    proc = subprocess.Popen(
        [sys.executable, str(SERVE), "--out", str(out_dir), "--port", "0", "--scheme", "verity/test/coins/v1",
         "--families", json.dumps(FAMILIES), "--key", str(key or tmp_path / "key"), "--params", "4,2",
         "--keygen-workers", "1", "--host", HOST, *extra],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    return proc, out_dir, proc.stdout.readline()


def finish(proc: subprocess.Popen) -> tuple[int, str]:
    _, err = proc.communicate()
    return proc.returncode, err


def session(address: str, roots: int = 1) -> None:
    with R.RemoteChallenger.connect(address) as client:
        record = C.LiveRecord(client, "positions", itertools.count().__next__)
        for i in range(roots):
            record.record(bytes([i]) * 64)


def test_the_challenger_signs_before_it_listens_and_the_verifier_checks_its_out_alone(tmp_path):
    proc, out, first = start(tmp_path)
    assert first.startswith("listening 127.0.0.1:")
    assert {p.name for p in out.iterdir()} >= {R.RECORD, R.KEY, R.RELEASES}
    session(first.split()[1], roots=2)
    assert finish(proc)[0] == 0

    key = C.ChallengerKey.from_json(json.loads((out / R.KEY).read_text()))
    assert key.public == (tmp_path / "key" / "public").read_bytes()
    commitment, roots = R.trusted_session(out, key, lms_verify)
    assert roots == {"positions": (bytes([0]) * 64, bytes([1]) * 64), "replay": ()}
    summary = json.loads((out / "challenger.json").read_text())
    assert summary["released"] == {"positions": 2, "replay": 0}
    assert summary["ended"] == "closed by the client" and summary["commitment"] == commitment.to_json()
    seed = (tmp_path / "key" / "seed").read_bytes()
    assert all(seed not in p.read_bytes() and seed.hex() not in p.read_text() for p in out.iterdir())


def test_a_second_session_signs_at_the_keys_next_leaf(tmp_path):
    leaves = []
    for out in ("one", "two"):
        proc, out_dir, first = start(tmp_path, out)
        session(first.split()[1])
        assert finish(proc)[0] == 0
        key = C.ChallengerKey.from_json(json.loads((out_dir / R.KEY).read_text()))
        R.trusted_session(out_dir, key, lms_verify)
        leaves.append(int(json.loads((out_dir / R.RECORD).read_text())["signature"][:8], 16))
    assert leaves == [0, 1]


def test_a_key_inside_out_is_refused(tmp_path):
    proc, out, first = start(tmp_path, "out", tmp_path / "out" / "key")
    code, err = finish(proc)
    assert (code, first) == (2, "") and "inside its published --out" in err
    assert not (tmp_path / "out" / "key").exists()


def test_a_broken_sequence_ends_the_session_with_exit_1(tmp_path):
    proc, out, first = start(tmp_path)
    host, port = first.split()[1].split(":")
    head = json.dumps({"type": "open", "label": "positions", "index": 0, "after": "00" * 64}).encode()
    with socket.create_connection((host, int(port))) as s:
        s.sendall(R.HEADER.pack(len(head), 0) + head)
        assert s.recv(1 << 16)
    code, _ = finish(proc)
    assert code == 1
    assert json.loads((out / "challenger.json").read_text())["ended"].startswith("a client's frames are hello")


def fake_ssh(tmp_path: Path, body: str) -> dict:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {tmp_path / "ssh.args"}\n{body}\n')
    ssh.chmod(0o755)
    peers = tmp_path / "peers"
    peers.write_text("# name, then ssh's options and destination\nvy-nebius-2 -i /k -o IdentitiesOnly=yes research@10.0.0.2\n")
    return {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}


def test_with_a_peer_the_tunnel_opens_once_the_challenger_listens(tmp_path):
    env = fake_ssh(tmp_path, "exec tail -f /dev/null")
    proc, out, first = start(tmp_path, "out", None, "--peer", "vy-nebius-2", "--peers", str(tmp_path / "peers"),
                             env=env)
    port = first.split()[1].split(":")[1]
    session(first.split()[1])
    assert finish(proc)[0] == 0
    args = (tmp_path / "ssh.args").read_text().splitlines()
    assert args[args.index("-R") + 1] == f"127.0.0.1:{port}:127.0.0.1:{port}"
    assert args[-6:] == ["-i", "/k", "-o", "IdentitiesOnly=yes", "research@10.0.0.2", "sleep 86400"]
    assert json.loads((out / "address.json").read_text())["peer"] == "vy-nebius-2"


def test_a_tunnel_that_ends_before_a_client_connects_ends_the_session(tmp_path):
    env = fake_ssh(tmp_path, "exit 255")
    proc, out, first = start(tmp_path, "out", None, "--peer", "vy-nebius-2", "--peers", str(tmp_path / "peers"),
                             env=env)
    assert first.startswith("listening")
    code, _ = finish(proc)
    assert code == 1
    assert json.loads((out / "challenger.json").read_text())["ended"] == (
        "the tunnel to vy-nebius-2 ended before a client connected")


def test_a_peer_missing_from_the_peers_file_is_refused(tmp_path):
    fake_ssh(tmp_path, "exit 0")
    proc, out, first = start(tmp_path, "out", None, "--peer", "vy-nebius-9", "--peers", str(tmp_path / "peers"))
    code, err = finish(proc)
    assert (code, first) == (2, "") and "no peer vy-nebius-9" in err
