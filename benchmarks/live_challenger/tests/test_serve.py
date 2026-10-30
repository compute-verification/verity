"""The challenger's host process (`serve.py`) over loopback with a test key and the Lean verifier's coins stubbed
(`stub_serve.py`): it samples every coin and salt from those coins, signs and writes its record before it listens, serves one
session, signs its release log when it ends and publishes nothing secret; the verifier checks the session from the
process's `out/` and the auditor's signed ledger alone.  A Lean that can't answer starts no session, a second session
signs at the key's next leaves, a session id the key has signed for is refused, a key inside `out/` is refused, a refused
hello is logged and the next connection taken, a broken sequence after the hello ends the session with exit 1, signed as
such, a salt opens only after `--root`, and with `--peer` (development sessions alone: a window's, with `--root`, is
refused it) the tunnel opens once the challenger listens and a tunnel that ends first ends it."""

from __future__ import annotations

import fcntl
import hashlib
import itertools
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from verity.core.protocols.sampled_proofs import challenger as C
from verity.core.protocols.sampled_proofs import remote as R
from verity.ml.crypto import lms

SERVE = Path(__file__).resolve().parents[1] / "serve.py"
STUB = Path(__file__).resolve().parent / "stub_serve.py"
FAMILIES = [["positions", 2], ["replay", 1]]
HOST = "vy-nebius-cpu-1"
#: the test key: eight leaves, enough for two sessions and the auditor's open and end of each
PARAMS = lms.Params(64, 4, 3)


#: the test key: eight leaves, enough for two sessions and the auditor's open and end of each
PARAMS = lms.Params(64, 4, 3)


def lms_verify(key: C.ChallengerKey, message: bytes, signature: bytes) -> bool:
    p = lms.Params(64, key.W, key.H)
    return lms.verify(p, key.public, message, lms.Signature.decode(p, signature))


class Auditor:
    """The auditor's ledger beside the runs in `tmp_path`, under the key they sign with, which it makes first (its seed and
    leaf state in `key/`, which serve.py then reads): each entry signed through the key's leaf state, as the host's ledger
    signs, a session's open before serve.py's record and its end after serve.py's close."""

    def __init__(self, tmp_path: Path) -> None:
        directory = tmp_path / "key"
        directory.mkdir(mode=0o700)
        seed = bytes([5]) * PARAMS.N
        fd = os.open(directory / "seed", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.write(fd, seed)
        os.close(fd)
        self.private = lms.keygen(PARAMS, seed)
        self.public = lms.public_key(self.private)
        self.state = directory / "state.json"
        lms.LeafState.create(self.state, self.public, PARAMS.H)
        self.path, self.lines, self.windows = tmp_path / "windows.jsonl", [], {}

    def sign(self, message: bytes) -> bytes:
        with lms.LeafState(self.state, self.public, PARAMS.H) as state:
            return lms.sign(self.private, state, message).encode()

    def append(self, kind: str, session: str, body: dict) -> None:
        self.lines.append(C.ledger_line(self.lines, kind, session, body, self.public[:16], self.sign))
        self.path.write_text("".join(line + "\n" for line in self.lines))

    def open(self, session: str, root: str | None = None) -> None:
        self.windows[session] = {"root": root or hashlib.sha512(session.encode()).hexdigest()}
        self.append("open", session, {"window": self.windows[session], "host": HOST, "pinned": "ab" * 20,
                                      "key_sha512": hashlib.sha512(self.public).hexdigest(), "by": "proofs"})

    def end(self, session: str, out: Path, closed: bool = True) -> None:
        record = json.loads((out / R.RECORD).read_text())
        self.append("end", session, {
            "closed": closed, "rc": 0 if closed else 1, "complete": closed,
            "files": {p.name: hashlib.sha512(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()},
            "record_leaf": int(record["signature"][:8], 16), "commitment": record["commitment"]["root"],
            "refused_hellos": len((out / R.HELLOS).read_text().splitlines())})


def start(tmp_path: Path, out: str = "out", key: Path | None = None, *extra: str, env=None, session: str | None = None,
          auditor: Auditor | None = None, root: str | None = None) -> tuple:
    """serve.py on loopback for session `session` (s-OUT), writing `out`; `auditor`'s ledger opens it first, for `root`."""
    out_dir = tmp_path / out
    if auditor is not None:
        auditor.open(session or f"s-{out}", root)
    proc = subprocess.Popen(
        [sys.executable, str(STUB), str(SERVE), "--out", str(out_dir), "--port", "0", "--scheme", "verity/test/coins/v1",
         "--families", json.dumps(FAMILIES), "--session", session or f"s-{out}", "--key", str(key or tmp_path / "key"),
         "--params", f"{PARAMS.W},{PARAMS.H}", "--keygen-workers", "1", "--host", HOST, *extra],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={"COINS_LOG": str(tmp_path / "coins.log"), **(os.environ if env is None else env)})
    return proc, out_dir, proc.stdout.readline()


def finish(proc: subprocess.Popen) -> tuple[int, str]:
    _, err = proc.communicate()
    return proc.returncode, err


def trusted(out: Path, auditor: Auditor, session: str = "s-out", closed: bool = True):
    """The verifier's read of `out`, once `auditor`'s ledger ends `session` as `closed` says, sealing `out`'s files, under
    the key the run wrote (pinned from the auditor's key, outside `out`)."""
    key = C.ChallengerKey.from_json(json.loads((out / R.KEY).read_text()))
    assert key.public == auditor.public
    auditor.end(session, out, closed)
    return R.trusted_session(out, [auditor.path], auditor.windows[session], key, lms_verify)


def close(out: Path) -> dict:
    return json.loads((out / R.CLOSE).read_text())


def session(address: str, roots: int = 1) -> None:
    with R.RemoteChallenger.connect(address) as client:
        record = C.LiveRecord(client, "positions", itertools.count().__next__)
        for i in range(roots):
            record.record(bytes([i]) * 64)


def test_the_challenger_signs_before_it_listens_and_the_verifier_checks_its_out_alone(tmp_path):
    auditor = Auditor(tmp_path)
    proc, out, first = start(tmp_path, auditor=auditor)
    assert first.startswith("listening 127.0.0.1:")
    assert {p.name for p in out.iterdir()} >= {R.RECORD, R.KEY, R.RELEASES}
    session(first.split()[1], roots=2)
    assert finish(proc)[0] == 0

    key = C.ChallengerKey.from_json(json.loads((out / R.KEY).read_text()))
    assert key.public == (tmp_path / "key" / "public").read_bytes()
    commitment, roots = trusted(out, auditor)
    assert roots == {"positions": (bytes([0]) * 64, bytes([1]) * 64), "replay": ()}
    assert commitment == C.ChallengerRecord.from_json(json.loads((out / R.RECORD).read_text())).commitment
    assert (close(out)["session"], close(out)["ended"], close(out)["lines"]) == ("s-out", None, 5)
    seed = (tmp_path / "key" / "seed").read_bytes()
    assert all(seed not in p.read_bytes() and seed.hex() not in p.read_text() for p in out.iterdir())


def test_every_coin_and_salt_is_the_lean_verifiers(tmp_path):
    """The challenger reads its coins from `one_stage.verifier.coins` alone: one coin and one salt request per coin of the
    layout, in order, and no other source."""
    proc, out, first = start(tmp_path)
    session(first.split()[1])
    assert finish(proc)[0] == 0
    asked = [int(n) for n in (tmp_path / "coins.log").read_text().split()]
    assert asked == [C.COIN_BYTES, C.HM96.salt_bytes] * sum(n for _, n in FAMILIES)


def test_a_lean_that_cannot_answer_starts_no_session(tmp_path):
    proc, out, first = start(tmp_path, env={**os.environ, "COINS_REFUSE": "1"})
    code, err = finish(proc)
    assert (code, first) == (2, "") and "the challenger refused to start" in err and "refused 64 coin bytes" in err
    assert not out.exists() and not (tmp_path / "key").exists()


def test_a_source_no_claim_instance_names_starts_no_session(tmp_path):
    """The host's coins are named by a claim instance (the Lean's, `uniform/io-getrandombytes`), or it samples
    none: a source without one starts no session and asks it for no byte."""
    proc, out, first = start(tmp_path, env={**os.environ, "COINS_UNNAMED": "1"})
    code, err = finish(proc)
    assert (code, first) == (2, "") and "the challenger refused to start" in err and "coins-without-claim" in err
    assert not out.exists() and not (tmp_path / "coins.log").exists()


def test_a_second_session_signs_at_the_keys_next_leaves(tmp_path):
    """Each session's record and close sign at the key's next leaves, after the ledger's open of it and before its end."""
    auditor, leaves = Auditor(tmp_path), []
    for out in ("one", "two"):
        proc, out_dir, first = start(tmp_path, out, auditor=auditor)
        session(first.split()[1])
        assert finish(proc)[0] == 0
        trusted(out_dir, auditor, f"s-{out}")
        leaves += [int(json.loads((out_dir / f).read_text())["signature"][:8], 16) for f in (R.RECORD, R.CLOSE)]
    assert leaves == [1, 2, 5, 6]


def test_the_key_is_held_only_while_it_signs_so_the_hosts_other_signers_sign_during_a_session(tmp_path):
    """On the auditor host every signer of the key takes `sign.lock` (`--sign-lock`) and the leaf state only to sign: a
    session holds neither while it serves, so the ledger and the host's other sessions sign meanwhile, and its close
    signs at the next leaf after theirs."""
    lock = tmp_path / "sign.lock"
    proc, out, first = start(tmp_path, "out", None, "--sign-lock", str(lock))
    key = C.ChallengerKey.from_json(json.loads((out / R.KEY).read_text()))
    private = lms.keygen(lms.Params(64, key.W, key.H), (tmp_path / "key" / "seed").read_bytes())
    with open(lock, "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with lms.LeafState(tmp_path / "key" / "state.json", key.public, key.H) as state:
            assert lms.sign(private, state, b"another signer").q == 1
    session(first.split()[1])
    assert finish(proc)[0] == 0
    assert (out / "hellos.jsonl").read_text() == ""
    assert int(close(out)["signature"][:8], 16) == 2


def test_a_session_id_the_key_has_signed_for_is_refused(tmp_path):
    proc, _, first = start(tmp_path, "one", session="s-1")
    session(first.split()[1])
    assert finish(proc)[0] == 0
    proc, out, first = start(tmp_path, "two", session="s-1")
    code, err = finish(proc)
    assert (code, first) == (2, "") and "is already started" in err
    assert not (out / R.RECORD).exists()


def test_a_key_inside_out_is_refused(tmp_path):
    proc, out, first = start(tmp_path, "out", tmp_path / "out" / "key")
    code, err = finish(proc)
    assert (code, first) == (2, "") and "inside its published --out" in err
    assert not (tmp_path / "out" / "key").exists()


def test_a_broken_sequence_after_the_hello_ends_the_session_with_exit_1(tmp_path):
    auditor = Auditor(tmp_path)
    proc, out, first = start(tmp_path, auditor=auditor)
    host, port = first.split()[1].split(":")
    hello = json.dumps({"type": "hello"}).encode()
    head = json.dumps({"type": "close", "extra": 1}).encode()
    with socket.create_connection((host, int(port))) as s:
        s.sendall(R.HEADER.pack(len(hello), 0) + hello)
        assert s.recv(1 << 16)
        s.sendall(R.HEADER.pack(len(head), 0) + head)
        assert s.recv(1 << 16)
    code, _ = finish(proc)
    assert code == 1
    assert close(out)["ended"].startswith("a client's frames are hello")
    with pytest.raises(C.CoinError, match="ended without its client closing it: a client's frames are hello"):
        trusted(out, auditor)


def test_refused_hellos_are_logged_and_the_next_connection_is_taken_until_one_is_accepted_then_none(tmp_path):
    """A refused hello releases nothing, so it can't spend the window (proofs, on the thread): the session takes the next
    connection, and once a hello is accepted it stops listening."""
    secret = bytes(range(32))
    (tmp_path / "secret").write_text(secret.hex() + "\n")
    root = "ab" * 64
    auditor = Auditor(tmp_path)
    proc, out, first = start(tmp_path, "out", None, "--secret-file", str(tmp_path / "secret"), "--root", root,
                             auditor=auditor, root=root)
    address = first.split()[1]
    host, port = address.split(":")
    with pytest.raises(C.CoinError, match="the session's secret"):
        R.RemoteChallenger.connect(address)
    with pytest.raises(C.CoinError, match="not the session's"):
        R.RemoteChallenger.connect(address, bytes(32))
    socket.create_connection((host, int(port))).close()
    with R.RemoteChallenger.connect(address, secret) as client:
        with pytest.raises(ConnectionRefusedError):
            socket.create_connection((host, int(port)))
        C.LiveRecord(client, "positions", itertools.count().__next__).record(bytes(64))
    assert finish(proc)[0] == 0
    assert close(out)["ended"] is None
    record = C.ChallengerRecord.from_json(json.loads((out / R.RECORD).read_text()))
    assert {k: len(v) for k, v in R.released_roots(out / R.RELEASES, record.commitment).items()} == {
        "positions": 1, "replay": 0}
    logged = [json.loads(ln) for ln in (out / "hellos.jsonl").read_text().splitlines()]
    assert [h["n"] for h in logged] == [0, 1, 2] and "secret" in logged[0]["reason"] and "closed" in logged[2]["reason"]
    assert trusted(out, auditor)[0] == record.commitment
    assert json.loads(auditor.lines[-1])["body"]["refused_hellos"] == 3

    for bad in (["--root", root.upper()], ["--root", root + "\n"]):
        proc, out, first = start(tmp_path, "bad", None, *bad)
        assert finish(proc)[0] == 2 and first == ""


SALTED = ["--families", json.dumps([["pouw/salt", 1], ["pouw/round", 1]])]


@pytest.mark.parametrize("after, served", [("ab", True), ("cd", False)])
def test_with_a_root_a_salt_opens_only_after_it(tmp_path, after, served):
    """A salt opens only after the session's root: one that followed a root unrelated to its window's Commit is refused."""
    proc, out, first = start(tmp_path, "out", None, "--root", "ab" * 64, *SALTED)
    with R.RemoteChallenger.connect(first.split()[1]) as client:
        record = C.LiveRecord(client, "pouw/salt", itertools.count().__next__)
        if served:
            record.record(bytes.fromhex(after * 64))
        else:
            with pytest.raises(C.CoinError, match="only after the window's Commit root"):
                record.record(bytes.fromhex(after * 64))
    code, _ = finish(proc)
    record = C.ChallengerRecord.from_json(json.loads((out / R.RECORD).read_text()))
    released = len(R.released_roots(out / R.RELEASES, record.commitment)["pouw/salt"])
    assert (code, released) == ((0, 1) if served else (1, 0))
    if not served:
        assert close(out)["ended"].startswith("a pouw/salt coin opens only after the window's Commit root")


def test_without_a_root_a_salt_opens_after_any_root(tmp_path):
    proc, out, first = start(tmp_path, "out", None, *SALTED)
    with R.RemoteChallenger.connect(first.split()[1]) as client:
        C.LiveRecord(client, "pouw/salt", itertools.count().__next__).record(bytes(64))
    assert finish(proc)[0] == 0


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
    assert close(out)["ended"] == "the tunnel to vy-nebius-2 ended before a client's hello was accepted"
    assert close(out)["lines"] == 1


def test_a_peer_missing_from_the_peers_file_is_refused(tmp_path):
    fake_ssh(tmp_path, "exit 0")
    proc, out, first = start(tmp_path, "out", None, "--peer", "vy-nebius-9", "--peers", str(tmp_path / "peers"))
    code, err = finish(proc)
    assert (code, first) == (2, "") and "no peer vy-nebius-9" in err


def test_a_windows_session_never_opens_its_own_tunnel(tmp_path):
    """A window's session (`--root`) reaches its peer through the peer's certifier coin link alone
    (`CoinsConfidential`), so `--peer` with it is refused before any key or coin is touched."""
    env = fake_ssh(tmp_path, "exec tail -f /dev/null")
    proc, out, first = start(tmp_path, "out", None, "--root", "ab" * 64, "--peer", "vy-nebius-2", "--peers",
                             str(tmp_path / "peers"), env=env)
    code, err = finish(proc)
    assert (code, first) == (2, "") and "certifier coin link, never its own ssh -R" in err
    assert not out.exists() and not (tmp_path / "key").exists() and not (tmp_path / "ssh.args").exists()
