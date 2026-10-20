"""The auditor host's service (`auditor.py`) on a test key, with the gate's root steps (chown, systemctl) and the clock injected:
a window gets one session, its open entry signed before serve.py runs; the session's end entry seals its out/; a session whose pin
moved doesn't run; the gate refuses what it can't check; and the reference check refuses a ledger that was edited, cut, reordered
or given a second session for a window."""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from verity.core.protocols.sampled_proofs import challenger as C
from verity.core.protocols.sampled_proofs import remote as R
from verity.ml.crypto import lms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import auditor as A  # noqa: E402

P = lms.Params(64, 4, 4)
T0 = datetime(2026, 10, 20, 9, 30, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 10, 20, 9, 40, 0, tzinfo=timezone.utc)
COMMIT = "ab" * 64
PIN = "c" * 40
FAMILIES = [["positions", 2], ["replay", 1]]


@pytest.fixture
def host(tmp_path) -> A.Host:
    (tmp_path / "var" / "sessions").mkdir(parents=True)
    (tmp_path / "peers").write_text("# peers\nvy-challenger-1 research@10.0.0.1\nvy-nebius-1 research@10.0.0.2\n")
    return A.Host(root=tmp_path / "var", peers=tmp_path / "peers")


@pytest.fixture
def signer(host):
    return A.key_signer(host, P, workers=1)


def verify_under(host):
    public = (host.key / "public").read_bytes()
    return public[:16], lambda m, sig: lms.verify(P, public, m, lms.Signature.decode(P, sig))


def open_window(host, signer, *extra, token="ab12", active=lambda s: False, period="p0", commit=COMMIT):
    argv = ["open", "--period", period, "--commit", commit, "--peer", "vy-nebius-1", "--scheme", "verity/test/coins/v1",
            "--families", json.dumps(FAMILIES), *extra]
    return A.gate(argv, host, by="research", now=T0, token=token, own=lambda d: None, start=lambda s: None, active=active,
                  admit=lambda s: A.admit(host, s, signer=signer, pinned=PIN, now=T0, active=active))


def test_a_window_gets_one_session_whose_open_entry_is_signed_before_it_serves(host, signer):
    rc, out = open_window(host, signer, "--lane", "vllm")
    doc = json.loads(out)
    assert rc == 0 and doc == {"session": "s20261020T093000Z-ab12", "port": 7811, "peer": "vy-nebius-1", "seq": 0,
                               "host": "vy-challenger-1.s20261020T093000Z-ab12"}
    I, verify = verify_under(host)
    windows = A.verify_ledger(A.read_ledger(host.ledger), I, verify)
    (e,) = [w["open"] for w in windows.values()]
    assert list(windows) == [("p0", COMMIT)] and windows[("p0", COMMIT)]["end"] is None
    assert e["body"]["pinned"] == PIN and e["body"]["lane"] == "vllm" and e["body"]["by"] == "research"
    assert e["body"]["key_sha512"] == hashlib.sha512((host.key / "public").read_bytes()).hexdigest()
    assert e["body"]["started_utc"] == "2026-10-20T09:30:00Z" and e["prev"] == A.ZERO

    rc, out = open_window(host, signer, token="cd34")
    assert rc == 3 and "already has its session" in json.loads(out)["refused"]
    assert not (host.sessions / "s20261020T093000Z-cd34").exists()

    # a request that passed the gate's look at the ledger before another took the window is refused by admit, under its lock
    d = host.sessions / "s20261020T093000Z-ef56"
    d.mkdir()
    (d / "request.json").write_text((host.sessions / doc["session"] / "request.json").read_text())
    with pytest.raises(A.Refused, match="one session a window"):
        A.admit(host, d.name, signer=signer, pinned=PIN, now=T0, active=lambda s: False)
    assert len(A.read_ledger(host.ledger)) == 1

    rc, out = open_window(host, signer, token="0a0b", commit="cd" * 64, active=lambda s: True)
    assert rc == 0 and json.loads(out)["port"] == 7812
    rc, out = open_window(host, signer, token="0c0d", commit="ef" * 64, active=lambda s: False)
    assert rc == 0 and json.loads(out)["port"] == 7811


def test_a_session_serves_from_the_pinned_tree_and_its_end_entry_seals_its_out(host, signer, tmp_path, monkeypatch):
    monkeypatch.setenv("COINS_LOG", str(tmp_path / "coins.log"))
    session = json.loads(open_window(host, signer)[1])["session"]
    body = A.sessions_of(A.read_ledger(host.ledger))[session]["open"]["body"]
    argv = A.serve_argv(host, session, body)
    i = argv.index("--peer")
    del argv[i:i + 2]
    argv[argv.index("--port") + 1] = "0"
    argv = [sys.executable, str(HERE / "stub_serve.py"), *argv[2:], "--params", "4,4", "--keygen-workers", "1"]
    out = host.sessions / session / "out"

    def client(proc, seconds):
        assert seconds == 120 * 60
        address = (out / "stdout.log").read_text().split()[1]
        with R.RemoteChallenger.connect(address) as c:
            record = C.LiveRecord(c, "positions", itertools.count().__next__)
            record.record(bytes(64))
        return proc.wait()

    end = A.run(host, session, signer=signer, pinned=PIN, now=lambda: T1, argv=argv, wait=client)
    assert end["body"]["rc"] == 0 and end["body"]["closed"] and end["body"]["ended_utc"] == "2026-10-20T09:40:00Z"
    assert set(end["body"]["files"]) >= {"record.json", "key.json", "releases.jsonl", "challenger.json"}
    assert all(hashlib.sha512((out / n).read_bytes()).hexdigest() == h for n, h in end["body"]["files"].items())

    I, verify = verify_under(host)
    assert A.verify_ledger(A.read_ledger(host.ledger), I, verify)[("p0", COMMIT)]["end"] == end
    key = C.ChallengerKey.from_json(json.loads((out / R.KEY).read_text()))
    assert key.public == (host.key / "public").read_bytes()
    R.trusted_session(out, key, lambda k, m, s: lms.verify(P, k.public, m, lms.Signature.decode(P, s)))
    assert json.loads((out / R.RECORD).read_text())["host"] == body["host"] == f"vy-challenger-1.{session}"

    rc, tgz = A.gate(["fetch", session], host)
    with tarfile.open(fileobj=io.BytesIO(tgz)) as tar:
        assert {f"{session}/request.json", f"{session}/out/record.json", f"{session}/out/releases.jsonl"} <= set(tar.getnames())
    rc, st = A.gate(["status", session], host, active=lambda s: True)
    assert rc == 0 and json.loads(st)["end"] == end and not json.loads(st)["active"]
    with pytest.raises(A.Refused, match="has ended"):
        A.run(host, session, signer=signer, pinned=PIN, now=lambda: T1, argv=argv, wait=client)


def test_a_session_whose_pin_moved_after_its_admission_does_not_run(host, signer):
    session = json.loads(open_window(host, signer)[1])["session"]
    end = A.run(host, session, signer=signer, pinned="d" * 40, now=lambda: T1, argv=["/nonexistent/serve"])
    assert end["body"]["rc"] == 2 and not end["body"]["closed"] and "the pin moved" in end["body"]["ended"]
    assert end["body"]["files"] == {} and not (host.sessions / session / "out").exists()
    with pytest.raises(A.Refused, match="not a pinned tree"):
        A.pinned_commit(Path("/workspace"))
    assert A.pinned_commit(Path("/opt/vy-challenger") / PIN) == PIN


@pytest.mark.parametrize("bad, why", [
    (["--commit", "AB" * 64], "128 lowercase hex"),
    (["--commit", "ab" * 32], "128 lowercase hex"),
    (["--peer", "vy-challenger-1"], "--peer"),
    (["--peer", "vy-nebius-9"], "--peer"),
    (["--families", "[]"], "--families"),
    (["--families", '[["positions", 0]]'], "--families"),
    (["--families", "not json"], "--families"),
    (["--period", "-x"], "--period"),
    (["--period", "p 0"], "--period"),
    (["--minutes", "0"], "--minutes"),
    (["--minutes", "721"], "--minutes"),
    (["--epoch", "-1"], "--epoch"),
    (["--lane", "a/b"], "--lane"),
    (["--scheme", "Has Caps"], "--scheme"),
])
def test_the_gate_refuses_a_request_it_cant_check(host, signer, bad, why):
    good = {"--period": "p0", "--commit": COMMIT, "--peer": "vy-nebius-1", "--scheme": "verity/test/coins/v1",
            "--families": json.dumps(FAMILIES)}
    good.update(dict(zip(bad[::2], bad[1::2])))
    argv = ["open", *itertools.chain.from_iterable(good.items())]
    rc, out = A.gate(argv, host, by="research", now=T0, token="ab12", own=lambda d: None, start=lambda s: None,
                     active=lambda s: False, admit=lambda s: pytest.fail("admitted"))
    assert rc == 3 and why in json.loads(out)["refused"]
    assert A.read_ledger(host.ledger) == [] and list(host.sessions.iterdir()) == []


def test_the_gate_takes_its_request_from_ssh_as_shell_words_and_refuses_any_other_verb(host, signer):
    words = (f"open --period p0 --commit {COMMIT} --peer vy-nebius-1 --scheme verity/test/coins/v1 "
             f"--families '{json.dumps(FAMILIES)}'")
    rc, out = A.gate(["--ssh", words], host, by="research", now=T0, token="ab12", own=lambda d: None, start=lambda s: None,
                     active=lambda s: False, admit=lambda s: A.admit(host, s, signer=signer, pinned=PIN, now=T0,
                                                                     active=lambda s: False))
    assert rc == 0 and json.loads(out)["session"] == "s20261020T093000Z-ab12"
    for argv in (["--ssh", "open --period 'p0"], ["--ssh", "x" * 8193], ["--ssh", "bash -c id"], ["--ssh", ""],
                 ["status", "../../etc"], ["status", "s20261020T093000Z-ffff"], ["fetch", "s20261020T093000Z-ffff"]):
        rc, out = A.gate(argv, host, active=lambda s: False)
        assert rc == 3 and json.loads(out)["refused"]
    assert A.gate(["ledger"], host) == (0, host.ledger.read_bytes())


def test_the_reference_check_refuses_an_edited_cut_reordered_or_doubled_ledger(host, signer, tmp_path):
    for token, commit in (("ab12", COMMIT), ("cd34", "cd" * 64)):
        assert open_window(host, signer, token=token, commit=commit)[0] == 0
    lines = A.read_ledger(host.ledger)
    I, verify = verify_under(host)
    assert len(A.verify_ledger(lines, I, verify)) == 2

    e = json.loads(lines[1])
    e["body"]["port"] = 7899
    edited = [lines[0], json.dumps(e, sort_keys=True, separators=(",", ":")).encode()]
    for bad, why in ((edited, "signature"), (lines[1:], "entry 0"), (lines[::-1], "entry 0")):
        with pytest.raises(ValueError, match=why):
            A.verify_ledger(bad, I, verify)

    _, sign, _ = signer
    first = A.sessions_of(lines)["s20261020T093000Z-ab12"]["open"]["body"]
    A.append(host, "open", "s20261020T093000Z-ef56", lambda ls: first, I, sign)
    with pytest.raises(ValueError, match="a second open"):
        A.verify_ledger(A.read_ledger(host.ledger), I, verify)
    other = A.Host(root=tmp_path / "other", peers=host.peers)
    other.root.mkdir()
    A.append(other, "end", "s20261020T093000Z-0000", lambda ls: {}, I, sign)
    with pytest.raises(ValueError, match="no open"):
        A.verify_ledger(A.read_ledger(other.ledger), I, verify)
