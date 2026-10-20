"""RT4-INT's witnesses on the auditor ledger v1 (`auditor.py`, cursor/infra-challenger-host-98f5 @ d2ad7bdea6), run as
`test_auditor.py` runs it: a test key, the gate's root steps and the clock injected, serve.py as a process with its coins
stubbed (`stub_serve.py`).

Each test asserts what the code does today; one that passes where a finding says the code admits something is that
finding's witness, and flips to the refusal when the code is fixed.

* RT4-INT-24: a window is the two strings its requester names, so one served window gets a session per (period, commit)
  pair. Two sessions for one Commit under two period labels both end closed, both pass `verify_ledger` and
  `trusted_session`, and their coins differ; serve.py is never told the window, so a session's salt follows any root; and
  every open names `by` "research" whatever lane asked.
* RT4-INT-25: `closed` is the client's word. A client that takes its salt and one of three round coins, then closes, ends
  `closed: true`, which the ledger's rule doesn't read as a refused window.
* RT4-INT-26: nothing checks the `end` entry's seal. A session whose release log is rewritten after its end still passes
  `verify_ledger` (which reads no session) and `trusted_session` (which reads no ledger); only the comparison no function
  makes catches it.
* RT4-INT-27: a restore of the host's key state and ledger forks the ledger at a reused LMS leaf, and `verify_ledger`
  passes both heads.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
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
SALT, ROUND = "pouw/salt", "pouw/round"
FAMILIES = [[SALT, 1], [ROUND, 3]]


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


def open_window(host, signer, *, token, period="p0", commit=COMMIT, lane=None):
    argv = ["open", "--period", period, "--commit", commit, "--peer", "vy-nebius-1", "--scheme", "verity/test/coins/v1",
            "--families", json.dumps(FAMILIES), *(["--lane", lane] if lane else [])]
    rc, out = A.gate(argv, host, by="research", now=T0, token=token, own=lambda d: None, start=lambda s: None,
                     active=lambda s: False,
                     admit=lambda s: A.admit(host, s, signer=signer, pinned=PIN, now=T0, active=lambda s: False))
    assert rc == 0, out
    return json.loads(out)["session"]


def serve(host, signer, session, client, tmp_path, monkeypatch) -> dict:
    """Run the admitted session through `auditor.run` with stub_serve.py, `client(RemoteChallenger)` its one client; the
    session's `end` entry."""
    monkeypatch.setenv("COINS_LOG", str(tmp_path / "coins.log"))
    body = A.sessions_of(A.read_ledger(host.ledger))[session]["open"]["body"]
    argv = A.serve_argv(host, session, body)
    i = argv.index("--peer")
    del argv[i:i + 2]
    argv[argv.index("--port") + 1] = "0"
    argv = [sys.executable, str(HERE / "stub_serve.py"), *argv[2:], "--params", "4,4", "--keygen-workers", "1"]
    out = host.sessions / session / "out"

    def wait(proc, seconds):
        address = (out / "stdout.log").read_text().split()[1]
        with R.RemoteChallenger.connect(address) as c:
            client(c)
        return proc.wait()

    return A.run(host, session, signer=signer, pinned=PIN, now=lambda: T1, argv=argv, wait=wait)


def trusted(host, session):
    out = host.sessions / session / "out"
    key = C.ChallengerKey.from_json(json.loads((out / R.KEY).read_text()))
    return R.trusted_session(out, key, lambda k, m, s: lms.verify(P, k.public, m, lms.Signature.decode(P, s)))


def test_one_commit_gets_a_session_per_period_label_and_both_pass_the_ledger_and_the_session_checks(host, signer, tmp_path,
                                                                                                     monkeypatch):
    root = bytes.fromhex(COMMIT)
    first = open_window(host, signer, token="ab12", period="p0", lane="vllm")
    again = open_window(host, signer, token="cd34", period="p0-again", lane="hash")
    other = open_window(host, signer, token="ef56", period="p0", commit="cd" * 64)

    def window(c):
        c.open(SALT, 0, root)
        for s in range(3):
            c.open(ROUND, s, hashlib.sha512(b"round %d" % s).digest())

    ends = [serve(host, signer, s, window, tmp_path, monkeypatch) for s in (first, again)]
    assert [e["body"]["closed"] for e in ends] == [True, True]

    I, verify = verify_under(host)
    windows = A.verify_ledger(A.read_ledger(host.ledger), I, verify)
    assert set(windows) == {("p0", COMMIT), ("p0-again", COMMIT), ("p0", "cd" * 64)}
    assert windows[("p0", COMMIT)]["end"] == ends[0] and windows[("p0-again", COMMIT)]["end"] == ends[1]

    (ca, ra), (cb, rb) = trusted(host, first), trusted(host, again)
    assert ra[SALT] == rb[SALT] == (root,) and ra[ROUND] == rb[ROUND]
    assert ca.root != cb.root

    opens = {s: windows[w]["open"]["body"] for s, w in ((first, ("p0", COMMIT)), (again, ("p0-again", COMMIT)),
                                                       (other, ("p0", "cd" * 64)))}
    assert {b["by"] for b in opens.values()} == {"research"}
    assert [opens[s]["lane"] for s in (first, again, other)] == ["vllm", "hash", None]
    for s in (first, again):
        assert json.loads((host.sessions / s / "out" / R.RECORD).read_text())["host"] == opens[s]["host"]


def test_a_session_serves_a_salt_after_any_root_and_a_client_that_closes_after_one_round_coin_ends_closed(
        host, signer, tmp_path, monkeypatch):
    session = open_window(host, signer, token="ab12")
    elsewhere = hashlib.sha512(b"not the window's commit").digest()

    def dropped(c):
        c.open(SALT, 0, elsewhere)
        c.open(ROUND, 0, hashlib.sha512(b"round 0").digest())

    end = serve(host, signer, session, dropped, tmp_path, monkeypatch)
    assert end["body"]["rc"] == 0 and end["body"]["closed"] is True and end["body"]["ended"] == "closed by the client"

    I, verify = verify_under(host)
    assert A.verify_ledger(A.read_ledger(host.ledger), I, verify)[("p0", COMMIT)]["end"] == end
    summary = json.loads((host.sessions / session / "out" / "challenger.json").read_text())
    assert summary["released"] == {SALT: 1, ROUND: 1} and summary["commitment"]["families"] == FAMILIES
    _, received = trusted(host, session)
    assert received[SALT] == (elsewhere,) and elsewhere.hex() != COMMIT


def test_a_release_log_rewritten_after_its_end_entry_passes_both_checks_and_only_the_unchecked_seal_sees_it(
        host, signer, tmp_path, monkeypatch):
    session = open_window(host, signer, token="ab12")
    early = [hashlib.sha512(b"early %d" % s).digest() for s in range(3)]

    def window(c):
        c.open(SALT, 0, bytes.fromhex(COMMIT))
        for s in range(3):
            c.open(ROUND, s, early[s])

    end = serve(host, signer, session, window, tmp_path, monkeypatch)
    out = host.sessions / session / "out"
    log = out / R.RELEASES
    late = [hashlib.sha512(b"chosen after every coin %d" % s).digest() for s in range(3)]
    text = log.read_text()
    for e, x in zip(early, late):
        text = text.replace(e.hex(), x.hex())
    log.write_text(text)

    I, verify = verify_under(host)
    assert A.verify_ledger(A.read_ledger(host.ledger), I, verify)[("p0", COMMIT)]["end"] == end
    _, received = trusted(host, session)
    assert received[ROUND] == tuple(late)
    sealed = end["body"]["files"]
    assert hashlib.sha512(log.read_bytes()).hexdigest() != sealed[R.RELEASES]
    assert "stdout.log" not in sealed and (out / "stdout.log").is_file()
    assert "request.json" not in sealed and (host.sessions / session / "request.json").is_file()


def test_a_restored_key_state_and_ledger_fork_the_chain_at_a_reused_leaf_and_verify_ledger_passes_both_heads(host, signer,
                                                                                                           tmp_path):
    open_window(host, signer, token="ab12", period="p0")
    snap = tmp_path / "snapshot"
    snap.mkdir()
    saved = [host.key / "state.json", host.key / "state.json.journal", host.ledger]
    for p in saved:
        shutil.copy2(p, snap / p.name)
    open_window(host, signer, token="cd34", period="p1")
    head_a = A.read_ledger(host.ledger)
    for p in saved:
        shutil.copy2(snap / p.name, p)
    open_window(host, signer, token="ef56", period="p1-other")
    head_b = A.read_ledger(host.ledger)

    I, verify = verify_under(host)
    assert set(A.verify_ledger(head_a, I, verify)) == {("p0", COMMIT), ("p1", COMMIT)}
    assert set(A.verify_ledger(head_b, I, verify)) == {("p0", COMMIT), ("p1-other", COMMIT)}
    assert head_a[0] == head_b[0] and head_a[1] != head_b[1]
    a, b = json.loads(head_a[1]), json.loads(head_b[1])
    assert (a["seq"], a["prev"]) == (b["seq"], b["prev"])
    q = [lms.Signature.decode(P, bytes.fromhex(e["sig"])).q for e in (a, b)]
    assert q[0] == q[1]
