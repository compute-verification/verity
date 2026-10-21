"""`challengers.Gate` against the auditor host's own gate (`benchmarks/live_challenger/auditor.py`), in this process: its
requests go to `auditor.gate` as the lane key's forced command would pass them, and each opened session is served by
`auditor.run` through `serve.py` (the Lean's coins stubbed, `stub_serve.py`) on a thread, with the gate's root steps
(chown, systemctl) and the clock injected. An honest session whose coins the gate's session released is accepted and
held to the ledger, and refused when the verifier has no copy of main's first-parent history or the host's pinned tree
isn't in it; a challenger key other than the pinned one, a reserve outside the audit's period, a second session for the
same window and a challenger whose commitment isn't the open's are each refused before a coin leaves."""
from __future__ import annotations

import hashlib
import itertools
import json
import shlex
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
LIVE = ROOT / "benchmarks" / "live_challenger"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(LIVE))
sys.path.insert(0, str(ROOT / "verity" / "core" / "protocols" / "remote_bandwidth" / "tests"))

import auditor as A  # noqa: E402
import challengers  # noqa: E402
from test_remote_bandwidth import SCHEDULE, drive, finish, params, store  # noqa: E402,F401
from verity.core.protocols.remote_bandwidth import coins  # noqa: E402
from verity.core.protocols.sampled_proofs.challenger import NO_MAIN, CoinError  # noqa: E402
from verity.ml.crypto import lms  # noqa: E402

P = lms.Params(64, 4, 4)
T0 = datetime(2026, 10, 21, 1, 0, 0, tzinfo=timezone.utc)
PIN = "c" * 40
LANE = "memory-accounting"
SECRET = bytes(range(32))
CIRCUIT = hashlib.sha512(b"verity/remote-bandwidth/v0").hexdigest()


class Host:
    """The auditor host in `root`: its gate under the lane's key, and the sessions its opens started, each `auditor.run`
    on a thread until its client closes it."""

    def __init__(self, root: Path) -> None:
        (root / "var" / "sessions").mkdir(parents=True)
        (root / "peers").write_text("vy-challenger-1 research@10.0.0.1\nvy-nebius-1 research@10.0.0.2\n")
        self.host = A.Host(root=root / "var", peers=root / "peers")
        self.signer = A.key_signer(self.host, P, workers=1)
        public = (self.host.key / "public").read_bytes()
        self.pin = hashlib.sha512(public).hexdigest()
        (self.host.root / "pubkey.json").write_text(json.dumps(
            {"W": P.W, "H": P.H, "public": public.hex(), "public_sha512": self.pin}))
        self.tokens = (f"{i:04x}" for i in itertools.count())
        self.runs: dict[str, threading.Thread] = {}
        self.main = root / "main.txt"
        self.main.write_text(f"{'0f' * 20}\n{PIN}\n")

    def call(self, words: list[str]) -> tuple[int, bytes]:
        host, signer = self.host, self.signer
        return A.gate(["--ssh", shlex.join(words), "--key", LANE], host, now=T0, token=next(self.tokens),
                      own=lambda p: None, start=self.start, active=lambda s: False, secret=SECRET,
                      reserve=lambda s: A.reserve(host, s, signer=signer, pinned=PIN, now=T0),
                      admit=lambda s: A.admit(host, s, signer=signer, pinned=PIN, now=T0, active=lambda s: False),
                      period=lambda args: A.period(host, args, signer=signer, pinned=PIN, now=T0,
                                                   active=lambda s: False))

    def start(self, session: str) -> None:
        """The session's unit: serve.py on a port of its own (no tunnel), returning once it listens."""
        body = A.sessions_of(A.read_ledger(self.host.ledger))[session]["open"]["body"]
        argv = A.serve_argv(self.host, session, body)
        i = argv.index("--peer")
        del argv[i:i + 2]
        argv[argv.index("--port") + 1] = "0"
        argv = [sys.executable, str(LIVE / "tests" / "stub_serve.py"), *argv[2:], "--params", "4,4",
                "--keygen-workers", "1"]
        listening = threading.Event()

        def wait(proc, seconds):
            listening.set()
            return proc.wait()

        self.runs[session] = threading.Thread(target=A.run, args=(self.host, session), kwargs=dict(
            signer=self.signer, pinned=PIN, now=lambda: T0, argv=argv, wait=wait))
        self.runs[session].start()
        listening.wait()

    def address(self, answer: dict) -> str:
        return (self.host.sessions / answer["session"] / "out" / "stdout.log").read_text().split()[1]

    def pause(self) -> None:
        for t in self.runs.values():
            t.join()

    def gate(self, out: Path, *, pin: str | None = None, period: str = "p-1", call=None,
             main: Path | None = None) -> challengers.Gate:
        return challengers.Gate(out, call or self.call, "vy-nebius-1", pin or self.pin, period, main or self.main,
                                address=self.address, pause=self.pause, polls=3)

    def period(self, name: str = "p-1") -> None:
        rc, out = self.call(["period-open", "--name", name, "--cap", "4", "--circuit", CIRCUIT])
        assert rc == 0, out

    def ledger(self) -> list[dict]:
        return A.entries(A.read_ledger(self.host.ledger))


@pytest.fixture
def host(tmp_path, monkeypatch) -> Host:
    monkeypatch.setenv("COINS_LOG", str(tmp_path / "coins.log"))
    return Host(tmp_path / "host")


def session(store, gate: challengers.Gate):
    """An honest session over `store` whose coins come from `gate`'s session for window 0: the session and verdicts."""
    p = params()
    client = gate.open(store.registration.digest, 0, p.windows, 8)
    c = coins.Coins(client, store.registration.digest, 0, p.windows, clock=itertools.count().__next__)
    s, prover, passes, clock = drive(store, p, SCHEDULE, coins_=c)
    return s, finish(s, prover, passes, clock)


def test_a_session_whose_coins_the_gates_session_released_is_held_to_the_ledger_and_accepted(store, host, tmp_path):
    host.period()
    gate = host.gate(tmp_path / "out")
    s, verdicts = session(store, gate)
    assert all(v.accepted for v in verdicts) and s.outcome == "accepted"
    doc = gate.finish(s)
    assert doc["reason"] is None, doc["reason"]
    sid = doc["session"]
    assert doc["closed"] is None and doc["end"]["closed"] and doc["end"]["rc"] == 0 and "secret" not in doc["open"]
    assert doc["open"]["root"] == coins.window(store.registration.digest, 0)["root"]
    kinds = [(e["kind"], e["session"]) for e in host.ledger()]
    assert kinds == [("period-open", None), ("reserve", sid), ("open", sid), ("end", sid)]
    reserve = host.ledger()[1]["body"]
    assert (reserve["by"], reserve["peer"], reserve["scheme"], reserve["epoch"]) == (LANE, "vy-nebius-1", coins.SCHEME, 0)
    assert reserve["families"] == [list(f) for f in coins.families(params().windows, 8)]
    assert (tmp_path / "out" / f"{sid}.ledger.jsonl").read_bytes() == host.host.ledger.read_bytes()
    assert not (tmp_path / "out" / sid / "secret").exists()
    key = gate.key()
    heads = [tmp_path / "out" / f"{sid}.ledger.jsonl"]
    assert s.trusted_reason(tmp_path / "out" / sid / "out", heads, key, [PIN], challengers.lms_verify, "p-2") == (
        "the challenger's session for this session's window is not trusted: the auditor's ledger holds the session's "
        "window in period p-1, not the audit's period p-2")


def test_a_session_is_refused_without_the_verifiers_copy_of_main_or_when_its_host_ran_a_commit_outside_it(
        store, host, tmp_path):
    host.period()
    gate = host.gate(tmp_path / "out", main=tmp_path / "none.txt")
    s, verdicts = session(store, gate)
    assert all(v.accepted for v in verdicts)
    doc = gate.finish(s)
    assert doc["reason"] == f"{NO_MAIN} (none at {tmp_path / 'none.txt'})"
    out, heads = tmp_path / "out" / doc["session"] / "out", [tmp_path / "out" / f"{doc['session']}.ledger.jsonl"]
    assert s.trusted_reason(out, heads, gate.key(), ["0f" * 20], challengers.lms_verify, "p-1") == (
        f"the challenger's session for this session's window is not trusted: the auditor's ledger's reserve entry 1 "
        f"was written by the host's tree at {PIN}, which is not a commit of main's first-parent history: repin the "
        "host to a main commit")
    assert s.trusted_reason(out, heads, gate.key(), [PIN], challengers.lms_verify, "p-1") is None


def test_a_challenger_key_other_than_the_pinned_one_is_refused_before_a_reserve(store, host, tmp_path):
    host.period()
    with pytest.raises(CoinError, match="not the pinned"):
        host.gate(tmp_path / "out", pin="0" * 128).open(store.registration.digest, 0, params().windows, 8)
    assert [e["kind"] for e in host.ledger()] == ["period-open"]


def test_a_reserve_outside_the_audits_period_is_refused_before_its_open(store, host, tmp_path):
    host.period("p-0")
    with pytest.raises(CoinError, match="in period p-0, not the audit's period p-1: it expires unopened"):
        host.gate(tmp_path / "out").open(store.registration.digest, 0, params().windows, 8)
    assert [e["kind"] for e in host.ledger()] == ["period-open", "reserve"]


def test_a_second_session_for_the_window_is_refused_at_its_open(store, host, tmp_path):
    host.period()
    gate = host.gate(tmp_path / "out")
    s, verdicts = session(store, gate)
    assert gate.finish(s)["reason"] is None
    with pytest.raises(CoinError, match="one session a root, ever"):
        host.gate(tmp_path / "again").open(store.registration.digest, 0, params().windows, 8)
    assert [e["kind"] for e in host.ledger()] == ["period-open", "reserve", "open", "end", "reserve"]


def test_a_challenger_whose_commitment_isnt_the_opens_is_refused_before_its_first_coin(store, host, tmp_path):
    def call(words):
        rc, out = host.call(words)
        if words[0] == "open" and rc == 0:
            out = json.dumps({**json.loads(out), "commitment": "ab" * 64}).encode()
        return rc, out

    host.period()
    with pytest.raises(CoinError, match="refused before its first coin"):
        host.gate(tmp_path / "out", call=call).open(store.registration.digest, 0, params().windows, 8)
    host.pause()
    end = host.ledger()[-1]["body"]
    assert host.ledger()[-1]["kind"] == "end" and end["released"] == {label: 0 for label in coins.LABELS}


def test_the_gates_ports_are_the_hosts():
    assert challengers.GATE_PORTS == A.PORTS
