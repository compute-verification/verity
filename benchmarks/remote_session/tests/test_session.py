"""The verifier's and the prover's processes over a loopback socket, the driver's reference as the prover's backend and
a clock that advances by one tick per pass: an honest session is accepted and its γ terms computed, a prover that
dropped blocks is rejected at its openings, and a second session waits out the first's deadline."""
from __future__ import annotations

import os
import queue
import random
import socket
import sys
import threading
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2]))

from verity.core.protocols.remote_bandwidth import audit  # noqa: E402
from verity.core.protocols.remote_bandwidth import codec as rcodec  # noqa: E402

import prover  # noqa: E402
import verifier  # noqa: E402
import wire  # noqa: E402

P512 = rcodec.Codec(rcodec.P512, 2, False)


class PassClock:
    """Time is the prover's passes: one tick as each pass ends. A wait returns the inbox's next message or None once
    the clock reaches its end, woken by either; once the prover has said its passes are done, time jumps to the wait's
    end."""

    def __init__(self) -> None:
        self.t, self.done, self.inbox = 0, False, None

    def now(self) -> int:
        return self.t

    def tick(self) -> None:
        if self.inbox is None:
            self.t += 1
            return
        with self.inbox.mutex:
            self.t += 1
            self.inbox.not_empty.notify_all()

    def wait(self, inbox: queue.Queue, until_ns):
        self.inbox = inbox
        with inbox.not_empty:
            while True:
                if until_ns is not None and self.done:
                    self.t = max(self.t, until_ns)
                if until_ns is not None and self.t >= until_ns:
                    return None
                if inbox._qsize():
                    break
                inbox.not_empty.wait()
            return inbox._get()

    def idle(self) -> None:
        with self.inbox.mutex:
            self.done = True
            self.inbox.not_empty.notify_all()

    def busy(self) -> None:
        self.done = False


class TickingBackend(prover.CpuBackend):
    def __init__(self, clock: PassClock, *args, **kw) -> None:
        super().__init__(*args, **kw)
        self.clock = clock

    def run_pass(self, seed: bytes):
        out = super().run_pass(seed)
        self.clock.tick()
        return out


class ArchivingBackend(TickingBackend):
    def __init__(self, *args, **kw) -> None:
        super().__init__(*args, **kw)
        self.archived = []

    def archive_window(self, handles):
        self.archived.append(len(handles))
        return handles


def run(plan: str, *, blocks: int = 16, windows: int = 3, samples: int = 6, sessions: int = 1, checks=None,
        backend_cls=TickingBackend):
    clock = PassClock()
    a, b = socket.socketpair()
    vconn, pconn = wire.Conn(a, clock=clock.now), wire.Conn(b)
    backend = backend_cls(clock, blocks, random.Random(7), read_bytes_per_s=1e20)
    p = prover.ProverSide(pconn, backend, b"test-node", prover.parse_plan(plan), seed=11, log=lambda m: None)
    results = []

    def prove() -> None:
        p.register(codec=P512)
        for mode, fraction in p.plan:
            head, _, _ = p.expect("session")
            results.append(p.session(head, mode, fraction))

    th = threading.Thread(target=prove, daemon=True)
    th.start()
    v = verifier.Verifier(vconn, source=os.urandom(32), windows=windows, window_ns=20, samples=samples,
                          delta=2.0 ** -40, jitter_ns=10 ** 9, clock=clock, log=lambda m: None,
                          checks=checks)
    v.register()
    allowance = v.allowance(v.ping(4))
    out, not_before = [], None
    for k in range(sessions):
        r = v.session(k, allowance, not_before)
        out.append(r)
        not_before = r.get("end_ns", clock.now())
    th.join()
    return out, results, backend


def test_an_honest_session_is_accepted_with_its_ledger(monkeypatch):
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    (r,), (pr,), _ = run("honest")
    assert r["accepted"] and pr["accepted"] and r["failure"] is None
    assert [w["passes"] for w in r["windows_seen"]] == pr["passes"] and min(pr["passes"]) >= 1
    lg = r["ledger"]["spec"]
    assert set(lg["terms"]) == set(audit.LEDGER_SOURCES) and lg["gamma"] > 0
    assert all(len(d) == 6 for d in r["draws"].values())
    assert all(w["slack_ns"] >= 0 for w in r["windows_seen"])


def test_a_prover_that_dropped_blocks_is_rejected_at_an_opening():
    (r,), (pr,), _ = run("skip:0.5", samples=8)
    assert not r["accepted"] and not pr["accepted"]
    assert r["failure"]["failure"] == "receipt"
    i, p = r["failure"]["at"]
    assert p in pr["dropped"] and pr["opened"][-1]["on_dropped"] >= 1


def test_a_second_session_waits_out_the_first_and_is_accepted():
    rs, prs, _ = run("honest,honest", sessions=2)
    assert [r["accepted"] for r in rs] == [True, True]
    assert rs[1]["windows_seen"][0]["released_ns"] >= rs[0]["end_ns"]
    assert rs[0]["coins"] != rs[1]["coins"]


def test_each_completed_window_is_archived_before_deferred_opening():
    (r,), (pr,), backend = run("honest", backend_cls=ArchivingBackend)
    assert r["accepted"] and pr["accepted"]
    assert backend.archived == pr["passes"]


@pytest.fixture(scope="module")
def pool():
    from checks import Checks

    c = Checks(2)
    yield c
    c.close()


def test_worker_checks_give_the_verdicts_of_session_verify(monkeypatch, pool):
    monkeypatch.setattr(prover, "OPENINGS_PART", 2)
    monkeypatch.setattr(verifier, "CHECK_BATCH", 3)
    (r,), (pr,), _ = run("honest", checks=pool)
    assert r["accepted"] and pr["accepted"] and r["check_workers"] == 2
    assert all(c > 0 for c in r["verify_cpu_ns"].values())
    (r,), (pr,), _ = run("skip:0.5", samples=8, checks=pool)
    assert not r["accepted"] and r["failure"]["failure"] == "receipt" and r["failure"]["at"][1] in pr["dropped"]


def test_openings_round_trip_on_the_wire_and_a_short_body_is_none():
    o = audit.Opening(3, 17, bytes(range(256)) * 16, b"\1" * 1024, b"\2" * 192, (b"\3" * 64,) * 5, b"\4" * 192,
                      (b"\5" * 64,) * 9, b"")
    body = wire.encode_openings([o, o])
    assert wire.decode_openings(body) == [o, o]
    assert wire.decode_openings(body[:-1]) is None and wire.decode_openings(b"") is None


@pytest.mark.parametrize("name", ["p1024/k3/masked", "p1024/k2/masked", "p512/k2"])
def test_codec_names_round_trip(name):
    assert wire.codec_named(name).name == name


def test_the_predicted_rate_is_the_draws_chance_to_miss_every_dropped_receipt():
    import summary

    assert summary.miss(100, 0, 10) == 1.0 and summary.miss(10, 1, 10) == 0.0
    assert summary.miss(20, 2, 2) == pytest.approx(18 / 20 * 17 / 19)
    U, f, s = 68 << 22, 2.0 ** -14, 2773
    assert summary.miss(U, round(f * U), s) == pytest.approx((1 - f) ** s, rel=1e-4)
