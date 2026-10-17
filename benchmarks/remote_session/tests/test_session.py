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

from verity.core.protocols.remote_bandwidth import audit, timing  # noqa: E402
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
    def __init__(self, clock: PassClock, *args, tiers: dict | None = None, **kw) -> None:
        super().__init__(*args, **kw)
        self.clock, self.tiers = clock, tiers

    def calibrate(self) -> dict:
        return {**super().calibrate(), **({"read_tiers": self.tiers} if self.tiers else {})}

    def run_pass(self, seed: bytes):
        out = super().run_pass(seed)
        self.clock.tick()
        return out


def run(plan: str, *, blocks: int = 16, windows: int = 3, samples: int = 6, sessions: int = 1, checks=None,
        read_bytes_per_s: float = 1e20, tiers: dict | None = None):
    clock = PassClock()
    a, b = socket.socketpair()
    vconn, pconn = wire.Conn(a, clock=clock.now), wire.Conn(b)
    backend = TickingBackend(clock, blocks, random.Random(7), read_bytes_per_s=read_bytes_per_s, tiers=tiers)
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
    return out, results


def test_an_honest_session_is_accepted_with_its_ledger(monkeypatch):
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    (r,), (pr,) = run("honest")
    assert r["accepted"] and pr["accepted"] and r["failure"] is None
    assert [w["passes"] for w in r["windows_seen"]] == pr["passes"] and min(pr["passes"]) >= 1
    assert pr["discarded"] >= r["windows"]
    lg = r["ledger"]["spec"]
    assert set(lg["terms"]) == set(audit.LEDGER_SOURCES) and lg["gamma"] > 0
    assert all(len(d) == 6 for d in r["draws"].values())
    assert all(w["slack_ns"] >= 0 for w in r["windows_seen"])


def test_a_prover_reporting_a_slow_read_cannot_lower_either_ledgers_capacity(monkeypatch):
    """MEM-24: the capacity is the verifier's, and the prover's calibration can only raise it."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    (slow,), _ = run("honest", read_bytes_per_s=1.0)
    (fast,), _ = run("honest", read_bytes_per_s=1e20)
    assert slow["ledger"]["inputs"]["reported_read_bytes_per_s"] == 1.0
    assert slow["ledger"]["measured"]["capacity_bytes_per_s"] == timing.READ_BYTES_PER_S
    assert slow["ledger"]["spec"]["capacity_bytes_per_s"] == timing.SPEC_BYTES_PER_S
    assert fast["ledger"]["measured"]["capacity_bytes_per_s"] == fast["ledger"]["spec"]["capacity_bytes_per_s"] == 1e20


def test_the_ledgers_are_the_nodes_whatever_share_of_it_the_store_is_on(monkeypatch):
    """MEM-26: R is every GPU's HBM and the host's DRAM, S_chip every GPU's chip and the host's caches, and the squaring
    ceiling the node's, for a store on one GPU too: its calibration's one-GPU read lowers none of them, and its
    certified bytes are the same at either R."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    (r,), _ = run("honest", read_bytes_per_s=timing.GPU_READ_BYTES_PER_S)
    lg = r["ledger"]
    node_read = timing.GPUS * timing.GPU_READ_BYTES_PER_S + timing.HOST_READ_BYTES_PER_S
    node_spec = timing.GPUS * timing.GPU_SPEC_BYTES_PER_S + timing.HOST_SPEC_BYTES_PER_S
    assert lg["inputs"]["reported_read_bytes_per_s"] == timing.GPU_READ_BYTES_PER_S
    assert lg["measured"]["capacity_bytes_per_s"] == lg["inputs"]["measured_read_bytes_per_s"] == node_read
    assert lg["spec"]["capacity_bytes_per_s"] == lg["inputs"]["spec_bytes_per_s"] == node_spec
    assert lg["inputs"]["chip_bytes"] == timing.GPUS * timing.GPU_CHIP_BYTES + timing.HOST_CHIP_BYTES == 3_045_746_688
    assert lg["inputs"]["squarings_per_s"] == timing.GPUS * timing.GPU_SQUARINGS_PER_S["p512"]
    assert lg["spec"]["certified_bytes"] == pytest.approx(lg["measured"]["certified_bytes"], rel=1e-9)


def test_the_timing_parts_reproduce_the_whole_node_sessions_split_and_sum_to_its_timing_term():
    """r20261015-203743-d101 (15 Oct), 8 GPUs' HBM and a host tier, every pass the whole store: its parts are #2089's
    split, and at either R they sum to the ledger's timing term."""
    d101 = {"span_ns": 420_074_057_129, "passes": 6430, "reads_bytes": 4_888_145_229_250_560, "pass_ns": 65_259_603,
            "tiers": {"hbm": {"bytes": 751_619_276_800, "ms": 61.502974},
                      "host": {"bytes": 8_589_934_592, "ms": 25.468371}}}
    for r in (timing.READ_BYTES_PER_S, timing.SPEC_BYTES_PER_S):
        parts = verifier.timing_parts(capacity_bytes_per_s=r, **d101)
        assert sum(parts.values()) == pytest.approx(1 - d101["reads_bytes"] / (r * d101["span_ns"] / 1e9), abs=1e-12)
    parts = verifier.timing_parts(capacity_bytes_per_s=timing.READ_BYTES_PER_S, **d101)
    assert sum(parts.values()) == pytest.approx(0.09926702029063428, abs=1e-12)
    assert {k: round(v * 100, 2) for k, v in parts.items()} == {
        "above_stated_tiers": 0.0, "host_dram_idle": 3.68, "hbm_below_stated": 0.66, "receipts_beyond_read": 5.48,
        "boundaries_and_allowance": 0.10, "blocks_left_out": 0.0}
    spec = verifier.timing_parts(capacity_bytes_per_s=timing.SPEC_BYTES_PER_S, **d101)
    assert spec["above_stated_tiers"] == pytest.approx(1 - timing.READ_BYTES_PER_S / timing.SPEC_BYTES_PER_S)


def test_a_sessions_ledgers_carry_the_timing_parts_when_its_calibration_reads_each_tier(monkeypatch):
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    block = rcodec.BLOCK_BYTES
    tiers = {"hbm": {"bytes": 12 * block, "ms": 1e-6}, "host": {"bytes": 4 * block, "ms": 1e-6}}
    (r,), _ = run("honest", read_bytes_per_s=1.0, tiers=tiers)
    lg = r["ledger"]
    assert lg["inputs"]["timing_parts"]["tiers"] == tiers and lg["inputs"]["timing_parts"]["pass_ns"] >= 0
    for name in ("spec", "measured"):
        assert sum(lg[name]["timing_parts"].values()) == pytest.approx(lg[name]["terms"]["timing"], abs=1e-12)
    assert lg["measured"]["timing_parts"]["above_stated_tiers"] == pytest.approx(0, abs=1e-15)
    for short in ({"hbm": {"bytes": 15 * block, "ms": 1e-6}}, None):
        (r,), _ = run("honest", read_bytes_per_s=1.0, tiers=short)
        assert "timing_parts" not in r["ledger"]["measured"] and "timing_parts" not in r["ledger"]["inputs"]


def test_a_second_session_waits_out_the_first_and_is_accepted():
    rs, prs = run("honest,honest", sessions=2)
    assert [r["accepted"] for r in rs] == [True, True]
    assert rs[1]["windows_seen"][0]["released_ns"] >= rs[0]["end_ns"]
    assert rs[0]["coins"] != rs[1]["coins"]


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
