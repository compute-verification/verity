"""The verifier's and the prover's processes over a loopback socket, the driver's reference as the prover's backend and
a clock that advances by one tick per pass with both sides settled around each tick, so a session runs one schedule:
an honest session is accepted and its γ terms computed, a prover that dropped blocks is rejected at its openings, a
second session waits out the first's deadline, and KV pages appended and released mid-session (a memory map's steps)
are audited, accepted when held and rejected when dropped. Sessions run under the verifier's default verdict, `both`:
the Lean verifier decides, `audit.Session` runs beside it, and the two must agree at every event and window."""
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


VERIFIER, PROVER = 0, 1


class Counted(wire.Conn):
    """One side's end of the loopback, stamping with the pass clock and telling it of each frame it sends and each
    time it asks for its next: a side asking for its (n + 1)-th frame has handed on its n-th."""

    def __init__(self, sock: socket.socket, clock: PassClock, side: int) -> None:
        super().__init__(sock, clock=clock.now)
        self.pass_clock, self.side, self.asked = clock, side, 0

    def send(self, head: dict, body: bytes = b"") -> None:
        self.pass_clock.sending(self.side, head["type"])
        super().send(head, body)

    def recv(self):
        self.pass_clock.reading(self.side, self.asked)
        self.asked += 1
        return super().recv()


class PassClock:
    """Time is the prover's passes: one tick as each pass ends, both sides settled around it. A tick waits until the
    verifier's reader has handed on every frame the prover sent, the verifier is waiting on its clock with an empty
    inbox (or has ended the session) and the prover's reader has handed on every frame the verifier sent, and again
    once time has moved, so whatever either side sends in answer reaches the other at the tick it was sent, and a
    session runs one schedule however its threads are scheduled. A wait returns the inbox's next message or None once
    the clock reaches its end, woken by either; once the prover has said its passes are done, time jumps to the wait's
    end."""

    def __init__(self) -> None:
        self.t, self.done, self.inbox, self.parked, self.idle_at, self.ended = 0, False, None, None, None, False
        self.frames = threading.Condition()
        self.sent, self.asked = [0, 0], [0, 0]

    def conns(self) -> tuple[Counted, Counted]:
        a, b = socket.socketpair()
        return Counted(a, self, VERIFIER), Counted(b, self, PROVER)

    def now(self) -> int:
        return self.t

    def sending(self, side: int, kind: str) -> None:
        with self.frames:
            self.sent[side] += 1
        if side == VERIFIER and kind in ("session", "end"):
            if self.inbox is None:
                self.ended = kind == "end"
                return
            with self.inbox.mutex:
                self.ended = kind == "end"
                self.parked.notify_all()

    def reading(self, side: int, n: int) -> None:
        with self.frames:
            self.asked[side] = n
            self.frames.notify_all()

    def tick(self) -> None:
        self._settle()
        with self.inbox.mutex:
            self.t += 1
            self.inbox.not_empty.notify_all()
        self._settle()

    def _settle(self) -> None:
        with self.frames:
            self.frames.wait_for(lambda: self.inbox is not None and self.asked[VERIFIER] == self.sent[PROVER])
        with self.inbox.mutex:
            self.parked.wait_for(lambda: self.ended or (self.idle_at == self.t and not self.inbox._qsize()))
        with self.frames:
            self.frames.wait_for(lambda: self.asked[PROVER] == self.sent[VERIFIER])

    def wait(self, inbox: queue.Queue, until_ns):
        if self.inbox is None:
            with self.frames:
                self.inbox, self.parked = inbox, threading.Condition(inbox.mutex)
                self.frames.notify_all()
        with inbox.not_empty:
            while True:
                if until_ns is not None and self.done:
                    self.t = max(self.t, until_ns)
                if until_ns is not None and self.t >= until_ns:
                    return None
                if inbox._qsize():
                    return inbox._get()
                self.idle_at = self.t
                self.parked.notify_all()
                inbox.not_empty.wait()
                self.idle_at = None

    def idle(self) -> None:
        with self.inbox.mutex:
            self.done = True
            self.inbox.not_empty.notify_all()

    def busy(self) -> None:
        self.done = False


class TickingBackend(prover.CpuBackend):
    def __init__(self, clock: PassClock, *args, tiers: dict | None = None, layout: list | None = None, **kw) -> None:
        super().__init__(*args, **kw)
        self.clock, self.tiers, self._layout = clock, tiers, layout

    def layout(self) -> list | None:
        return self._layout

    def calibrate(self) -> dict:
        return {**super().calibrate(), **({"read_tiers": self.tiers} if self.tiers else {})}

    def run_pass(self, seed: bytes, plan=None):
        out = super().run_pass(seed, plan)
        self.clock.tick()
        return out


def run(plan: str, *, blocks: int = 16, windows: int = 3, samples: int = 6, sessions: int = 1, checks=None,
        read_bytes_per_s: float = 1e20, tiers: dict | None = None, lifetime: int = audit.LIFETIME,
        registry: audit.Registry | None = None, layout: list | None = None, host_bytes: int = 0,
        verdict: str = verifier.VERDICT, then=None, transcript: Path | None = None, **kv):
    """The verifier's sessions and the prover's over one registration, under `verdict`; `then(v)` once they end."""
    clock = PassClock()
    vconn, pconn = clock.conns()
    backend = TickingBackend(clock, blocks, random.Random(7), read_bytes_per_s=read_bytes_per_s, tiers=tiers,
                             layout=layout)
    p = prover.ProverSide(pconn, backend, b"test-node", prover.parse_plan(plan), seed=11, log=lambda m: None, **kv)
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
                          checks=checks, lifetime=lifetime, registry=registry, host_bytes=host_bytes, verdict=verdict,
                          transcript=transcript)
    try:
        v.register()
        allowance = v.allowance(v.ping(4))
        out, not_before = [], None
        for k in range(sessions):
            r = v.session(k, allowance, not_before)
            out.append(r)
            not_before = r.get("end_ns", clock.now())
        th.join()
        if then is not None:
            then(v)
    finally:
        v.stop()
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


def test_a_session_on_the_pass_clock_runs_one_schedule(monkeypatch):
    """Each tick settles both sides, so an honest session's windows are the same at every run: window 0's boundary
    falls at tick 20, as its 20th pass ends, and each later window starts one tick after its coin, the pass in flight
    at the boundary discarded, so it holds 19 passes; each root reaches the verifier at the tick its pass ended, and a
    window's draws go out as the next window's first root arrives."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    (r,), (pr,) = run("honest")
    assert r["accepted"] and pr["passes"] == [20, 19, 19] and pr["discarded"] == 3
    seen = r["windows_seen"]
    assert [(w["released_ns"], w["boundary_ns"]) for w in seen] == [(0, 20), (20, 40), (40, 60)]
    assert [w["arrivals_ns"] for w in seen] == [list(range(1, 21)), list(range(22, 41)), list(range(42, 61))]
    assert [w["draws_sent_ns"] for w in seen[:2]] == [22, 42]


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


def test_kv_pages_appended_and_released_mid_session_are_audited(monkeypatch):
    """Two sessions over one registration, each beside a serving trace of memory-map/v1 steps (2 pages every 4th pass
    over a pool of 10 slots): an honest one is accepted, its record counting the appends, releases and post-seed reads
    the verifier saw, which its ledger leaves uncredited without a property verdict; then one whose prover drops the
    pages it appends, whose appends start again from N, is rejected at a draw on a dropped page.  At 16 draws a window
    the second's chance of drawing no dropped page is below 2^-30 at the passes these sessions make."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    (honest, dropping), (ph, pd) = run("kv:2,kvdrop:2", sessions=2, samples=16, kv_every=4, kv_pool=8)
    assert honest["accepted"] and ph["accepted"] and honest["failure"] is None
    kv = ph["kv"]
    assert kv["pool_slots"] == 10 and kv["steps"] >= 6 and kv["appended"] == 2 * kv["steps"]
    assert kv["sent_appended"] == honest["appended"] >= 2 * (kv["steps"] - 2)
    assert kv["sent_released"] == honest["released"] > 8
    assert kv["pass_appended"][0] == 0 and 0 < max(kv["pass_appended"]) <= 10
    assert all(8 < n <= 18 for n in kv["pass_blocks"])
    pub = honest["record"]["public"]
    assert (pub["appended"], pub["released"]) == (honest["appended"], honest["released"])
    assert sum(pub["post_seed"]) == sum(kv["pass_appended"]) > 0
    steps = [s for w in honest["record"]["profile"]["windows"] for s in w["steps"]]
    assert len(steps) == sum(pub["passes"]) and all(a < b for a, b in zip(steps, steps[1:]))
    lg = honest["ledger"]
    assert lg["inputs"]["post_seed_receipts"] == sum(pub["post_seed"]) and not lg["inputs"]["post_seed_credited"]
    reads = lg["inputs"]["reads_bytes"]
    whole = lg["spec"]["capacity_bytes_per_s"] * pub["span_ns"] / 1e9
    credited = reads - sum(pub["post_seed"]) * rcodec.BLOCK_BYTES
    assert lg["spec"]["terms"]["timing"] == pytest.approx(1 - credited / whole, rel=1e-12)
    assert not dropping["accepted"] and not pd["accepted"]
    assert dropping["failure"]["failure"] == "receipt" and pd["failure_on_dropped"] is True
    assert pd["kv"]["appended"] > 0 and min(pd["dropped"]) == 16 and dropping["appended"] > 0


def test_kv_pages_are_checked_by_worker_processes_under_seeds_that_bind_the_appends(monkeypatch):
    """Each pass's seed binds the appends sent before its root, and the verifier's check workers recompute it from
    the window's snapshot (`checks.Checks.snapshot`, the appends' head at each root among it): a kv session checked
    by one worker process is accepted, as `Session.verify` accepts it."""
    from checks import Checks

    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    checks = Checks(1)
    try:
        (r,), (p,) = run("kv:2", samples=16, kv_every=2, kv_pool=8, checks=checks)
    finally:
        checks.close(wait=True)
    assert r["accepted"] and p["accepted"] and r["failure"] is None and r["appended"] > 0


def test_the_check_workers_bound_each_windows_wrong_receipts_as_its_profile_does():
    from checks import Checks

    from verity.core.protocols.profile import IntegrityProfile, Subset

    profiles = [IntegrityProfile(("receipt",), (n,), {"receipt": Subset(k)}, 2.0 ** -40, drawn={"receipt": k})
                for n, k in ((4096, 64), (10 ** 6, 300), (10 ** 6, 301))]
    checks = Checks(2)
    try:
        assert checks.bounds(profiles) == [p.bound() for p in profiles]
    finally:
        checks.close(wait=True)


def test_kv_pages_past_the_lifetime_are_re_keyed_and_a_prover_that_keeps_them_is_rejected(monkeypatch):
    """Under L = 1 window, sent with the session, every appended page is stale in the window after the one it joined.
    The honest prover re-keys each before that window's first pass (writes its payload again in its slot, at a fresh
    index) and is accepted. One that never re-keys is rejected at that pass's root (`lifetime`)."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    (r,), (p,) = run("kv:2", samples=8, kv_every=4, kv_pool=8, lifetime=1)
    assert r["accepted"] and p["accepted"] and r["record"]["public"]["lifetime"] == 1
    kv = p["kv"]
    assert kv["lifetime"] == 1 and kv["rekeyed"] > 0 and kv["appended"] == 2 * kv["steps"] + kv["rekeyed"]
    monkeypatch.setattr(prover.ServingTrace, "stale", lambda self, window: [])
    (r,), (p,) = run("kv:2", samples=8, kv_every=4, kv_pool=8, lifetime=1)
    assert not r["accepted"] and not p["accepted"] and p["kv"]["rekeyed"] == 0
    assert r["failure"]["failure"] == "lifetime" and r["failure"]["t"] == 1


def test_serving_runs_at_most_k_steps_past_the_pass_in_flight_and_reuses_a_freed_block_once_no_pass_needs_it():
    """`Ahead` at k = 2, serving scripted: it runs steps 0..2 and waits on the bound, so the first pass takes step 3
    with their records; serving then runs on to the bound past it, 5, but its step 4 reuses a block freed at step 3,
    which the pass at 3 holds, so it waits until that pass has read.  Then serving runs to 6, the next pass takes 7
    with the records of 3..6, and serving runs to 9.  Once the session ends serving runs free, and a serving error
    reaches the next take."""
    a, ran = prover.Ahead(2, fill=True), []

    def serving() -> None:
        s = 0
        while a.begin(s):
            if s == 4:
                a.reuse(3)
            ran.append(s)
            a.hand(s, f"record {s}")
            s += 1

    th = threading.Thread(target=serving, daemon=True)
    th.start()
    assert a.take() == (3, ["record 0", "record 1", "record 2"])
    assert ran == [0, 1, 2, 3] and a.waits == {"bound": 1, "fence": 1}
    a.passed(3)
    assert a.take() == (7, ["record 3", "record 4", "record 5", "record 6"])
    assert ran == list(range(10)) and a.waits == {"bound": 3, "fence": 1}
    a.end()
    th.join()
    assert ran == list(range(10)) and a.leads == [1] and a.stats()["steps"] == 10
    b = prover.Ahead(0)
    b.fail(ValueError("the engine raised"))
    with pytest.raises(RuntimeError, match="serving raised"):
        b.take()
    with pytest.raises(ValueError):
        prover.Ahead(-1)


@pytest.mark.slow
@pytest.mark.parametrize("k", [0, 2])
def test_a_pipelined_session_with_serving_up_to_k_steps_past_its_passes_is_accepted(monkeypatch, k):
    """Serving on its own thread (`Ahead`, its furthest schedule) writes a page a step into a slot freed k + 1 steps
    before, each pass covers the newest step serving has handed and reads every block in its slot, and pages stale
    under L = 1 are re-keyed at each window's first pass: the session is accepted, no pass read a block serving had
    overwritten, serving ran more steps than there were passes, every pass read with serving at most k steps past its
    own and some at exactly k, and the fence held serving once: at the first step, whose write overwrites the pool's
    filler in place and waits on its own step's pass (the filler the next k + 1 steps write is freed there)."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    (r,), (p,) = run("kv:1", windows=2, samples=16, kv_every=1, kv_pool=8, lifetime=1, ahead=k, fill=True,
                     kv_headroom=k + 1)
    assert r["accepted"] and p["accepted"] and r["failure"] is None and p["misread"] == []
    a, kv = p["ahead"], p["kv"]
    assert all(0 <= lead <= k for lead in a["leads"]) and max(a["leads"]) == k and a["fence_waits"] == 1
    assert len(a["leads"]) == sum(p["passes"]) + p["discarded"] < a["steps"]
    assert kv["rekeyed"] > 0 and kv["appended"] == kv["steps"] + kv["rekeyed"]
    assert r["appended"] == kv["sent_appended"] > 0 and r["released"] == kv["sent_released"] > 0


@pytest.mark.slow
def test_a_pipelined_prover_that_overwrites_a_freed_block_before_its_pass_reads_it_is_rejected_at_that_receipt(
        monkeypatch):
    """Serving up to 2 steps past the pass in flight, each write into a slot freed the step before: an honest
    session's fence holds serving's writes until no pass to come holds the block, and it is accepted; one whose writes
    skip the fence (`kvoverwrite`) lands pages over blocks the pass in flight still has to read, and is rejected at a
    receipt of one of them.  Each of its passes but the second (whose steps write the pool's empty slots) misreads 2
    of its at most 17 blocks, so at 96 draws a window its chance of drawing none in either window is below 2^-30."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    (honest, over), (ph, po) = run("kv:1,kvoverwrite:1", sessions=2, windows=2, samples=96, kv_every=1, kv_pool=8,
                                   ahead=2, fill=True, kv_headroom=1)
    assert honest["accepted"] and ph["accepted"] and ph["misread"] == []
    assert ph["ahead"]["fence_waits"] > 0 and all(0 <= lead <= 2 for lead in ph["ahead"]["leads"])
    assert not over["accepted"] and not po["accepted"] and po["misread"]
    assert over["failure"]["failure"] == "receipt" and po["failure_on_misread"] is True


class CarrierVerdict:
    """A property verdict that holds, its carriers at most `carrier` bytes a window (`audit.PostSeedVerdict`)."""

    def __init__(self, carrier: float, holds: bool = True) -> None:
        self.carrier, self.holds = carrier, holds

    def carrier_bytes(self, read_bytes: int) -> float:
        return self.carrier


def test_the_memory_line_credits_the_labelled_weights_and_kv_only_on_a_verdict(monkeypatch):
    """The ledger's memory line on a kv session whose hello labels its store: 8 blocks of weights, then the KV pool's 8
    pages, in HBM.  At a draw of every receipt the floor is 0, so each window credits the weights every pass holds,
    less the blocks the node's squarings could rebuild in its span (none here: the clock's spans are ticks, so the
    session runs at no squarings, and a rate is put back below); the KV the session appended is held and read but
    credited only on the property's verdict, then less its carriers; the share and residual are of the node's HBM.
    Unlabelled, the whole registered store is filler and no weights are credited."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    monkeypatch.setattr(timing, "SQUARINGS_PER_S", {"p512": 0.0})
    seen = {}
    memory = verifier.memory

    def keep(*a, **kw):
        seen["args"] = a, kw
        return memory(*a, **kw)

    monkeypatch.setattr(verifier, "memory", keep)
    layout = [{"kind": "weights", "tier": "hbm", "first": 0, "blocks": 8},
              {"kind": "kv", "tier": "hbm", "first": 8, "blocks": 8}]
    (r,), (p,) = run("kv:2", samples=4096, kv_every=4, kv_pool=8, layout=layout)
    assert r["accepted"] and p["accepted"] and r["appended"] > 0
    B, mem, pub = rcodec.BLOCK_BYTES, r["ledger"]["memory"], r["record"]["public"]
    assert mem["labels"] and not mem["kv_credited"] and mem["hbm_bytes"] == verifier.HBM_BYTES == 8 * 96 * 2 ** 30
    for w, posts in zip(mem["windows"], (w["post_seed"] for w in r["record"]["profile"]["windows"])):
        assert w["floor"] == 0 == w["lost_blocks"] == w["epsilon"] and w["held"]["weights"] == 8 * B
        assert w["held"]["kv"] == min(posts) * B and w["read"]["kv"] == sum(posts) * B and w["held"]["host"] == 0
        assert 8 * B <= w["held"]["hbm"] == w["basis"] == w["credited"]["hbm"] <= 16 * B
        assert w["credited"]["weights"] == 8 * B and w["credited"]["kv"] == w["credited"]["kv_read"] == 0
    assert mem["read"]["kv"] == sum(pub["post_seed"]) * B > 0 and mem["kv_bytes"] == mem["kv_read_bytes"] == 0
    assert mem["credited_bytes"] == min(w["credited"]["hbm"] for w in mem["windows"]) and mem["weights_bytes"] == 8 * B
    assert mem["share"] == mem["credited_bytes"] / mem["hbm_bytes"] and mem["residual"] == 1 - mem["share"]
    assert mem["delta"] == 2.0 ** -40 and mem["epsilon"] == 0

    (record, plans, regions), kw = seen["args"]
    assert regions == (("weights", "hbm", 0, 8), ("kv", "hbm", 8, 8))
    span = max(w.span_ns for w in record.windows)
    rate = record.codec.encode_squarings * 1e9 / span / 2
    rebuilt = verifier.memory(record, plans, regions, hbm_bytes=kw["hbm_bytes"],
                              rebuild=audit.Rebuild(record.codec, rate))
    for w, x in zip(record.windows, rebuilt["windows"]):
        assert x["lost_blocks"] == pytest.approx(w.span_ns / span / 2) and x["lost_blocks"] <= 0.5
        assert x["credited"]["weights"] == pytest.approx(8 * B - x["lost_blocks"] * B)
        assert x["epsilon"] == pytest.approx(x["lost_blocks"] * B / x["basis"])
        assert x["credited"]["hbm"] == pytest.approx((1 - x["epsilon"]) * x["basis"])
    whole = verifier.memory(record, plans, regions, credit_post_seed=CarrierVerdict(0.0), **kw)
    carried = verifier.memory(record, plans, regions, credit_post_seed=CarrierVerdict(B / 2), **kw)
    assert whole["kv_credited"] and whole["credited_bytes"] >= mem["credited_bytes"]
    for w, c, posts in zip(whole["windows"], carried["windows"], (w.post_seed for w in record.windows)):
        lost = w["lost_blocks"] * B
        assert w["credited"]["kv_read"] == pytest.approx(max(0, sum(posts) * B - lost))
        assert c["credited"]["kv_read"] == pytest.approx(max(0, sum(posts) * B - B / 2 * sum(map(bool, posts)) - lost))
        assert w["credited"]["kv"] == pytest.approx(max(0, min(posts) * B - lost))
    assert whole["kv_read_bytes"] > carried["kv_read_bytes"] > 0
    with pytest.raises(ValueError, match="fails"):
        verifier.memory(record, plans, regions, credit_post_seed=CarrierVerdict(0.0, holds=False), **kw)

    bare = verifier.memory(record, plans, None, **kw)
    assert not bare["labels"] and bare["weights_bytes"] == 0 and bare["credited_bytes"] == mem["credited_bytes"]


def test_the_node_line_credits_each_kind_against_the_nodes_hbm_and_host_dram(monkeypatch):
    """With the node's host DRAM given, the memory line is also by kind against HBM and DRAM: a store of 8 weights
    blocks and 4 of filler in HBM, then 4 of filler in host memory, every pass reading all 16, credits each kind what
    every pass held less the window's lost blocks, and in all every held block less them once, KV nothing without a
    verdict; each share is of HBM plus DRAM, the residual what the total leaves, and the line is its worst window's."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    monkeypatch.setattr(timing, "SQUARINGS_PER_S", {"p512": 0.0})
    seen, memory = {}, verifier.memory

    def keep(*a, **kw):
        seen["args"] = a, kw
        return memory(*a, **kw)

    monkeypatch.setattr(verifier, "memory", keep)
    layout = [{"kind": "weights", "tier": "hbm", "first": 0, "blocks": 8},
              {"kind": "filler", "tier": "hbm", "first": 8, "blocks": 4},
              {"kind": "filler", "tier": "host", "first": 12, "blocks": 4}]
    host = 1843 * 10 ** 9
    (r,), (p,) = run("honest", samples=4096, layout=layout, host_bytes=host)
    assert r["accepted"] and p["accepted"]
    B, node, whole = rcodec.BLOCK_BYTES, r["ledger"]["memory"]["node"], verifier.HBM_BYTES + host
    assert node["node_bytes"] == whole and node["host_bytes"] == host and node["lost_bytes"] == 0
    assert node["held"] == {"weights": 8 * B, "filler_hbm": 4 * B, "host": 4 * B}
    assert node["credited"] == {"weights": 8 * B, "filler_hbm": 4 * B, "host": 4 * B, "kv": 0}
    assert node["credited_bytes"] == 16 * B and node["residual"] == 1 - 16 * B / whole
    assert node["share"] == {k: v / whole for k, v in node["credited"].items()}

    (record, plans, regions), kw = seen["args"]
    span = max(w.span_ns for w in record.windows)
    rate = record.codec.encode_squarings * 1e9 / span / 2
    lossy = verifier.memory(record, plans, regions, hbm_bytes=kw["hbm_bytes"], rebuild=audit.Rebuild(record.codec, rate))
    line = verifier.node_line(lossy, host_bytes=host)
    worst = max(lossy["windows"], key=lambda x: x["lost_blocks"])
    lost = worst["lost_blocks"] * B
    assert line["window"] == worst["window"] and line["lost_bytes"] == lost == pytest.approx(B / 2)
    assert line["credited"]["weights"] == pytest.approx(8 * B - lost)
    assert line["credited"]["filler_hbm"] == pytest.approx(4 * B - lost) == line["credited"]["host"]
    assert line["credited_bytes"] == pytest.approx(16 * B - lost) and line["held"] == node["held"]


def test_an_engines_arena_puts_each_weight_at_a_block_boundary_and_refuses_what_does_not_fit():
    import engine

    assert engine.pack([1, 4096, 4097, 10], 5 * 4096) == [0, 4096, 8192, 16384]
    with pytest.raises(ValueError, match="more than the arena"):
        engine.pack([1, 4096, 4097, 10], 5 * 4096 - 1)


def test_aligned_engines_step_at_one_instant_of_the_host_clock_whatever_their_perf_counters_read():
    import engine

    due = [clock + engine.aligned(perf, clock, 0.5) - perf for perf, clock in ((3.0, 1000.12), (77.25, 1000.31))]
    assert due == [pytest.approx(1000.5)] * 2
    assert engine.aligned(10.0, 1001.0, 0.5) == pytest.approx(10.5)


@pytest.mark.parametrize("layout", [
    [{"kind": "weights", "tier": "hbm", "first": 0, "blocks": 8}],
    [{"kind": "weights", "tier": "hbm", "first": 0, "blocks": 8},
     {"kind": "kv", "tier": "hbm", "first": 9, "blocks": 7}],
    [{"kind": "code", "tier": "hbm", "first": 0, "blocks": 16}],
    [{"kind": "kv", "tier": "nvme", "first": 0, "blocks": 16}],
    [{"kind": "kv", "tier": "hbm", "first": 0, "blocks": 0}, {"kind": "kv", "tier": "hbm", "first": 0, "blocks": 16}],
    {"kind": "kv", "tier": "hbm", "first": 0, "blocks": 16},
    ["kv"],
])
def test_a_layout_that_is_not_the_stores_regions_in_order_is_refused(layout):
    with pytest.raises(ValueError):
        verifier.regions_of(layout, 16)


def registering(registry: audit.Registry) -> tuple[str, str]:
    """A prover's hello to a verifier holding `registry`, for a node with a live registration there: the verifier's
    refusal, and the reason the prover read in its `bye`."""
    a, b = socket.socketpair()
    p = prover.ProverSide(wire.Conn(b), prover.CpuBackend(16, random.Random(7)), b"test-node",
                          prover.parse_plan("honest"), seed=11, log=lambda m: None)
    heard = []

    def prove() -> None:
        try:
            p.register(codec=P512)
        except wire.Closed as e:
            heard.append(str(e))

    th = threading.Thread(target=prove, daemon=True)
    th.start()
    v = verifier.Verifier(wire.Conn(a), source=os.urandom(32), windows=3, window_ns=20, samples=6, delta=2.0 ** -40,
                          jitter_ns=10 ** 9, log=lambda m: None, registry=registry)
    with pytest.raises(audit.RegistryError) as refused:
        v.register()
    th.join()
    v.stop()
    return str(refused.value), heard[0]


def greeted(lifetime: int, deferred: bool) -> tuple[verifier.Verifier, ValueError | None, list[str]]:
    """A prover whose backend says `deferred` greeting a verifier of lifetime cap `lifetime`: the verifier, its refusal
    if it refused, and the reason the prover read in its `bye`."""
    a, b = socket.socketpair()
    backend = prover.CpuBackend(16, random.Random(7))
    backend.deferred = deferred
    p = prover.ProverSide(wire.Conn(b), backend, b"test-node", prover.parse_plan("honest"), seed=11, log=lambda m: None)
    heard = []

    def prove() -> None:
        try:
            p.register(codec=P512)
        except wire.Closed as e:
            heard.append(str(e))

    th = threading.Thread(target=prove, daemon=True)
    th.start()
    v = verifier.Verifier(wire.Conn(a), source=os.urandom(32), windows=3, window_ns=20, samples=6, delta=2.0 ** -40,
                          jitter_ns=10 ** 9, log=lambda m: None, lifetime=lifetime)
    refused = None
    try:
        v.register()
    except ValueError as e:
        refused = e
    th.join()
    v.stop()
    return v, refused, heard


def test_a_prover_that_defers_residency_is_refused_under_a_lifetime_of_one_window():
    """The hello says whether the prover defers an appended block's residency to its encode: a block appended in window
    w is then resident from w or w + 1, so its re-key falls at age L − 1, and under L = 1 the verifier refuses it with
    `bye` before drawing a registration.  At L = 2, or not deferring, the session's parameters carry it."""
    v, refused, heard = greeted(1, True)
    assert refused is not None and "lifetime of at least 2 windows, not 1" in str(refused)
    assert heard == [f"refused: {refused}"] and v.registry.drawn(b"test-node") == 0
    for lifetime, deferred in ((2, True), (1, False)):
        v, refused, heard = greeted(lifetime, deferred)
        assert refused is None and heard == [] and v.deferred is deferred and v.registry.drawn(b"test-node") == 1


def test_a_node_with_a_live_registration_is_refused_until_the_verifier_closes_it_with_its_sessions_counted(monkeypatch):
    """The red team's KB-4: a second registration is a retry that redraws the codec seed.  Over one registry, a
    second hello from the node while its registration is live draws nothing and is answered `bye` with the reason;
    closing the registration counts its sessions, the dropping prover's as its failure, and only then does the node
    register again, R = 2."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    registry = audit.Registry()
    (honest, dropping), _ = run("honest,skip:0.5", sessions=2, registry=registry)
    assert honest["accepted"] and not dropping["accepted"] and dropping["failure"]["failure"] == "receipt"
    assert registry.live(b"test-node") == bytes.fromhex(honest["registration"])
    refused, heard = registering(registry)
    assert "live registration" in refused and heard == f"refused: {refused}"
    assert registry.refused == {b"test-node": 1} and registry.drawn(b"test-node") == 1
    closed = registry.close(b"test-node")
    assert closed.sessions == ("accepted", "receipt") and closed.failed == 1
    (again,), _ = run("honest", registry=registry)
    assert again["accepted"] and again["registration"] != honest["registration"] and registry.drawn(b"test-node") == 2


def test_the_verifier_closes_a_registration_left_live_from_its_registry_file(tmp_path, monkeypatch, capsys):
    """`--registry R --close NODE`: a registration a verifier process left live (its session in flight when it ended)
    closes with that session unended, counted as failed, and the node may register again."""
    registry = audit.Registry.from_json({"schema": audit.REGISTRY_SCHEMA, "closed": [], "refused": {},
                                         "live": {b"test-node".hex(): {"registration": "ab" * 64,
                                                                        "sessions": ["accepted", audit.UNENDED]}}})
    path = tmp_path / "registry.json"
    verifier.save_registry(path, registry)
    monkeypatch.setattr(sys, "argv", ["verifier.py", "--registry", str(path), "--close", b"test-node".hex()])
    assert verifier.main() == 0
    back = verifier.load_registry(path)
    assert back.live(b"test-node") is None and back.closed == [
        audit.Closed(b"test-node", bytes.fromhex("ab" * 64), ("accepted", audit.UNENDED))]
    assert "closed" in capsys.readouterr().out


@pytest.mark.parametrize("verdict", verifier.VERDICTS)
def test_each_verdict_accepts_an_honest_session_and_rejects_a_dropping_one(monkeypatch, verdict):
    """`--verdict`: the Lean verifier decides under `lean` and `both` (its check's time is each window's
    `lean_check_ns`), with `audit.Session` checking beside it under `both`, and `audit.Session` alone under `python`,
    with no Lean process. Each accepts the honest session, rejects the prover that dropped half its blocks at a receipt,
    and closes the registration with the two sessions counted so in both registries."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    closed, lean = [], []

    def then(v):
        lean.append(v.lean)
        closed.append(v.close_registration(b"test-node"))

    (honest, dropping), _ = run("honest,skip:0.5", sessions=2, verdict=verdict, then=then)
    assert honest["accepted"] and not dropping["accepted"] and dropping["failure"]["failure"] == "receipt"
    assert honest["verdict"] == verdict and honest["disagreement"] is None and dropping["disagreement"] is None
    ns = list(honest["lean_check_ns"].values())
    if verdict == "python":
        assert lean == [None] and ns == [None] * 3
    else:
        assert lean[0] is not None and len(ns) == 3 and all(type(x) is int and x > 0 for x in ns)
    assert closed[0]["sessions"] == ["accepted", "receipt"] and "python" not in closed[0]


def test_a_window_the_verifiers_judge_differently_is_refused_as_disagreement(monkeypatch):
    """Under `both`, a Lean verdict that differs from `audit.Session`'s (the Lean's accepting window 1 read as a
    rejection at a draw) refuses the session at that window as `disagreement`, which both registries count."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    settle = verifier.Pair.settle

    def wrong(self, check):
        res = settle(self, check)
        if res.get("accepted") != 1:
            return res
        return {"failure": "receipt", "window": 1, "at": [0, 0], "openings": res["openings"], "ns": res["ns"]}

    monkeypatch.setattr(verifier.Pair, "settle", wrong)
    closed = []
    (r,), (p,) = run("honest", verdict="both", then=lambda v: closed.append(v.close_registration(b"test-node")))
    assert not r["accepted"] and not p["accepted"] and r["failure"] == {"t": 1, "failure": "disagreement", "at": None}
    assert r["disagreement"]["event"] == "verify" and r["disagreement"]["python"] == {"accepted": 1}
    assert closed[0]["sessions"] == ["disagreement"] and "python" not in closed[0]


@pytest.mark.parametrize("verdict", ["both", "python"])
def test_a_transcript_replays_each_session_to_its_verdict_through_both_verifiers(tmp_path, monkeypatch, verdict):
    """`--transcript DIR`: each session's difftest case, written as it runs, which `replay.py transcript` replays
    through `RemoteAudit.Session` and `audit.Session` alike: the honest session accepted, every window verified, and
    the prover that dropped half its blocks rejected at the window and draw it was rejected at live, at a receipt,
    which only a transcript's openings can show. The file is the path, bytes and sha256 verifier.json gives it."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    monkeypatch.syspath_prepend(str(HERE.parents[2] / "verity/core/protocols/remote_bandwidth/lean/tests"))
    import hashlib
    import json

    import replay

    (honest, dropping), _ = run("honest,skip:0.5", sessions=2, verdict=verdict, transcript=tmp_path)
    assert honest["accepted"] and dropping["failure"]["failure"] == "receipt"
    f = dropping["failure"]
    rows, verified = [], []
    with replay.server() as s:
        for r in (honest, dropping):
            t = r["transcript"]
            raw = Path(t["path"]).read_bytes()
            assert Path(t["path"]).parent == tmp_path and t["bytes"] == len(raw)
            assert t["sha256"] == hashlib.sha256(raw).hexdigest()
            case = json.loads(raw)
            verified.append([e["t"] for e in case["events"] if e["kind"] == "verify"])
            rows.append(replay.replay(case, s))
    assert all(row["agree"] for row in rows), rows
    assert sorted(verified[0]) == [0, 1, 2] and verified[1][-1] == f["t"]
    assert rows[0]["lean"]["outcome"] == "accepted" and rows[0]["lean"]["rejected"] is None
    assert rows[1]["lean"]["rejected"] == {"failure": "receipt", "window": f["t"], "at": list(f["at"])}


def test_an_event_the_verifiers_answer_differently_is_refused_as_disagreement(monkeypatch):
    """Each event goes to both verifiers at one reading of the clock, and their results are compared: a root the Lean
    reads without its step is its `step` rejection where `audit.Session` takes the root, and the session is refused
    at that window as `disagreement`."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    event = verifier.lv.receive_event

    def stepless(t, i, root, change, step):
        e = event(t, i, root, change, step)
        return {**e, "step": None} if (t, i) == (1, 3) else e

    monkeypatch.setattr(verifier.lv, "receive_event", stepless)
    closed = []
    (r,), _ = run("honest", verdict="both", then=lambda v: closed.append(v.close_registration(b"test-node")))
    assert not r["accepted"] and r["failure"] == {"t": 1, "failure": "disagreement", "at": None}
    d = r["disagreement"]
    assert d["event"] == "receive" and d["lean"]["failure"] == "step" and d["python"] == {"ok": True}
    assert closed[0]["sessions"] == ["disagreement"] and "python" not in closed[0]


def test_an_append_is_answered_with_its_coin_and_a_root_message_is_applied_in_order():
    """`Verifier.append` answers an append's message with its first index and its coin, drawn then, once: the same
    root sent again draws none and is the session's `coin` rejection.  `Verifier.root`: a message's releases, then
    its root with its plan change and its step; a plan that still holds a block the message released, or one
    appended by no message, is the session's `plan` rejection, a root with no step or one not above the last pass's
    is its `step` rejection, and a message the session cannot read, a root's carrying appends among them, is
    malformed."""
    reg = audit.register(os.urandom(32), b"test-node", 8, bytes(64), P512)
    params = audit.Params(8, 10 ** 9, 1, 2, 2.0 ** -40, window_ns=10, codec=P512, receipt=reg.receipt, rows=reg.rows)

    def session():
        from verity.core.protocols.remote_bandwidth import coins

        s = audit.Session(params, reg, None, coins.Challenger(reg.digest, 0, params.coins), clock=lambda: 0,
                          not_before_ns=0)
        s.start()
        return s

    def appended(s):
        return verifier.Verifier.append(s, {"type": "append", "root": root, "count": 2})

    root = (b"\7" * 64).hex()
    s = session()
    reply = appended(s)
    assert set(reply) == {"type", "first", "coin"} and (reply["type"], reply["first"]) == ("appended", 8)
    assert bytes.fromhex(reply["coin"]) == s._appends[0][3] and len(reply["coin"]) == 64
    msg = {"t": 0, "i": 0, "root": root, "released": [[3, 1]], "change": [[[8, 2]], [[3, 1]]], "step": 5}
    assert verifier.Verifier.root(s, msg) is None
    assert [p.runs for p in s.windows[0].plans] == [((0, 3), (4, 6))] and s.released == 1 and s.extent == 10
    assert s.windows[0].steps == [5]
    assert verifier.Verifier.root(s, {"t": 0, "i": 1, "root": root, "change": [[], []], "step": 9}) is None
    assert appended(s) == audit.Verdict(0, False, "coin") and s.challenger.appends == 1
    for step in (5, 4):
        s = session()
        appended(s)
        assert verifier.Verifier.root(s, msg) is None
        v = verifier.Verifier.root(s, {"t": 0, "i": 1, "root": root, "change": [[], []], "step": step})
        assert v is not None and v.failure == "step"
    s = session()
    appended(s)
    v = verifier.Verifier.root(s, {k: x for k, x in msg.items() if k != "step"})
    assert v is not None and v.failure == "step"
    s = session()
    appended(s)
    v = verifier.Verifier.root(s, {**msg, "change": [[[8, 2]], []]})
    assert v is not None and v.failure == "plan"
    s = session()
    v = verifier.Verifier.root(s, {**msg, "released": []})
    assert v is not None and v.failure == "plan"
    with pytest.raises(ValueError):
        verifier.Verifier.append(session(), {"type": "append", "root": "zz", "count": 2})
    with pytest.raises(ValueError, match="message of its own"):
        verifier.Verifier.root(session(), {**msg, "appends": [[root, 2]]})


def test_a_root_message_names_the_withdrawn_blocks_among_its_releases():
    """`Verifier.root` hands a message's withdrawn runs to its release (`Session.release(released, withdrawn)`): an
    appended block no plan held, freed before its encode landed, is withdrawn; one a plan held, a registered one, or a
    withdrawal outside the message's releases is the session's `release` rejection."""
    from verity.core.protocols.remote_bandwidth import coins

    reg = audit.register(os.urandom(32), b"test-node", 8, bytes(64), P512)
    params = audit.Params(8, 10 ** 9, 1, 2, 2.0 ** -40, window_ns=10, codec=P512, receipt=reg.receipt, rows=reg.rows,
                          lifetime=2, deferred=True)
    root = (b"\7" * 64).hex()

    def session():
        s = audit.Session(params, reg, None, coins.Challenger(reg.digest, 0, params.coins), clock=lambda: 0,
                          not_before_ns=0)
        s.start()
        verifier.Verifier.append(s, {"type": "append", "root": root, "count": 3})
        return s

    msg = {"t": 0, "i": 0, "root": root, "released": [[9, 1]], "withdrawn": [[9, 1]],
           "change": [[[8, 1], [10, 1]], []], "step": 5}
    s = session()
    assert verifier.Verifier.root(s, msg) is None and s.released == 1
    for bad in ({"released": [[3, 1]], "withdrawn": [[3, 1]], "change": [[[8, 3]], [[3, 1]]]},
                {"released": [[3, 1]], "change": [[[8, 3]], [[3, 1]]]},
                {"released": []}):
        v = verifier.Verifier.root(session(), {**msg, **bad})
        assert v is not None and v.failure == "release"
    s = session()
    assert verifier.Verifier.root(s, {**msg, "released": [], "withdrawn": [], "change": [[[8, 3]], []]}) is None
    v = verifier.Verifier.root(s, {**msg, "i": 1, "change": [[], [[9, 1]]], "step": 6})
    assert v is not None and v.failure == "release"


def test_a_kv_plan_needs_its_pages_a_step():
    assert prover.parse_plan("honest,kv:4,kvdrop:2,skip:0.5,kvoverwrite:2") == [
        ("honest", 0.0), ("kv", 4.0), ("kvdrop", 2.0), ("skip", 0.5), ("kvoverwrite", 2.0)]
    for bad in ("kv", "kv:0", "kv:1.5", "kvdrop:", "evict:2", "kvoverwrite"):
        with pytest.raises(ValueError):
            prover.parse_plan(bad)


def test_openings_round_trip_on_the_wire_and_a_short_body_is_none():
    o = audit.Opening(3, 17, bytes(range(256)) * 16, b"\1" * 1024, b"\2" * 192, (b"\3" * 64,) * 5, b"\4" * 192,
                      (b"\5" * 64,) * 9, b"")
    appended = audit.Opening(4, 18, bytes(range(256)) * 16, b"\1" * 1024, b"\2" * 192, (b"\3" * 64,) * 5, b"\4" * 192,
                             (b"\5" * 64,) * 3, b"", b"\6" * 64)
    body = wire.encode_openings([o, appended, o])
    assert wire.decode_openings(body) == [o, appended, o]
    assert wire.decode_openings(body[:-1]) is None and wire.decode_openings(b"") is None
    with pytest.raises(ValueError):
        wire.encode_openings([audit.Opening(3, 17, o.block, o.leaf, o.leaf_salt, o.receipt_path, o.payload_salt,
                                            o.payload_path, b"", b"\6" * 63)])


@pytest.mark.parametrize("name", ["p1024/k3/masked", "p1024/k2/masked", "p512/k2"])
def test_codec_names_round_trip(name):
    assert wire.codec_named(name).name == name


def test_the_predicted_rate_is_the_draws_chance_to_miss_every_dropped_receipt():
    import summary

    assert summary.miss(100, 0, 10) == 1.0 and summary.miss(10, 1, 10) == 0.0
    assert summary.miss(20, 2, 2) == pytest.approx(18 / 20 * 17 / 19)
    U, f, s = 68 << 22, 2.0 ** -14, 2773
    assert summary.miss(U, round(f * U), s) == pytest.approx((1 - f) ** s, rel=1e-4)
