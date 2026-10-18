"""The verifier's and the prover's processes over a loopback socket, the driver's reference as the prover's backend and
a clock that advances by one tick per pass: an honest session is accepted and its γ terms computed, a prover that
dropped blocks is rejected at its openings, a second session waits out the first's deadline, and KV pages appended and
released mid-session (a memory map's steps) are audited, accepted when held and rejected when dropped."""
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
        paces: list | None = None, **kv):
    clock = PassClock()
    a, b = socket.socketpair()
    vconn, pconn = wire.Conn(a, clock=clock.now), wire.Conn(b)
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
                          checks=checks, lifetime=lifetime, registry=registry, host_bytes=host_bytes)
    v.register()
    allowance = v.allowance(v.ping(4))
    out, not_before = [], None
    for k in range(len(paces) if paces else sessions):
        r = v.session(k, allowance, not_before, paces[k] if paces else None)
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


def test_the_memory_line_held_on_average_credits_what_the_worst_pass_form_cannot(monkeypatch):
    """Both forms of the memory line on a store of 8 weights blocks and 8 of filler, every pass reading all 16, at 200
    draws a window of its ~320 receipts, so f_t ≈ 0.14.  At every pass (the worst-pass form) a window loses f_t·U_t
    ≈ 44 blocks, more than the store, and credits nothing.  On average a window loses f_t·U_t/P_t ≈ 2 blocks, its
    rebuilds and the chip's block, so it credits the weights less those; with the rebuilds spread over its passes, less
    again only f_t·N and the chip.  The node line carries both forms and both rebuild readings, the line's error
    counts AuditRO's slack a pass beside δ, and a recorded session recomputes the same average line
    (`recorded_memory`)."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    monkeypatch.setattr(timing, "SQUARINGS_PER_S", {"p512": 0.0})
    seen, memory = {}, verifier.memory

    def keep(*a, **kw):
        seen["args"] = a, kw
        return memory(*a, **kw)

    monkeypatch.setattr(verifier, "memory", keep)
    layout = [{"kind": "weights", "tier": "hbm", "first": 0, "blocks": 8},
              {"kind": "filler", "tier": "hbm", "first": 8, "blocks": 8}]
    (r,), (p,) = run("honest", samples=200, layout=layout)
    assert r["accepted"] and p["accepted"]
    (record, plans, regions), kw = seen["args"]
    assert kw["chip_bytes"] == timing.CHIP_BYTES
    B, span = rcodec.BLOCK_BYTES, max(w.span_ns for w in record.windows)
    rate = record.codec.encode_squarings * 1e9 / span / 2
    m = verifier.memory(record, plans, regions, hbm_bytes=kw["hbm_bytes"], rebuild=audit.Rebuild(record.codec, rate),
                        chip_bytes=B)
    assert m["form"] == verifier.WORST_PASS and m["average"]["form"] == verifier.AVERAGE and m["chip_bytes"] == B
    floors = audit.floors(record)
    assert max(floors) > 0.1
    for w, x, f in zip(record.windows, m["windows"], floors):
        rebuilt, a = w.span_ns / span / 2, x["average"]
        assert x["lost_blocks"] == pytest.approx(f * w.receipts + rebuilt) and w.receipts == 16 * w.passes
        assert x["credited"]["weights"] == max(0.0, 8 * B - x["lost_blocks"] * B)
        assert a["form"] == verifier.AVERAGE and a["held"] == {"weights": 8 * B, "hbm": 16 * B, "host": 0}
        assert a["wrong_blocks"] == pytest.approx(f * 16) and a["rebuild_blocks"] == pytest.approx(rebuilt)
        assert a["chip_blocks"] == 1 and a["lost_blocks"] == pytest.approx(f * 16 + rebuilt + 1)
        assert a["credited"]["weights"] == pytest.approx(max(0.0, 8 * B - (f * 16 + rebuilt + 1) * B))
        assert a["credited"]["hbm"] == pytest.approx(max(0.0, 16 * B - (f * 16 + rebuilt + 1) * B))
        assert a["credited_rebuild_per_pass"]["weights"] == pytest.approx(8 * B - (f * 16 + rebuilt / w.passes + 1) * B)
        assert a["epsilon"] == pytest.approx(a["lost_blocks"] / 16)
    assert m["credited_bytes"] == min(x["credited"]["hbm"] for x in m["windows"]) == 0
    avg = m["average"]
    assert avg["credited_bytes"] == min(x["average"]["credited"]["hbm"] for x in m["windows"]) > 0
    assert avg["weights_bytes"] == min(x["average"]["credited"]["weights"] for x in m["windows"]) > m["weights_bytes"]
    assert avg["share"] == avg["credited_bytes"] / m["hbm_bytes"] and avg["residual"] == 1 - avg["share"]
    assert avg["weights_bytes_rebuild_per_pass"] > avg["weights_bytes"]

    host = 1843 * 10 ** 9
    line = verifier.node_line(m, host_bytes=host)
    assert line["form"] == verifier.WORST_PASS and line["credited_bytes"] == 0
    a = line["average"]
    lost = max(x["average"]["lost_blocks"] for x in m["windows"]) * B
    assert a["form"] == verifier.AVERAGE and a["lost_bytes"] == pytest.approx(lost)
    assert a["held"] == {"weights": 8 * B, "filler_hbm": 8 * B, "host": 0}
    assert a["credited"]["weights"] == pytest.approx(8 * B - lost) == a["credited"]["filler_hbm"]
    assert a["credited_bytes"] == pytest.approx(16 * B - lost) and a["share"]["weights"] == a["credited"]["weights"] / (
        verifier.HBM_BYTES + host)
    pp = line["average_rebuild_per_pass"]
    lost_pp = max(x["average"]["lost_blocks_rebuild_per_pass"] for x in m["windows"]) * B
    assert pp["form"] == verifier.AVERAGE and pp["lost_bytes"] == pytest.approx(lost_pp) and lost_pp < lost
    assert pp["credited"]["weights"] == pytest.approx(8 * B - lost_pp) == pytest.approx(
        avg["weights_bytes_rebuild_per_pass"])

    lg = r["ledger"]
    reading = next(lg[n] for n in ("spec", "measured") if "error" in lg[n])
    assert lg["memory"]["error"] == verifier.memory_error(lg) == pytest.approx(
        reading["error"]["horizon"] + 2.0 ** lg["sketch_failure"]["log2"])
    assert lg["memory"]["error"] > lg["memory"]["delta"] + record.passes * 2.0 ** -audit.SLACK_LOG2 * 0.99

    def same(x, y) -> bool:
        if isinstance(x, dict):
            return x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
        return x == pytest.approx(y) if isinstance(x, float) else x == y

    again = verifier.recorded_memory({"ledger": {"memory": m}, "record": r["record"]}, chip_bytes=B, host_bytes=host)
    assert all(same(x["average"], y["average"]) for x, y in zip(again["windows"], m["windows"]))
    assert same(again["average"], avg) and same(again["node"]["average"], a)


@pytest.mark.parametrize("k", [1, 3])
def test_a_paced_session_puts_each_group_of_k_passes_under_a_live_coin_and_a_dropper_is_still_rejected(k):
    pace = verifier.pace_of(f"{k}:{4 * k}:40:2", 1.0)
    assert pace["groups"] == 4 and pace["draws"] == 10
    (r, r2), (pr, pr2) = run("honest,skip:0.25", paces=[pace, pace])
    assert r["accepted"] and pr["accepted"] and r["windows"] == 4 * 3
    seen = r["windows_seen"]
    assert [w["passes"] for w in seen] == [k] * 12 and pr["discarded"] == 0
    for g in range(1 if k > 1 else 2, len(seen)):
        # A group's coin leaves once the group before has k − 1 roots, or for k = 1 (one coin ahead) once the one
        # before that has its root.
        assert seen[g]["released_ns"] >= seen[g - 1]["arrivals_ns"][k - 2] if k > 1 else \
            seen[g]["released_ns"] >= seen[g - 2]["arrivals_ns"][-1]
    pl = r["ledger"]["paced"]
    assert len(pl["stretches"]) == 2 and pl["groups"] == 4
    assert {pl["snapshot"]["form"], pl["time"]["form"]} == {verifier.AVERAGE, verifier.TIME_HELD}
    assert not r2["accepted"] and r2["failure"]["failure"] == "receipt" and pr2["failure_on_dropped"]


def test_held_over_time_charges_what_a_burst_frees_in_one_groups_slack_and_the_snapshot_does_not():
    """Two stretches of four groups of two passes over N blocks, every pass reading all of them: stretching one
    group's interval (a burst then idle until its deadline) lowers the time-held credit by its absent block-time,
    rebuilds at 2·L_max each and the wrong receipts at Δ_g/n_g (L_max/n in the group after the stretch), and the
    snapshot's only by the rebuilds its longer reads allow; a block in from the disks or a link counts as a rebuild."""
    N, b = 1000, 4096
    held = {"weights": 600 * b, "hbm": N * b, "host": 0}
    mem = {"windows": [{"held": held, "average": {"held": held}} for _ in range(12)]}

    def seen(stretch: int) -> list[dict]:
        out, t = [], 0
        for g in range(12):
            span = 10 * (stretch if g == 5 else 1)
            out.append({"released_ns": t, "passes": 2, "arrivals_ns": [t + span // 2, t + span]})
            t += span
        return out

    def line(stretch: int, rebuild_per_ns: float = 0.0, wrong: float = 0.0, ingress_per_ns: float = 0.0) -> dict:
        return verifier.paced_line(mem, seen(stretch), groups=4, wrong=wrong, blocks=N, rebuild_per_ns=rebuild_per_ns,
                                   chip_bytes=0, block_bytes=b, node_bytes=10 * N * b, ingress_per_ns=ingress_per_ns)

    even, burst = line(1, 1e-3, 50.0), line(9, 1e-3, 50.0)
    assert even["snapshot"]["lost_blocks"] == pytest.approx(burst["snapshot"]["lost_blocks"] - 1e-3 * (80 + 2 * 80) / 8)
    s = burst["stretches"][1]
    assert s["span_ns"] == 10 * 4 + 80 and s["l_max_ns"] == 90
    assert s["rebuild_absent_blocks"] == pytest.approx(2 * 90 * 1e-3 * (120 + 180) / 120)
    assert s["wrong_absent_blocks"] == pytest.approx(50 * 90 / 2 / 120)
    assert burst["time"]["weights_bytes"] < even["time"]["weights_bytes"] <= 600 * b
    assert line(1)["time"]["weights_bytes"] == line(1)["snapshot"]["weights_bytes"] == 600 * b
    linked = line(9, 1e-3, 50.0, ingress_per_ns=2e-3)["stretches"][1]
    assert linked["ingress_blocks"] == pytest.approx(2e-3 * (120 + 180))
    assert linked["rebuild_absent_blocks"] == pytest.approx(2 * 90 * 3e-3 * (120 + 180) / 120)
    assert linked[verifier.AVERAGE]["lost_blocks"] == pytest.approx((50 + 3e-3 * 300) / 8)


def test_the_paced_line_refuses_a_session_any_of_whose_passes_leaves_a_registered_block_out():
    """The paced line pays each absent stretch with a wrong read or a rebuild, which needs every pass to read every
    registered block: a group whose least pass leaves one out, in either tier and in the closing stretch too, is
    refused with the group and its count, and a paced session whose KV steps free registered slots carries that
    refusal in its ledger."""
    N, b = 1000, 4096
    seen = [{"released_ns": 10 * g, "passes": 2, "arrivals_ns": [10 * g + 5, 10 * g + 10]} for g in range(12)]
    whole = {"weights": 600 * b, "hbm": 700 * b, "host": 300 * b}

    def line(at: int = 0, short: dict | None = None, groups: int = 12) -> dict:
        held = [whole | (short or {}) if g == at else whole for g in range(groups)]
        return verifier.paced_line({"windows": [{"held": h, "average": {"held": h}} for h in held]}, seen, groups=4,
                                   wrong=0.0, blocks=N, rebuild_per_ns=0.0, chip_bytes=0, block_bytes=b,
                                   node_bytes=10 * N * b)

    assert line()["time"]["weights_bytes"] == line()["snapshot"]["weights_bytes"] == 600 * b
    with pytest.raises(ValueError, match="group 2's least pass read 990 of the 1000 registered blocks"):
        line(2, {"hbm": 690 * b})
    with pytest.raises(ValueError, match="group 10's least pass read 999 of the 1000 registered blocks"):
        line(10, {"host": 299 * b})
    with pytest.raises(ValueError, match="a memory window for each group: 11 for 12"):
        line(groups=11)

    (r,), (p,) = run("kv:2", paces=[verifier.pace_of("1:4:40:2", 1.0)], kv_every=1, kv_pool=8)
    assert r["accepted"] and p["accepted"] and r["released"] > 0
    assert min(x["held"]["hbm"] for x in r["ledger"]["memory"]["windows"]) < 16 * rcodec.BLOCK_BYTES
    assert "least pass read" in r["ledger"]["paced"]["refused"]


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


def test_a_kv_plan_needs_its_pages_a_step():
    assert prover.parse_plan("honest,kv:4,kvdrop:2,skip:0.5") == [("honest", 0.0), ("kv", 4.0), ("kvdrop", 2.0),
                                                                  ("skip", 0.5)]
    for bad in ("kv", "kv:0", "kv:1.5", "kvdrop:", "evict:2"):
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
