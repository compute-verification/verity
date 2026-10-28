"""The verifier's and the prover's processes over a loopback socket, the driver's reference as the prover's backend and
a clock that advances by one tick per pass with both sides settled around each tick, so a session runs one schedule:
an honest session is accepted and its γ terms computed, a prover that dropped blocks is rejected at its openings, a
second session waits out the first's deadline, and KV pages appended and released mid-session (a memory map's steps)
are audited, accepted when held and rejected when dropped. The Lean verifier decides, `audit.AuditRun` keeps the books
beside it, and the two must agree at every event, window verdict and record field."""
from __future__ import annotations

import dataclasses
import json
import math
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

from verity.core.protocols.remote_bandwidth import audit, coins, ledger, receipts as rb  # noqa: E402
from verity.core.protocols.remote_bandwidth import codec as rcodec  # noqa: E402
from verity.core import site  # noqa: E402

import prover  # noqa: E402
import verifier  # noqa: E402
import wire  # noqa: E402

P512 = rcodec.Codec(rcodec.P512, 2, False)
NODE = verifier.NODE
SITE = verifier.one_host("test-node")
"""The Site the prover's node, `test-node`, is a host of."""


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
        return self._layout if self._layout is not None else super().layout()

    def calibrate(self) -> dict:
        return {**super().calibrate(), **({"read_tiers": self.tiers} if self.tiers else {})}

    def run_pass(self, seed: bytes, plan=None):
        out = super().run_pass(seed, plan)
        self.clock.tick()
        return out


def no_squarings(monkeypatch, *, chip: bool = True) -> None:
    """Run the verifier's node at no squarings, and with no on-chip storage unless `chip`."""
    def bare(d):
        return dataclasses.replace(d, squarings_per_s={w: 0.0 for w in d.squarings_per_s},
                                   chip_bytes=d.chip_bytes if chip else 0)
    host = NODE.host if chip else dataclasses.replace(NODE.host, chip_bytes=0)
    monkeypatch.setattr(verifier, "NODE", dataclasses.replace(NODE, devices=tuple(map(bare, NODE.devices)), host=host))


def run(plan: str, *, blocks: int = 16, windows: int = 3, samples: int = 6, sessions: int = 1,
        read_bytes_per_s: float = 1e20, tiers: dict | None = None, lifetime: int | None = None,
        layout: list | None = None, host_bytes: int = 0, paces: list | None = None, then=None,
        transcript: Path | None = None, challenger=None, appends: int | None = None, store=None,
        binding=None, groups_per_coin: int = 1, **kv):
    """The verifier's sessions and the prover's over one registration, each paced by its entry in
    `paces` if given, its coins from `challenger` (the verifier's own by default), `appends` append coins a session
    (by default as the prover's hello says, `Verifier.appends_of`), `groups_per_coin` windows a draw; `then(v)` once
    they end. With `store` (a `remote_bandwidth.store.Store`) the prover's blocks are that store's, and with `binding`
    the verifier binds the registration to its pinned model."""
    clock = PassClock()
    vconn, pconn = clock.conns()
    backend = TickingBackend(clock, store.blocks if store else blocks, random.Random(7),
                             read_bytes_per_s=read_bytes_per_s, tiers=tiers, layout=layout, store=store)
    p = prover.ProverSide(pconn, backend, b"test-node", prover.parse_plan(plan), seed=11, log=lambda m: None, **kv)
    results = []

    def prove() -> None:
        p.register(codec=P512)
        for mode, fraction in p.plan:
            head, _, _ = p.expect("session")
            results.append(p.session(head, mode, fraction))

    th = threading.Thread(target=prove, daemon=True)
    th.start()
    v = verifier.Verifier(vconn, source=os.urandom(32), site=SITE, windows=windows, window_ns=20, samples=samples,
                          delta=2.0 ** -40, jitter_ns=10 ** 9, clock=clock, log=lambda m: None,
                          lifetime=lifetime, host_bytes=host_bytes, transcript=transcript,
                          challenger=challenger, appends=appends, binding=binding, groups_per_coin=groups_per_coin)
    try:
        v.register()
        allowance = v.allowance(v.ping(4))
        out, not_before = [], None
        for k in range(len(paces) if paces else sessions):
            r = v.session(k, allowance, not_before, paces[k] if paces else None)
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
    """The capacity is the verifier's, and the prover's calibration can only raise it."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    (slow,), _ = run("honest", read_bytes_per_s=1.0)
    (fast,), _ = run("honest", read_bytes_per_s=1e20)
    assert slow["ledger"]["inputs"]["reported_read_bytes_per_s"] == 1.0
    assert slow["ledger"]["measured"]["capacity_bytes_per_s"] == NODE.read_bytes_per_s
    assert slow["ledger"]["spec"]["capacity_bytes_per_s"] == NODE.spec_bytes_per_s
    assert fast["ledger"]["measured"]["capacity_bytes_per_s"] == fast["ledger"]["spec"]["capacity_bytes_per_s"] == 1e20


def test_the_ledgers_are_the_nodes_whatever_share_of_it_the_store_is_on(monkeypatch):
    """R is every GPU's HBM and the host's DRAM, S_chip every GPU's chip and the host's caches, and the squaring
    ceiling the node's, for a store on one GPU too: its calibration's one-GPU read lowers none of them, and its
    certified bytes are the same at either R."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    gpu = NODE.devices[0]
    (r,), _ = run("honest", read_bytes_per_s=gpu.read_bytes_per_s)
    lg = r["ledger"]
    node_read = math.fsum([8 * gpu.read_bytes_per_s, NODE.host.read_bytes_per_s])
    node_spec = math.fsum([8 * gpu.spec_bytes_per_s, NODE.host.spec_bytes_per_s])
    assert lg["inputs"]["reported_read_bytes_per_s"] == gpu.read_bytes_per_s
    assert lg["measured"]["capacity_bytes_per_s"] == lg["inputs"]["measured_read_bytes_per_s"] == node_read
    assert lg["spec"]["capacity_bytes_per_s"] == lg["inputs"]["spec_bytes_per_s"] == node_spec
    assert lg["inputs"]["chip_bytes"] == 8 * gpu.chip_bytes + NODE.host.chip_bytes == 3_105_652_736
    assert lg["inputs"]["squarings_per_s"] == 8 * gpu.squarings_per_s["p512"]
    assert lg["spec"]["certified_bytes"] == pytest.approx(lg["measured"]["certified_bytes"], rel=1e-9)


def test_the_audited_node_is_the_registered_hosts_read_off_its_cards(tmp_path):
    """`--site LOG HOST`: the node is host HOST's in the Site its registration log registers, the log checked against
    the card files' sheets (`verifier.site_node`), so eight RTX PRO 6000s registered under a host are a vy-nebius
    node, and their memory the memory line's capacity; an id the Site registers as no host is refused, and so is a
    log the check refuses."""
    log = verifier.vy_nebius_log("node0")
    path = tmp_path / "log.json"
    path.write_text(json.dumps(log.to_json()))
    node = verifier.site_node(path, "node0")
    assert node == NODE and verifier.device_memory(node) == verifier.HBM_BYTES == 8 * 96 * 2 ** 30
    with pytest.raises(ValueError, match="is no host the Site registers"):
        verifier.site_node(path, "gpu0")
    path.write_text(json.dumps(site.Registration("remote", (log.entries[0], log.entries[0])).to_json()))
    with pytest.raises(site.Refused):
        verifier.site_node(path, "node0")


def test_the_timing_parts_reproduce_the_whole_node_sessions_split_and_sum_to_its_timing_term():
    """A whole-node session (`d101`), 8 GPUs' HBM and a host tier, every pass the whole store: its parts are the timing
    split, and at either R they sum to the ledger's timing term."""
    d101 = {"span_ns": 420_074_057_129, "passes": 6430, "reads_bytes": 4_888_145_229_250_560, "pass_ns": 65_259_603,
            "tiers": {"hbm": {"bytes": 751_619_276_800, "ms": 61.502974},
                      "host": {"bytes": 8_589_934_592, "ms": 25.468371}}}
    for r in (NODE.read_bytes_per_s, NODE.spec_bytes_per_s):
        parts = verifier.timing_parts(capacity_bytes_per_s=r, **d101)
        assert sum(parts.values()) == pytest.approx(1 - d101["reads_bytes"] / (r * d101["span_ns"] / 1e9), abs=1e-12)
    parts = verifier.timing_parts(capacity_bytes_per_s=NODE.read_bytes_per_s, **d101)
    assert sum(parts.values()) == pytest.approx(0.09926702029063428, abs=1e-12)
    assert {k: round(v * 100, 2) for k, v in parts.items()} == {
        "above_stated_tiers": 0.0, "host_dram_idle": 3.68, "hbm_below_stated": 0.66, "receipts_beyond_read": 5.48,
        "boundaries_and_allowance": 0.10, "blocks_left_out": 0.0}
    spec = verifier.timing_parts(capacity_bytes_per_s=NODE.spec_bytes_per_s, **d101)
    assert spec["above_stated_tiers"] == pytest.approx(1 - NODE.read_bytes_per_s / NODE.spec_bytes_per_s)


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


def test_an_unpaced_session_declares_its_snapshot_line_and_no_time_held_line(monkeypatch):
    """The declared outputs are the protocol's (`ledger.declared`): γ at both R, as the readings beside them, and the
    snapshot line; an unpaced session's coins come a stated window apart, so it has no time-held line."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    (r,), _ = run("honest")
    lg = r["ledger"]
    d = lg["declared"]
    assert d["accepted"] and d["windows"] == 3
    assert d["gamma"]["spec"]["gamma"] == lg["spec"]["gamma"] and d["gamma"]["measured"]["gamma"] == lg["measured"]["gamma"]
    assert [w["window"] for w in d["gamma"]["spec"]["windows"]] == [0, 1, 2]
    assert 0 <= d["memory"][ledger.SNAPSHOT]["epsilon"] <= 1 and d["memory"][ledger.SNAPSHOT]["delta"] > 0
    assert "unpaced" in d["memory"][ledger.TIME_HELD]["refused"]


@pytest.mark.parametrize("k, schedule", [
    (1, [(0, 1, [1]), (1, 2, [2]), (2, 3, [3]), (3, 4, [4]), (4, 5, [5]), (5, 6, [6])]),
    (3, [(0, 2, [1, 2, 3]), (2, 5, [4, 5, 6]), (5, 8, [7, 8, 9]), (8, 11, [10, 11, 12]), (11, 14, [13, 14, 15]),
         (14, 17, [16, 17, 18])]),
])
def test_a_paced_session_runs_k_passes_a_coin_and_declares_its_time_held_line(monkeypatch, k, schedule):
    """Paced (`pace_of`, `audit.Pace(k, T)`: two stretches of two windows and the closing stretch), the prover runs k
    passes under each coin and waits for the next, and the verifier sends each boundary as the window's (k − 1)-th
    root arrives (`AuditRun.next_boundary_ns`; k = 1: as the window before's root does), so on the pass clock window t
    holds k passes, none discarded; γ is declared over the fixed windows of T from the first challenge to the last
    deadline, and the session declares all three memory forms.  Its lifetime cap counts groups, those of their nominal
    length (the stretch over its groups) in 30 s."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    T = 10**9
    pace = verifier.pace_of(f"{k}:{T // 10**6}:12:2", 500 / k)
    assert pace["groups"] == 2 and pace["draws"] == 6
    (r,), (pr,) = run("honest", paces=[pace])
    assert r["accepted"] and pr["accepted"] and r["windows"] == 6 and r["pace"] == pace
    assert r["record"]["public"]["pace"] == {"k": k, "window_ns": T}
    assert r["lifetime"] == r["record"]["public"]["lifetime"] == audit.lifetime_windows(T // 2) == 60
    assert pr["passes"] == [k] * 6 and pr["discarded"] == 0
    assert [(w["released_ns"], w["boundary_ns"], w["arrivals_ns"]) for w in r["windows_seen"]] == [
        (s, b, list(a)) for s, b, a in schedule]
    ws = r["ledger"]["declared"]["gamma"]["spec"]["windows"]
    span = r["windows_seen"][-1]["deadline_ns"] - schedule[0][0]
    assert [w["window"] for w in ws] == list(range(-(-span // T)))
    assert [w["span_ns"] for w in ws] == [T] * (len(ws) - 1) + [span - (len(ws) - 1) * T]
    d = r["ledger"]["declared"]
    for form in ledger.FORMS:
        assert 0 <= d["memory"][form]["epsilon"] <= 1 and d["memory"][form]["delta"] > 0, form


@pytest.mark.parametrize("k", [1, 3])
def test_a_paced_session_at_one_window_in_flight_sends_each_coin_after_the_root_before_it(monkeypatch, k):
    """At F = 1 (`pace_of`'s fifth field, `audit.Pace.in_flight`) each boundary leaves once its window has its k roots,
    so every coin leaves after the root before it; the honest prover still runs k passes a coin, none discarded, the
    Lean verifier agrees at every event, and the record names one window in flight."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    T = 10**9
    pace = verifier.pace_of(f"{k}:{T // 10**6}:12:2:1", 500 / k)
    assert pace == verifier.pace_of(f"{k}:{T // 10**6}:12:2", 500 / k) | {"in_flight": 1}
    with pytest.raises(ValueError, match="1 or 2"):
        verifier.pace_of(f"{k}:{T // 10**6}:12:2:3", 500 / k)
    (r,), (pr,) = run("honest", paces=[pace])
    assert r["accepted"] and pr["accepted"] and r["windows"] == 6
    assert r["record"]["public"]["pace"] == {"k": k, "window_ns": T, "in_flight": 1}
    assert pr["passes"] == [k] * 6 and pr["discarded"] == 0
    seen = r["windows_seen"]
    assert all(len(w["arrivals_ns"]) == k and w["boundary_ns"] >= w["arrivals_ns"][-1] for w in seen)
    assert all(b["released_ns"] == a["boundary_ns"] for a, b in zip(seen, seen[1:]))


def test_a_paced_prover_that_dropped_blocks_is_rejected():
    (r,), (pr,) = run("skip:0.5", paces=[verifier.pace_of("2:1000:12:2", 250.0)])
    assert not r["accepted"] and not pr["accepted"] and "declared" not in r.get("ledger", {})


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


def test_a_session_reserves_append_coins_only_when_its_plan_appends_and_an_append_past_them_is_refused(monkeypatch):
    """Each session's coins are W challenges, W openings coins and its append coins: none, and no append family, under
    a plan of honest sessions (the hello says so), and `coins.APPENDS` under one with a kv session, on which both are
    accepted.  At `appends=0` the kv session's first append has no coin: the books reject it as `coin` and the Lean,
    which takes only the append coins released, otherwise, so the session is refused there as `disagreement`, the
    honest one accepted."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    import challengers

    layouts, local = [], challengers.Local.open

    def opened(self, registration, k, windows, appends):
        c = local(self, registration, k, windows, appends)
        layouts.append(c.commitment.families)
        return c

    monkeypatch.setattr(challengers.Local, "open", opened)
    pair = ((coins.CHALLENGE, 3), (coins.OPENINGS, 3))
    (r,), (p,) = run("honest", samples=8)
    assert r["accepted"] and p["accepted"] and layouts == [pair]
    layouts.clear()
    (honest, kv), (ph, pk) = run("honest,kv:2", sessions=2, samples=8, kv_every=4, kv_pool=8)
    assert honest["accepted"] and kv["accepted"] and ph["accepted"] and pk["accepted"] and kv["appended"] > 0
    assert layouts == [pair + ((coins.APPEND, coins.APPENDS),)] * 2
    layouts.clear()
    (honest, kv), (ph, pk) = run("honest,kv:2", sessions=2, samples=8, kv_every=4, kv_pool=8, appends=0)
    assert layouts == [pair] * 2
    assert honest["accepted"] and ph["accepted"] and honest["challenger"]["reason"] is None
    assert not kv["accepted"] and kv["failure"]["failure"] == "disagreement" and kv["appended"] == 0
    assert kv["disagreement"]["event"] == "append" and kv["disagreement"]["python"]["failure"] == "coin"
    assert kv["challenger"]["reason"] is None and not pk["accepted"]


def test_a_window_with_no_draws_settles_malformed_whatever_its_batches():
    """A window with no draws (none a session makes) settles `malformed` whatever its batches' results, as
    `AuditRun.verify` and the Lean's settle give it."""
    from types import SimpleNamespace

    from checks import Checks

    class Books:
        _fail = audit.AuditRun._fail

        def __init__(self) -> None:
            self.windows, self.failed = [SimpleNamespace(draws=(), verdict=None)], None

        def _live(self) -> None:
            pass

    for results in ([], [{"failure": None, "at": None}]):
        books = Books()
        assert Checks.settle(books, 0, results, 0) == audit.Verdict(0, False, "malformed") == books.windows[0].verdict


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


def logged_appends(monkeypatch) -> list:
    """Each append the serving trace's records make, (step, first, count, root), in order."""
    out, record = [], prover.ServingTrace.record

    def logging(self, s, window, freed, writes, rekey=True):
        n = len(self.appends)
        change = record(self, s, window, freed, writes, rekey)
        out.extend((s, a.first, a.count, a.root) for a in self.appends[n:])
        return change

    monkeypatch.setattr(prover.ServingTrace, "record", logging)
    return out


def heads_follow_the_steps(record: dict, appends: list) -> bool:
    """Whether each pass's public head is the chain through `appends` of the steps below its own and the record's
    chain is through them all, as the memory-map unit checks it (`RequirementLink.mapped`)."""

    def chain(below: int) -> str | None:
        head = None
        for s, first, count, root in appends:
            if s < below:
                head = audit.append_head(head, first, count, root)
        return None if head is None else head.hex()

    pub, windows = record["public"], record["profile"]["windows"]
    return pub["appends"] == chain(audit.STEP_LIMIT) and all(
        h == chain(s) for w, hs in zip(windows, pub["heads"], strict=True) for s, h in zip(w["steps"], hs, strict=True))


def test_kv_pages_past_the_lifetime_are_re_keyed_and_a_prover_that_keeps_them_is_rejected(monkeypatch):
    """Under L = 1 window, sent with the session, every appended page is stale in the window after the one it joined.
    The honest prover re-keys each before that window's first pass (writes its payload again in its slot, at a fresh
    index) and is accepted, each pass's head the chain through the appends of the steps below its own, those of the
    queued passes each boundary discards among them. One that never re-keys is rejected at that pass's root
    (`lifetime`)."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    appends = logged_appends(monkeypatch)
    (r,), (p,) = run("kv:2", samples=8, kv_every=4, kv_pool=8, lifetime=1)
    assert r["accepted"] and p["accepted"] and r["record"]["public"]["lifetime"] == 1
    assert appends and heads_follow_the_steps(r["record"], appends) and p["discarded"] > 0
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
    filler in place and waits on its own step's pass (the filler the next k + 1 steps write is freed there).  Each
    pass's head is the chain through the appends of the steps below its own, the steps serving ran past among them."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    appends = logged_appends(monkeypatch)
    (r,), (p,) = run("kv:1", windows=2, samples=16, kv_every=1, kv_pool=8, lifetime=1, ahead=k, fill=True,
                     kv_headroom=k + 1)
    assert r["accepted"] and p["accepted"] and r["failure"] is None and p["misread"] == []
    assert appends and heads_follow_the_steps(r["record"], appends)
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
    credited only on the property's verdict and the session's schedule check (its verdict stood in here: the check's
    own tests are `remote_bandwidth`'s), then less its carriers; the share and residual are of the node's HBM.
    Unlabelled, the whole registered store is filler and no weights are credited.  The node has no on-chip storage
    here, which would hold the whole store."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    no_squarings(monkeypatch, chip=False)
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
    with pytest.raises(ValueError, match="schedule check"):
        verifier.memory(record, plans, regions, credit_post_seed=CarrierVerdict(0.0), **kw)
    other = audit.Scheduled(record.registration, bytes(len(record.coins)), 0)
    with pytest.raises(ValueError, match="schedule check"):
        verifier.memory(record, plans, regions, credit_post_seed=CarrierVerdict(0.0), schedule=other, **kw)
    checked = audit.Scheduled(record.registration, record.coins, 0)
    whole = verifier.memory(record, plans, regions, credit_post_seed=CarrierVerdict(0.0), schedule=checked, **kw)
    carried = verifier.memory(record, plans, regions, credit_post_seed=CarrierVerdict(B / 2), schedule=checked, **kw)
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
    verdict; each share is of HBM plus DRAM, the residual what the total leaves, and the line is its worst window's.
    The node has no on-chip storage here."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    no_squarings(monkeypatch, chip=False)
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
    ≈ 44 blocks, its rebuilds and the chip's block, more than the store, and credits nothing.  On average a window
    loses f_t·U_t/P_t ≈ 2 blocks, its rebuilds and the chip's block, so it credits the weights less those; with the
    rebuilds spread over its passes, less again only f_t·N and the chip.  The node line carries both forms and both
    rebuild readings, the line's error counts AuditRO's slack a pass beside δ, and a recorded session recomputes the
    same line in both forms (`recorded_memory`), a record written before the worst-pass form charged the chip
    included."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    no_squarings(monkeypatch)
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
    assert kw["chip_bytes"] == NODE.chip_bytes
    B, span = rcodec.BLOCK_BYTES, max(w.span_ns for w in record.windows)
    rate = record.codec.encode_squarings * 1e9 / span / 2
    m = verifier.memory(record, plans, regions, hbm_bytes=kw["hbm_bytes"], rebuild=audit.Rebuild(record.codec, rate),
                        chip_bytes=B)
    assert m["form"] == verifier.WORST_PASS and m["average"]["form"] == verifier.AVERAGE and m["chip_bytes"] == B
    floors = audit.floors(record)
    assert max(floors) > 0.1
    for w, x, f in zip(record.windows, m["windows"], floors):
        rebuilt, a = w.span_ns / span / 2, x["average"]
        assert x["lost_blocks"] == pytest.approx(f * w.receipts + rebuilt + 1) and w.receipts == 16 * w.passes
        assert x["chip_blocks"] == 1
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
    assert lg["memory"]["error"] == verifier.memory_error(lg) == (
        reading["error"]["horizon"] + 2.0 ** lg["sketch_failure"]["log2"] + lg["delta"]["hiding"]
        + lg["delta"]["binding"] + lg["delta"]["rejection"])
    assert lg["delta"]["hiding"] == ledger.hiding_over(len(record.windows)) > 0
    assert lg["delta"]["binding"] > 0
    assert lg["delta"]["rejection"] == len(record.windows) * record.samples / 2 ** 64 > 0
    assert record.samples == r["samples"] == 200 and record.public()["samples"] == 200
    assert lg["memory"]["error"] > lg["memory"]["delta"] + record.passes * 2.0 ** -audit.SLACK_LOG2 * 0.99
    assert lg["inputs"]["slack_log2"] == audit.SLACK_LOG2

    def same(x, y) -> bool:
        if isinstance(x, dict):
            return x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
        return x == pytest.approx(y) if isinstance(x, float) else x == y

    again = verifier.recorded_memory({"ledger": {"memory": m}, "record": r["record"]}, chip_bytes=B, host_bytes=host)
    keys = ("chip_blocks", "lost_blocks", "epsilon", "credited", "average")
    assert all(same({k: x[k] for k in keys}, {k: y[k] for k in keys}) for x, y in zip(again["windows"], m["windows"]))
    assert same(again["average"], avg) and same(again["node"]["average"], a)
    old = m | {"windows": [{k: v for k, v in x.items() if k not in ("chip_blocks", "average")}
                           | {"lost_blocks": x["lost_blocks"] - 1} for x in m["windows"]]}
    then = verifier.recorded_memory({"ledger": {"memory": old}, "record": r["record"]}, chip_bytes=B, host_bytes=host)
    assert all(same({k: x[k] for k in keys}, {k: y[k] for k in keys}) for x, y in zip(then["windows"], m["windows"]))
    assert same(then["node"], again["node"])


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


@pytest.mark.parametrize("k", [1, 3])
def test_a_paced_group_is_drawn_as_its_k_th_root_arrives(k):
    """A paced window takes at most k roots, so its k-th makes them final once its boundary has left
    (`audit.AuditRun.close`), which the loop sends before that root: each group's draws go out as that root arrives, the
    last group's too, so the openings coin's request on the challenger's log names the group's last root when it came,
    not one root or a deadline later."""
    (r,), _ = run("honest", paces=[verifier.pace_of(f"{k}:{4 * k}:40:2", 1.0)])
    assert r["accepted"]
    assert [w["draws_sent_ns"] for w in r["windows_seen"]] == [w["arrivals_ns"][-1] for w in r["windows_seen"]]


@pytest.mark.parametrize("k", [1, 3])
def test_a_k_1_paced_root_asks_the_challenger_for_its_two_coins_in_one_round_trip(tmp_path, monkeypatch, k):
    """At k = 1 a group's root closes it and the next boundary is due at once, so the group's openings coin and the
    challenge the boundary releases, σ_(g+2), go to the challenger in one round trip, in that order, both after that
    root (`coins.SessionCoins.ahead`); the last two groups' boundaries release no coin, so their openings coins go alone. At
    k = 3 a coin leaves at the following group's second root, so every coin goes alone. The session's transcript
    replays to its verdict through both verifiers. The clock ticks as each pass starts as well as when it ends, so
    time passes between a coin leaving and the root after it, as on the verifier's own clock: at one tick a pass, a
    paced window's 1 ns is a whole pass and each boundary waits for the tick its next root comes at."""
    monkeypatch.syspath_prepend(str(HERE.parents[2] / "verity/core/protocols/remote_bandwidth/lean/tests"))
    ticking = TickingBackend.run_pass

    def run_pass(self, seed: bytes, plan=None):
        self.clock.tick()
        return ticking(self, seed, plan)

    monkeypatch.setattr(TickingBackend, "run_pass", run_pass)
    import json

    import challengers
    import replay
    from verity.core.protocols.sampled_proofs import challenger as C

    asked = []

    class Asked(C.LiveChallenger):
        def open(self, label, index, after):
            return self.open_many([(label, index, after)])[0]

        def open_many(self, requests):
            asked.append([(label, index, bytes(after)) for label, index, after in requests])
            return [C.LiveChallenger.open(self, *r) for r in requests]

    monkeypatch.setattr(challengers.Local, "open", lambda self, registration, e, windows, appends: Asked(
        coins.SCHEME, e, coins.families(windows, appends), os.urandom))
    (r,), _ = run("honest", paces=[verifier.pace_of(f"{k}:{4 * k}:40:2", 1.0)], transcript=tmp_path)
    assert r["accepted"] and r["challenger"]["reason"] is None
    case = json.loads(Path(r["transcript"]["path"]).read_bytes())
    roots = [bytes.fromhex(e["root"]) for e in case["events"] if e["kind"] == "receive"]
    n, last = r["windows"], [roots[k * g + k - 1] for g in range(r["windows"])]
    window = asked[0][0][2]
    if k == 1:
        assert asked == ([[("challenge", 0, window)], [("challenge", 1, window)]]
                         + [[("openings", g, last[g]), ("challenge", g + 2, last[g])] for g in range(n - 2)]
                         + [[("openings", g, last[g])] for g in (n - 2, n - 1)])
    else:
        assert all(len(b) == 1 for b in asked) and len(asked) == 2 * n
        assert [b[0][2] for b in asked if b[0][0] == "openings"] == last
    with replay.server() as s:
        row = replay.replay(case, s)
    assert row["agree"] and row["lean"]["outcome"] == "accepted" and row["lean"]["refused"] == 0, row


def test_the_auditor_line_reads_each_groups_coin_and_last_root_off_the_challengers_log():
    """`auditor_line.auditor_seen`: group g's coin at challenge coin g's release, and its last root at the first request
    on the challenger's log that names the root openings coin g was asked after. On a log that is the verifier's stamps
    shifted, the line's groups are the verifier's; a later openings request lengthens them by as much; and an earlier
    request naming that root, σ_(g+2)'s at k = 1, is the one read."""
    import json

    import auditor_line as al

    got = {}
    (r,), _ = run("honest", paces=[verifier.pace_of("1:4:40:2", 1.0)],
                  then=lambda v: got.update(reg=wire.registration_json(v.registration)))
    seen = r["windows_seen"]
    n, R, A = len(seen), [w["released_ns"] for w in seen], [w["arrivals_ns"][-1] for w in seen]

    def log(late: int, named: bool) -> list[str]:
        lines = [json.dumps({"event": "commitment", "root": "0" * 128})]
        for g in range(n):
            for label, after, at in (("challenge", f"{g - 2:0128x}" if named and g >= 2 else "f" * 128, R[g] - 1),
                                     ("openings", f"{g:0128x}", A[g] + late)):
                lines.append(json.dumps({"event": "root", "label": label, "index": g, "root": after,
                                         "received_at": at + 100}))
                lines.append(json.dumps({"event": "release", "label": label, "index": g, "after": after,
                                         "released_at": at + 101}))
        return lines

    def lmax(s: dict) -> list[int]:
        return [x["l_max_ns"] for x in verifier.recorded_paced(s, got["reg"], host_bytes=0)["stretches"]]

    shifted, how = al.auditor_seen(log(0, False), seen)
    assert [w["released_ns"] for w in shifted] == [x + 100 for x in R]
    assert [w["arrivals_ns"] for w in shifted] == [[x + 100] for x in A]
    assert how == {"last_root_first_named_by": {"openings": n}}
    assert al.group_l(shifted) == al.group_l(seen) and lmax({**r, "windows_seen": shifted}) == lmax(r)
    late, _ = al.auditor_seen(log(3, False), seen)
    assert al.group_l(late) == [max(A[g] + 103, R[g + 1] + 100) - R[g] - 100 for g in range(n - 1)]
    named, how = al.auditor_seen(log(3, True), seen)
    assert [w["arrivals_ns"] for w in named] == ([[R[g + 2] + 99] for g in range(n - 2)]
                                                 + [[A[g] + 103] for g in (n - 2, n - 1)])
    assert how == {"last_root_first_named_by": {"challenge": n - 2, "openings": 2}}
    with pytest.raises(ValueError, match="not one of each for each of"):
        al.auditor_seen(log(0, False)[:-4], seen)


def test_the_line_of_record_reads_the_lean_verifiers_record_of_the_session():
    """`verifier.recorded_inputs`: the line's groups' passes, windows and registered blocks are the Lean verifier's
    record of the session (`lean_record`, which a run from before sessions kept it reads back from its `record` and
    `windows_seen`, field for field), and its line is the line of the same groups and windows given. The pooled
    bound's inputs its record's profile gives (`recorded_windows`) are the ones the line reads off the Lean's record
    and the session. A record of another registration, or windows seen that aren't the record's, is refused."""
    line = verifier.line
    got = {}
    (r,), _ = run("honest", paces=[verifier.pace_of("1:4:40:2", 1.0)],
                  then=lambda v: got.update(reg=wire.registration_json(v.registration)))
    assert r["accepted"] and r["verdict"] == "lean" and r["lean_record"]["registration"] == got["reg"]["digest"]
    old = {k: v for k, v in r.items() if k != "lean_record"}
    assert verifier.lean_record(old) == r["lean_record"]
    rates = verifier.recorded_rates(r, got["reg"], disk_bytes_per_s=4.4e9, link_bytes_per_s=5e10)
    mem = r["ledger"]["memory"]
    windows = [(sum(w["lengths"]), w["drawn"]) for w in r["lean_record"]["windows"]]
    given = line.inputs_json(
        verifier.paced_groups(mem, r["windows_seen"], block_bytes=mem["block_bytes"]), per=r["pace"]["groups"],
        blocks=got["reg"]["blocks"], block_bytes=mem["block_bytes"], chip_bytes=mem["chip_bytes"],
        node_bytes=mem["hbm_bytes"], delta=r["delta"], rates=rates, windows=windows)
    want = line.run_json(given)
    assert "refused" not in want
    for s in (r, old):
        assert verifier.recorded_windows(s) == (windows, s["delta"])
        assert line.record_run_json(verifier.recorded_inputs(s, got["reg"], host_bytes=0, rates=rates)) == want
        assert verifier.recorded_line(s, got["reg"], host_bytes=0, rates=rates).to_json() == want
    with pytest.raises(ValueError, match="of registration"):
        verifier.recorded_inputs(r, {**got["reg"], "digest": "00" * 64}, host_bytes=0, rates=rates)
    with pytest.raises(ValueError, match="not its record's"):
        verifier.lean_record({**old, "windows_seen": old["windows_seen"][:-1]})


def test_the_line_of_record_deducts_at_least_the_auditors_s_chip():
    """`verifier.recorded_chip_bytes`: a record its verifier wrote at an S_chip under `NODE.chip_bytes` (gate runs 7
    to 9's 3,045,746,688 B) is deducted at `NODE.chip_bytes`, in the line of record's inputs (`recorded_inputs`, which
    the Lean and line.py both read) and in the verifier's diagnostic (`recorded_paced`): every stretch, in both forms
    and both readings, charges the difference in whole blocks more as absent than the record's own S_chip would. A
    record at a larger S_chip keeps its own."""
    line = verifier.line
    got = {}
    (r,), _ = run("honest", paces=[verifier.pace_of("1:4:40:2", 1.0)],
                  then=lambda v: got.update(reg=wire.registration_json(v.registration)))
    mem = r["ledger"]["memory"]
    b, short = mem["block_bytes"], 3_045_746_688
    assert mem["chip_bytes"] == NODE.chip_bytes > short
    rates = verifier.recorded_rates(r, got["reg"], disk_bytes_per_s=4.4e9, link_bytes_per_s=5e10)

    def at(chip: int) -> dict:
        return {**r, "ledger": {**r["ledger"], "memory": {**mem, "chip_bytes": chip}}}

    inputs = verifier.recorded_inputs(at(short), got["reg"], host_bytes=0, rates=rates)
    assert inputs["chip_bytes"] == NODE.chip_bytes
    assert inputs == verifier.recorded_inputs(r, got["reg"], host_bytes=0, rates=rates)
    charged, own = line.record_run_json(inputs), line.record_run_json({**inputs, "chip_bytes": short})
    more = -(-NODE.chip_bytes // b) - -(-short // b)
    assert more > 0
    for f in line.FORMS:
        assert charged[f]["stretches"] and len(charged[f]["stretches"]) == len(own[f]["stretches"])
        for x, y in zip(charged[f]["stretches"], own[f]["stretches"]):
            for reading in line.READINGS:
                assert line.number(x[reading]["lost_blocks"]) == line.number(y[reading]["lost_blocks"]) + more
    assert verifier.recorded_paced(at(short), got["reg"], host_bytes=0)["chip_blocks"] == -(-NODE.chip_bytes // b)
    larger = NODE.chip_bytes + b
    assert verifier.recorded_inputs(at(larger), got["reg"], host_bytes=0, rates=rates)["chip_bytes"] == larger
    assert verifier.recorded_paced(at(larger), got["reg"], host_bytes=0)["chip_blocks"] == -(-larger // b)


def test_the_auditor_line_reads_a_last_root_by_its_request_not_by_its_value():
    """A prover that repeats a root doesn't make a group's last root early on the challenger's clock: window 2's last
    root is also its boundary σ_3's (asked before it came), and window 4's is window 1's, but `auditor_seen` reads each
    at its openings request, so every L is the honest 1.5 coin intervals."""
    import json

    import auditor_line as al

    n, X, Y = 6, "a" * 128, "b" * 128
    seen = [{"released_ns": 1000 * g + 1000, "arrivals_ns": [1000 * g + 2500]} for g in range(n)]
    lines = [json.dumps({"event": "commitment", "root": "0" * 128})]
    for g in range(n):
        for label, after, at in (("challenge", X if g == 3 else f"{2 * g:0128x}", 1000 * g + 999),
                                 ("openings", {2: X, 4: Y, 1: Y}.get(g, f"{2 * g + 1:0128x}"), 1000 * g + 2500)):
            lines.append(json.dumps({"event": "root", "label": label, "index": g, "root": after, "received_at": at}))
            lines.append(json.dumps({"event": "release", "label": label, "index": g, "after": after,
                                     "released_at": at + 1}))
    got, how = al.auditor_seen(lines, seen)
    assert [w["arrivals_ns"] for w in got] == [[1000 * g + 2500] for g in range(n)]
    assert al.group_l(got) == [1500] * (n - 1) and how == {"last_root_first_named_by": {"openings": n}}


def test_a_paced_session_whose_openings_come_beside_its_passes_gets_the_same_verdicts_from_both_verifiers():
    """`node.py --overlap-openings`: the prover opens each window as its draws arrive instead of after its last pass,
    so both verifiers take openings between a paced session's roots and coins; the honest session is still accepted
    and the dropper rejected, with no disagreement."""
    pace = verifier.pace_of("1:4:40:2", 1.0)
    (r, r2), (pr, pr2) = run("honest,skip:0.25", paces=[pace, pace], defer_openings=False)
    assert r["verdict"] == "lean" and r["disagreement"] is None and r2["disagreement"] is None
    assert r["accepted"] and pr["accepted"] and len(r["lean_check_ns"]) == r["windows"] == 12
    assert not r2["accepted"] and r2["failure"]["failure"] == "receipt" and pr2["failure_on_dropped"]


def test_held_over_time_charges_what_a_burst_frees_in_one_groups_slack_and_the_snapshot_does_not():
    """Two stretches of four groups of two passes over N blocks, every pass reading all of them: stretching one
    group's interval (a burst then idle until its deadline) lowers the time-held credit by its absent block-time,
    rebuilds at 2·L_max each and the wrong receipts at Δ_g/n_g (L_max/n in the group after the stretch), and the
    snapshot's only by the rebuilds its longer reads allow; a block in from the disks or a link counts as a rebuild,
    and so does each block of the routes' burst."""
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

    def line(stretch: int, rebuild_per_ns: float = 0.0, wrong: float = 0.0, ingress_per_ns: float = 0.0,
             burst: float = 0.0) -> dict:
        return verifier.paced_line(mem, seen(stretch), groups=4, wrong=wrong, blocks=N, rebuild_per_ns=rebuild_per_ns,
                                   chip_bytes=0, block_bytes=b, node_bytes=10 * N * b, burst_bytes=burst * b,
                                   ingress_per_ns=ingress_per_ns)

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
    bursty, T, A = line(9, 1e-3, 50.0, burst=7.0)["stretches"][1], verifier.TIME_HELD, verifier.AVERAGE
    assert bursty["burst_absent_blocks"] == pytest.approx(2 * 90 * 7 / 120)
    assert bursty[T]["lost_blocks"] == pytest.approx(s[T]["lost_blocks"] + 2 * 90 * 7 / 120)
    assert bursty[A]["lost_blocks"] == pytest.approx(s[A]["lost_blocks"] + 7 / 8)


def test_a_paced_lines_ingress_charge_brings_the_bursts_the_ledger_states_for_its_routes():
    """`ingress_bursts` is `ledger.Node.stated`'s B: the rebuild batch in flight always, one second at the disks'
    ceiling for the charge up to it, and a link window once the charge passes the disks (the link-charged line)."""
    assert verifier.ingress_bursts(0.0) == NODE.chip_bytes
    assert verifier.ingress_bursts(4.4e9) == NODE.chip_bytes + 4.4e9
    assert verifier.ingress_bursts(54.4e9) == NODE.chip_bytes + 4.4e9 + NODE.host.link_window_bytes


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
                                   node_bytes=10 * N * b, burst_bytes=0.0)

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


def greeted(lifetime: int | None, deferred: bool, *, hello: dict | None = None, drop: tuple[str, ...] = (),
            plan: str = "honest", node: bytes = b"test-node",
            hosts: dict = SITE) -> tuple[verifier.Verifier, ValueError | None, list[str]]:
    """A prover of `plan` on `node` whose backend says `deferred` greeting a verifier of lifetime cap `lifetime` under
    the Site `hosts`, its hello with `hello`'s fields too and without `drop`'s: the verifier, its refusal if it
    refused, and the reason the prover read in its `bye`."""
    a, b = socket.socketpair()
    backend = prover.CpuBackend(16, random.Random(7))
    backend.deferred = deferred
    p = prover.ProverSide(wire.Conn(b), backend, node, prover.parse_plan(plan), seed=11, log=lambda m: None)
    send = p.conn.send

    def greet(head: dict) -> dict:
        return {k: x for k, x in (head | (hello or {})).items() if k not in drop}

    p.conn.send = lambda head, body=b"": send(greet(head) if head["type"] == "hello" else head, body)
    heard = []

    def prove() -> None:
        try:
            p.register(codec=P512)
        except wire.Closed as e:
            heard.append(str(e))

    th = threading.Thread(target=prove, daemon=True)
    th.start()
    v = verifier.Verifier(wire.Conn(a), source=os.urandom(32), site=hosts, windows=3, window_ns=20, samples=6,
                          delta=2.0 ** -40, jitter_ns=10 ** 9, log=lambda m: None, lifetime=lifetime)
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
    `bye` before drawing a registration.  At L = 2, or not deferring, the session's parameters carry it, and so does
    the default, whose count of windows (`audit.lifetime_windows`) is at least 2 for a prover that defers."""
    v, refused, heard = greeted(1, True)
    assert refused is not None and "lifetime of at least 2 windows, not 1" in str(refused)
    assert heard == [f"refused: {refused}"] and v.registry.drawn(b"test-node") == 0
    for lifetime, deferred in ((2, True), (1, False), (None, True), (None, False)):
        v, refused, heard = greeted(lifetime, deferred)
        assert refused is None and heard == [] and v.deferred is deferred and v.registry.drawn(b"test-node") == 1


def test_only_a_hello_whose_plan_appends_nothing_reserves_no_append_coins():
    """The hello says whether any session of the prover's plan appends: a plan of honest and skip sessions says not,
    and its sessions reserve no append coins; any other plan, or a hello that doesn't say, keeps `coins.APPENDS`, and
    a verifier given `appends` keeps its own.  A hello whose `appends` isn't a bool is refused with `bye`."""
    for plan, drop, appends in (("honest,skip:0.25", (), 0), ("honest,kv:2", (), coins.APPENDS),
                                ("serve", (), coins.APPENDS), ("honest", ("appends",), coins.APPENDS)):
        v, refused, heard = greeted(None, False, plan=plan, drop=drop)
        assert refused is None and heard == [] and v.appends_of() == appends, plan
    v, refused, _ = greeted(None, False, plan="honest")
    v.appends = 8
    assert v.appends_of() == 8
    v, refused, heard = greeted(None, False, hello={"appends": "no"})
    assert refused is not None and "true or false" in str(refused) and heard == [f"refused: {refused}"]
    assert v.registry.drawn(b"test-node") == 0


def test_a_hello_asking_for_receipts_that_bind_tiles_is_refused():
    """A hello asking that its receipts bind PoUW's tile outputs (`bind`): the Lean covers receipts of blocks alone, so
    both registries refuse it before drawing a registration (`audit.Registry.register` as `Registry.register` does),
    and the prover reads why in its `bye`."""
    v, refused, heard = greeted(None, False, hello={"bind": True})
    assert refused is not None and "tile outputs" in str(refused)
    assert heard == [f"refused: {refused}"] and v.registry.drawn(b"test-node") == 0


def test_a_hello_naming_no_host_the_site_registers_is_refused_before_either_registry_draws():
    """The hello's node is a host of the verifier's Site, its id in UTF-8 (`audit.host_of`, the Lean's `hostOf`):
    another id, the site's own and bytes that aren't UTF-8 are each refused by both registries with `bye`, drawing
    nothing, and the same node registers under a Site that registers it."""
    for node, why in ((b"other-node", "other-node is no host"), (b"site", "site is no host"),
                      (b"\xfftest-node", "a host's id in UTF-8")):
        v, refused, heard = greeted(None, False, node=node)
        assert refused is not None and why in str(refused) and heard == [f"refused: {refused}"], node
        assert "disagreement" not in str(refused) and v.registry.drawn(node) == 0
    v, refused, heard = greeted(None, False, node=b"other-node", hosts=verifier.one_host("other-node"))
    assert refused is None and heard == [] and v.registry.drawn(b"other-node") == 1


def test_a_session_needs_the_site_its_hello_is_held_to(tmp_path, monkeypatch, capsys):
    """`--site LOG HOST` names the hosts a hello may name: without it the verifier refuses to start a session before it
    listens, while `--close` and `--memory-of` still run."""
    monkeypatch.setattr(sys, "argv", ["verifier.py", "--out", str(tmp_path / "out")])
    with pytest.raises(SystemExit) as e:
        verifier.main()
    assert e.value.code == 2 and "--site LOG HOST" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


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


def test_the_lean_verifier_accepts_an_honest_session_and_rejects_a_dropping_one(monkeypatch):
    """The Lean verifier decides each window (its check's time is the window's `lean_check_ns`), `audit.AuditRun`
    taking its verdicts: it accepts the honest session, rejects the prover that dropped half its blocks at a receipt,
    and closes the registration with the two sessions counted so in both registries."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    closed = []
    (honest, dropping), _ = run("honest,skip:0.5", sessions=2,
                                then=lambda v: closed.append(v.close_registration(b"test-node")))
    assert honest["accepted"] and not dropping["accepted"] and dropping["failure"]["failure"] == "receipt"
    assert honest["verdict"] == "lean" and honest["disagreement"] is None and dropping["disagreement"] is None
    ns = list(honest["lean_check_ns"].values())
    assert len(ns) == 3 and all(type(x) is int and x > 0 for x in ns)
    assert closed[0]["sessions"] == ["accepted", "receipt"] and "python" not in closed[0]


def test_a_window_the_lean_rejects_is_rejected_on_the_books_at_its_draw(monkeypatch):
    """A window's verdict is the Lean's, set on `audit.AuditRun` (`Checks.settle`): the Lean's accepting window 1 read
    as a rejection at a draw rejects the session there, at that draw, with no disagreement."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    settle = verifier.Pair.settle

    def changed(res):
        if res.get("accepted") != 1:
            return res
        return {"failure": "receipt", "window": 1, "at": [0, 0], "openings": res["openings"], "ns": res["ns"]}

    def wrong(self, check, answered=None):
        if answered is None:
            return changed(settle(self, check))
        return settle(self, check, lambda res: answered(changed(res)))

    monkeypatch.setattr(verifier.Pair, "settle", wrong)
    (r,), (p,) = run("honest")
    assert not r["accepted"] and not p["accepted"] and r["failure"] == {"t": 1, "failure": "receipt", "at": [0, 0]}
    assert r["disagreement"] is None


def test_a_transcript_replays_each_session_to_its_verdict_through_both_verifiers(tmp_path, monkeypatch):
    """`--transcript DIR`: each session's difftest case, written as it runs, which `replay.py transcript` replays
    through `RemoteAudit.Session` and `audit.AuditRun` alike: the honest session accepted, every window verified, and
    the prover that dropped half its blocks rejected at the window and draw it was rejected at live, at a receipt,
    which only a transcript's openings can show. The file is the path, bytes and sha256 verifier.json gives it."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    monkeypatch.syspath_prepend(str(HERE.parents[2] / "verity/core/protocols/remote_bandwidth/lean/tests"))
    import hashlib
    import json

    import replay

    (honest, dropping), _ = run("honest,skip:0.5", sessions=2, transcript=tmp_path)
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


def test_a_paced_sessions_transcript_names_its_pace_and_replays_paced(tmp_path, monkeypatch):
    """A paced session's transcript names its pace (`audit.Pace`'s k and T), which both verifiers hold each window's
    roots to, so its replay is the paced session's: accepted, with its record's pace."""
    monkeypatch.syspath_prepend(str(HERE.parents[2] / "verity/core/protocols/remote_bandwidth/lean/tests"))
    import json

    import replay

    (r,), _ = run("honest", paces=[verifier.pace_of("2:8:40:2", 1.0)], transcript=tmp_path)
    assert r["accepted"] and [w["passes"] for w in r["windows_seen"]] == [2] * 12
    case = json.loads(Path(r["transcript"]["path"]).read_bytes())
    assert case["params"]["pace"] == {"k": 2, "window_ns": 8_000_000} == r["record"]["public"]["pace"]
    with replay.server() as s:
        row = replay.replay(case, s)
    assert row["agree"] and row["lean"]["outcome"] == "accepted" and row["lean"]["refused"] == 0, row


def test_a_session_drawn_four_windows_a_coin_is_accepted_through_both_verifiers_and_a_skipper_is_not(
        tmp_path, monkeypatch):
    """`--groups-per-coin 4` over five windows: two draws, windows 0–3 and window 4, each drawn at its last window's
    close on its own openings coin (`audit.Params.groups_per_coin`), the challenger reserving ⌈5/4⌉ = 2 of them. The
    first draw's two receipts leave at least two of its windows a share of none, which the close accepts and no
    openings follow; the Lean verifier agrees at every event and on the record, which names g, and the session's
    transcript replays to the same verdict through both verifiers. A prover that dropped half its blocks is rejected
    at a receipt."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    monkeypatch.syspath_prepend(str(HERE.parents[2] / "verity/core/protocols/remote_bandwidth/lean/tests"))
    import json

    import replay

    (r,), (pr,) = run("honest", windows=5, samples=2, groups_per_coin=4, transcript=tmp_path)
    assert r["accepted"] and pr["accepted"] and r["disagreement"] is None, r["failure"]
    assert r["record"]["public"]["groups_per_coin"] == 4 and r["lean_record"]["groups_per_coin"] == 4
    assert r["challenger"]["reason"] is None
    seen = r["windows_seen"]
    assert len({w["draws_sent_ns"] for w in seen[:4]}) == 1 and seen[4]["draws_sent_ns"] >= seen[3]["draws_sent_ns"]
    first = [len(r["draws"].get(t, [])) for t in range(4)]
    assert sum(first) == 2 and first.count(0) >= 2 and len(r["draws"][4]) == 2
    assert sorted(t for t, v in r["verify_latency_ns"].items()) == list(range(5))
    case = json.loads(Path(r["transcript"]["path"]).read_bytes())
    assert case["params"]["groups_per_coin"] == 4 and len(case["coins"]) == 5 + 2
    with replay.server() as s:
        row = replay.replay(case, s)
    assert row["agree"] and row["lean"]["outcome"] == "accepted", row
    (bad,), (pbad,) = run("skip:0.5", windows=5, samples=12, groups_per_coin=4)
    assert not bad["accepted"] and not pbad["accepted"] and bad["failure"]["failure"] == "receipt"
    assert bad["disagreement"] is None


def test_a_session_drawn_two_windows_a_coin_at_one_window_in_flight_is_refused_before_it_starts(
        tmp_path, monkeypatch, capsys):
    """`--groups-per-coin 2` with a `--paced` session at F = 1: no property covers one window in flight at g > 1, so
    the Lean verifier's `session` request refuses it, and the driver refuses the pair before it listens."""
    monkeypatch.setattr(sys, "argv", ["verifier.py", "--out", str(tmp_path / "out"), "--groups-per-coin", "2",
                                      "--paced", "2:8:12:2", "2:8:12:2:1"])
    with pytest.raises(SystemExit) as e:
        verifier.main()
    assert e.value.code == 2 and "one window in flight" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_the_leans_line_of_a_transcript_is_the_line_of_the_record_session_run_makes_of_it(tmp_path):
    """`line_record` with the session's transcript (`session`) runs `Session.run` on it and reads the line off the
    record that run makes (`Line.lineOfRun`, `LineHeld`'s `lineOf`): the line of the session's kept record, given
    beside it or not, with the times seen on the verifier's clock the record's own. A kept record that isn't the run's,
    a record or transcript of another registration than the one given, a transcript cut short of its verdict,
    `samples` other than its parameters' and times seen off the record's are refused, and a transcript that names two
    windows a coin beside one openings coin a window is malformed; on the challenger's clock the times aren't held to
    the record."""
    import copy
    import json

    import lean_verifier as lv

    got = {}
    (r,), _ = run("honest", paces=[verifier.pace_of("2:8:40:2", 1.0)], transcript=tmp_path,
                  then=lambda v: got.update(reg=wire.registration_json(v.registration)))
    case = json.loads(Path(r["transcript"]["path"]).read_bytes())
    rates = verifier.recorded_rates(r, got["reg"], disk_bytes_per_s=4.4e9, link_bytes_per_s=5e10)
    inputs = verifier.recorded_inputs(r, got["reg"], host_bytes=0, rates=rates)
    want = verifier.line.record_run_json(inputs)
    assert "refused" not in want and inputs["clock"] == "record"
    assert verifier.recorded_inputs(r, got["reg"], host_bytes=0, rates=rates, transcript=case)["session"] is case
    moved = copy.deepcopy(inputs)
    moved["record"]["windows"][0]["stale"][0] += 1
    late = copy.deepcopy(inputs)
    late["seen"][1]["released_ns"] += 1
    lean = lv.Lean(lv.build(), lambda _: None)
    try:
        def call(x: dict) -> dict:
            return lean.call({"op": "line_record", **x})

        bare = {k: x for k, x in inputs.items() if k != "record"}
        assert call(inputs) == call({**inputs, "session": case}) == call({**bare, "session": case}) == want
        assert call({**moved, "session": case}) == {
            "refused": "the record is not the one Session.run makes of the session's transcript"}
        other = {**inputs, "registration": {**inputs["registration"], "payload_root": "01" * 64}}
        assert call({**other, "session": case}) == call(other) == verifier.line.record_run_json(other) == {
            "refused": "the record is not of the registration given"}
        assert call({**bare, "session": {**case, "events": case["events"][:-1]}}) == {
            "refused": "the Lean verifier gave the session no record"}
        with pytest.raises(lv.LeanError, match="samples: the session's"):
            call({**bare, "session": case, "samples": inputs["samples"] + 1})
        with pytest.raises(lv.LeanError, match="coins of 32 bytes"):
            call({**bare, "session": {**case, "params": {**case["params"], "groups_per_coin": 2}}})
        why = verifier.line.record_run_json(late)["refused"]
        assert why.startswith("window 1 was seen released at") and call({**late, "session": case}) == {"refused": why}
        assert "refused" not in call({**late, "session": case, "clock": "challenger"})
    finally:
        lean.close()


def test_the_leans_line_of_a_transcript_drawn_four_windows_a_coin_reads_the_draws_of_the_record_it_makes(tmp_path):
    """At `--groups-per-coin 4` and two windows in flight, a paced session's twelve windows are three draws. Its line
    of record reads the same twelve groups and, for its pooled bound, the three draws, each four windows' receipts and
    draws summed (`line.draws_of`); `line_record` with the transcript runs `Shared.run 4` on it and reads the line off
    the record that run makes, which names g: the line of the kept record, given beside it or not, line.py's. With
    the record's pace at one window in flight it has no line (`line.SHARED`)."""
    import copy
    import json

    import lean_verifier as lv

    got = {}
    (r,), (pr,) = run("honest", paces=[verifier.pace_of("2:8:40:2", 1.0)], groups_per_coin=4, transcript=tmp_path,
                      then=lambda v: got.update(reg=wire.registration_json(v.registration)))
    assert r["accepted"] and pr["accepted"] and r["windows"] == 12 and r["record"]["public"]["groups_per_coin"] == 4
    case = json.loads(Path(r["transcript"]["path"]).read_bytes())
    assert case["params"]["groups_per_coin"] == 4 and len(case["coins"]) == 12 + 3
    rates = verifier.recorded_rates(r, got["reg"], disk_bytes_per_s=4.4e9, link_bytes_per_s=5e10)
    inputs = verifier.recorded_inputs(r, got["reg"], host_bytes=0, rates=rates)
    record = inputs["record"]
    assert record["groups_per_coin"] == 4 and len(record["windows"]) == 12
    draws = verifier.line.draws_of(4, record["windows"])
    assert [d for _, d in draws] == [min(inputs["samples"], u) for u, _ in draws] and len(draws) == 3
    assert draws == verifier.recorded_windows(r)[0]
    want = verifier.line.record_run_json(inputs)
    per_window = copy.deepcopy(inputs)
    del per_window["record"]["groups_per_coin"]
    assert "refused" not in want and verifier.line.record_run_json(per_window)["refused"].startswith("window ")
    one = copy.deepcopy(inputs)
    one["record"]["pace"] = {**one["record"]["pace"], "in_flight": 1}
    lean = lv.Lean(lv.build(), lambda _: None)
    try:
        def call(x: dict) -> dict:
            return lean.call({"op": "line_record", **x})

        bare = {k: x for k, x in inputs.items() if k != "record"}
        assert call(inputs) == call({**inputs, "session": case}) == call({**bare, "session": case}) == want
        assert call(one) == verifier.line.record_run_json(one) == {"refused": verifier.line.SHARED}
    finally:
        lean.close()


def test_books_whose_record_is_not_the_leans_refuse_the_session_as_disagreement(monkeypatch):
    """An accepted session's record is held to the Lean's whole, each pass's arrival, head and stale bound among what
    the ledger reads: books whose record has another stale bound refuse the session at its end as `disagreement`."""
    import dataclasses

    record = audit.AuditRun.record

    def stale(self):
        rec = record(self)
        w = rec.windows[0]
        return dataclasses.replace(rec, windows=(dataclasses.replace(w, stale=tuple(b + 1 for b in w.stale)),
                                                 *rec.windows[1:]))

    monkeypatch.setattr(audit.AuditRun, "record", stale)
    (r,), _ = run("honest")
    assert not r["accepted"] and r["failure"]["failure"] == "disagreement", r["failure"]
    assert r["disagreement"]["event"] == "record"


LEAN = verifier.lv.Lean


class HeldLean(LEAN):
    """The Lean process with every request from its first event on held back until `release` (the test calls it as
    the prover's passes end): a Lean as slow to answer and to check as can be. A request the driver waits on while
    they're held is noted in `waited`, and releases them."""

    made: list[HeldLean] = []

    def __init__(self, exe, done) -> None:
        self.hold, self.held, self.waited, self.holding, self.released = threading.Lock(), [], [], False, False
        super().__init__(exe, done)
        HeldLean.made.append(self)

    def _put(self, request, body, answered) -> None:
        with self.hold:
            self.holding = self.holding or request["op"] == "event"
            if self.holding and not self.released:
                self.held.append((request, body, answered))
                return
        super()._put(request, body, answered)

    def call(self, request, body=b""):
        with self.hold:
            waiting = self.holding and not self.released
            if waiting:
                self.waited.append(request["op"])
        if waiting:
            self.release()
        return super().call(request, body)

    def release(self) -> None:
        with self.hold:
            self.released = True
            for item in self.held:
                super()._put(*item)
            self.held.clear()


@pytest.fixture
def held_lean(monkeypatch):
    """Every Lean process the test starts is a `HeldLean`, released as the prover says its passes are done."""
    HeldLean.made = []
    idle = PassClock.idle

    def released(self) -> None:
        for lean in HeldLean.made:
            lean.release()
        idle(self)

    monkeypatch.setattr(verifier.lv, "Lean", HeldLean)
    monkeypatch.setattr(PassClock, "idle", released)
    return HeldLean.made


SCHEDULE = ("released_ns", "boundary_ns", "deadline_ns", "passes", "arrivals_ns", "slack_ns", "draws_sent_ns")


@pytest.mark.parametrize("defer_openings", [True, False])
def test_a_lean_that_answers_nothing_until_the_passes_end_delays_no_coin_root_or_draw(monkeypatch, held_lean,
                                                                                     defer_openings):
    """The driver waits on no Lean answer while a session runs: with every request to the Lean from the session's
    first event held back until the prover's passes are done (`HeldLean`), with its openings after its passes or
    beside them (`--overlap-openings`), a paced session runs the schedule it runs with a Lean that answers at once,
    every coin, boundary, root and draw at the same tick, and is still accepted only on the Lean's verdict of each of
    its windows and its record."""
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    pace = verifier.pace_of("1:4:40:2", 1.0)
    monkeypatch.setattr(verifier.lv, "Lean", LEAN)
    (plain,), _ = run("honest", paces=[pace], defer_openings=defer_openings)
    monkeypatch.setattr(verifier.lv, "Lean", HeldLean)
    (r,), (pr,) = run("honest", paces=[pace], defer_openings=defer_openings)
    assert len(held_lean) == 1 and held_lean[0].waited == [] and held_lean[0].released
    assert r["accepted"] and pr["accepted"] and r["disagreement"] is None and r["verdict"] == "lean"
    assert len(r["lean_check_ns"]) == r["windows"] == 12 and all(type(x) is int and x > 0
                                                                for x in r["lean_check_ns"].values())
    assert [{k: w[k] for k in SCHEDULE} for w in r["windows_seen"]] == \
        [{k: w[k] for k in SCHEDULE} for w in plain["windows_seen"]]
    assert r["record"]["public"]["pace"] == plain["record"]["public"]["pace"]


@pytest.mark.parametrize("held", [False, True])
def test_an_event_the_verifiers_answer_differently_is_refused_as_disagreement(monkeypatch, held_lean, held):
    """Each event goes to both verifiers at one reading of the clock, and their results are compared: a root the Lean
    reads without its step is its `step` rejection where `audit.AuditRun` takes the root, and the session is refused
    at that window as `disagreement`, also when the Lean's answer comes only after the prover's last pass."""
    if not held:
        monkeypatch.setattr(verifier.lv, "Lean", LEAN)
    monkeypatch.setattr(prover, "OPENINGS_PART", 4)
    event = verifier.lv.receive_event

    def stepless(t, i, root, change, step):
        e = event(t, i, root, change, step)
        return {**e, "step": None} if (t, i) == (1, 3) else e

    monkeypatch.setattr(verifier.lv, "receive_event", stepless)
    closed = []
    (r,), _ = run("honest", then=lambda v: closed.append(v.close_registration(b"test-node")))
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
        s = audit.AuditRun(params, reg, None, coins.local(reg.digest, 0, params.windows, os.urandom), clock=lambda: 0,
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
    assert appended(s) == audit.Verdict(0, False, "coin") and s.coins.appends == 1
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


def test_a_plan_change_is_the_runs_one_plan_adds_and_removes_from_the_other():
    rng = random.Random(3)
    for _ in range(200):
        n, m = rng.randrange(1, 60), rng.randrange(0, 20)
        before = rb.Plan.of(sorted(rng.sample(range(n), rng.randrange(1, n + 1))), n)
        after = rb.Plan.of(sorted(rng.sample(range(n + m), rng.randrange(1, n + m + 1))), n + m)
        added, removed = prover.plan_change(before, after)
        a, b = set(before), set(after)
        assert added == prover.runs_of(sorted(b - a)) and removed == prover.runs_of(sorted(a - b))
        assert before.changed(added, removed, n + m) == after


def test_a_root_message_names_the_withdrawn_blocks_among_its_releases():
    """`Verifier.root` hands a message's withdrawn runs to its release (`AuditRun.release(released, withdrawn)`): an
    appended block no plan held, freed before its encode landed, is withdrawn; one a plan held, a registered one, or a
    withdrawal outside the message's releases is the session's `release` rejection."""
    reg = audit.register(os.urandom(32), b"test-node", 8, bytes(64), P512)
    params = audit.Params(8, 10 ** 9, 1, 2, 2.0 ** -40, window_ns=10, codec=P512, receipt=reg.receipt, rows=reg.rows,
                          lifetime=2, deferred=True)
    root = (b"\7" * 64).hex()

    def session():
        s = audit.AuditRun(params, reg, None, coins.local(reg.digest, 0, params.windows, os.urandom), clock=lambda: 0,
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


def test_a_reader_apart_hands_on_each_frame_in_order_and_ends_with_the_stream_or_the_prover():
    """The prover's reader in its own process: frames come through its pipe as sent, each with its stamp; a frame over
    the limits is refused as on a reader thread; the stream's close is `Closed`; and the reader exits once the
    prover's end of its pipe closes, so it never holds the stream open past the prover."""
    a, b = socket.socketpair()
    side, apart = wire.Conn(a), wire.Conn(b)
    apart.read_apart()
    sent = [({"type": "coin", "t": 1}, b""), ({"type": "draws", "t": 1}, os.urandom(1 << 20)),
            ({"type": "close"}, b"x")]
    for head, body in sent:
        side.send(head, body)
    got = [apart.recv() for _ in sent]
    assert [(h, body) for h, body, _ in got] == sent and all(type(s) is int for _, _, s in got)
    assert apart.received == side.sent
    a.sendall(wire.HEADER.pack(wire.MAX_HEAD + 1, 0))
    with pytest.raises(ValueError, match="over the limit"):
        apart.recv()
    assert apart.reader.wait() == 0

    a, b = socket.socketpair()
    side, apart = wire.Conn(a), wire.Conn(b)
    apart.read_apart()
    a.close()
    with pytest.raises(wire.Closed):
        apart.recv()
    assert apart.reader.wait() == 0

    a, b = socket.socketpair()
    apart = wire.Conn(b)
    apart.read_apart()
    os.close(apart._pipe)
    assert apart.reader.wait() == 0
    with pytest.raises(ValueError, match="monotonic"):
        wire.Conn(a, clock=lambda: 0).read_apart()
    a.close()
    b.close()


@pytest.mark.parametrize("name", ["p1024/k3/masked", "p1024/k2/masked", "p512/k2"])
def test_codec_names_round_trip(name):
    assert wire.codec_named(name).name == name


def test_the_predicted_rate_is_the_draws_chance_to_miss_every_dropped_receipt():
    import summary

    assert summary.miss(100, 0, 10) == 1.0 and summary.miss(10, 1, 10) == 0.0
    assert summary.miss(20, 2, 2) == pytest.approx(18 / 20 * 17 / 19)
    U, f, s = 68 << 22, 2.0 ** -14, 2773
    assert summary.miss(U, round(f * U), s) == pytest.approx((1 - f) ** s, rel=1e-4)
