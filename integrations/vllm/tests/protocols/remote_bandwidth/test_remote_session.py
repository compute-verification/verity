"""The remote bandwidth audit run over the node's `Store` end to end on the CPU: `remote.prove` (the prover's side, the
Commit's hook) connects over localhost TCP to `benchmarks/remote_session`'s verifier, registers the store, records its
memory map against the registration, and runs each pass after an engine step of a serving-like trace (pages written
over two steps, held, freed and reused, and one long-lived page re-keyed under L = 2 windows).  The honest audit run is
accepted with the appends, releases, re-keys and steps the verifier saw, each pass's seed chaining the appends sent
before its root, each append encoded under its own coin, and each chunk resident only once its encode has landed
(deferred residency: read as plaintext and uncredited until then); one whose prover seeds without them, encodes under
a zero or stale coin, re-sends an append, appends after the last close or withholds a chunk it appended is rejected; a
second registration of a node the verifier holds live is refused with its reason.  The verifier's clock is the prover's
passes, a tick each (`PassClock`)."""
from __future__ import annotations

import contextlib
import json
import os
import queue
import random
import socket
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

from verity.core.protocols.remote_bandwidth import audit, codec as rcodec, memory_map, receipts as rb
from verity_vllm.protocols.remote_bandwidth import memory_map as store_map, remote
from verity_vllm.protocols.remote_bandwidth.interface import Append, PageLayout
from verity_vllm.protocols.remote_bandwidth.registry import ArrayBacking, Store

from .test_memory_map import BLOCK, PAGES, SLOT_BYTES, SLOTS, kv_payload
from .test_registry import CODEC, RECEIPT

P512 = rcodec.Codec(rcodec.P512, 2, False)
verifier, wire = remote.session_module("verifier"), remote.session_module("wire")
VLLM = Path(__file__).resolve().parents[3]


class PassClock:
    """Time is the prover's passes: one tick as each ends.  A wait returns the inbox's next message, or None once the
    clock reaches its end; once the verifier has said the passes are done, time jumps to the wait's end."""

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


@contextlib.contextmanager
def verifying(sock: socket.socket, clock: PassClock, **kw) -> Iterator[verifier.Verifier]:
    """`verifier.Verifier` over the accepted ``sock`` on ``clock``, torn down however the body ends: the socket shut,
    so a prover waiting on it sees the audit run end (the verifier's reader thread would hold it open), and the Lean
    verifier's process (`remote-audit-verifier`) terminated, killed and reaped, so a failing test leaves none waiting
    on its stdin."""
    v = None
    try:
        v = verifier.Verifier(wire.Conn(sock, clock=clock.now), source=os.urandom(32), delta=2.0 ** -40,
                              jitter_ns=10 ** 9, clock=clock, log=lambda m: None, **kw)
        yield v
    finally:
        with contextlib.suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)
        sock.close()
        if v is not None and v.lean is not None:
            v.lean.proc.terminate()
            v.lean.proc.kill()
            v.stop()


def store() -> Store:
    """A weights chunk, one KV layer of 4 pages (K0 K1 V0 V1 each, page p at store chunks 1 + 4p …), and a chunk of
    host filler: N = 18.  Each chunk a backend encodes is 128 square roots at p512/k2, so the store is small."""
    s = Store()
    w = np.random.default_rng(0).integers(0, 256, BLOCK, np.uint8)
    s.register("w", "hbm", "weights", ArrayBacking(w), block_bytes=BLOCK, codec=CODEC, receipt=RECEIPT, source="w")
    s.register_kv_pool("kv", [ArrayBacking(np.zeros(PAGES * 2 * SLOTS * SLOT_BYTES, np.uint8))],
                       PageLayout(PAGES, SLOTS, SLOT_BYTES), block_bytes=BLOCK, codec=CODEC, receipt=RECEIPT)
    s.register("dram", "host", "filler", ArrayBacking(np.zeros(BLOCK, np.uint8)), block_bytes=BLOCK, codec=CODEC,
               receipt=RECEIPT, source="zeros")
    return s


class TickingBackend(remote.StoreBackend):
    def __init__(self, clock: PassClock, *args, **kw) -> None:
        super().__init__(*args, **kw)
        self.clock, self.heads, self.coins, self.plans = clock, [], [], []

    def launch(self, i: int, after, plan=None) -> dict:
        ticket = super().launch(i, after, plan)
        self.plans.append((ticket["step"], plan))
        return ticket

    def encode_append(self, first: int, coin: bytes) -> None:
        self.coins.append((first, coin))
        super().encode_append(first, coin)

    def wait(self, ticket: dict):
        self.heads.append(ticket["head"])
        return super().wait(ticket)

    def run_pass(self, seed: bytes, plan=None):
        out = super().run_pass(seed, plan)
        self.clock.tick()
        return out


class HeadlessBackend(TickingBackend):
    """A prover whose seeds chain only the root before each pass, not the appends sent before it."""

    def launch(self, i: int, after, plan=None) -> dict:
        return {**super().launch(i, after, plan), "head": None}


class ZeroCoinBackend(TickingBackend):
    """A prover that encodes each append under a zero coin, not the verifier's."""

    def encode_append(self, first: int, coin: bytes) -> None:
        super().encode_append(first, bytes(32))


class StaleCoinBackend(TickingBackend):
    """A prover that encodes each append under the coin of the append before it (the first under its own)."""

    def encode_append(self, first: int, coin: bytes) -> None:
        super().encode_append(first, self.coins[-1][1] if self.coins else coin)


class Tampering:
    """The prover's connection, each message it sends replaced by ``tamper(head)``'s list."""

    def __init__(self, conn, tamper) -> None:
        self.conn, self.tamper = conn, tamper

    def send(self, head: dict, body: bytes = b"") -> None:
        for h in self.tamper(head):
            self.conn.send(h, body)

    def __getattr__(self, name: str):
        return getattr(self.conn, name)


def tampering(monkeypatch, tamper) -> None:
    connect = wire.connect
    monkeypatch.setattr(wire, "connect", lambda host, port: Tampering(connect(host, port), tamper))


def engine(s: Store, at: int = 0):
    """A serving trace in the binder's order: pages 0 to 2 in turn, each a request's for three steps (slots 0 to 3 at
    its first, 4 to 7 at its second, read at its third), freed before the step that reuses it, which is after the
    forward of its last step, its last structural reader; page 3 a request's that never finishes, written at steps 0 and 1.  The
    engine numbers its steps from ``at`` (one whose binder numbered steps before the audit run)."""

    lives, n = {3: 0}, [0]

    def step(_: int) -> None:
        k, n[0] = n[0], n[0] + 1
        p = k % 3
        if k >= 3:
            s.kv_free([p])
        s.kv_alloc([p] + ([3] if k == 0 else []))
        lives[p] = k
        appends = [Append(p, 0, 4)] + ([Append((k - 1) % 3, 4, 4)] if k >= 1 else []) + (
            [Append(3, 4 * k, 4)] if k < 2 else [])
        s.kv_write_reference(appends, lambda li, page, x: kv_payload(li, page, x, lives[page]), k + at)
    return step


def run(plan: str, *, windows: int = 2, window_ns: int = 5, samples: int = 4096, lifetime: int = 2,
        backend=TickingBackend, at: int = 0, **kw):
    """One audit run of ``plan``, every receipt of a window sampled at the default ``samples``; the backend keeps the
    regions the verifier read from its hello (``backend.regions``)."""
    clock = PassClock()
    s = store()
    backend = backend(clock, s, engine(s, at), randbytes=random.Random(7).randbytes, **kw)
    lsock = wire.listen("127.0.0.1", 0)
    results = []
    th = threading.Thread(target=lambda: results.extend(remote.prove(
        backend, f"127.0.0.1:{lsock.getsockname()[1]}", plan, node=b"test-node", codec=P512, log=lambda m: None)),
        daemon=True)
    th.start()
    sock, _ = lsock.accept()
    lsock.close()
    with verifying(sock, clock, windows=windows, window_ns=window_ns, samples=samples, lifetime=lifetime) as v:
        v.register()
        backend.regions = v.regions
        r = v.session(0, v.allowance(v.ping(4)), None)
        th.join()
    return r, results[0], s, backend


@pytest.mark.slow
def test_an_honest_store_session_is_accepted_with_its_frees_re_keys_and_steps():
    r, p, s, backend = run("serve", windows=3)
    assert r["accepted"] and p["accepted"] and r["failure"] is None
    # the hello carries the store's layout, which the verifier takes as the memory line's labels
    assert backend.regions == (("weights", "hbm", 0, 1), ("kv", "hbm", 1, 16), ("filler", "host", 17, 1))
    # step 0's completed chunks are appended once it has run, before pass 0's root, so every pass's seed chains the
    # head of the appends sent before its root (`audit.chained`)
    assert len(backend.heads) > 2 and all(h is not None for h in backend.heads)
    kv, pub = p["kv"], r["record"]["public"]
    # each append sent was answered with its first index and a coin of its own, under which its chunks were encoded
    sent = backend.trace.appends[:len(backend.coins)]
    assert [f for f, _ in backend.coins] == [a.first for a in sent] and len({c for _, c in backend.coins}) == len(sent)
    assert sum(a.count for a in sent) == kv["sent_appended"]
    assert kv["lifetime"] == pub["lifetime"] == 2 and kv["rekeyed"] > 0
    assert kv["sent_appended"] == pub["appended"] > 0 and kv["sent_released"] == pub["released"] > 0
    assert sum(pub["post_seed"]) == sum(kv["pass_appended"]) > 0
    steps = [x for w in r["record"]["profile"]["windows"] for x in w["steps"]]
    assert len(steps) == sum(pub["passes"]) and all(a < b for a, b in zip(steps, steps[1:])) and steps[-1] < kv["steps"]
    head, lines = memory_map.parse(s.memory_map.text().splitlines())
    lines = list(lines)
    assert head.registration.hex() == pub["registration"] and head.blocks == 18
    # page 0's first request, written at steps 0 and 1 and read at 2, is freed before step 3's write.  Its K0 V0
    # (chunks 1, 3), written at step 0, and its K1 V1 (2, 4), completed at step 1, appended and encoded before step 2
    # runs (one pass in flight), so written at step 1, are all freed at step 2, the step whose forward read them last
    by = {x.step: x for x in lines}
    assert {1, 3} <= {w.slot for w in by[0].written} and {2, 4} <= {w.slot for w in by[1].written}
    assert {1, 2, 3, 4} <= set(by[2].freed)
    # page 3, written over steps 0 and 1 and never freed, is written again whole in one step (re-keyed) once two
    # windows old, its old indices freed in the line before; its first writes were split, so every line that writes all
    # four of its chunks is a re-key
    rekeys = [x.step for x in lines if {13, 14, 15, 16} <= {w.slot for w in x.written}]
    assert rekeys and min(rekeys) > 1 and kv["rekeyed"] >= 4 * len(rekeys)
    assert next(x.step for x in lines if {13, 14, 15, 16} <= set(x.freed)) < min(rekeys)
    # the map, replayed, holds the audit run's live chunks, its writes are the appends' digests, and the store holds them
    agree = store_map.agreement(backend)
    assert agree["agrees"] and agree["held_appended"] > 0 and agree["held_wrong"] == []


@pytest.mark.slow
def test_a_completed_block_is_uncredited_until_its_encode_lands_and_resident_from_the_step_after():
    """Deferred residency end to end: each step's completed chunks go out as one append once the step has run, and
    their coin comes back before the next step runs (one pass in flight), when the device encodes them.  The map
    writes them at the step whose line is open then, the completing step's, so the passes up to it hold none of them
    (serving reads them as plaintext, uncredited), and the pass after it holds them all, encoded under their own
    coin."""
    r, p, s, backend = run("serve")
    assert r["accepted"] and p["accepted"]
    trace, plans = backend.trace, dict(backend.plans)
    agree = store_map.agreement(backend)
    _, lines = memory_map.parse(s.memory_map.text().splitlines())
    wrote = {w.index: x.step for x in lines for w in x.written}
    late = trace.deferral()["steps_late"]
    assert set(late) == {0} and sum(late.values()) == len(trace.landed) > 4
    for a, made, landed in zip(trace.appends, trace.made, trace.landed):
        idx = set(range(a.first, a.first + a.count))
        assert landed == made and {wrote[j] for j in idx} == {landed}
        assert not any(idx & set(plan) for st, plan in plans.items() if st <= landed)
        assert landed + 1 not in plans or idx <= set(plans[landed + 1])
        assert all(backend.audit.blocks[j] is not None for j in idx)
    for a in trace.appends[len(trace.landed):]:
        assert all(j not in wrote and backend.audit.blocks[j] is None for j in range(a.first, a.first + a.count))
    assert agree["agrees"] and agree["unrecorded"] == sum(a.count for a in trace.appends[len(trace.landed):])


def test_a_pass_whose_plan_holds_a_block_not_yet_encoded_is_refused():
    s = store()
    b = remote.StoreBackend(s, engine(s))
    reg = audit.register(bytes(32), b"n", b.blocks, b.payload_root(b"n"), P512, nonce=bytes(32))
    b.encode(reg)
    a = b.append([b"\x01" * rcodec.BLOCK_BYTES], [bytes(192)])
    plan = rb.Plan(((0, 19),), 19)
    with pytest.raises(RuntimeError, match=r"blocks \[18\], appended and not yet encoded"):
        b.run_pass(bytes(64), plan)
    b.encode_append(a.first, bytes(range(32)))
    assert b.run_pass(bytes(64), plan)[1][3] == plan


@pytest.mark.slow
def test_a_session_on_an_engine_whose_steps_were_numbered_before_it_adopts_its_first_step():
    """The Commit's binder numbers the engine's steps from its first run; an audit run after its runs takes the next."""
    r, p, s, backend = run("serve", at=40)
    assert r["accepted"] and p["accepted"]
    head, lines = memory_map.parse(s.memory_map.text().splitlines())
    steps = [x.step for x in lines]
    assert steps[0] >= 40 and steps == sorted(set(steps))
    assert all(x >= 40 for w in r["record"]["profile"]["windows"] for x in w["steps"])


class Device:
    """kv.cu's encode on the host (`remote.KvDevice`'s interface): each word by `codec.encode_word` under the key
    set and the offset handed in for its chunk, so `remote.DeviceEncoded`'s plumbing runs without a GPU.  ``shift``
    moves every offset (a device that keys appends wrongly)."""

    def __init__(self, shift: int = 0) -> None:
        self.shift, self.key, self.calls = shift, None, []

    def set_key(self, key: rcodec.Key) -> None:
        self.key = key

    def encode(self, payloads, first: int, offsets=None) -> list[bytes]:
        self.calls.append((first, len(payloads), offsets is not None))
        k, f = self.key, self.key.codec.field
        out = []
        for u, m in enumerate(payloads):
            i0, b = f.block_words * rcodec._index(first + u, k.codec), 0 if offsets is None else offsets[u] + self.shift
            out.append(rcodec._block([rcodec.encode_word(x, i0 + w, k, b) for w, x in enumerate(rcodec._words(m, f))], f))
        return out


@pytest.mark.slow
def test_a_session_whose_store_is_encoded_on_the_device_is_accepted_and_each_encode_gated():
    r, p, _, backend = run("serve", device=(d := Device()))
    assert r["accepted"] and p["accepted"]
    appends = len(backend.coins)
    assert backend.gates["device_encode"] == {"registered": 18, "appends": appends, "blocks_checked": 2 + appends}
    assert d.calls[0] == (0, 18, False) and [c[0] for c in d.calls[1:]] == [f for f, _ in backend.coins]
    assert all(c[2] for c in d.calls[1:])


def test_a_device_that_keys_an_append_off_its_bound_offset_fails_its_gate():
    s = store()
    b = remote.StoreBackend(s, engine(s), device=Device(shift=1))
    reg = audit.register(bytes(32), b"n", b.blocks, b.payload_root(b"n"), P512, nonce=bytes(32))
    b.encode(reg)
    a = b.audit.append([b"\x01" * rcodec.BLOCK_BYTES], [bytes(192)])
    with pytest.raises(RuntimeError, match="not codec.encode's"):
        b.audit.encode(a.first, bytes(range(32)))


def test_the_commits_session_refuses_engines_it_does_not_serve_and_sets_the_rest():
    cs = remote.CommitSession("127.0.0.1:7781", kv_pages=8)
    assert cs.engine_args({"max_model_len": 128}) == {"max_model_len": 128, "async_scheduling": False,
                                                       "enable_prefix_caching": False, "num_gpu_blocks_override": 8}
    for bad, why in [({"tensor_parallel_size": 2}, "parallelism"), ({"async_scheduling": True}, "async"),
                     ({"enable_prefix_caching": True}, "prefix caching")]:
        with pytest.raises(SystemExit, match=why):
            cs.engine_args(bad)
    cs.target(type("T", (), {"async_scheduling": False})())
    for t in (None, type("T", (), {"async_scheduling": None})()):
        with pytest.raises(SystemExit, match="declare \"async_scheduling\": false"):
            cs.target(t)
    for kw, err in [({"plan": "serve,serve"}, "one mode"), ({"kv_pages": 0}, "a page"), ({"plan": "nope"}, "a session's mode"),
                    ({"plan": "servedrop:2"}, "appends nothing"), ({"filler_gib": -1}, "nonnegative"),
                    ({"node": "n" * 65}, "node id")]:
        with pytest.raises(Exception, match=err):
            remote.CommitSession("127.0.0.1:7781", **kw)
    assert 1 <= len(cs.node.encode()) <= audit.NODE_BYTES


class Stats:
    def stats(self) -> dict:
        return {}


def served_store() -> Store:
    """The Commit's store: two weights chunks and two of filler, no KV pool (N = 4)."""
    s = Store()
    w = np.random.default_rng(1).integers(0, 256, 2 * BLOCK, np.uint8)
    s.register("w", "hbm", "weights", ArrayBacking(w), block_bytes=BLOCK, codec=CODEC, receipt=RECEIPT, source="w")
    s.register("filler", "hbm", "filler", ArrayBacking(np.zeros(2 * BLOCK, np.uint8)), block_bytes=BLOCK, codec=CODEC,
               receipt=RECEIPT, source="filler:zeros")
    return s


def served_engine(s: Store):
    """The binder's steps with the KV outside the store (`Binder(kv=False)`): each numbered, writing no chunk of it."""
    def step(k: int) -> None:
        s.kv_append([], k)
        s.end_step()
    return step


def lockstep(tmp_path, monkeypatch, run_steps: int, windows: int = 2, setup: Callable[[], None] | None = None,
             each: Callable[[int], None] | None = None, pace: dict | None = None, step_ms: float = 0.0,
             served: bool = False):
    """The Commit's audit run (`CommitSession.begin` … `end`) over a CPU store, its verifier on a thread at L = 2
    windows (paced at ``pace``, `verifier.pace_of`'s), the run ``run_steps`` steps of the scripted engine, each
    through the audit run's `Lockstep` and after ``each(k)``, after ``setup()``, the run's work between `begin` and its
    first step.  ``served``: the Commit's store and steps (`served_store`, `served_engine`), else a KV pool written as
    `engine` writes it."""
    clock = PassClock()
    monkeypatch.setattr(remote, "StoreBackend", lambda *a, **kw: TickingBackend(clock, *a, **kw))
    lsock = wire.listen("127.0.0.1", 0)
    cs = remote.CommitSession(f"127.0.0.1:{lsock.getsockname()[1]}", out=tmp_path, codec=P512, log=lambda m: None,
                              step_ms=step_ms)
    cs.store, cs.binder = served_store() if served else store(), Stats()
    seen: list = []

    def verify() -> None:
        sock, _ = lsock.accept()
        lsock.close()
        try:
            with verifying(sock, clock, windows=windows, window_ns=5, samples=4096, lifetime=2) as v:
                v.register()
                seen.append(v.session(0, v.allowance(v.ping(4)), None, pace=pace))
        except Exception as e:  # noqa: BLE001  -- a side that stops mid-run drops the connection (`verifying`)
            seen.append(e)

    th = threading.Thread(target=verify, daemon=True)
    th.start()
    steps: list[int] = []
    try:
        lock = cs.begin()
        if setup is not None:
            setup()
        step = (served_engine if served else engine)(cs.store)

        def run(k: int):
            steps.append(k)
            if each is not None:
                each(k)
            return step(k)

        for k in range(run_steps):
            lock.step(lambda: run(k))
        rec = cs.end()
    except BaseException:
        # a run that fails ends its audit run, so the verifier's thread ends with it and reaps its process
        if cs._thread is not None:
            if cs.record is None:
                cs.end()
            th.join()
        raise
    th.join()
    return rec, seen[0], cs, steps


@pytest.mark.slow
def test_the_commits_session_runs_in_lockstep_with_its_run_and_its_map_stops_with_it(tmp_path, monkeypatch):
    """Each pass's engine step is the run's next step: the audit run's steps are the run's first steps, from its first,
    and the run's steps after the audit run are unaudited and in no map line."""
    rec, r, cs, steps = lockstep(tmp_path, monkeypatch, 40, windows=3)
    assert r["accepted"] and rec["accepted"] and rec["error"] is None and steps == list(range(40))
    mm, run = rec["memory_map"], rec["run"]
    assert mm["first_step"] == 0 and mm["steps"] == run["audited"] and mm["agrees"] and mm["rekeyed"] > 0
    assert 0 < run["audited"] < 40 and run["audited"] + run["unaudited"] == 40
    head, lines = memory_map.parse((tmp_path / "memory_map.jsonl").read_text().splitlines(keepends=True))
    # the last audited step may run past the close: its pass is dropped and its append never encoded, so its line,
    # with nothing freed or written, is not written
    last = max(x for w in r["record"]["profile"]["windows"] for x in w["steps"])
    assert last <= max(st.step for st in lines) <= run["audited"] - 1 and cs.store.memory_map is None
    assert mm["deferral"]["waiting"] == mm["unrecorded"]
    assert json.loads((tmp_path / "remote_session.json").read_text())["run"] == run


@pytest.mark.slow
def test_the_commits_session_arms_its_first_window_once_its_runs_first_step_has_run(tmp_path, monkeypatch):
    """The run's work between `begin` and its first step, and that step, pass 0's, are in no window: the audit run is
    registered by then, and arms its first window (the verifier's first coin follows `armed`) once the step has run,
    so the window's clock starts at pass 0's appends and read."""
    sent, at = [], {}
    send = wire.Conn.send
    monkeypatch.setattr(wire.Conn, "send", lambda self, head, *a, **kw: sent.append(head["type"]) or send(self, head, *a, **kw))
    rec, r, _, _ = lockstep(tmp_path, monkeypatch, 40, windows=3, setup=lambda: at.setdefault("setup", list(sent)),
                            each=lambda k: at.setdefault(k, list(sent)))
    assert r["accepted"] and rec["accepted"]
    assert "ready" in at["setup"] and not {"armed", "coin"} & {*at["setup"], *at[0]}
    assert "root" in at[1] and sent.index("armed") < sent.index("coin") < sent.index("append") < sent.index("root")
    steps = [x for w in r["record"]["profile"]["windows"] for x in w["steps"]]
    assert steps[0] == rec["memory_map"]["first_step"] == 0


@pytest.mark.slow
def test_a_paced_commits_session_runs_a_step_a_coin_and_tags_each_pass_with_its_step(tmp_path, monkeypatch):
    """At ``--paced 1:…`` each window is one pass under a coin of its own, its engine step inside it (no pass moves into
    the next window); the audit run's record tags each pass with the step the verifier's record has for it, a step of
    the map; the calibration's pass times carry ``step_ms``, so the allowance covers the step.  The Commit's store is
    the weights and filler (the KV outside it), so its steps write no registered chunk, every pass reads all N, and the
    memory line is declared in its time-held form as well as its snapshot."""
    pace = verifier.pace_of("1:1000:4096:2", 1000 / 3)
    assert pace["groups"] == 3
    rec, r, cs, _ = lockstep(tmp_path, monkeypatch, 40, pace=pace, step_ms=2.5, served=True)
    assert r["accepted"] and rec["accepted"] and r["record"]["public"]["pace"] == {"k": 1, "window_ns": 10**9}
    wins = r["record"]["profile"]["windows"]
    assert len(wins) == 9 and all(len(w["steps"]) == 1 for w in wins) and rec["prover"]["discarded"] == 0
    mm = rec["memory_map"]
    assert mm["pass_steps"] == [[w["window"], 0, w["steps"][0]] for w in wins] and mm["pass_steps_mapped"]
    assert mm["agrees"] and mm["recorded"] == 0 and mm["deferral"]["appends"] == 0
    assert list(memory_map.parse((tmp_path / "memory_map.jsonl").read_text().splitlines(keepends=True))[1]) == []
    assert {p.length for _, p in cs.backend.plans} == {4}
    assert rec["step_ms"] == 2.5 and cs.backend.calibrate()["step_ns"] == 2_500_000
    memory = r["ledger"]["declared"]["memory"]
    assert 0 <= memory["snapshot"]["epsilon"] <= 1
    assert "refused" not in memory["time-held"] and 0 <= memory["time-held"]["epsilon"] <= 1


def test_a_run_that_ends_before_its_session_fails_the_session(tmp_path, monkeypatch):
    rec, r, _, steps = lockstep(tmp_path, monkeypatch, 3)
    assert not rec["accepted"] and "the run ended after 3 steps" in rec["error"] and steps == [0, 1, 2]
    assert rec["run"] == {"steps": 3, "audited": 3, "unaudited": 0}
    assert isinstance(r, Exception) or not r["accepted"]


def test_a_commits_session_that_fails_is_recorded_rejected_and_its_run_goes_on_unaudited(monkeypatch, tmp_path):
    """The Commit writes its verdict whatever the audit run does: an audit run that raises is rejected, its error recorded,
    and the run's steps run unaudited."""
    def fails(backend, *a, **kw):
        raise ConnectionRefusedError("no verifier")

    monkeypatch.setattr(remote, "prove", fails)
    cs = remote.CommitSession("127.0.0.1:7781", out=tmp_path)
    cs.store, cs.binder = store(), Stats()
    lock, steps = cs.begin(), []
    for k in range(4):
        lock.step(lambda: steps.append(k))
    rec = cs.end()
    assert rec["accepted"] is False and rec["error"] == "ConnectionRefusedError: no verifier" and rec["memory_map"] is None
    assert steps == [0, 1, 2, 3] and rec["run"] == {"steps": 4, "audited": 0, "unaudited": 4}
    assert (tmp_path / "remote_session.json").exists()
    with pytest.raises(RuntimeError, match="one session"):
        cs.begin()


@pytest.mark.slow
def test_a_store_session_whose_prover_seeds_without_the_appends_head_is_rejected():
    r, p, _, backend = run("serve", backend=HeadlessBackend)
    assert not r["accepted"] and not p["accepted"] and all(h is None for h in backend.heads)
    assert r["failure"]["failure"] == "receipt-path"


@pytest.mark.slow
@pytest.mark.parametrize("backend", [ZeroCoinBackend, StaleCoinBackend])
def test_a_store_session_whose_prover_encodes_its_appends_under_another_coin_is_rejected(backend):
    r, p, _, b = run("serve", backend=backend)
    assert not r["accepted"] and not p["accepted"] and len(b.coins) > 1
    assert r["failure"]["failure"] == "codec"


def test_a_store_session_whose_prover_re_sends_an_append_is_rejected_with_coin(monkeypatch):
    sent = []

    def resend(head: dict) -> list[dict]:
        if head["type"] != "append" or sent:
            return [head]
        sent.append(head)
        return [head, head]

    tampering(monkeypatch, resend)
    r, p, _, _ = run("serve")
    assert sent and not r["accepted"] and not p["accepted"] and r["failure"]["failure"] == "coin"


@pytest.mark.slow
def test_a_store_session_whose_prover_appends_after_the_last_close_is_rejected_with_coin(monkeypatch):
    """An append sent after the verifier's last `close`, before the prover's `closed`: no root follows its coin."""

    def late(head: dict) -> list[dict]:
        if head["type"] != "closed":
            return [head]
        return [{"type": "append", "root": os.urandom(64).hex(), "count": 1}, head]

    tampering(monkeypatch, late)
    r, p, _, _ = run("serve")
    assert not r["accepted"] and not p["accepted"] and r["failure"]["failure"] == "coin"
    assert r["failure"]["t"] == 1


@pytest.mark.slow
def test_a_store_session_whose_prover_withholds_an_appended_block_is_rejected():
    r, p, _, _ = run("servedrop:1")
    assert not r["accepted"] and not p["accepted"]
    assert r["failure"]["failure"] == "receipt" and p["failure_on_dropped"] is True and p["dropped"] == [18]


def test_a_verifier_address_is_host_port_and_a_store_serves_one_session():
    assert remote.address("10.0.0.7:7781") == ("10.0.0.7", 7781)
    for bad in ("7781", "host:", ":7781", "host:0", "host:65536", "host:x"):
        with pytest.raises(ValueError, match="HOST:PORT"):
            remote.address(bad)
    s = store()
    b = remote.StoreBackend(s, engine(s))
    reg = audit.register(bytes(32), b"n", b.blocks, b.payload_root(b"n"), P512, nonce=bytes(32))
    b.encode(reg)
    b.begin_session()
    with pytest.raises(ValueError, match="at least 2 windows"):
        b.serving(reg, 1)
    b.serving(reg, 2)
    with pytest.raises(RuntimeError, match="one session"):
        b.begin_session()


def test_a_second_registration_of_a_live_node_is_refused_with_the_verifiers_reason():
    """One live registration a node (`audit.Registry`): a hello for a node whose registration the verifier holds live
    is answered `bye`, and `prove` raises `remote.Refused` with the verifier's reason."""
    registry, s = audit.Registry(), store()
    b = remote.StoreBackend(s, engine(s))
    registry.register(os.urandom(32), b"test-node", b.blocks, b.payload_root(b"test-node"), P512)
    lsock = wire.listen("127.0.0.1", 0)
    address = f"127.0.0.1:{lsock.getsockname()[1]}"
    raised = []

    def prover() -> None:
        try:
            remote.prove(b, address, node=b"test-node", codec=P512, log=lambda m: None)
        except remote.Refused as e:
            raised.append(e)

    th = threading.Thread(target=prover, daemon=True)
    th.start()
    sock, _ = lsock.accept()
    lsock.close()
    with verifying(sock, PassClock(), windows=1, window_ns=5, samples=64, registry=registry) as v:
        with pytest.raises(audit.RegistryError, match="live registration") as refused:
            v.register()
        th.join()
    (e,) = raised
    assert e.why == f"refused: {refused.value}" and address in str(e) and e.why in str(e)
    assert registry.refused == {b"test-node": 1} and registry.drawn(b"test-node") == 1


def test_a_serving_store_defers_residency_so_a_lifetime_of_one_window_is_refused():
    """The store's appended chunks are resident once their encodes land, and its map re-keys a window ahead, so at
    L = 1 every line would re-key every live chunk: its hello says `deferred`, and a verifier of lifetime 1 answers it
    `bye` before sampling a registration."""
    registry, s = audit.Registry(), store()
    b = remote.StoreBackend(s, engine(s))
    lsock = wire.listen("127.0.0.1", 0)
    address = f"127.0.0.1:{lsock.getsockname()[1]}"
    raised = []

    def prover() -> None:
        try:
            remote.prove(b, address, node=b"test-node", codec=P512, log=lambda m: None)
        except remote.Refused as e:
            raised.append(e)

    th = threading.Thread(target=prover, daemon=True)
    th.start()
    sock, _ = lsock.accept()
    lsock.close()
    with verifying(sock, PassClock(), windows=1, window_ns=5, samples=64, registry=registry, lifetime=1) as v:
        with pytest.raises(ValueError, match="at least 2 windows"):
            v.register()
        th.join()
    (e,) = raised
    assert "at least 2 windows" in e.why and registry.drawn(b"test-node") == 0


FAILING = '''
import os
from pathlib import Path

from tests.protocols.remote_bandwidth import test_remote_session as T


def deliberate(*a, **kw):
    raise AssertionError("deliberate")


def test_a_session_that_fails_on_the_tests_thread(monkeypatch):
    monkeypatch.setattr(T.verifier.Verifier, "session", deliberate)
    T.run("serve")


def test_a_registration_that_fails_on_the_lockstep_verifiers_thread(tmp_path, monkeypatch):
    monkeypatch.setattr(T.verifier.Verifier, "register", deliberate)
    _, r, _, _ = T.lockstep(tmp_path, monkeypatch, 3)
    assert not isinstance(r, Exception), r


def test_a_run_that_fails_in_its_first_step(tmp_path, monkeypatch):
    T.lockstep(tmp_path, monkeypatch, 3, each=deliberate)


def test_no_verifier_process_is_left():
    comm = T.verifier.lv.EXE[:15]
    left = []
    for d in Path("/proc").iterdir():
        try:
            stat = (d / "stat").read_text() if d.name.isdigit() else ""
        except OSError:
            continue
        name, rest = stat.split("(", 1)[-1].rsplit(")", 1) if stat else ("", "")
        if name == comm and int(rest.split()[1]) == os.getpid():
            left.append(int(d.name))
    assert left == []
'''


@pytest.mark.slow
def test_a_failing_test_leaves_no_verifier_process(tmp_path):
    """A pytest whose tests fail with the Lean verifier's process live (the verifier's audit run raising on the test's
    thread, its registration raising on `lockstep`'s verifier thread, which hung the suite while the prover waited on
    the socket, and the run's first step raising) finishes, and leaves no `remote-audit-verifier` child, running or
    unreaped, to the test after them."""
    (tmp_path / "test_failing.py").write_text(FAILING)
    r = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "no:xdist",
                        f"--junitxml={tmp_path / 'report.xml'}", str(tmp_path / "test_failing.py")],
                       cwd=VLLM, capture_output=True, text=True)
    out = r.stdout + r.stderr
    assert r.returncode == 1, out
    cases = {c.get("name"): [x.get("message", "") for x in c if x.tag == "failure"]
             for c in ElementTree.parse(tmp_path / "report.xml").iter("testcase")}
    failed = ["test_a_session_that_fails_on_the_tests_thread",
              "test_a_registration_that_fails_on_the_lockstep_verifiers_thread",
              "test_a_run_that_fails_in_its_first_step"]
    assert set(cases) == {*failed, "test_no_verifier_process_is_left"}, out
    assert all(len(cases[n]) == 1 and "deliberate" in cases[n][0] for n in failed), cases
    assert cases["test_no_verifier_process_is_left"] == [], out
