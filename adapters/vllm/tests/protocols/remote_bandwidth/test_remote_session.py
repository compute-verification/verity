"""PoUB, run over the node's `NodeMemory` end to end on the CPU: `remote.prove` (the prover's side, the
Commit's hook) connects over localhost TCP to PoUB's verifier (`verity/protocols/accounting/poub/verifier/session`),
registers the store, records its memory map against the registration, and runs each pass after an engine step of a
serving-like trace (pages written over two steps, held, freed and reused, and one long-lived page re-keyed under L = 2
windows).  The honest audit run is
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
import select
import socket
import subprocess
import sys
import threading
import traceback
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
import pytest
from verity.protocols.accounting.poub.verifier import audit, memory_map
from verity.protocols.accounting.poub.verifier import codec as rcodec
from verity.protocols.accounting.poub.verifier import receipts as rb
from verity.protocols.accounting.poub.verifier.session import verifier, wire

from verity_vllm.errors import Refused
from verity_vllm.protocols.remote_bandwidth import binder, remote
from verity_vllm.protocols.remote_bandwidth import memory_map as store_map
from verity_vllm.protocols.remote_bandwidth.interface import Append, PageLayout
from verity_vllm.protocols.remote_bandwidth.registry import ArrayBacking, NodeMemory

from .test_memory_map import BLOCK, PAGES, SLOT_BYTES, SLOTS, kv_payload
from .test_registry import CODEC, RECEIPT

P512 = rcodec.Codec(rcodec.P512, 2, False)
VLLM = Path(__file__).resolve().parents[3]


class PassClock:
    """Time is the prover's passes: one tick as each ends.  A wait returns the inbox's next message, or None once the
    clock reaches its end; once the verifier has said the passes are done, time jumps to the wait's end."""

    def __init__(self) -> None:
        self.t, self.done = 0, False
        self.inbox: queue.Queue | None = None

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
        assert self.inbox is not None, "idle before any wait"
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


def store() -> NodeMemory:
    """A weights chunk, one KV layer of 4 pages (K0 K1 V0 V1 each, page p at store chunks 1 + 4p …), and a chunk of
    host filler: N = 18.  Each chunk a backend encodes is 128 square roots at p512/k2, so the store is small."""
    s = NodeMemory()
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
        self.clock = clock
        self.heads: list[bytes | None] = []
        self.coins: list[tuple[int, bytes]] = []
        self.plans: list[tuple[int | None, Any]] = []

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


def engine(s: NodeMemory, at: int = 0):
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


def answered(lsock: socket.socket, over: socket.socket) -> socket.socket | None:
    """The connection made to ``lsock``, or None once ``over`` reads its stream's end and none was made: the side that
    would connect is over, so none will come, and a bare ``accept()`` would wait for ever.  It waits on those two
    events and on no clock, and closes both sockets."""
    try:
        return lsock.accept()[0] if lsock in select.select([lsock, over], [], [])[0] else None
    finally:
        lsock.close()
        over.close()


class Prover(threading.Thread):
    """The prover's side on a thread: ``prove()``, which connects to the test's listener.  `accept` answers that
    connection, and `outcome` what ``prove`` returned.  Each fails the test with what ``prove`` raised, where a bare
    ``accept()`` would wait for ever on a prover that ended before it connected.  The thread raises it as well, as a
    bare thread would."""

    def __init__(self, prove: Callable[[], Any]) -> None:
        super().__init__(daemon=True)
        self.prove = prove
        self.returned: Any = None
        self.raised: str | None = None
        # `over` reads the end of its stream once the thread is over, whatever ended it (`answered`)
        self.over, self._running = socket.socketpair()

    def run(self) -> None:
        try:
            self.returned = self.prove()
        except BaseException:
            self.raised = traceback.format_exc()
            raise
        finally:
            self._running.close()

    def accept(self, lsock: socket.socket) -> socket.socket:
        """Starts the thread and answers its connection to ``lsock``, which is closed either way."""
        self.start()
        sock = answered(lsock, self.over)
        if sock is not None:
            return sock
        self.join()
        raise pytest.fail.Exception("the prover ended before it connected: " + (
            f"it returned {self.returned!r}" if self.raised is None else f"it raised\n{self.raised}"))

    def outcome(self) -> Any:
        """What ``prove`` returned, once the thread is over."""
        self.join()
        if self.raised is not None:
            pytest.fail(f"the prover raised\n{self.raised}")
        return self.returned


def run(plan: str, *, windows: int = 2, window_ns: int = 5, samples: int = 4096, lifetime: int = 2,
        backend=TickingBackend, at: int = 0, **kw):
    """One audit run of ``plan``, every receipt of a window sampled at the default ``samples``; the backend keeps the
    regions the verifier read from its hello (``backend.regions``)."""
    clock = PassClock()
    s = store()
    backend = backend(clock, s, engine(s, at), randbytes=random.Random(7).randbytes, **kw)
    lsock = wire.listen("127.0.0.1", 0)
    address = f"127.0.0.1:{lsock.getsockname()[1]}"
    th = Prover(lambda: remote.prove(backend, address, plan, node=b"test-node", codec=P512, log=lambda m: None))
    sock = th.accept(lsock)
    with verifying(sock, clock, windows=windows, window_ns=window_ns, samples=samples, lifetime=lifetime) as v:
        v.register()
        backend.regions = v.regions
        r = v.session(0, v.allowance(v.ping(4)), None)
        th.join()
    return r, th.outcome()[0], s, backend


def test_a_prover_that_ends_before_it_connects_fails_the_test_where_an_accept_would_wait(monkeypatch):
    """`Prover.accept`: a prover that raises before it connects fails the test with what it raised, and one that
    returns before it connects with what it returned.  The listener is closed either way, and the thread raised what
    ``prove`` raised."""
    hooked: list = []
    monkeypatch.setattr(threading, "excepthook", hooked.append)

    def lost() -> None:
        raise FileNotFoundError("no session files")

    lsock = wire.listen("127.0.0.1", 0)
    with pytest.raises(pytest.fail.Exception, match="^the prover ended before it connected: it raised\n") as failed:
        Prover(lost).accept(lsock)
    assert "in lost" in str(failed.value) and str(failed.value).endswith("FileNotFoundError: no session files\n")
    assert lsock.fileno() == -1
    assert [(a.exc_type, str(a.exc_value)) for a in hooked] == [(FileNotFoundError, "no session files")]

    lsock = wire.listen("127.0.0.1", 0)
    with pytest.raises(pytest.fail.Exception, match=r"^the prover ended before it connected: it returned \['nothing'\]$"):
        Prover(lambda: ["nothing"]).accept(lsock)
    assert lsock.fileno() == -1 and len(hooked) == 1


def test_a_prover_that_connects_is_answered_while_it_is_at_work_and_its_outcome_is_what_it_returned(monkeypatch):
    """`Prover.accept` answers the prover's connection while the prover is still at work, with the listener closed,
    and `outcome` what the prover returned, once it is over."""
    hooked: list = []
    monkeypatch.setattr(threading, "excepthook", hooked.append)
    lsock = wire.listen("127.0.0.1", 0)
    port = lsock.getsockname()[1]

    def reads() -> bytes:
        with socket.create_connection(("127.0.0.1", port)) as conn:
            return conn.recv(1)  # held here until the test's side sends

    th = Prover(reads)
    with th.accept(lsock) as sock:
        assert lsock.fileno() == -1 and th.is_alive()
        sock.sendall(b"k")
        assert th.outcome() == b"k"
    assert not hooked


def test_a_prover_that_raises_after_it_connects_fails_the_test_at_its_outcome(monkeypatch):
    """`Prover.outcome`: a prover that connects and then raises is answered by `accept` as any other, and fails the
    test at its outcome with what it raised.  The thread raised it as well."""
    hooked: list = []
    monkeypatch.setattr(threading, "excepthook", hooked.append)
    lsock = wire.listen("127.0.0.1", 0)
    port = lsock.getsockname()[1]

    def drops() -> None:
        socket.create_connection(("127.0.0.1", port)).close()
        raise ValueError("mid-session")

    th = Prover(drops)
    with th.accept(lsock) as sock:
        assert sock.recv(1) == b""
        with pytest.raises(pytest.fail.Exception, match="^the prover raised\n") as failed:
            th.outcome()
    assert "in drops" in str(failed.value) and str(failed.value).endswith("ValueError: mid-session\n")
    assert [(a.exc_type, str(a.exc_value)) for a in hooked] == [(ValueError, "mid-session")]


def test_an_audit_run_whose_prover_cannot_load_the_sessions_files_fails_with_that(monkeypatch):
    """`run`: `remote.prove` loads the session's prover before it connects, so a tree without that file ends the
    prover's thread there.  The run fails with what the load raised, where its accept would wait for ever."""
    load = remote.session_module

    def missing(name: str, *directory: Any) -> Any:
        if name == "prover":
            raise FileNotFoundError("no prover.py")
        return load(name, *directory)

    monkeypatch.setattr(remote, "session_module", missing)
    monkeypatch.setattr(threading, "excepthook", lambda raised: None)
    with pytest.raises(pytest.fail.Exception, match="^the prover ended before it connected: it raised\n") as failed:
        run("serve")
    assert str(failed.value).endswith("FileNotFoundError: no prover.py\n")


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


def served(coins: tuple[int, ...]) -> tuple[NodeMemory, remote.StoreBackend]:
    """A store whose trace ran `engine`'s first ``len(coins)`` steps with no session: step 0 before the first window, as
    `Serving.ready` runs it, the rest in window 0.  Once step k has run, the first ``coins[k]`` appends, in the order
    made, are encoded, each under a coin of its own: the line open then writes an append's blocks, and none before."""
    s = store()
    b = remote.StoreBackend(s, engine(s), randbytes=random.Random(7).randbytes)
    reg = audit.register(bytes(32), b"n", b.blocks, b.payload_root(b"n"), P512, nonce=bytes(32))
    b.encode(reg)
    trace = b.serving(reg, 2)
    for k, landed in enumerate(coins):
        if k == 1:
            b.begin_window(bytes(32), None)
        assert trace.advance(0 if k else None) == k
        for a in trace.appends[len(trace.landed):landed]:
            b.encode_append(a.first, bytes([a.first]) * 32)
    return s, b


def test_a_map_agrees_with_its_trace_and_its_store_up_to_the_appends_not_yet_encoded():
    """`memory_map.agreement` after a trace with no session, whose step 0's append was encoded at once (its line writes
    pages 0 and 3's K0 V0, the store's blocks 1, 3, 13 and 15), step 1's only once step 2 had run (the line open at
    the end writes it, which the agreement closes), and step 2's never: the map, replayed, holds the trace's live
    blocks, its writes are the appends' digests in order up to the one not encoded (`unrecorded`), and each slot it
    holds at an appended index holds that index's bytes in the store."""
    s, b = served((1, 1, 2))
    first, late, waiting = b.trace.appends
    assert store_map.agreement(b) == {
        "live_agrees": True, "appends_agree": True, "recorded": first.count + late.count, "unrecorded": waiting.count,
        "held_appended": first.count + late.count, "held_wrong": [], "agrees": True}
    assert {x: s.memory_map.replay.holds(x) for x in (1, 3, 13, 15)} == dict(
        zip((1, 3, 13, 15), range(first.first, first.first + first.count)))
    assert b.trace.deferral()["encodes"] == [[0, 0, first.first, first.count], [1, 2, late.first, late.count],
                                             [2, None, waiting.first, waiting.count]]


@pytest.mark.parametrize("part", ["live", "appends", "store"])
def test_a_map_that_disagrees_with_its_trace_or_its_store_says_in_which_part(part):
    """Each part of `memory_map.agreement` decides alone, after steps 0 and 1 with step 0's append encoded: a trace
    whose live blocks are not the map's (`live_agrees`), one whose appended digests are not the map's writes
    (`appends_agree`), and a store whose bytes at a slot the map holds at an appended index are not that index's
    (`held_wrong`, the slot)."""
    s, b = served((1, 1))
    landed, waiting = b.trace.appends
    if part == "live":
        b.trace.live[0] = False
    elif part == "appends":
        b.trace.digests[0] = bytes(len(b.trace.digests[0]))
    else:
        s.backing("kv/0").array[(1 - s.region("kv/0").first) * BLOCK] ^= 1
    assert store_map.agreement(b) == {
        "live_agrees": part != "live", "appends_agree": part != "appends", "recorded": landed.count,
        "unrecorded": waiting.count, "held_appended": landed.count, "held_wrong": [1] if part == "store" else [],
        "agrees": False}


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
        self.shift = shift
        self.key: rcodec.Key | None = None
        self.calls: list[tuple[int, int, bool]] = []

    def set_key(self, key: rcodec.Key) -> None:
        self.key = key

    def encode(self, payloads, first: int, offsets=None) -> list[bytes]:
        self.calls.append((first, len(payloads), offsets is not None))
        k = self.key
        assert k is not None, "encode before set_key"
        f = k.codec.field
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


def test_a_device_encoded_store_takes_the_registered_payloads_and_each_appends_coin_once():
    """`remote.DeviceEncoded` refuses payloads that are not the registration's, by their count or by their root, before
    the device encodes a block.  An append is encoded once, by the coin for its first block: the store then holds the
    codec's bound encoding of each of its blocks, and the gate counts the append and its first block."""
    s = store()
    b = remote.StoreBackend(s, engine(s), device=(d := Device()))
    reg = audit.register(bytes(32), b"n", b.blocks, b.payload_root(b"n"), P512, nonce=bytes(32))
    with pytest.raises(ValueError, match="the registration is of 18 blocks"):
        remote.DeviceEncoded(reg, b.payloads[:-1], b.salts[:-1], d)
    with pytest.raises(ValueError, match="these payloads are not the registered ones"):
        remote.DeviceEncoded(reg, [*b.payloads[:-1], b"\x01" * rcodec.BLOCK_BYTES], b.salts, d)
    assert d.calls == []
    b.encode(reg)
    payloads, coin = [b"\x01" * rcodec.BLOCK_BYTES, b"\x02" * rcodec.BLOCK_BYTES], bytes(range(32))
    a = b.append(payloads, [bytes(192), bytes(192)])
    with pytest.raises(ValueError, match="no append from block 19 awaits its coin"):
        b.encode_append(a.first + 1, coin)
    assert b.audit.blocks[18:] == [None, None]
    b.encode_append(a.first, coin)
    bound = [rcodec.encode(m, 18 + u, reg.key, bound=True, coin=coin) for u, m in enumerate(payloads)]
    assert b.audit.blocks[18:] == bound
    assert b.gates["device_encode"] == {"registered": 18, "appends": 1, "blocks_checked": 3}
    assert d.calls == [(0, 18, False), (18, 2, True)]
    with pytest.raises(ValueError, match="no append from block 18 awaits its coin"):
        b.encode_append(a.first, coin)


def test_a_store_that_changed_between_its_hello_and_its_registration_is_not_encoded():
    """`StoreBackend.encode` holds the store to the payloads its hello committed: a byte changed since, in the first
    block or the last, refuses the registration, and the store as it was at the hello is encoded."""
    s = store()
    b = remote.StoreBackend(s, engine(s))
    reg = audit.register(bytes(32), b"n", b.blocks, b.payload_root(b"n"), P512, nonce=bytes(32))
    for region, at in (("w", 0), ("dram", -1)):
        held = s.backing(region).array
        held[at] ^= 1
        with pytest.raises(RuntimeError, match="the store changed between its hello and its registration"):
            b.encode(reg)
        held[at] ^= 1
    assert b.encode(reg) == {"encode_bytes_per_s": 1.0} and len(b.audit.blocks) == 18


def test_a_paced_backends_calibration_adds_its_step_to_each_pass_time():
    """`StoreBackend.with_step`: with no step to wait on a calibration is returned as it is; with one, each pass time
    (the maximum, the median and each quantile of ``pass_ns``) grows by ``step_ns``, which the result names, every
    other entry stays, and the calibration handed in is not changed."""
    s = store()
    b = remote.StoreBackend(s, engine(s), read_bytes_per_s=5e9)
    cal = {"pass_ns_max": 7, "pass_ns_median": 5, "pass_ns": {"p50": 5, "max": 7}, "gap_ns_max": 3}
    assert b.with_step(cal) is cal
    assert b.calibrate() == {"pass_ns_max": 0, "gap_ns_max": 0, "read_bytes_per_s": 5e9}
    b.step_ns = 40
    assert b.with_step(cal) == {"pass_ns_max": 47, "pass_ns_median": 45, "pass_ns": {"p50": 45, "max": 47},
                                "gap_ns_max": 3, "step_ns": 40}
    assert cal == {"pass_ns_max": 7, "pass_ns_median": 5, "pass_ns": {"p50": 5, "max": 7}, "gap_ns_max": 3}
    assert b.with_step({"gap_ns_max": 3}) == {"gap_ns_max": 3, "step_ns": 40}
    assert b.calibrate() == {"pass_ns_max": 40, "gap_ns_max": 0, "read_bytes_per_s": 5e9, "step_ns": 40}


def test_a_trace_runs_a_step_only_in_its_passs_window_and_none_before_the_first_once_one_began():
    """`Serving.advance` refuses a pass's step while the store's open window is not the pass's, or is closed, and a
    step before the first window once a window has begun, open or closed.  No refused step runs the engine."""
    s, b = served(())
    trace = b.trace
    asked: list[int] = []
    b.engine = asked.append
    with pytest.raises(RuntimeError, match="the pass is in window 0, the store's open window is -1"):
        trace.advance(0)
    b.begin_window(bytes(32), None)
    with pytest.raises(RuntimeError, match="the pass is in window 1, the store's open window is 0"):
        trace.advance(1)
    with pytest.raises(RuntimeError, match="a step before the first window, and window 0 has begun"):
        trace.advance(None)
    s.end_window()
    with pytest.raises(RuntimeError, match="the pass is in window 0, the store's open window is 0"):
        trace.advance(0)
    with pytest.raises(RuntimeError, match="a step before the first window, and window 0 has begun"):
        trace.advance(None)
    assert asked == [] and (trace.step, trace.steps, trace.map.last) == (-1, 0, -1)


def test_a_traces_engine_step_opens_one_memory_map_line_and_its_first_takes_the_engines_number():
    """An engine step that opens no memory-map line is refused, the session's first or a later one.  The first step is
    numbered as the engine numbers it (here from 40).  After it, a step that opens two lines for one ask is refused
    too, and a refused step is not counted."""
    s, b = served(())
    trace, step = b.trace, engine(s, 40)
    b.engine = lambda k: None
    with pytest.raises(RuntimeError, match=r"step after -1 opens one memory-map line \(kv_append\(0\)\)"):
        trace.advance(None)
    b.engine = step
    assert trace.advance(None) == 40 and (trace.first, trace.step, trace.steps) == (40, 40, 1)
    b.begin_window(bytes(32), None)
    for run in (lambda k: None, lambda k: (step(k), step(k))):
        b.engine = run
        with pytest.raises(RuntimeError, match=r"step after 40 opens one memory-map line \(kv_append\(41\)\)"):
            trace.advance(0)
    assert (trace.first, trace.step, trace.steps) == (40, 40, 1)


def test_a_trace_takes_an_append_and_a_closed_lines_writes_only_as_its_memory_map_gave_them():
    """`Serving._apply` stops a step whose append is not the memory map's, of other bytes than the map digested or at
    another index, and one that closes a line whose writes are not the digests appended at its indices."""
    for other in ("bytes", "index"):
        s, b = served(())
        honest = b.append
        if other == "bytes":
            b.append = lambda payloads, salts, honest=honest: honest([bytes(len(m)) for m in payloads], salts)
        else:
            b.append = lambda payloads, salts, honest=honest: (honest(payloads, salts), honest(payloads, salts))[1]
        with pytest.raises(AssertionError, match="the store's append is not its memory map's"):
            b.trace.advance(None)
        assert b.trace.appends == [] and b.trace.steps == 0, other
    s, b = served((1,))
    b.trace.digests[0] = bytes(len(b.trace.digests[0]))
    b.begin_window(bytes(32), None)
    with pytest.raises(AssertionError, match="step 0's writes from 18 are not the blocks appended there"):
        b.trace.advance(0)
    assert b.trace.steps == 1


def test_a_backends_next_window_ends_the_stores_open_one_and_keeps_the_root_before_it():
    """`StoreBackend.begin_window` opens the store's next window under the challenge, ending the one still open, and
    keeps the previous root it was last given when a window brings none."""
    s, b = served(())
    b.begin_window(bytes(32), None)
    assert (s.window, s.window_open, b.challenge, b.previous) == (0, True, bytes(32), None)
    b.begin_window(bytes(range(32)), b"\x01" * 64)
    assert (s.window, s.window_open, b.challenge, b.previous) == (1, True, bytes(range(32)), b"\x01" * 64)
    b.begin_window(bytes(32), None)
    assert (s.window, s.window_open, b.challenge, b.previous) == (2, True, bytes(32), b"\x01" * 64)


def test_a_backends_openings_are_each_positions_block_its_leaf_and_its_payloads_commitment():
    """`StoreBackend.open_window` answers each pass's positions in order, over a plan that skips blocks 2 and 3: the
    block the plan reads at the position, as the audit store holds it; its leaf's receipts and salt, which verify
    against the pass's root at the position and hold the block's receipt; and the payload's salt, path and digest, a
    registered block's digest empty and an appended block's its append's."""
    _, b = served((1,))
    (a,), reg, seed = b.trace.appends, b.registration, bytes(range(64))
    plan = rb.Plan(((0, 2), (4, 18)), 22)
    root, handle = b.run_pass(seed, plan)
    first, second = b.open_window([(handle, [0, 2, 15, 16]), (handle, [19])])
    assert (len(first), len(second)) == (4, 1) and second == b.open(handle, [19])
    receipt = rb.receipter(reg.receipt, seed, reg.rows)
    for p, j, opening in zip([0, 2, 15, 16, 19], [0, 4, 17, 18, 21], first + second):
        block, leaf, salt, path, *committed = opening
        assert block == b.audit.blocks[j] and tuple(committed) == b.audit.committed(j), p
        assert rb.verify_leaf(seed, plan, root, p, leaf, salt, path), p
        assert rb.leaf_receipt(leaf, p, 20) == receipt(j, block), p
        assert committed[2] == (b"" if j < 18 else a.digests[j - 18]), p


def test_a_session_module_that_fails_to_load_is_not_kept(tmp_path):
    """`session_module` loads a module once from its file and keeps it in ``sys.modules``.  One whose file is missing,
    or whose body raises, is not kept, so the next call loads the file again; once kept, it is returned whatever the
    directory asked."""
    name = "session_module_under_test"
    assert name not in sys.modules
    try:
        with pytest.raises(FileNotFoundError):
            remote.session_module(name, tmp_path)
        assert name not in sys.modules
        (tmp_path / f"{name}.py").write_text("VALUE = 1\nraise RuntimeError('half loaded')\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="half loaded"):
            remote.session_module(name, tmp_path)
        assert name not in sys.modules
        (tmp_path / f"{name}.py").write_text("VALUE = 2\n", encoding="utf-8")
        module = remote.session_module(name, tmp_path)
        assert module.VALUE == 2 and sys.modules[name] is module
        assert remote.session_module(name, tmp_path / "elsewhere") is module
    finally:
        sys.modules.pop(name, None)


def test_a_pass_interval_is_closed_with_its_phases_its_wall_and_what_they_leave(monkeypatch):
    """`Phases.close` ends the interval open since the close before it: each phase's time in it, summed over its runs,
    ``wall`` from that close to this one, and ``other``, the wall no phase took.  The first close ends no interval and
    drops what ran before it.  The clock is the test's: each reading is the next of its list, and the first close
    reads 0, a reading like any other."""

    class Clock:
        def __init__(self, *readings: int) -> None:
            self.readings = iter(readings)

        def perf_counter_ns(self) -> int:
            return next(self.readings)

    monkeypatch.setattr(remote, "time", Clock(-9, -2, 0, 10, 30, 40, 45, 50, 60, 100, 160))
    phases = remote.Phases()
    with phases.time("engine"):
        pass
    phases.close()
    assert phases.closed == [] and phases.since == 0
    for name in ("engine", "plan", "engine"):
        with phases.time(name):
            pass
    phases.close()
    phases.close()
    assert phases.closed == [{"engine": 30, "plan": 5, "wall": 100, "other": 65}, {"wall": 60, "other": 60}]


def test_a_run_whose_step_raises_fails_the_sessions_ask_with_its_reason():
    """`Lockstep.step`: the run's engine step that raises answers the session's ask with what it raised, so the session
    fails there, and raises in the run as it was; the step is counted neither audited nor unaudited."""
    lock, failed = remote.Lockstep(), []

    def session() -> None:
        try:
            lock(0)
        except RuntimeError as e:
            failed.append(str(e))

    def step() -> None:
        raise ValueError("no KV to write")

    th = threading.Thread(target=session, daemon=True)
    th.start()
    with pytest.raises(ValueError, match="no KV to write"):
        lock.step(step)
    th.join()
    assert failed == ["the run's engine step raised ValueError: no KV to write"]
    assert (lock.audited, lock.unaudited) == (0, 0)


def test_a_registration_cut_off_is_a_refusal_only_when_the_verifier_refused(monkeypatch):
    """`prove` raises `Refused`, with the verifier's address and reason, for a registration the wire closed with
    ``refused``; closed for any other reason it raises as the wire raised.  The connection is shut and closed either
    way."""
    done: list[str] = []

    class Sock:
        def shutdown(self, how: int) -> None:
            done.append("shutdown")

        def close(self) -> None:
            done.append("close")

    class Wire:
        class Closed(Exception):
            pass

        @staticmethod
        def connect(host: str, port: int) -> Any:
            return type("Conn", (), {"sock": Sock()})()

    def closing(why: str) -> None:
        """The prover's module is one whose registration the wire closes with ``why``."""

        class Side:
            def __init__(self, conn, backend, node, plan, log) -> None:
                self.plan = plan

            def register(self, codec) -> None:
                raise Wire.Closed(why)

        class Prover:
            wire, ProverSide = Wire, Side
            parse_plan = staticmethod(lambda plan: [(plan, None)])

        monkeypatch.setattr(remote, "session_module", lambda name: {"prover": Prover}[name])

    closing("the peer closed the connection")
    with pytest.raises(Wire.Closed, match="the peer closed the connection"):
        remote.prove(object(), "10.0.0.7:7781", log=lambda m: None)
    closing("refused: a live registration")
    with pytest.raises(Refused, match="at 10.0.0.7:7781 refused the store's registration") as refused:
        remote.prove(object(), "10.0.0.7:7781", log=lambda m: None)
    assert type(refused.value) is Refused and refused.value.reasons == (
        "the verifier at 10.0.0.7:7781 refused the store's registration: refused: a live registration",)
    assert str(refused.value) == refused.value.reasons[0]
    assert done == ["shutdown", "close"] * 2


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
    refused: list[tuple[dict[str, Any], str]] = [
        ({"plan": "serve,serve"}, "one mode"), ({"kv_pages": 0}, "a page"), ({"plan": "nope"}, "a session's mode"),
        ({"plan": "servedrop:2"}, "appends nothing"), ({"filler_gib": -1}, "nonnegative"), ({"node": "n" * 65}, "node id")]
    for kw, err in refused:
        with pytest.raises(Exception, match=err):
            remote.CommitSession("127.0.0.1:7781", **kw)
    assert 1 <= len(cs.node.encode()) <= audit.NODE_BYTES


def test_the_phases_summary_is_each_phases_quantiles_over_the_closed_intervals_in_ms():
    """`Phases.summary` over the closed intervals: a phase an interval didn't run counts 0 there, the quantile at q is
    the sorted list's entry at ⌊q·n⌋ (its last at 1.0, and at 0.99 too up to 100 intervals), the mean is over every
    interval, all in ms to 3 places, and the phases come sorted by name."""
    phases = remote.Phases()
    assert phases.summary() == {}
    phases.closed = [{"encode": 2_000_000, "wall": 10_000_000, "other": 8_000_000}, {"wall": 4_000_000, "other": 4_000_000},
                     {"encode": 1_500_000, "send": 500_000, "wall": 6_000_000, "other": 4_000_000}]
    assert list(phases.summary()) == ["encode", "other", "send", "wall"]
    assert phases.summary() == {"encode": {"p50": 1.5, "p99": 2.0, "max": 2.0, "mean": 1.167},
                                "other": {"p50": 4.0, "p99": 8.0, "max": 8.0, "mean": 5.333},
                                "send": {"p50": 0.0, "p99": 0.5, "max": 0.5, "mean": 0.167},
                                "wall": {"p50": 6.0, "p99": 10.0, "max": 10.0, "mean": 6.667}}
    phases.closed = [{"wall": 1_234_567}]
    assert phases.summary() == {"wall": {"p50": 1.235, "p99": 1.235, "max": 1.235, "mean": 1.235}}
    phases.closed = [{"wall": ms * 1_000_000} for ms in range(101, 0, -1)]
    assert phases.summary() == {"wall": {"p50": 51.0, "p99": 100.0, "max": 101.0, "mean": 51.0}}


def test_a_prover_that_fails_closes_its_connection_even_when_the_shutdown_fails(monkeypatch):
    """`prove` drops the connection of a prover that fails, so the verifier sees the session end there: it shuts the
    socket down both ways and closes it, closes it all the same when the shutdown raises ``OSError`` (a peer already
    gone), and raises the prover's own exception either way."""
    done: list[Any] = []

    class Sock:
        def __init__(self, gone: bool) -> None:
            self.gone = gone

        def shutdown(self, how: int) -> None:
            done.append(("shutdown", how))
            if self.gone:
                raise OSError("not connected")

        def close(self) -> None:
            done.append("close")

    class Wire:
        def __init__(self, gone: bool) -> None:
            self.gone = gone

        def connect(self, host: str, port: int) -> Any:
            done.append(("connect", host, port))
            return type("Conn", (), {"sock": Sock(self.gone)})()

    def prover_side(conn, backend, node, plan, log, **kw):
        raise ValueError(f"deliberate: {node!r} {plan!r} {kw!r}")

    for gone in (False, True):
        del done[:]
        wire_ = Wire(gone)
        side = type("Prover", (), {"wire": wire_, "ProverSide": staticmethod(prover_side),
                                   "parse_plan": staticmethod(lambda plan: [(plan, None)])})()
        monkeypatch.setattr(remote, "session_module", lambda name, side=side: {"prover": side}[name])
        with pytest.raises(ValueError, match=r"deliberate: b'n' \[\('serve', None\)\] \{'windows': 2\}"):
            remote.prove(object(), "10.0.0.7:7781", node=b"n", windows=2)
        assert done == [("connect", "10.0.0.7", 7781), ("shutdown", socket.SHUT_RDWR), "close"]


class Binding:
    """A stand-in for the store's `Binder` that records how the Commit's session makes and uses it."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.made = (args, kwargs)
        self.calls: list[Any] = []

    def install(self) -> Binding:
        self.calls.append("install")
        return self

    def register_filler(self, nbytes: int, device: Any) -> None:
        self.calls.append(("register_filler", nbytes, device))

    def stats(self) -> dict[str, Any]:
        self.calls.append("stats")
        return {"weights": 3}


def test_the_commits_session_installs_a_binder_of_its_store_before_the_build(monkeypatch):
    """`CommitSession.install` makes the store's binder with no passes of its own and the KV outside the store, over
    the session's store and its ``weight_blocks``, and keeps what its `install` returns; `before_build` sets the engine
    arguments, checks the target and installs, in that order, and installs nothing for a target it refuses."""
    monkeypatch.setattr(binder, "Binder", Binding)
    cs = remote.CommitSession("127.0.0.1:7781", kv_pages=8, weight_blocks=5)
    assert cs.binder is None
    cs.install()
    assert isinstance(cs.binder, Binding) and cs.binder.calls == ["install"]
    assert cs.binder.made == ((cs.store,), {"passes": False, "weight_blocks": 5, "kv": False})

    cs = remote.CommitSession("127.0.0.1:7781", kv_pages=None)
    ekw = {"engine_args": {"max_model_len": 128}, "target": type("T", (), {"async_scheduling": False})()}
    cs.before_build(ekw)
    assert ekw["engine_args"] == {"max_model_len": 128, "async_scheduling": False, "enable_prefix_caching": False}
    assert isinstance(cs.binder, Binding) and cs.binder.made == ((cs.store,), {"passes": False, "weight_blocks": None, "kv": False})

    cs = remote.CommitSession("127.0.0.1:7781")
    with pytest.raises(SystemExit, match="declare \"async_scheduling\": false"):
        cs.before_build({"engine_args": {}, "target": None})
    assert cs.binder is None


def built(core: Any = True, asynchronous: bool = False, prefix: bool = False, tp: int = 1, client: bool = True) -> Any:
    """A stand-in for a built engine as `CommitSession.check` reads it: the engine's client of its core (none without
    ``client``) with the core it holds in this process (``core``, None when it runs in another), and the scheduler's,
    the cache's and the parallel configuration's one field each."""
    def names(**kw: Any) -> Any:
        return type("N", (), kw)()

    config = names(scheduler_config=names(async_scheduling=asynchronous), cache_config=names(enable_prefix_caching=prefix),
                   parallel_config=names(tensor_parallel_size=tp))
    return names(llm_engine=names(vllm_config=config, **({"engine_core": names(engine_core=core)} if client else {})))


def test_the_commits_session_checks_the_built_engine_and_registers_its_filler():
    """`CommitSession.check` after the build: it refuses, with every reason in order, an engine whose core is in
    another process, that schedules asynchronously, caches prefixes or has a tensor parallel size other than 1, and a
    store that registered a KV pool; else it registers ``filler_gib`` GiB of filler on the runner's device, in whole
    bytes rounded down (none at 0), logs the binder's stats and returns them."""
    said: list[str] = []
    cs = remote.CommitSession("127.0.0.1:7781", filler_gib=0.3, log=said.append)
    cs.binder = Binding()
    runner = type("Runner", (), {"device": "cuda:0"})()
    assert cs.check(built(), runner) == {"weights": 3}
    assert cs.binder.calls == [("register_filler", 322_122_547, "cuda:0"), "stats"]
    assert said == ["[delta] remote session: the store binder on the engine, {'weights': 3}"]

    cs = remote.CommitSession("127.0.0.1:7781", log=said.append)
    cs.binder = Binding()
    assert cs.check(built(), runner) == {"weights": 3} and cs.binder.calls == ["stats"] and len(said) == 2

    head = "--remote-verifier: the engine is not one the session serves: "
    whys = [("the engine core in another process", built(core=None)), ("the engine core in another process", built(client=False)),
            ("async scheduling resolved on", built(asynchronous=True)), ("prefix caching on", built(prefix=True)),
            ("tensor parallel size 2", built(tp=2)), ("tensor parallel size 0", built(tp=0)),
            ("the engine core in another process; async scheduling resolved on; prefix caching on; tensor parallel size 4",
             built(core=None, asynchronous=True, prefix=True, tp=4))]
    for why, llm in whys:
        with pytest.raises(SystemExit) as e:
            cs.check(llm, runner)
        assert str(e.value) == head + why
    cs.store = store()
    with pytest.raises(SystemExit) as e:
        cs.check(built(), runner)
    assert str(e.value) == head + "a KV pool registered (served writes stay outside the registration)"
    assert cs.binder.calls == ["stats"] and len(said) == 2


def test_the_commits_session_names_the_step_that_comes_first():
    """`CommitSession.check` before `install`, and `end` before `begin`, each raise a `RuntimeError` naming the step
    that comes first; an engine the session doesn't serve is refused before `check` asks for its binder."""
    cs = remote.CommitSession("127.0.0.1:7781", log=lambda m: None)
    runner = type("Runner", (), {"device": "cuda:0"})()
    with pytest.raises(RuntimeError) as early:
        cs.check(built(), runner)
    assert str(early.value) == "the Commit's session checks its engine once its binder is installed (`install`)"
    with pytest.raises(SystemExit, match="^--remote-verifier: the engine is not one the session serves: prefix caching on$"):
        cs.check(built(prefix=True), runner)
    with pytest.raises(RuntimeError) as early:
        cs.end()
    assert str(early.value) == "the Commit's session ends once it has begun (`begin`)"
    assert cs.binder is None and cs.lockstep is None and cs.record is None


class Stats:
    def stats(self) -> dict:
        return {}


def served_store() -> NodeMemory:
    """The Commit's store: two weights chunks and two of filler, no KV pool (N = 4)."""
    s = NodeMemory()
    w = np.random.default_rng(1).integers(0, 256, 2 * BLOCK, np.uint8)
    s.register("w", "hbm", "weights", ArrayBacking(w), block_bytes=BLOCK, codec=CODEC, receipt=RECEIPT, source="w")
    s.register("filler", "hbm", "filler", ArrayBacking(np.zeros(2 * BLOCK, np.uint8)), block_bytes=BLOCK, codec=CODEC,
               receipt=RECEIPT, source="filler:zeros")
    return s


def served_engine(s: NodeMemory):
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
    over, ended = socket.socketpair()

    def verify() -> None:
        sock = answered(lsock, over)
        if sock is None:
            return
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
            lock.step(lambda k=k: run(k))
        rec = cs.end()
    except BaseException:
        # a run that fails ends its audit run, so the verifier's thread ends with it and reaps its process
        if cs._thread is not None and cs.record is None:
            cs.end()
        raise
    finally:
        # the session is over: a verifier's thread it never connected to ends here (`answered`)
        ended.close()
        th.join()
    if not seen:
        pytest.fail(f"the Commit's session ended before it connected to its verifier: {rec['error']}")
    return rec, seen[0], cs, steps


def test_a_commits_session_that_cannot_connect_fails_its_test_where_the_verifiers_thread_would_wait(tmp_path, monkeypatch):
    """`lockstep`: a session whose prover cannot connect records why and its run goes on unaudited, so no connection
    comes to the verifier's thread.  The helper ends that thread once the session is over, and fails the test with
    what the session recorded; a run that fails as well raises as it did.  Each would wait for ever to join a thread
    at ``accept()``."""
    def unreachable(host: str, port: int) -> None:
        raise ConnectionRefusedError("no verifier")

    def stops(k: int) -> None:
        raise ValueError(f"the run's step {k}")

    monkeypatch.setattr(remote.session_module("prover").wire, "connect", unreachable)
    with pytest.raises(pytest.fail.Exception) as failed:
        lockstep(tmp_path, monkeypatch, 3)
    assert str(failed.value) == ("the Commit's session ended before it connected to its verifier: "
                                 "ConnectionRefusedError: no verifier")
    assert json.loads((tmp_path / "remote_session.json").read_text())["run"] == {"steps": 3, "audited": 0, "unaudited": 3}
    with pytest.raises(ValueError, match="^the run's step 0$"):
        lockstep(tmp_path, monkeypatch, 3, each=stops)


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
    sent: list[str] = []
    at: dict[object, list[str]] = {}
    send = wire.Conn.send

    def logged(self, head, *a, **kw):
        sent.append(head["type"])
        return send(self, head, *a, **kw)

    def setup() -> None:
        at.setdefault("setup", list(sent))

    def each(k: int) -> None:
        at.setdefault(k, list(sent))

    monkeypatch.setattr(wire.Conn, "send", logged)
    rec, r, _, _ = lockstep(tmp_path, monkeypatch, 40, windows=3, setup=setup, each=each)
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
        lock.step(lambda k=k: steps.append(k))
    rec = cs.end()
    assert rec["accepted"] is False and rec["error"] == "ConnectionRefusedError: no verifier" and rec["memory_map"] is None
    assert steps == [0, 1, 2, 3] and rec["run"] == {"steps": 4, "audited": 0, "unaudited": 4}
    assert (tmp_path / "remote_session.json").exists()
    with pytest.raises(RuntimeError, match="one session"):
        cs.begin()


@pytest.mark.parametrize(("result", "steps"), [(None, []), ({"accepted": True}, []), ({"accepted": True, "pass_steps": None}, []),
                                               ({"accepted": True, "pass_steps": []}, []),
                                               ({"accepted": True, "pass_steps": [[0, 0, 5]]}, [[0, 0, 5]])])
def test_a_session_whose_result_leaves_its_pass_steps_out_records_none(monkeypatch, tmp_path, result, steps):
    """The audit run's record of a session whose trace began: its map's ``pass_steps`` are the prover's result's, and the
    empty list where the result leaves the field out, holds None there, or was never returned (the prover raised:
    ``result`` None here); with no step in the map, none of them is mapped."""
    def proves(backend, *a, **kw):
        reg = audit.register(bytes(32), b"n", backend.blocks, backend.payload_root(b"n"), P512, nonce=bytes(32))
        backend.encode(reg)
        backend.begin_session()
        backend.serving(reg, 2)
        if result is None:
            raise ConnectionResetError("the verifier went away")
        return [result]

    monkeypatch.setattr(remote, "prove", proves)
    cs = remote.CommitSession("127.0.0.1:7781", out=tmp_path, codec=P512, log=lambda m: None)
    cs.store, cs.binder = store(), Stats()
    cs.begin()
    rec = cs.end()
    mm = rec["memory_map"]
    assert mm["pass_steps"] == steps and mm["pass_steps_mapped"] is False and mm["first_step"] is None and mm["steps"] == 0
    assert rec["prover"] == ({} if result is None else result) and rec["run"] == {"steps": 0, "audited": 0, "unaudited": 0}
    assert rec["error"] == ("ConnectionResetError: the verifier went away" if result is None else None)
    assert rec["accepted"] is (result is not None) and mm.get("agrees") is (None if result is None else True)
    assert json.loads((tmp_path / "remote_session.json").read_text())["memory_map"]["pass_steps"] == steps


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
    sent: list[dict] = []

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
    is answered `bye`, and `prove` raises `Refused` with the verifier's reason."""
    registry, s = audit.Registry(), store()
    b = remote.StoreBackend(s, engine(s))
    registry.register(os.urandom(32), b"test-node", b.blocks, b.payload_root(b"test-node"), P512)
    lsock = wire.listen("127.0.0.1", 0)
    address = f"127.0.0.1:{lsock.getsockname()[1]}"
    raised = []

    def prover() -> None:
        try:
            remote.prove(b, address, node=b"test-node", codec=P512, log=lambda m: None)
        except Refused as e:
            raised.append(e)

    th = Prover(prover)
    sock = th.accept(lsock)
    with verifying(sock, PassClock(), windows=1, window_ns=5, samples=64, registry=registry) as v:
        with pytest.raises(audit.RegistryError, match="live registration") as refused:
            v.register()
        th.outcome()
    (e,) = raised
    assert e.reasons == (f"the verifier at {address} refused the store's registration: refused: {refused.value}",)
    assert str(e) == e.reasons[0]
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
        except Refused as e:
            raised.append(e)

    th = Prover(prover)
    sock = th.accept(lsock)
    with verifying(sock, PassClock(), windows=1, window_ns=5, samples=64, registry=registry, lifetime=1) as v:
        with pytest.raises(ValueError, match="at least 2 windows") as short:
            v.register()
        th.outcome()
    (e,) = raised
    assert e.reasons == (f"the verifier at {address} refused the store's registration: refused: {short.value}",)
    assert registry.drawn(b"test-node") == 0


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
