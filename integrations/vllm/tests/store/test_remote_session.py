"""The remote bandwidth session over the node's `Store` end to end on the CPU: `remote.prove` (the prover's side, the
Commit's hook) connects over localhost TCP to `benchmarks/remote_session`'s verifier, registers the store, records its
memory map against the registration, and runs each pass after an engine step of a serving-like trace (pages written
over two steps, held, freed and reused, and one long-lived page re-keyed under L = 1 window).  The honest session is
accepted with the appends, releases, re-keys and steps the verifier saw; one whose prover withholds a block it appended
is rejected at a draw on that block.  The verifier's clock is the prover's passes, a tick each (`PassClock`)."""
from __future__ import annotations

import os
import queue
import random
import threading

import numpy as np
import pytest

from verity.core.protocols.remote_bandwidth import audit, codec as rcodec, memory_map
from verity_vllm.store import remote
from verity_vllm.store.interface import Append, PageLayout
from verity_vllm.store.registry import ArrayBacking, Store

from .test_memory_map import BLOCK, PAGES, SLOT_BYTES, SLOTS, kv_payload
from .test_registry import CODEC, RECEIPT

P512 = rcodec.Codec(rcodec.P512, 2, False)
verifier, wire = remote.session_module("verifier"), remote.session_module("wire")


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


def store() -> Store:
    """A weights block, one KV layer of 4 pages (K0 K1 V0 V1 each, page p at store blocks 1 + 4p …), and a block of
    host filler: N = 18.  Each block a backend encodes is 128 square roots at p512/k2, so the store is small."""
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
        self.clock = clock

    def run_pass(self, seed: bytes, plan=None):
        out = super().run_pass(seed, plan)
        self.clock.tick()
        return out


def engine(s: Store):
    """A serving trace in the binder's order: pages 0 to 2 in turn, each a request's for three steps (slots 0 to 3 at
    its first, 4 to 7 at its second, read at its third), freed before the step that reuses it, which is after the
    forward of its last step, its last read; page 3 a request's that never finishes, written at steps 0 and 1."""

    lives = {3: 0}

    def step(k: int) -> None:
        p = k % 3
        if k >= 3:
            s.kv_free([p])
        s.kv_alloc([p] + ([3] if k == 0 else []))
        lives[p] = k
        appends = [Append(p, 0, 4)] + ([Append((k - 1) % 3, 4, 4)] if k >= 1 else []) + (
            [Append(3, 4 * k, 4)] if k < 2 else [])
        s.kv_write_reference(appends, lambda li, page, x: kv_payload(li, page, x, lives[page]), k)
    return step


def run(plan: str, *, windows: int = 2, window_ns: int = 5, samples: int = 4096, lifetime: int = 1):
    """One session of ``plan``, every receipt of a window drawn at the default ``samples``."""
    clock = PassClock()
    s = store()
    backend = TickingBackend(clock, s, engine(s), randbytes=random.Random(7).randbytes)
    lsock = wire.listen("127.0.0.1", 0)
    results = []
    th = threading.Thread(target=lambda: results.extend(remote.prove(
        backend, f"127.0.0.1:{lsock.getsockname()[1]}", plan, node=b"test-node", codec=P512, log=lambda m: None)),
        daemon=True)
    th.start()
    sock, _ = lsock.accept()
    lsock.close()
    v = verifier.Verifier(wire.Conn(sock, clock=clock.now), source=os.urandom(32), windows=windows, window_ns=window_ns,
                          samples=samples, delta=2.0 ** -40, jitter_ns=10 ** 9, clock=clock, log=lambda m: None,
                          lifetime=lifetime)
    v.register()
    r = v.session(0, v.allowance(v.ping(4)), None)
    th.join()
    sock.close()
    return r, results[0], s


def test_an_honest_store_session_is_accepted_with_its_frees_re_keys_and_steps():
    r, p, s = run("serve")
    assert r["accepted"] and p["accepted"] and r["failure"] is None
    kv, pub = p["kv"], r["record"]["public"]
    assert kv["lifetime"] == pub["lifetime"] == 1 and kv["rekeyed"] > 0
    assert kv["sent_appended"] == pub["appended"] > 0 and kv["sent_released"] == pub["released"] > 0
    assert sum(pub["post_seed"]) == sum(kv["pass_appended"]) > 0
    steps = [x for w in r["record"]["profile"]["windows"] for x in w["steps"]]
    assert len(steps) == sum(pub["passes"]) and all(a < b for a, b in zip(steps, steps[1:])) and steps[-1] < kv["steps"]
    head, lines = memory_map.parse(s.memory_map.text().splitlines())
    lines = list(lines)
    assert head.registration.hex() == pub["registration"] and head.blocks == 18
    # page 0's first request, written at steps 0 and 1 and read at 2, is freed before step 3's write: at step 2, the
    # step whose forward read it last
    assert min(x.step for x in lines if {1, 2, 3, 4} <= set(x.freed)) == 2
    # page 3, written over steps 0 and 1 and never freed, is written again whole in one step (re-keyed) once a window
    # old; its first writes were split, so every line that writes all four of its blocks is a re-key
    rekeys = [x.step for x in lines if {13, 14, 15, 16} <= {w.slot for w in x.written}]
    assert rekeys and min(rekeys) > 1 and kv["rekeyed"] >= 4 * len(rekeys)


def test_a_store_session_whose_prover_withholds_an_appended_block_is_rejected():
    r, p, _ = run("servedrop:1")
    assert not r["accepted"] and not p["accepted"]
    assert r["failure"]["failure"] == "receipt" and p["failure_on_dropped"] is True and p["dropped"] == [18]


def test_the_hook_takes_a_verifier_address_and_one_session_a_store(monkeypatch):
    """`serve`, the Commit's hook, proves the store with one engine step a pass; the session is one a store."""
    seen, steps = [], []
    monkeypatch.setattr(remote, "prove", lambda backend, verifier, **kw: seen.append((backend, verifier, kw)) or [])
    s = store()
    remote.serve("10.0.0.7:7781", s, lambda: steps.append(1), plan="serve")
    (backend, verifier, kw), = seen
    backend.engine(0)
    assert backend.store is s and verifier == "10.0.0.7:7781" and kw == {"plan": "serve"} and steps == [1]
    assert remote.address("10.0.0.7:7781") == ("10.0.0.7", 7781)
    for bad in ("7781", "host:", ":7781", "host:0", "host:65536", "host:x"):
        with pytest.raises(ValueError, match="HOST:PORT"):
            remote.address(bad)
    s = store()
    b = remote.StoreBackend(s, engine(s))
    reg = audit.register(bytes(32), b"n", b.blocks, b.payload_root(b"n"), P512, nonce=bytes(32))
    b.encode(reg)
    b.begin_session()
    b.serving(reg, 1)
    with pytest.raises(RuntimeError, match="one session"):
        b.begin_session()
