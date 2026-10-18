"""The remote bandwidth session over the node's `Store` end to end on the CPU: `remote.prove` (the prover's side, the
Commit's hook) connects over localhost TCP to `benchmarks/remote_session`'s verifier, registers the store, records its
memory map against the registration, and runs each pass after an engine step of a serving-like trace (pages written
over two steps, held, freed and reused, and one long-lived page re-keyed under L = 1 window).  The honest session is
accepted with the appends, releases, re-keys and steps the verifier saw, each pass's seed chaining the appends sent
before its root, each append encoded under its own coin; one whose prover seeds without them, encodes under a zero
or stale coin, re-sends an append, appends after the last close or withholds a block it appended is rejected; a second
registration of a node the verifier holds live is refused with its reason.  The verifier's clock is the prover's
passes, a tick each (`PassClock`)."""
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
        self.clock, self.heads, self.coins = clock, [], []

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
    """A prover whose seeds chain only the root before each pass, not the appends sent before it (the seed before
    the red team's KB-1)."""

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
    forward of its last step, its last read; page 3 a request's that never finishes, written at steps 0 and 1.  The
    engine numbers its steps from ``at`` (one whose binder numbered steps before the session)."""

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


def run(plan: str, *, windows: int = 2, window_ns: int = 5, samples: int = 4096, lifetime: int = 1,
        backend=TickingBackend, at: int = 0, **kw):
    """One session of ``plan``, every receipt of a window drawn at the default ``samples``; the backend keeps the
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
    v = verifier.Verifier(wire.Conn(sock, clock=clock.now), source=os.urandom(32), windows=windows, window_ns=window_ns,
                          samples=samples, delta=2.0 ** -40, jitter_ns=10 ** 9, clock=clock, log=lambda m: None,
                          lifetime=lifetime)
    v.register()
    backend.regions = v.regions
    r = v.session(0, v.allowance(v.ping(4)), None)
    th.join()
    sock.close()
    return r, results[0], s, backend


def test_an_honest_store_session_is_accepted_with_its_frees_re_keys_and_steps():
    r, p, s, backend = run("serve")
    assert r["accepted"] and p["accepted"] and r["failure"] is None
    # the hello carries the store's layout, which the verifier takes as the memory line's labels
    assert backend.regions == (("weights", "hbm", 0, 1), ("kv", "hbm", 1, 16), ("filler", "host", 17, 1))
    # pass 0 has no append before it (step 0's line closes when step 1 opens), so its seed is the unchained one; every
    # later pass's seed chains the head of the appends sent before its root (`audit.chained`)
    assert backend.heads[0] is None and len(backend.heads) > 2 and all(h is not None for h in backend.heads[1:])
    kv, pub = p["kv"], r["record"]["public"]
    # each append sent was answered with its first index and a coin of its own, under which its blocks were encoded
    sent = backend.trace.appends[:len(backend.coins)]
    assert [f for f, _ in backend.coins] == [a.first for a in sent] and len({c for _, c in backend.coins}) == len(sent)
    assert sum(a.count for a in sent) == kv["sent_appended"]
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
    # the map, replayed, holds the session's live blocks, its writes are the appends' digests, and the store holds them
    agree = remote.agreement(backend)
    assert agree["agrees"] and agree["held_appended"] > 0 and agree["held_wrong"] == []


def test_a_session_on_an_engine_whose_steps_were_numbered_before_it_adopts_its_first_step():
    """The Commit's binder numbers the engine's steps from its first run; a session after its runs takes the next."""
    r, p, s, backend = run("serve", at=40)
    assert r["accepted"] and p["accepted"]
    head, lines = memory_map.parse(s.memory_map.text().splitlines())
    steps = [x.step for x in lines]
    assert steps[0] >= 40 and steps == sorted(set(steps))
    assert all(x >= 40 for w in r["record"]["profile"]["windows"] for x in w["steps"])


class Device:
    """kv.cu's encode on the host (`remote.KvDevice`'s interface): each word by `codec.encode_word` under the key
    set and the offset handed in for its block, so `remote.DeviceEncoded`'s plumbing runs without a GPU.  ``shift``
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
    for bad, why in [({"tensor_parallel_size": 2}, "parallelism"), ({"kv_cache_dtype": "fp8"}, "kv_cache_dtype"),
                     ({"async_scheduling": True}, "async"), ({"enable_prefix_caching": True}, "prefix caching")]:
        with pytest.raises(SystemExit, match=why):
            cs.engine_args(bad)
    for kw, err in [({"plan": "serve,serve"}, "one mode"), ({"kv_pages": 0}, "a page"), ({"plan": "nope"}, "a session's mode")]:
        with pytest.raises(Exception, match=err):
            remote.CommitSession("127.0.0.1:7781", **kw)


def test_a_store_session_whose_prover_seeds_without_the_appends_head_is_rejected():
    r, p, _, backend = run("serve", backend=HeadlessBackend)
    assert not r["accepted"] and not p["accepted"] and all(h is None for h in backend.heads)
    assert r["failure"]["failure"] == "receipt-path"


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


def test_a_store_session_whose_prover_withholds_an_appended_block_is_rejected():
    r, p, _, _ = run("servedrop:1")
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
    v = verifier.Verifier(wire.Conn(sock, clock=PassClock().now), source=os.urandom(32), windows=1, window_ns=5,
                          samples=64, delta=2.0 ** -40, jitter_ns=10 ** 9, clock=PassClock(), log=lambda m: None,
                          registry=registry)
    with pytest.raises(audit.RegistryError, match="live registration") as refused:
        v.register()
    th.join()
    sock.close()
    (e,) = raised
    assert e.why == f"refused: {refused.value}" and address in str(e) and e.why in str(e)
    assert registry.refused == {b"test-node": 1} and registry.drawn(b"test-node") == 1
