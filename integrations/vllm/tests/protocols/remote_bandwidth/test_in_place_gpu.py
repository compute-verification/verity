"""The in-place session (`in_place.InPlaceBackend`) on a GPU against `benchmarks/remote_session`'s verifier over
localhost: weights and a KV pool in CUDA tensors registered as the binder registers vLLM's (`TorchBacking`, and
`PagedBacking` over FlashAttention's cache layout), read where they are by session.cu and encoded in place by kv.cu,
under the serving trace of `test_remote_session` with each step's KV written after its `kv_append`, as a forward
writes it.  The payload tree is the driver's, the encode `codec.encode`'s; the honest session is accepted with blocks
the step's forward rewrote read from their copies; a prover that withholds a block, keeps no copies, or whose
tensors moved is rejected.  The int8 attention's credited read over the int8 backend's cache, encoded in place, gives
its plaintext windows, before and after an append's encode moves its slots' offsets.  Runs where CUDA, session.cu's
build (``VERITY_SESSION_SO``) and kv.cu's at p1024, masked (``VERITY_KV_SO``) are; the credited read where paged.cu's is
too (``VERITY_PAGED_SO``, at p1024, masked, FOLD6, 32 query heads)."""
from __future__ import annotations

import os
import random
import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")
SESSION_SO, KV_SO = os.environ.get("VERITY_SESSION_SO"), os.environ.get("VERITY_KV_SO")
PAGED_SO = os.environ.get("VERITY_PAGED_SO")
pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and SESSION_SO and KV_SO),
                                reason="a GPU, VERITY_SESSION_SO (session.cu) and VERITY_KV_SO (kv.cu at p1024, masked)")

from verity.core.protocols.remote_bandwidth import audit, codec as rcodec  # noqa: E402
from verity_vllm.protocols.remote_bandwidth import int8_kv, memory_map as store_map, remote  # noqa: E402
from verity_vllm.protocols.remote_bandwidth.binder import PagedBacking, TorchBacking, page_layout, storage_bytes  # noqa: E402
from verity_vllm.protocols.remote_bandwidth.in_place import InPlaceBackend  # noqa: E402
from verity_vllm.protocols.remote_bandwidth.interface import IDENTITY, Append, PageLayout  # noqa: E402
from verity_vllm.protocols.remote_bandwidth.registry import NodeMemory  # noqa: E402

from .test_memory_map import kv_payload  # noqa: E402
from .test_registry import RECEIPT  # noqa: E402
from .test_remote_session import PassClock, verifying, wire  # noqa: E402

BLOCK = rcodec.BLOCK_BYTES
LAYERS, PAGES, SLOTS, HEADS, DIM = 2, 6, 8, 4, 128      # a slot 1 KiB of K, 1 KiB of V; a plane-page 2 blocks


def gpu_store() -> tuple[NodeMemory, list]:
    """Five weight blocks, then two KV layers of 6 pages (FlashAttention's (2, pages, slots, heads, dim) bf16): N = 53."""
    s = NodeMemory()
    w = torch.randint(0, 256, (5 * BLOCK,), dtype=torch.uint8, device="cuda")
    s.register("w", "hbm", "weights", TorchBacking(w, 5 * BLOCK), block_bytes=BLOCK, codec=IDENTITY, receipt=RECEIPT,
               source="w", encode=False)
    caches = [torch.zeros(2, PAGES, SLOTS, HEADS, DIM, dtype=torch.bfloat16, device="cuda") for _ in range(LAYERS)]
    phys = page_layout(caches[0], PAGES, SLOTS)
    backings = [PagedBacking(storage_bytes(t), t.storage_offset() * t.element_size(), phys) for t in caches]
    s.register_kv_pool("kv", backings, PageLayout(PAGES, SLOTS, phys.slot_bytes, phys.planes, slot_major=phys.slot_major),
                       block_bytes=BLOCK, codec=IDENTITY, receipt=RECEIPT, encode=False)
    return s, caches


def engine(s: NodeMemory, caches: list):
    """`test_remote_session.engine`'s trace, each step's KV written into the caches after its `kv_append` (the
    forward's write, after the schedule), then `end_step`."""
    lives, n = {3: 0}, [0]

    def write(a: Append) -> None:
        for li, t in enumerate(caches):
            u = t.view(torch.uint8)
            for x in range(a.first, a.first + a.count):
                kv = torch.frombuffer(bytearray(kv_payload(li, a.page, x, lives[a.page])), dtype=torch.uint8).cuda()
                u[0, a.page, x].copy_(kv[:BLOCK // 4].view(HEADS, 2 * DIM))
                u[1, a.page, x].copy_(kv[BLOCK // 4:].view(HEADS, 2 * DIM))

    def step(_: int) -> None:
        k, n[0] = n[0], n[0] + 1
        p = k % 3
        if k >= 3:
            s.kv_free([p])
        s.kv_alloc([p] + ([3] if k == 0 else []))
        lives[p] = k
        appends = [Append(p, 0, 4)] + ([Append((k - 1) % 3, 4, 4)] if k >= 1 else []) + (
            [Append(3, 4 * k, 4)] if k < 2 else [])
        s.kv_append(appends, k)
        for a in appends:
            write(a)
        s.end_step()
    return step


class Ticking(InPlaceBackend):
    def __init__(self, clock: PassClock, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.clock = clock

    def run_pass(self, seed: bytes, plan=None, step=None):
        out = super().run_pass(seed, plan, step)
        self.clock.tick()
        return out


def backend(s: NodeMemory, caches: list, clock: PassClock | None = None, **kw) -> Ticking:
    return Ticking(clock or PassClock(), s, engine(s, caches), [s.backing(r.name) for r in s.layout().regions],
                   session_lib=SESSION_SO, kv_lib=KV_SO, randbytes=random.Random(7).randbytes, **kw)


def run(plan: str, *, windows: int = 3, copies: bool = True, **kw):
    """One session of ``plan`` at L = 60 windows (no re-keys): the verifier's result, and the prover's or what it
    raised."""
    clock, (s, caches) = PassClock(), gpu_store()
    b = backend(s, caches, clock, **kw)
    if not copies:
        s.on_write = None
    lsock = wire.listen("127.0.0.1", 0)
    got: list = []

    def prove() -> None:
        try:
            got.extend(remote.prove(b, f"127.0.0.1:{lsock.getsockname()[1]}", plan, node=b"gpu-node",
                                    log=lambda m: None))
        except Exception as e:  # noqa: BLE001  -- the session's failure, which the test reads
            got.append(e)

    th = threading.Thread(target=prove, daemon=True)
    th.start()
    sock, _ = lsock.accept()
    lsock.close()
    with verifying(sock, clock, "gpu-node", windows=windows, window_ns=5, samples=512, lifetime=60) as v:
        v.register()
        try:
            r = v.session(0, v.allowance(v.ping(4)), None)
        except Exception as e:  # noqa: BLE001  -- a prover that fails drops the connection
            r = {"accepted": False, "error": e}
        th.join()
    return r, got[0], s, b


def test_the_payload_tree_is_the_drivers_and_the_store_is_encoded_where_it_is():
    s, caches = gpu_store()
    b = backend(s, caches)
    N = b.blocks
    payloads = [s.payload(x) for x in range(N)]
    ref = audit.payload_tree(b"n", payloads, b.append_salts(0, N))
    assert N == 53 and b.payload_root(b"n") == ref.commitment.root
    assert all(b.payload_path(j) == ref.open(j).path for j in (0, 4, 5, 6, 29, N - 1))
    reg = audit.register(bytes(32), b"n", N, ref.commitment.root, nonce=bytes(32))
    b.encode(reg)
    held = b._gather(np.arange(N)).cpu().numpy()
    assert [rcodec.decode(held[j].tobytes(), j, reg.key) for j in range(N)] == payloads
    assert b.gates["device_encode"]["blocks_checked"] == 5


def test_calibration_reports_the_passes_in_place_and_the_bare_read_as_the_verifiers_allowance_reads_them():
    s, caches = gpu_store()
    b = backend(s, caches, calibration_passes=64, calibration_warm=4)
    N = b.blocks
    b.payload_root(b"n")
    b.encode(audit.register(bytes(32), b"n", N, bytes(64), nonce=bytes(32)))
    cal = b.calibrate()
    assert cal["passes"] == 63 and cal["warm_passes"] == 4
    assert 0 < cal["pass_ns"]["p50"] <= cal["pass_ns"]["p999"] <= cal["pass_ns_max"] == cal["pass_ns"]["max"]
    assert 0 < cal["gap_ns"]["p50"] <= cal["gap_ns_max"] and cal["read_tiers"]["hbm"]["bytes"] == N * BLOCK
    assert cal["read_bytes_per_s"] > 0 and b.pass_ns == [] and b.phases.closed == []
    assert b.warmup(10 ** 6)["passes"] >= 1


class Tiny(torch.nn.Module):
    """vLLM's shape in small: an embedding tied to an LM head that only a logits processor reads, and linears, one
    with a bias too small to register; the embedding's last 2304 bytes past its whole block stay plain."""

    class Logits(torch.nn.Module):
        def __init__(self, head) -> None:
            super().__init__()
            self.head = [head]

        def forward(self, h):
            return h @ self.head[0].weight.t()

    def __init__(self) -> None:
        super().__init__()
        kw = {"dtype": torch.bfloat16, "device": "cuda"}
        self.embed = torch.nn.Embedding(50, 64, **kw)
        self.up = torch.nn.Linear(64, 96, bias=False, **kw)
        self.down = torch.nn.Linear(96, 64, **kw)
        self.lm_head = torch.nn.Module()
        self.lm_head.weight = self.embed.weight
        self.logits_processor = Tiny.Logits(self.lm_head)

    def forward(self, ids):
        return self.logits_processor(self.down(torch.relu(self.up(self.embed(ids)))))


def test_weights_are_decoded_at_use_so_the_forward_is_the_models_and_decoded_back_where_they_are_at_close():
    torch.manual_seed(3)
    model, s, weights = Tiny(), NodeMemory(), {}
    for name, p in model.named_parameters():
        b = TorchBacking.of_tensor(p.data, BLOCK)
        if b is not None:
            s.register(name, "hbm", "weights", b, block_bytes=BLOCK, codec=IDENTITY, receipt=RECEIPT, source=name,
                       encode=False)
            weights[name] = p
    assert sorted(weights) == ["down.weight", "embed.weight", "up.weight"]
    plain = {n: p.detach().clone() for n, p in weights.items()}
    ids = torch.arange(50, device="cuda")
    want, where = model(ids), {n: p.data_ptr() for n, p in weights.items()}
    b = InPlaceBackend(s, lambda _: None, [s.backing(r.name) for r in s.layout().regions], session_lib=SESSION_SO,
                       kv_lib=KV_SO, randbytes=random.Random(7).randbytes, model=model)
    b.payload_root(b"n")
    b.encode(audit.register(bytes(32), b"n", b.blocks, bytes(64), nonce=bytes(32)))
    assert not torch.equal(model.up.weight, plain["up.weight"])
    assert b.gates["decode_at_use"]["modules"] == 4
    assert torch.equal(model(ids), want) and b.decoded.calls == 4
    assert {n: p.data_ptr() for n, p in weights.items()} == where
    b.close()
    assert all(torch.equal(weights[n], plain[n]) for n in weights)
    assert b.gates["decode_at_use"]["decoded_back"] is True and not b.decoded.handles
    assert torch.equal(model(ids), want) and b.decoded.calls == 4


def test_an_honest_in_place_session_is_accepted_reading_rewritten_blocks_from_their_copies():
    r, p, s, b = run("serve")
    assert r["accepted"] and p["accepted"], (r.get("failure"), p)
    st = p["device"]
    assert st["in_place"] and st["passes"] > 10 and st["pass_ms"]["max"] is not None
    assert st["pass_patched"] > 0 and st["open_patched"] > 0 and st["encoded_appends"] > 4
    agree = store_map.agreement(b)
    assert agree["agrees"] and agree["held_appended"] > 0 and agree["held_wrong"] == []


def test_an_in_place_prover_that_withholds_an_appended_block_is_rejected():
    r, p, _, b = run("servedrop:1")
    assert not r["accepted"] and not p["accepted"] and p["failure_on_dropped"] is True
    assert p["dropped"] == [b.blocks]


def test_a_block_rewritten_after_its_receipt_with_no_copy_is_stale():
    r, p, _, _ = run("serve", copies=False)
    assert not r["accepted"] and isinstance(p, Exception) and "Stale" in str(p)


def test_a_registered_tensor_moved_mid_session_fails_it_by_name():
    calls = [0]

    def moved() -> list[str]:
        calls[0] += 1
        return ["kv[1]"] if calls[0] > 4 else []

    r, p, _, _ = run("serve", moved=moved)
    assert not r["accepted"] and isinstance(p, RuntimeError) and "kv[1]" in str(p)


I8_PAGES, I8_SLOTS, I8_HEADS = 4, 16, 8


def int8_store() -> tuple[NodeMemory, list]:
    """Three weight blocks, then two layers of the int8 backend's cache (vLLM's view [pages, KV heads, slots, 2 x 128]
    over [pages][slots][KV heads][K ‖ V] bf16, a token-layer one block): N = 3 + 2 x 64."""
    s, g = NodeMemory(), torch.Generator().manual_seed(5)
    w = torch.randint(0, 256, (3 * BLOCK,), dtype=torch.uint8, device="cuda")
    s.register("w", "hbm", "weights", TorchBacking(w, 3 * BLOCK), block_bytes=BLOCK, codec=IDENTITY, receipt=RECEIPT,
               source="w", encode=False)
    caches = [torch.randn(I8_PAGES, I8_SLOTS, I8_HEADS, 2 * DIM, generator=g).to(torch.bfloat16).cuda().transpose(1, 2)
              for _ in range(2)]
    lay = int8_kv.token_layer_layout(caches[0], I8_PAGES, I8_SLOTS, DIM)
    backings = [PagedBacking(storage_bytes(t), t.storage_offset() * t.element_size(), lay) for t in caches]
    s.register_kv_pool("kv", backings, lay, block_bytes=BLOCK, codec=IDENTITY, receipt=RECEIPT, encode=False)
    return s, caches


def windows(read, cache, layer: int, lens: list[int], table, slot_mapping=None):
    """One prefill step's windows of ``layer`` through ``read``: K and V [rows][KV heads][128]."""
    n, wo, k, v = read.buffers(torch.tensor(lens, dtype=torch.int32, device="cuda"), len(lens), I8_HEADS, DIM)
    read.read(cache, layer, table, slot_mapping, n, wo, k, v)
    return k, v


def plain_windows(cache, lens: list[int], table):
    rows = cache.transpose(1, 2).reshape(-1, I8_HEADS, 2 * DIM)
    slot = torch.cat([table[r][torch.arange(n, device="cuda") // I8_SLOTS].long() * I8_SLOTS
                      + torch.arange(n, device="cuda") % I8_SLOTS for r, n in enumerate(lens)])
    return rows[slot, :, :DIM], rows[slot, :, DIM:]


@pytest.mark.skipif(not PAGED_SO, reason="VERITY_PAGED_SO (paged.cu at p1024, masked, FOLD6, 32 query heads)")
def test_the_credited_read_gives_the_int8_caches_plaintext_windows_through_registration_writes_and_appends():
    s, caches = int8_store()
    b = InPlaceBackend(s, lambda _: None, [s.backing(r.name) for r in s.layout().regions], session_lib=SESSION_SO,
                       kv_lib=KV_SO, randbytes=random.Random(7).randbytes, paged_lib=PAGED_SO)
    lens, table = [37, 20], torch.tensor([[2, 0, 3], [1, 3, 0]], dtype=torch.int32, device="cuda")
    plain = [c.clone() for c in caches]
    b.payload_root(b"n")
    b.encode(audit.register(bytes(32), b"n", b.blocks, bytes(64), nonce=bytes(32)))
    g = b.gates["credited_read"]
    assert (g["layers"], g["installed"], g["slots"], g["settled"]) == (2, 0, 2 * I8_PAGES * I8_SLOTS, 0)
    for li in (0, 1):
        assert not torch.equal(caches[li], plain[li])
        got, want = windows(b.credited, caches[li], li, lens, table), plain_windows(plain[li], lens, table)
        assert all(torch.equal(x.view(torch.int16), y.view(torch.int16)) for x, y in zip(got, want))

    # the step writes 4 slots of page 1 in both layers in the clear: read as written, flagged, then appended
    written = torch.arange(I8_SLOTS, I8_SLOTS + 4, device="cuda")
    for li, c in enumerate(caches):
        new = torch.randn(4, I8_HEADS, 2 * DIM, device="cuda").to(torch.bfloat16)
        c.transpose(1, 2)[1, :4] = new
        plain[li].transpose(1, 2)[1, :4] = new
        got, want = windows(b.credited, c, li, lens, table, written), plain_windows(plain[li], lens, table)
        assert all(torch.equal(x.view(torch.int16), y.view(torch.int16)) for x, y in zip(got, want))
        assert b.kv_tables[li][1][written].tolist() == [1] * 4
    firsts = {r.name: r.first for r in s.layout().regions}
    slots = np.array([firsts[f"kv/{li}"] + I8_SLOTS + x for li in (0, 1) for x in range(4)], np.int64)
    payloads = [m.tobytes() for m in b._gather(slots).cpu().numpy()]
    a = b.append(payloads, b.append_salts(b.blocks, len(slots)), slots)
    b.encode_append(a.first, bytes(range(rcodec.COIN_BYTES)))
    assert b.gates["credited_read"]["settled"] == 8
    for li, c in enumerate(caches):
        assert b.kv_tables[li][1][written].tolist() == [0] * 4
        assert not torch.equal(c.transpose(1, 2)[1, :4], plain[li].transpose(1, 2)[1, :4])
        got, want = windows(b.credited, c, li, lens, table), plain_windows(plain[li], lens, table)
        assert all(torch.equal(x.view(torch.int16), y.view(torch.int16)) for x, y in zip(got, want))
