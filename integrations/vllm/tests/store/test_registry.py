"""The store's registry on the CPU, with a fake codec and a fake receipt: registration, the KV pages' life (alloc,
append, free) without re-encoding, the pass plan, chained seeds, and openings the verifier's reference accepts."""
from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest

from verity_vllm.store.interface import IDENTITY, Append, Lifecycle, PageLayout, Stale, StoreView
from verity_vllm.store.registry import ArrayBacking, Store, TagReceipt, chain, verify_opening

WORD = 64
BLOCK = 256
SIGMA, SIGMA2 = bytes(range(32)), bytes(range(1, 33))


class FakeCodec:
    """XOR with a pad keyed by the word's index: invertible, and a word encodes differently at another index."""

    name = "fake-xor"
    word_bytes = WORD
    payload_bytes = WORD

    @staticmethod
    def _pad(index: int, n: int) -> bytes:
        return b"".join(hashlib.sha512(b"pad" + struct.pack(">Q", index + w)).digest() for w in range(n // WORD))

    def encode(self, payload: bytes, index: int) -> bytes:
        return bytes(a ^ b for a, b in zip(payload, self._pad(index, len(payload))))

    decode = encode


class FakeReceipt:
    """SHA-512 over the seed, the block index, the block and the bound input; four receipts a leaf."""

    name = "fake-sha512"
    receipt_bytes = 64
    per_leaf = 4
    binds = ("z",)

    def keyed(self, seed: bytes):
        def receipt(j: int, block: bytes, z: bytes | None = None) -> bytes:
            return hashlib.sha512(seed + struct.pack(">Q", j) + block + (z or b"")).digest()
        return receipt


CODEC, RECEIPT = FakeCodec(), FakeReceipt()


def weights(store: Store, name: str = "w", blocks: int = 6, seed: int = 0) -> tuple[ArrayBacking, bytes]:
    payload = np.random.default_rng(seed).integers(0, 256, blocks * BLOCK, np.uint8)
    b = ArrayBacking(payload.copy())
    store.register(name, "hbm", "weights", b, block_bytes=BLOCK, codec=CODEC, receipt=RECEIPT, source=f"sha:{seed}")
    return b, payload.tobytes()


def pool(store: Store, layers: int = 2, pages: int = 4, slots: int = 8, slot_bytes: int = 64, **kw):
    lay = PageLayout(pages, slots, slot_bytes, **kw)
    ps, ks = lay.strides()
    nbytes = max(ps * (pages - 1) + ks * (lay.planes - 1) + lay.plane_bytes, ps * pages)
    bs = [ArrayBacking(np.zeros(nbytes, np.uint8)) for _ in range(layers)]
    store.register_kv_pool("kv", bs, lay, block_bytes=BLOCK, codec=CODEC, receipt=RECEIPT)
    return bs, lay


def kv_payload(layer: int, page: int, slot: int) -> bytes:
    return hashlib.sha512(struct.pack(">QQQ", layer, page, slot)).digest() * 2


def test_the_store_is_both_interfaces():
    s = Store()
    assert isinstance(s, StoreView) and isinstance(s, Lifecycle)


def test_registration_encodes_in_place_and_logs_each_region():
    s = Store()
    b, payload = weights(s)
    info = s.region("w")
    assert (info.first, info.blocks, info.first_word, info.word_bytes) == (0, 6, 0, WORD)
    assert b.read(0, b.nbytes) != payload
    assert CODEC.decode(b.read(0, b.nbytes), 0) == payload
    r1 = s.registration()
    b2, _ = weights(s, "w2", 2, seed=1)
    assert s.region("w2").first == 6 and s.region("w2").first_word == 6 * BLOCK // WORD
    r2 = s.registration()
    assert (r1.count, r2.count) == (1, 2) and r1.root != r2.root
    t = Store()
    weights(t)
    weights(t, "w2", 2, seed=1)
    assert t.registration() == r2
    assert s.layout().bytes_by_tier == {"hbm": 8 * BLOCK}


def test_registration_refuses_partial_blocks_and_unknown_tiers():
    s = Store()
    with pytest.raises(ValueError):
        s.register("x", "hbm", "weights", ArrayBacking(np.zeros(BLOCK + WORD, np.uint8)), block_bytes=BLOCK,
                   codec=CODEC, receipt=RECEIPT, source="x")
    with pytest.raises(ValueError):
        s.register("x", "nvme", "filler", ArrayBacking(np.zeros(BLOCK, np.uint8)), block_bytes=BLOCK, codec=CODEC,
                   receipt=RECEIPT, source="x")


def test_page_layouts_page_major_and_plane_major():
    s = Store()
    pool(s, layers=1, pages=3, slots=8, slot_bytes=64)
    p = s.pool
    assert p.ids.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    assert (p.lo.tolist(), p.hi.tolist()) == ([0, 4, 0, 4], [4, 8, 4, 8])
    t = Store()
    pool(t, layers=1, pages=3, slots=8, slot_bytes=64, page_stride=512, plane_stride=3 * 512)
    assert t.pool.ids.tolist() == [[0, 1, 6, 7], [2, 3, 8, 9], [4, 5, 10, 11]]


def test_kv_life_alloc_append_free_without_reencoding():
    s = Store()
    bs, lay = pool(s)
    before = [b.read(0, b.nbytes) for b in bs]
    s.kv_alloc([2])
    assert [b.read(0, b.nbytes) for b in bs] == before
    live = s.begin_pass([2])
    assert all(len(b) == 0 for b in live.blocks)
    s.end_pass()
    s.kv_write_reference([Append(2, 0, 3)], kv_payload, step=0)
    assert s.pool.dirty[2].tolist() == [True, False, True, False]
    ps = s.begin_pass([2])
    assert all(len(b) == 0 for b in ps.blocks)
    s.end_pass()
    s.kv_write_reference([Append(2, 3, 2)], kv_payload, step=1)
    p = s.pool
    assert p.dirty[2].tolist() == [False, True, False, True]
    assert p.epoch[2].tolist() == [1, 0, 1, 0] and p.step[2].tolist() == [1, -1, 1, -1]
    ps = s.begin_pass([2])
    assert [b.tolist() for b in ps.blocks] == [[8, 10], [8, 10]]
    s.end_pass()
    k0 = bs[0].read(2 * 1024, 64)
    assert CODEC.decode(k0, s.region("kv/0").word_index(8)) == kv_payload(0, 2, 0)[:64]
    s.kv_write_reference([Append(2, 5, 3)], kv_payload, step=2)
    written = [b.read(0, b.nbytes) for b in bs]
    reg = s.registration()
    s.kv_free([2])
    assert [b.read(0, b.nbytes) for b in bs] == written
    assert s.registration() == reg
    assert set(range(8, 12)) <= set(s.idle_blocks("kv/0").tolist())
    s.kv_alloc([2])
    assert [b.read(0, b.nbytes) for b in bs] == written
    assert s.source("kv/0", 8).kind == "kv" and s.source("kv/0", 0).kind == "filler"


def test_a_page_freed_part_written_keeps_its_dirty_block_out():
    s = Store()
    pool(s)
    s.kv_alloc([1])
    s.kv_write_reference([Append(1, 0, 6)], kv_payload, step=0)
    s.kv_free([1])
    idle = set(s.idle_blocks("kv/0").tolist())
    assert {4, 6} <= idle and not {5, 7} & idle
    s.kv_alloc([1])
    assert set(s.idle_blocks("kv/0").tolist()) >= {4, 6}


def _window(s: Store, sigma: bytes, steps):
    s.begin_window(sigma)
    out = []
    for live, appends in steps:
        s.begin_pass(live)
        out.append(s.end_pass())
        if appends:
            s.kv_write_reference(appends, kv_payload, step=len(out))
    return out, s.end_window()


def test_passes_are_chained_from_the_challenge():
    def run(flip: bool):
        s = Store()
        b, _ = weights(s)
        if flip:
            b.write(0, bytes([b.read(0, 1)[0] ^ 1]))
        return _window(s, SIGMA, [((), None), ((), None), ((), None)])[1], s.registration().root
    (w, reg), (w2, _) = run(False), run(True)
    assert w.passes[0].seed == chain(SIGMA, reg, 0)
    assert [p.seed for p in w.passes[1:]] == [chain(SIGMA, p.root, p.index + 1) for p in w.passes[:-1]]
    assert w.passes[0].seed == w2.passes[0].seed and w.passes[1].seed != w2.passes[1].seed
    assert chain(SIGMA, reg, 0) == hashlib.sha512(b"verity/remote-bandwidth/seed/v0\0" + SIGMA + bytes(4) + reg).digest()


def test_windows_chain_through_the_session():
    s = Store()
    weights(s)
    _, w1 = _window(s, SIGMA, [((), None), ((), None)])
    _, w2 = _window(s, SIGMA2, [((), None)])
    assert w2.passes[0].seed == chain(SIGMA2, w1.passes[-1].root, 0)
    with pytest.raises(ValueError):
        s.begin_window(b"sigma")


def test_a_plan_is_its_runs_each_block_once():
    s = Store(filler_plan=lambda st, j: [("kv/0", [0, 1, 5])])
    weights(s)
    weights(s, "w2", 2, seed=1)
    pool(s)
    s.kv_alloc([2])
    s.kv_write_reference([Append(2, 0, 8)], kv_payload, step=0)
    (pc,), w = _window(s, SIGMA, [([2], None)])
    runs = [(0, 8), (16, 4), (32, 4), (8, 2), (13, 1)]
    assert s.plan(0).tolist() == [b for f, c in runs for b in range(f, f + c)]
    want = b"verity/remote-bandwidth/plan/v0\0" + struct.pack(">I", 5) + b"".join(struct.pack(">QQ", *r) for r in runs)
    assert pc.plan_digest == hashlib.sha512(want).digest()
    t = Store(filler_plan=lambda st, j: [("kv/0", [0, 8])])
    pool(t)
    t.kv_alloc([2])
    t.kv_write_reference([Append(2, 0, 8)], kv_payload, step=0)
    t.begin_window(SIGMA)
    with pytest.raises(ValueError):
        t.begin_pass([2])


def test_outside_a_window_a_pass_commits_nothing():
    s = Store()
    weights(s)
    s.begin_pass()
    assert s.end_pass() is None
    assert s.counters.calls == {"register": 1, "begin_pass": 1, "end_pass": 1}


def test_openings_verify_and_a_tampered_one_does_not():
    host = ArrayBacking(np.zeros(4 * BLOCK, np.uint8))
    s = Store(filler_plan=lambda st, j: [("dram", range(4)), ("kv/1", st.idle_blocks("kv/1")[:3])])
    _, payload = weights(s)
    bs, _ = pool(s)
    s.register("dram", "host", "filler", host, block_bytes=BLOCK, codec=CODEC, receipt=RECEIPT, source="zeros")
    s.kv_alloc([0, 3])
    s.kv_write_reference([Append(0, 0, 8), Append(3, 0, 4)], kv_payload, step=0)
    passes, w = _window(s, SIGMA, [([0, 3], None), ([0, 3], [Append(3, 4, 4)]), ([0, 3], None)])
    assert len(w.passes) == 3 and w.passes[2].blocks == 6 + 2 * 6 + 4 + 3 + 2 * 2
    lay = s.layout()
    for j, pc in enumerate(w.passes):
        plan = s.plan(j)
        assert len(plan) == pc.blocks and len(set(plan.tolist())) == len(plan)
        for o in s.open([(j, q) for q in range(pc.blocks)]):
            info = next(r for r in lay.regions if r.name == o.region)
            want = payload[(o.block - info.first) * BLOCK:][:BLOCK] if o.region == "w" else None
            if o.region == "dram":
                want = bytes(BLOCK)
            assert verify_opening(info, pc, plan, o, RECEIPT, CODEC.decode, want)
    o = s.open([(0, 1)])[0]
    info = s.region("w")
    bad = o.__class__(**{**o.__dict__, "data": bytes(BLOCK)})
    assert not verify_opening(info, w.passes[0], s.plan(0), bad, RECEIPT, CODEC.decode, None)
    assert not verify_opening(info, w.passes[0], s.plan(0), o, RECEIPT, CODEC.decode, bytes(BLOCK))


def test_an_opening_of_a_rewritten_block_is_stale():
    s = Store()
    pool(s)
    s.kv_alloc([0])
    s.kv_write_reference([Append(0, 0, 8)], kv_payload, step=0)
    _, w = _window(s, SIGMA, [([0], None)])
    s.kv_free([0])
    s.kv_alloc([0])
    s.kv_write_reference([Append(0, 0, 2)], kv_payload, step=1)
    with pytest.raises(Stale):
        s.open([(0, 0)])
    assert s.open([(0, 1)])[0].block == 1


def test_a_device_reader_hands_in_segment_roots():
    s = Store()
    weights(s)
    s.begin_window(SIGMA)
    s.begin_pass()
    pc = s.end_pass(roots={0: b"\x07" * 64})
    assert pc.segment_roots == (b"\x07" * 64,)
    s.end_window()


def test_identity_codec_and_tag_receipts():
    from verity.core.protocols.bandwidth import tags
    s = Store()
    raw = np.arange(2 * BLOCK, dtype=np.uint8)
    b = ArrayBacking(raw.copy())
    rec = TagReceipt(tags.get("ro-blake3/v1"))
    s.register("w", "hbm", "weights", b, block_bytes=BLOCK, codec=IDENTITY, receipt=rec, source="arange")
    assert b.read(0, b.nbytes) == raw.tobytes()
    _, w = _window(s, SIGMA, [((), None)])
    o = s.open([(0, 1)])[0]
    assert verify_opening(s.region("w"), w.passes[0], s.plan(0), o, rec, IDENTITY.decode, raw.tobytes()[BLOCK:])
