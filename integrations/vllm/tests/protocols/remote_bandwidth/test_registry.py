"""The store's registry on the CPU, with a fake codec and a fake receipt: registration, the KV pages' life (alloc,
append, free) without re-encoding, the pass plan, chained seeds, and openings the verifier's reference accepts."""
from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest

from verity.core.protocols.remote_bandwidth import receipts as rb
from verity_vllm.protocols.remote_bandwidth.interface import IDENTITY, Append, Lifecycle, PageLayout, Stale, StoreView
from verity_vllm.protocols.remote_bandwidth.registry import ArrayBacking, KindReceipt, NodeMemory, RemoteBandwidthReceipt, chain
from verity_vllm.protocols.remote_bandwidth.verify import verify_opening

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
    """SHA-512 over the seed, the block index, the block and the bound input."""

    name = "fake-sha512"
    receipt_bytes = 64
    binds = ("z",)

    def keyed(self, seed: bytes):
        def receipt(j: int, block: bytes, z: bytes | None = None) -> bytes:
            return hashlib.sha512(seed + struct.pack(">Q", j) + block + (z or b"")).digest()
        return receipt


CODEC, RECEIPT = FakeCodec(), FakeReceipt()


def weights(store: NodeMemory, name: str = "w", blocks: int = 6, seed: int = 0) -> tuple[ArrayBacking, bytes]:
    payload = np.random.default_rng(seed).integers(0, 256, blocks * BLOCK, np.uint8)
    b = ArrayBacking(payload.copy())
    store.register(name, "hbm", "weights", b, block_bytes=BLOCK, codec=CODEC, receipt=RECEIPT, source=f"sha:{seed}")
    return b, payload.tobytes()


def pool(store: NodeMemory, layers: int = 2, pages: int = 4, slots: int = 8, slot_bytes: int = 64, **kw):
    lay = PageLayout(pages, slots, slot_bytes, **kw)
    ps, ks = lay.strides()
    nbytes = max(ps * (pages - 1) + ks * (lay.planes - 1) + lay.plane_bytes, ps * pages)
    bs = [ArrayBacking(np.zeros(nbytes, np.uint8)) for _ in range(layers)]
    store.register_kv_pool("kv", bs, lay, block_bytes=BLOCK, codec=CODEC, receipt=RECEIPT)
    return bs, lay


def kv_payload(layer: int, page: int, slot: int) -> bytes:
    return hashlib.sha512(struct.pack(">QQQ", layer, page, slot)).digest() * 2


def test_the_store_is_both_interfaces():
    s = NodeMemory()
    assert isinstance(s, StoreView) and isinstance(s, Lifecycle)


def test_registration_encodes_in_place_and_logs_each_region():
    s = NodeMemory()
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
    t = NodeMemory()
    weights(t)
    weights(t, "w2", 2, seed=1)
    assert t.registration() == r2
    assert s.layout().bytes_by_tier == {"hbm": 8 * BLOCK}


def test_registration_refuses_partial_blocks_and_unknown_tiers():
    s = NodeMemory()
    with pytest.raises(ValueError):
        s.register("x", "hbm", "weights", ArrayBacking(np.zeros(BLOCK + WORD, np.uint8)), block_bytes=BLOCK,
                   codec=CODEC, receipt=RECEIPT, source="x")
    with pytest.raises(ValueError):
        s.register("x", "nvme", "filler", ArrayBacking(np.zeros(BLOCK, np.uint8)), block_bytes=BLOCK, codec=CODEC,
                   receipt=RECEIPT, source="x")


def test_page_layouts_page_major_and_plane_major():
    s = NodeMemory()
    pool(s, layers=1, pages=3, slots=8, slot_bytes=64)
    p = s.pool
    assert p.ids.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    assert (p.lo.tolist(), p.hi.tolist()) == ([0, 4, 0, 4], [4, 8, 4, 8])
    t = NodeMemory()
    pool(t, layers=1, pages=3, slots=8, slot_bytes=64, page_stride=512, plane_stride=3 * 512)
    assert t.pool.ids.tolist() == [[0, 1, 6, 7], [2, 3, 8, 9], [4, 5, 10, 11]]


def test_kv_life_alloc_append_free_without_reencoding():
    s = NodeMemory()
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
    s = NodeMemory()
    pool(s)
    s.kv_alloc([1])
    s.kv_write_reference([Append(1, 0, 6)], kv_payload, step=0)
    s.kv_free([1])
    idle = set(s.idle_blocks("kv/0").tolist())
    assert {4, 6} <= idle and not {5, 7} & idle
    s.kv_alloc([1])
    assert set(s.idle_blocks("kv/0").tolist()) >= {4, 6}


def _window(s: NodeMemory, sigma: bytes, steps):
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
        s = NodeMemory()
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
    s = NodeMemory()
    weights(s)
    _, w1 = _window(s, SIGMA, [((), None), ((), None)])
    _, w2 = _window(s, SIGMA2, [((), None)])
    assert w2.passes[0].seed == chain(SIGMA2, w1.passes[-1].root, 0)
    with pytest.raises(ValueError):
        s.begin_window(b"sigma")


def test_the_sessions_first_window_chains_from_the_verifiers_registration_digest():
    """The driver's `Registration.digest`, not the store's own log root, is the root before the session's first pass;
    a later window takes none."""
    from verity.core.protocols.remote_bandwidth import audit
    s = NodeMemory()
    weights(s)
    digest = audit.register(bytes(32), b"node", s.layout().blocks, bytes(64), nonce=bytes(32)).digest
    s.begin_window(SIGMA, digest)
    s.begin_pass()
    s.end_pass()
    w = s.end_window()
    assert w.passes[0].seed == rb.seed(SIGMA, 0, digest) != chain(SIGMA, s.registration().root, 0)
    with pytest.raises(ValueError, match="first window"):
        s.begin_window(SIGMA2, digest)


def test_a_plan_is_its_runs_each_block_once():
    s = NodeMemory(filler_plan=lambda st, j: [("kv/0", [0, 1, 5])])
    weights(s)
    weights(s, "w2", 2, seed=1)
    pool(s)
    s.kv_alloc([2])
    s.kv_write_reference([Append(2, 0, 8)], kv_payload, step=0)
    (pc,), w = _window(s, SIGMA, [([2], None)])
    runs = [(0, 8), (16, 4), (32, 4), (8, 2), (13, 1)]
    assert s.plan(0).tolist() == [b for f, c in runs for b in range(f, f + c)]
    want = b"verity/remote-bandwidth/plan/v0\0" + struct.pack(">I", 5) + b"".join(struct.pack(">QQ", *r) for r in runs)
    assert pc.plan_digest == hashlib.sha512(want).digest() == rb.Plan(tuple(runs), s.layout().blocks).digest
    t = NodeMemory(filler_plan=lambda st, j: [("kv/0", [0, 8])])
    pool(t)
    t.kv_alloc([2])
    t.kv_write_reference([Append(2, 0, 8)], kv_payload, step=0)
    t.begin_window(SIGMA)
    with pytest.raises(ValueError, match="reads a block twice"):
        t.begin_pass([2])


def test_outside_a_window_a_pass_commits_nothing():
    s = NodeMemory()
    weights(s)
    s.begin_pass()
    assert s.end_pass() is None
    assert s.counters.calls == {"register": 1, "begin_pass": 1, "end_pass": 1}


def test_openings_verify_and_a_tampered_one_does_not():
    host = ArrayBacking(np.zeros(4 * BLOCK, np.uint8))
    s = NodeMemory(filler_plan=lambda st, j: [("dram", range(4)), ("kv/1", st.idle_blocks("kv/1")[:3])])
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
    s = NodeMemory()
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


def test_a_windows_openings_outlast_the_next_windows_start():
    """The driver's openings coin for window t comes once window t + 1's first root is in, so window t's passes stay
    open through window t + 1 and go when window t + 2 begins."""
    s = NodeMemory()
    _, payload = weights(s)
    _, w0 = _window(s, SIGMA, [((), None), ((), None)])
    _, w1 = _window(s, SIGMA2, [((), None)])
    assert (w0.index, w1.index) == (0, 1)
    o = s.open([(1, 2)], window=0)[0]
    assert o.pass_index == 1 and o.receipt != s.open([(0, 2)])[0].receipt
    want = payload[2 * BLOCK:3 * BLOCK]
    assert verify_opening(s.region("w"), w0.passes[1], s.plan(1, window=0), o, RECEIPT, CODEC.decode, want)
    assert not verify_opening(s.region("w"), w1.passes[0], s.plan(0), o, RECEIPT, CODEC.decode, want)
    s.begin_window(bytes(32))
    with pytest.raises(KeyError, match="window 0"):
        s.open([(1, 2)], window=0)
    assert s.plan(0, window=1).tolist() == list(range(6))


def _aligned_store() -> tuple[NodeMemory, dict[str, ArrayBacking]]:
    """Weights of 40 and 20 blocks and two KV layers of 4 pages of 4 blocks: whole subtrees of 2, 1, 1 and 1 leaves,
    then a tail of 8 and 4 blocks."""
    s = NodeMemory(salt_key=bytes(32))
    w, _ = weights(s, "w", 40)
    w2, _ = weights(s, "w2", 20, seed=1)
    bs, _ = pool(s)
    s.kv_alloc([0, 1, 2, 3])
    s.kv_write_reference([Append(p, 0, 8) for p in range(4)], kv_payload, step=0)
    return s, {"w": w, "w2": w2, "kv/0": bs[0], "kv/1": bs[1]}


def _driver_root(s: NodeMemory, backings: dict[str, ArrayBacking], pc, receipt=RECEIPT, z=None) -> bytes:
    """The pass's root as the driver builds it (`rb.tree`): every receipt in plan order, sixteen to a hidden leaf."""
    plan, keyed, lay = s.plan(pc.index).tolist(), receipt.keyed(pc.seed), s.layout()
    ys = []
    for j in plan:
        info = next(r for r in lay.regions if r.first <= j < r.first + r.blocks)
        bb = info.block_bytes
        ys.append(keyed(j, backings[info.name].read((j - info.first) * bb, bb), z(info.name, j - info.first) if z else None))
    p = rb.Plan.of(plan, lay.blocks)
    return rb.tree(pc.seed, p, ys, [s.salt(pc.seed, c) for c in range(rb.leaves(p.length))]).commitment.root


def test_a_pass_lays_its_segments_out_as_subtrees_of_the_drivers_tree():
    from verity_vllm.protocols.remote_bandwidth.registry import aligned
    s, backings = _aligned_store()
    (pc,), _ = _window(s, SIGMA, [([0, 1, 2, 3], None)])
    got = [(x.region, x.count, x.position, aligned(x)) for x in pc.segments]
    assert got == [("w", 32, 0, True), ("w2", 16, 32, True), ("kv/0", 16, 48, True), ("kv/1", 16, 64, True),
                   ("w", 8, 80, False), ("w2", 4, 88, False)]
    plan = s.plan(0).tolist()
    assert plan == [*range(32), *range(40, 56), *range(60, 76), *range(76, 92), *range(32, 40), *range(56, 60)]
    assert pc.root == _driver_root(s, backings, pc)


def test_under_the_drivers_receipt_the_pass_is_the_drivers_byte_for_byte():
    """4 KiB blocks under `RemoteBandwidthReceipt`, bound to a z per block: 77 positions, so a short last leaf and three
    pads, and the root `rb.tree` gives over `rb.Receipter`'s receipts."""
    s, rec = NodeMemory(salt_key=bytes(32)), RemoteBandwidthReceipt()
    backings = {}
    for name, n in [("w", 40), ("w2", 37)]:
        backings[name] = ArrayBacking(np.random.default_rng(n).integers(0, 256, n * rb.BLOCK_BYTES, np.uint8))
        s.register(name, "hbm", "weights", backings[name], block_bytes=rb.BLOCK_BYTES, codec=CODEC, receipt=rec,
                   source=name)

    def z(region: str, b: int) -> bytes:
        return hashlib.sha512(region.encode() + struct.pack(">Q", b)).digest()

    s.begin_window(SIGMA)
    s.begin_pass()
    pc = s.end_pass(z=z)
    assert [(x.region, x.count) for x in pc.segments] == [("w", 32), ("w2", 32), ("w", 8), ("w2", 5)]
    assert pc.root == _driver_root(s, backings, pc, rec, z)
    for o in s.open([(0, 40), (0, 76)]):
        info = s.region(o.region)
        assert verify_opening(info, pc, s.plan(0), o, rec, CODEC.decode, None, z(o.region, o.block - info.first))


def _opened(**end) -> NodeMemory:
    s, _ = _aligned_store()
    s.begin_window(SIGMA)
    s.begin_pass([0, 1, 2, 3])
    s.end_pass(**end)
    return s


def test_a_device_reader_hands_in_its_aligned_segments_roots_or_the_root():
    ref = _opened()
    pc = ref.end_window().passes[0]
    s = _opened(roots={k: pc.segment_roots[k] for k in range(4)})
    given = s.end_window().passes[0]
    assert (given.root, given.segment_roots) == (pc.root, pc.segment_roots)
    assert s.open([(0, 3)]) == ref.open([(0, 3)])
    assert _opened(roots={1: bytes(64)}).end_window().passes[0].root != pc.root
    t, _ = _aligned_store()
    t.begin_window(SIGMA)
    t.begin_pass([0, 1, 2, 3])
    with pytest.raises(ValueError):
        t.end_pass(roots={4: bytes(64)})
    assert t.in_pass
    whole = _opened(root=b"\x07" * 64)
    (pw,) = whole.end_window().passes
    assert (pw.root, pw.segment_roots) == (b"\x07" * 64, ())
    with pytest.raises(RuntimeError):
        whole.open([(0, 0)])


def test_openings_of_subtrees_and_the_tail_verify_as_the_drivers():
    s, _ = _aligned_store()
    (pc,), _ = _window(s, SIGMA, [([0, 1, 2, 3], None)])
    plan, lay = s.plan(0), s.layout()
    driver_plan = rb.Plan.of(plan.tolist(), lay.blocks)
    for o in s.open([(0, q) for q in range(pc.blocks)]):
        info = next(r for r in lay.regions if r.name == o.region)
        assert verify_opening(info, pc, plan, o, RECEIPT, CODEC.decode, None)
        assert rb.verify_leaf(pc.seed, driver_plan, pc.root, o.position, b"".join(o.leaf_receipts), o.salt,
                              o.leaf_path + o.segment_path)
    o = s.open([(0, 85)])[0]
    assert not verify_opening(s.region("w"), pc, plan, o.__class__(**{**o.__dict__, "salt": bytes(192)}), RECEIPT,
                              CODEC.decode, None)


@pytest.mark.parametrize("kind", rb.KINDS)
def test_identity_codec_and_the_drivers_receipt_kinds_of_the_block_alone(kind):
    """`KindReceipt` (the binder's default without tiles, at sketch-sha512): a receipt is `rb.receipter`'s of the block
    under the whole 64-byte pass seed, a z it is handed left out; the pass's root is the driver's (`rb.tree`), and an
    opening verifies."""
    s, rec = NodeMemory(salt_key=bytes(32)), KindReceipt(kind)
    raw = np.random.default_rng(3).integers(0, 256, 3 * rb.BLOCK_BYTES, np.uint8)
    b = ArrayBacking(raw.copy())
    s.register("w", "hbm", "weights", b, block_bytes=rb.BLOCK_BYTES, codec=IDENTITY, receipt=rec, source="rng")
    assert b.read(0, b.nbytes) == raw.tobytes() and s.region("w").receipt == kind
    (pc,), _ = _window(s, SIGMA, [((), None)])
    block = raw[rb.BLOCK_BYTES:2 * rb.BLOCK_BYTES].tobytes()
    want = rb.receipter(kind, pc.seed)(1, block)
    assert len(pc.seed) == 64 and len(want) == rb.DIGEST_BYTES == rec.receipt_bytes
    assert rec.keyed(pc.seed)(1, block, b"z") == want
    assert pc.root == _driver_root(s, {"w": b}, pc, rec)
    o = s.open([(0, 1)])[0]
    assert o.receipt == want
    assert verify_opening(s.region("w"), pc, s.plan(0), o, rec, IDENTITY.decode, block)


def test_a_receipt_kind_the_driver_does_not_name_is_refused():
    with pytest.raises(ValueError, match="a receipt kind is one of"):
        KindReceipt("ro-blake3/v1")
