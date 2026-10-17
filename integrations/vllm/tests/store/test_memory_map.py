"""The store's memory map (memory-map/v1) from its KV events: a serving-like trace of allocs, appends that fill pages
over several steps, a free of a part-written page, re-allocation and overwrite.  The driver's `memory_map.parse` and
`Replay` accept every step, the live slots and the blocks they hold are the store's own (a dirty block is in no
slot), each digest is the payload's, alloc and free emit nothing, and indices run from N in record order."""
from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest

from verity.core.protocols.remote_bandwidth import audit, codec, memory_map
from verity.core.protocols.remote_bandwidth.memory_map import MemoryMapError, Replay
from verity_vllm.store.interface import Append, PageLayout
from verity_vllm.store.registry import ArrayBacking, Store

from .test_registry import CODEC, RECEIPT

BLOCK = codec.BLOCK_BYTES
LAYERS, PAGES, SLOTS, SLOT_BYTES = 2, 4, 8, 1024     # a plane-page is 2 blocks of 4 slots; a page 4 blocks (K0 K1 V0 V1)


def kv_payload(layer: int, page: int, slot: int, life: int = 0) -> bytes:
    """A slot's K and V planes, 1,024 bytes each; ``life`` tells a page's rewrites apart."""
    return b"".join(hashlib.sha512(struct.pack(">QQQQQ", layer, page, slot, life, u)).digest()
                    for u in range(2 * SLOT_BYTES // 64))


def store() -> tuple[Store, list[ArrayBacking]]:
    """Weights of 3 blocks, the KV pool's two layers of 16 blocks each, then 2 blocks of host filler: N = 37."""
    s = Store()
    w = np.random.default_rng(0).integers(0, 256, 3 * BLOCK, np.uint8)
    s.register("w", "hbm", "weights", ArrayBacking(w), block_bytes=BLOCK, codec=CODEC, receipt=RECEIPT, source="w")
    lay = PageLayout(PAGES, SLOTS, SLOT_BYTES)
    bs = [ArrayBacking(np.zeros(PAGES * 2 * SLOTS * SLOT_BYTES, np.uint8)) for _ in range(LAYERS)]
    s.register_kv_pool("kv", bs, lay, block_bytes=BLOCK, codec=CODEC, receipt=RECEIPT)
    s.register("dram", "host", "filler", ArrayBacking(np.zeros(2 * BLOCK, np.uint8)), block_bytes=BLOCK, codec=CODEC,
               receipt=RECEIPT, source="zeros")
    return s, bs


def registration(s: Store) -> audit.Registration:
    return audit.register(bytes(32), b"node", s.layout().blocks, bytes(64), nonce=bytes(32))


def expected_payload(s: Store, slot: int, page: int, life: int) -> bytes:
    """The payload the trace wrote to a KV block: its plane's slots ``[lo, hi)``, in slot order."""
    p = s.pool
    for li, rn in enumerate(p.regions):
        b = slot - s.region(rn).first
        if 0 <= b < s.region(rn).blocks:
            c = int(p.slot_of[b])
            k = c // (p.per_page // 2)
            return b"".join(kv_payload(li, page, x, life)[k * SLOT_BYTES:(k + 1) * SLOT_BYTES]
                            for x in range(int(p.lo[c]), int(p.hi[c])))
    raise AssertionError(f"slot {slot} is not a KV block")


def pool_blocks(s: Store):
    """Each KV block: (store index, page, column)."""
    p = s.pool
    for rn in p.regions:
        first = s.region(rn).first
        for page in range(PAGES):
            for c in range(p.per_page):
                yield first + int(p.ids[page, c]), page, c


# (frees and allocs before the step, the step's appends, the page's life the payloads are of)
TRACE = [
    ((), (1, 2), [Append(1, 0, 8), Append(2, 0, 3)], 0),   # page 1 whole; page 2's K0 V0 reached, dirty
    ((), (), [Append(2, 3, 3)], 0),                         # page 2's K0 V0 complete, K1 V1 reached
    ((), (), [], 0),                                        # a step that writes nothing
    ((1, 2), (2,), [Append(2, 0, 8)], 1),                   # page 2 freed part-written, reused: all four written
    ((), (1,), [Append(1, 0, 2)], 1),                       # page 1 (written at step 0) reused: K0 V0 reached
    ((), (), [Append(1, 2, 6)], 1),                         # page 1's K0 V0 into empty slots, K1 V1 over step 0's
]


def run_trace(s: Store, m, check=None) -> list[memory_map.Step]:
    seen = 0
    for step, (free, alloc, appends, life) in enumerate(TRACE):
        before = m.text()
        if free:
            s.kv_free(list(free))
        if alloc:
            s.kv_alloc(list(alloc))
        assert m.text() == before, "kv_free and kv_alloc emit nothing"
        s.kv_write_reference(appends, lambda li, p, x, life=life: kv_payload(li, p, x, life), step)
        _, steps = memory_map.parse(m.text().splitlines())
        steps = list(steps)
        new, seen = steps[seen:], len(steps)
        if check is not None:
            check(step, life, new)
    return steps


def test_a_serving_trace_is_a_memory_map_the_driver_replays_as_the_store_holds_it():
    s, _ = store()
    reg = registration(s)
    m = s.record_memory_map(registration=reg.digest)
    head = memory_map.parse(m.text().splitlines())[0]
    assert head == memory_map.Header(reg.digest, 37) and m.text().splitlines()[0] == (
        f'{{"block_bytes":4096,"blocks":37,"registration":"{reg.digest.hex()}","schema":"verity/memory-map/v1"}}')
    replay, p = Replay(head, reg), s.pool
    index_of: dict[tuple[int, int], int] = {}
    changes = []

    def check(step: int, life: int, new: list[memory_map.Step]) -> None:
        assert [x.step for x in new] == ([] if step == 2 else [step])
        for x in new:
            changes.append(replay.apply(x))
            assert list(x.freed) == sorted(x.freed) and [w.slot for w in x.written] == sorted(w.slot for w in x.written)
            for w in x.written:
                page = int(p.page_of[(w.slot - 3) % 16])
                assert w.digest == codec.digest(s.payload(w.slot)) == codec.digest(expected_payload(s, w.slot, page, life))
                index_of[(w.slot, step)] = w.index
        for slot, page, c in pool_blocks(s):
            held = replay.holds(slot)
            if p.dirty[page, c]:
                assert held is None, f"step {step}: slot {slot} is dirty in the store and live in the map"
            elif p.step[page, c] < 0:
                assert held == slot
            else:
                assert held == index_of[(slot, int(p.step[page, c]))]
        assert all(replay.holds(x) == x for x in (0, 1, 2, 35, 36))

    steps = run_trace(s, m, check)
    assert [x.step for x in steps] == [0, 1, 3, 4, 5]
    written = [w for x in steps for w in x.written]
    assert [w.index for w in written] == list(range(37, 37 + len(written)))
    # K0 K1 V0 V1 of page p in layer l are store blocks 3 + 16 l + 4 p + (0, 1, 2, 3)
    l0, l1 = (lambda p, *cs: [3 + 4 * p + c for c in cs]), (lambda p, *cs: [19 + 4 * p + c for c in cs])
    assert [(x.freed, tuple(w.slot for w in x.written)) for x in steps] == [
        (tuple(l0(2, 0, 2) + l1(2, 0, 2)), tuple(l0(1, 0, 1, 2, 3) + l1(1, 0, 1, 2, 3))),
        (tuple(l0(2, 1, 3) + l1(2, 1, 3)), tuple(l0(2, 0, 2) + l1(2, 0, 2))),
        ((), tuple(l0(2, 0, 1, 2, 3) + l1(2, 0, 1, 2, 3))),
        (tuple(l0(1, 0, 2) + l1(1, 0, 2)), ()),
        ((), tuple(l0(1, 0, 1, 2, 3) + l1(1, 0, 1, 2, 3))),
    ]
    # step 0 releases the filler page 1 held (written) and the filler page 2's K0 V0 held (freed); step 3 releases
    # step 1's K0 V0 (overwritten); step 5 releases step 0's K1 V1 of page 1
    assert changes[0].released == ((7, 5), (13, 1), (23, 5), (29, 1)) and (changes[0].first, changes[0].count) == (37, 8)
    assert changes[1].released == ((12, 1), (14, 1), (28, 1), (30, 1))
    assert changes[2].released == ((45, 4),)
    assert changes[3].released == ((37, 1), (39, 1), (41, 1), (43, 1))
    assert changes[4].released == ((38, 1), (40, 1), (42, 1), (44, 1))


def test_a_dirty_block_is_in_no_step_until_a_write_completes_it():
    s, _ = store()
    m = s.record_memory_map()
    s.kv_alloc([0])
    s.kv_write_reference([Append(0, 0, 3)], kv_payload, 0)
    s.kv_write_reference([Append(0, 3, 1)], kv_payload, 1)
    s.kv_free([0])
    _, steps = memory_map.parse(m.text().splitlines())
    first, second = list(steps)
    assert (first.freed, first.written) == ((3, 5, 19, 21), ())
    assert first.step == 0 and [w.slot for w in second.written] == [3, 5, 19, 21] and second.freed == ()
    assert m.text().count("\n") == 3


def test_digests_are_taken_once_the_steps_bytes_are_in_place():
    """`kv_append` before the bytes land (a device codec writes them in the step's forward): the step is recorded, and
    its digests are those of the bytes at the next `kv_append` or `flush`."""
    s, bs = store()
    m = s.record_memory_map()
    s.kv_alloc([3])
    s.kv_append([Append(3, 0, 4)], 0)
    assert m.text().count("\n") == 1
    bs[0].write(3 * 2 * SLOTS * SLOT_BYTES, b"\x07" * BLOCK)
    s.kv_append([], 1)
    (step,) = memory_map.parse(m.text().splitlines())[1]
    assert step.step == 0 and step.written[0].slot == 3 + 12
    assert step.written[0].digest == codec.digest(CODEC.decode(b"\x07" * BLOCK, s.region("kv/0").word_index(12)))
    assert m.flush() == []


def test_the_map_goes_to_the_provers_file(tmp_path):
    """Without the verifier's registration digest the header names the store's registration root when the map began."""
    s, _ = store()
    root = s.registration().root
    m = s.record_memory_map(tmp_path / "memory-map.jsonl")
    steps = run_trace(s, m)
    assert m.text() == (tmp_path / "memory-map.jsonl").read_text() == memory_map.dumps(m.header, steps)
    assert m.header.registration == root != s.registration().root


def test_the_map_needs_4096_byte_blocks_registered_first_and_increasing_steps():
    t = Store()
    t.register("w", "hbm", "weights", ArrayBacking(np.zeros(2048, np.uint8)), block_bytes=1024, codec=CODEC,
               receipt=RECEIPT, source="w")
    with pytest.raises(ValueError, match="block_bytes=4096"):
        t.record_memory_map()
    s, _ = store()
    m = s.record_memory_map()
    with pytest.raises(RuntimeError, match="already records"):
        s.record_memory_map()
    with pytest.raises(RuntimeError, match="register regions first"):
        s.register("late", "host", "filler", ArrayBacking(np.zeros(BLOCK, np.uint8)), block_bytes=BLOCK, codec=CODEC,
                   receipt=RECEIPT, source="zeros")
    s.kv_alloc([0])
    s.kv_write_reference([Append(0, 0, 4)], kv_payload, 5)
    fill = s.pool.fill.copy()
    with pytest.raises(MemoryMapError, match="strictly increase"):
        s.kv_write_reference([Append(0, 4, 4)], kv_payload, 5)
    assert (s.pool.fill == fill).all() and m.last == 5
