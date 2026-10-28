"""The store's memory map (memory-map/v1) from its KV events: a serving-like trace of allocs, appends that fill pages
over several steps, frees of written and part-written pages, re-allocation and overwrite.  The driver's
`memory_map.parse` and `Replay` accept every step, the live slots and the blocks they hold are the store's own (a
dirty or released block is in no slot, and in no plan), each digest is the payload's, a free lands in the step whose
forward ran last and frees each block once, a step's line closes when the next opens, indices run from N in record
order, and a block as old as the lifetime cap is re-keyed before the first pass that could not hold it.  In a session
(deferred residency) a step's completed blocks go out as one append, and the line open when its encode lands writes
them: one freed before then is written there and freed a line later, a re-key frees its old index at once, and an
encode that would write a slot its line writes already waits a line."""
from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest

from verity.core.protocols.poub import audit, codec, memory_map
from verity.core.protocols.poub.memory_map import MemoryMapError, Replay
from verity_vllm.protocols.poub.interface import Append, PageLayout
from verity_vllm.protocols.poub.memory_map import LIFETIME
from verity_vllm.protocols.poub.registry import ArrayBacking, Store

from .test_registry import CODEC, RECEIPT

BLOCK = codec.BLOCK_BYTES
LAYERS, PAGES, SLOTS, SLOT_BYTES = 2, 4, 8, 1024     # a plane-page is 2 blocks of 4 slots; a page 4 blocks (K0 K1 V0 V1)


def kv_payload(layer: int, page: int, slot: int, life: int = 0) -> bytes:
    """A slot's K and V planes, 1,024 bytes each; ``life`` tells a page's rewrites apart."""
    return b"".join(hashlib.sha512(struct.pack(">QQQQQ", layer, page, slot, life, u)).digest()
                    for u in range(2 * SLOT_BYTES // 64))


def store(s: Store | None = None) -> tuple[Store, list[ArrayBacking]]:
    """Weights of 3 blocks, the KV pool's two layers of 16 blocks each, then 2 blocks of host filler: N = 37."""
    s = Store() if s is None else s
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
    ((2,), (0,), [Append(0, 0, 8)], 1),                     # page 2 (written whole) freed; page 0, filler, written
]


def run_trace(s: Store, m, check=None) -> list[memory_map.Step]:
    """The trace, each step's frees (a finished request's after the step before, a preempted one's in this step's
    schedule) and allocs, then its write; ``check(step, lives, new, pages)`` after each write, with the lines it closed
    and the pages it wrote, whose blocks are in the line it opened; then the last line closed."""
    seen, lives = 0, {}
    for step, (free, alloc, appends, life) in enumerate(TRACE + [((), (), [], 0)]):
        if step == len(TRACE):
            m.flush()
        else:
            before = m.text()
            if free:
                s.kv_free(list(free))
            if alloc:
                s.kv_alloc(list(alloc))
            assert m.text() == before, "a free is in the open line, and an alloc in none"
            s.kv_write_reference(appends, lambda li, p, x, life=life: kv_payload(li, p, x, life), step)
            lives[step] = life
        _, steps = memory_map.parse(m.text().splitlines())
        steps = list(steps)
        new, seen = steps[seen:], len(steps)
        if check is not None:
            check(step, lives, new, {a.page for a in appends} if step < len(TRACE) else set())
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

    def check(step: int, lives: dict, new: list[memory_map.Step], open_pages: set) -> None:
        assert [x.step for x in new] == ([] if step in (0, 5) else [step - 1] if step < 7 else [6])
        for x in new:
            changes.append(replay.apply(x))
            assert list(x.freed) == sorted(x.freed) and [w.slot for w in x.written] == sorted(w.slot for w in x.written)
            for w in x.written:
                page = int(p.page_of[(w.slot - 3) % 16])
                assert w.digest == codec.digest(s.payload(w.slot)) == codec.digest(
                    expected_payload(s, w.slot, page, lives[x.step]))
                index_of[(w.slot, x.step)] = w.index
        for slot, page, c in pool_blocks(s):
            held = replay.holds(slot)
            if page in open_pages:
                continue
            if p.dirty[page, c]:
                assert held is None, f"step {step}: slot {slot} is dirty in the store and live in the map"
            elif p.step[page, c] < 0:
                assert held == slot
            else:
                assert held == index_of[(slot, int(p.step[page, c]))]
        assert all(replay.holds(x) == x for x in (0, 1, 2, 35, 36))

    steps = run_trace(s, m, check)
    assert [x.step for x in steps] == [0, 1, 2, 3, 5, 6]
    written = [w for x in steps for w in x.written]
    assert [w.index for w in written] == list(range(37, 37 + len(written)))
    # K0 K1 V0 V1 of page p in layer l are store blocks 3 + 16 l + 4 p + (0, 1, 2, 3)
    l0, l1 = (lambda p, *cs: [3 + 4 * p + c for c in cs]), (lambda p, *cs: [19 + 4 * p + c for c in cs])
    whole = lambda p: tuple(l0(p, 0, 1, 2, 3) + l1(p, 0, 1, 2, 3))
    assert [(x.freed, tuple(w.slot for w in x.written)) for x in steps] == [
        (tuple(l0(2, 0, 2) + l1(2, 0, 2)), whole(1)),
        (tuple(l0(2, 1, 3) + l1(2, 1, 3)), tuple(l0(2, 0, 2) + l1(2, 0, 2))),
        (tuple(sorted(whole(1) + tuple(l0(2, 0, 2) + l1(2, 0, 2)))), ()),
        ((), whole(2)),
        (whole(2), whole(1)),
        ((), whole(0)),
    ]
    # step 0 releases the filler page 1 held (written) and the filler page 2's K0 V0 held (reached); step 1 page 2's
    # K1 V1 filler (reached); step 2, whose forward read them last, every block of pages 1 and 2 but page 2's dirty
    # K1 V1 (freed by kv_free, once); step 3 nothing (page 2's released blocks are rewritten); step 5 page 2 (freed
    # before step 6's write) and nothing page 1's write overwrote; step 6 page 0's filler (written)
    assert changes[0].released == ((7, 5), (13, 1), (23, 5), (29, 1)) and (changes[0].first, changes[0].count) == (37, 8)
    assert changes[1].released == ((12, 1), (14, 1), (28, 1), (30, 1))
    assert changes[2].released == ((37, 12),) and changes[2].count == 0
    assert changes[3].released == ()
    assert changes[4].released == ((49, 8),)
    assert changes[5].released == ((3, 4), (19, 4))


def test_a_freed_page_is_released_out_of_every_plan_until_written_again():
    """The audit credits only blocks the circuit keeps live: a freed page's blocks leave the filler and the pass's plan
    (one that names them is refused) until a write completes them under a new index."""
    s = Store(filler_plan=lambda st, j: [("kv/0", st.idle_blocks("kv/0"))])
    store(s)
    reg = registration(s)
    s.record_memory_map(registration=reg.digest)
    s.kv_alloc([1])
    s.kv_write_reference([Append(1, 0, 8)], kv_payload, 0)
    page1 = {3 + 4 + c for c in range(4)}
    s.kv_free([1])
    assert not page1 & set((s.region("kv/0").first + s.idle_blocks("kv/0")).tolist())
    s.begin_window(bytes(32), reg.digest)
    s.begin_pass(None)
    assert s.end_pass() is not None and not page1 & set(s.plan(0).tolist())
    bad = Store(filler_plan=lambda st, j: [("kv/0", [4, 5])])
    store(bad)
    bad.record_memory_map(registration=registration(bad).digest)
    bad.kv_alloc([1])
    bad.kv_write_reference([Append(1, 0, 8)], kv_payload, 0)
    bad.kv_free([1])
    with pytest.raises(ValueError, match="dirty or released"):
        bad.begin_pass(None)
    s.kv_alloc([1])
    s.kv_write_reference([Append(1, 0, 8)], kv_payload, 1)
    s.begin_pass(None)
    s.end_pass()
    assert page1 <= set(s.plan(1).tolist())


def test_the_first_window_chains_from_the_registration_the_map_names():
    s, _ = store()
    reg = registration(s)
    with pytest.raises(ValueError, match="Registration.digest"):
        s.record_memory_map()
    s.record_memory_map(registration=reg.digest)
    with pytest.raises(ValueError, match="registration digest its memory map names"):
        s.begin_window(bytes(32))
    with pytest.raises(ValueError, match="registration digest its memory map names"):
        s.begin_window(bytes(32), bytes(64))
    s.begin_window(bytes(32), reg.digest)


def test_a_dirty_block_is_in_no_step_until_a_write_completes_it():
    """A dirty block is freed when a write first reaches it, so a `kv_free` of its page frees only its appended
    blocks; its untouched ones stay resident."""
    s, _ = store()
    m = s.record_memory_map(registration=registration(s).digest)
    s.kv_alloc([0, 1])
    s.kv_write_reference([Append(0, 0, 3)], kv_payload, 0)
    s.kv_write_reference([Append(0, 3, 1)], kv_payload, 1)
    s.kv_write_reference([Append(1, 0, 2)], kv_payload, 2)
    s.kv_free([0, 1])
    assert m.text().count("\n") == 3
    s.kv_append([], 3)
    m.flush()
    _, steps = memory_map.parse(m.text().splitlines())
    first, second, third = list(steps)
    assert (first.freed, first.written) == ((3, 5, 19, 21), ())
    assert first.step == 0 and [w.slot for w in second.written] == [3, 5, 19, 21] and second.freed == ()
    assert (third.step, third.written) == (2, ())
    # page 0's K0 V0 (appended at step 1), page 1's K0 V0 (reached at step 2); both pages' K1 V1 are untouched filler
    assert third.freed == (3, 5, 7, 9, 19, 21, 23, 25)
    assert all(m.replay.holds(x) == x for x in (4, 6, 8, 10, 20, 22, 24, 26))


def test_on_write_sees_each_registered_block_a_write_reaches_before_its_bytes_change():
    """`Store.on_write`: a step's `kv_append` hands it the pool blocks the step's write reaches while they are
    registered (a pass's plan may still hold them), before the forward writes them; a block already reached and not
    complete (dirty) is not handed again."""
    s, _ = store()
    seen = []
    slot = lambda k: s.region("kv/0").first + int(s.pool.ids[k // s.pool.per_page, k % s.pool.per_page])
    s.on_write = lambda step, keys: seen.append((step, keys.tolist(), [s.payload(slot(k)) for k in keys]))
    s.record_memory_map(registration=registration(s).digest)
    s.kv_alloc([0])
    s.kv_append([Append(0, 0, 3)], 0)
    s.kv_append([Append(0, 3, 3)], 1)
    s.kv_append([Append(0, 6, 2)], 2)
    p = s.pool
    k0, k1 = [c for c in range(p.per_page) if p.lo[c] == 0], [c for c in range(p.per_page) if p.lo[c] == 4]
    assert [(st, keys) for st, keys, _ in seen] == [(0, k0), (1, k1), (2, [])]
    assert all(b == bytes(BLOCK) for b in seen[0][2])


def test_on_write_sees_a_freed_pages_blocks_when_a_write_reaches_them_again():
    """A freed page's appended blocks are released (dirty) with their bytes as they were, and the plan of the step before
    the free still holds them: the page's next write hands them to `on_write` before it changes them, once."""
    s, _ = store()
    seen = []
    s.on_write = lambda step, keys: seen.append((step, keys.tolist()))
    s.record_memory_map(registration=registration(s).digest)
    s.kv_alloc([0])
    s.kv_append([Append(0, 0, SLOTS)], 0)
    s.end_step()
    s.kv_free([0])
    s.kv_alloc([0])
    s.kv_append([Append(0, 0, 3)], 1)
    s.kv_append([Append(0, 3, 3)], 2)
    p = s.pool
    k0, k1 = [c for c in range(p.per_page) if p.lo[c] == 0], [c for c in range(p.per_page) if p.lo[c] == 4]
    assert seen == [(0, list(range(p.per_page))), (1, k0), (2, k1)]
    assert not p.released.any()


def test_digests_are_taken_once_the_steps_bytes_are_in_place():
    """`kv_append` before the bytes land (a device codec writes them in the step's forward): the step is recorded, and
    its digests are those of the bytes at the next `kv_append` or `flush`."""
    s, bs = store()
    m = s.record_memory_map(registration=registration(s).digest)
    s.kv_alloc([3])
    s.kv_append([Append(3, 0, 4)], 0)
    assert m.text().count("\n") == 1
    bs[0].write(3 * 2 * SLOTS * SLOT_BYTES, b"\x07" * BLOCK)
    s.kv_append([], 1)
    (step,) = memory_map.parse(m.text().splitlines())[1]
    assert step.step == 0 and step.written[0].slot == 3 + 12
    assert step.written[0].digest == codec.digest(CODEC.decode(b"\x07" * BLOCK, s.region("kv/0").word_index(12)))
    assert m.flush() == []


def test_a_live_block_is_re_keyed_once_its_window_is_lifetime_old():
    """Under L = 2, page 0, written at step 0 before window 0 and never freed, is written again in its slots with its
    digests at fresh indices by the line that closes in window 2, the first whose passes could not hold it; page 1,
    freed in that line, is not."""
    s, _ = store()
    reg = registration(s)
    m = s.record_memory_map(registration=reg.digest, lifetime=2)
    s.kv_alloc([0, 1])
    s.kv_write_reference([Append(0, 0, 8), Append(1, 0, 8)], kv_payload, 0)
    for w, step in ((0, 1), (1, 2), (2, 3)):
        if w:
            s.end_window()
        s.begin_window(bytes(32), reg.digest if w == 0 else None)
        if step == 3:
            s.kv_free([1])
        s.kv_append([], step)
    m.flush()
    first, rekey = memory_map.parse(m.text().splitlines())[1]
    page0, page1 = [3, 4, 5, 6, 19, 20, 21, 22], [7, 8, 9, 10, 23, 24, 25, 26]
    assert (first.step, [w.slot for w in first.written]) == (0, sorted(page0 + page1))
    assert (rekey.step, rekey.freed, [w.slot for w in rekey.written]) == (2, tuple(page1), page0)
    assert [w.digest for w in rekey.written] == [w.digest for w in first.written if w.slot in page0]
    assert [w.index for w in rekey.written] == list(range(37 + 16, 37 + 24))
    assert m.rekeyed == 8 and m.joined == dict.fromkeys(page0, 2)
    assert all(m.replay.holds(w.slot) == w.index for w in rekey.written)


def test_a_block_written_and_freed_in_one_step_is_no_write():
    """A write that completes a block and a free of its page in the same step: the line drops the write, and frees
    the block only if its slot held one (a reached block was freed when the write first reached it)."""
    s, _ = store()
    m = s.record_memory_map(registration=registration(s).digest)
    s.kv_alloc([0])
    s.kv_write_reference([Append(0, 0, 3)], kv_payload, 0)
    s.kv_write_reference([Append(0, 3, 1)], kv_payload, 1)
    s.kv_free([0])
    m.flush()
    (first,) = memory_map.parse(m.text().splitlines())[1]
    assert (first.step, first.freed, first.written) == (0, (3, 5, 19, 21), ()) and m.last == 1
    # step 1's line is empty: K0 V0's write is dropped, and K1 V1, which no write reached, stay resident filler
    assert all(m.replay.holds(x) is None for x in (3, 5)) and all(m.replay.holds(x) == x for x in (4, 6))
    assert m.joined == {} and not s.pool.dirty[0, [1, 3]].any()


def test_a_freed_pages_untouched_blocks_stay_resident_until_a_write_reaches_them():
    """`kv_free` frees only a page's appended blocks: its untouched filler (registered before the seed, never written
    since) stays resident, in the filler a plan may read, until a write first reaches it (freed there, if the write
    leaves it part written) or completes it (released by that write, as the slot's previous block)."""
    s, _ = store()
    m = s.record_memory_map(registration=registration(s).digest)
    s.kv_alloc([1])
    s.kv_write_reference([Append(1, 0, 4)], kv_payload, 0)          # page 1's K0 V0 whole
    s.kv_append([], 1)
    s.kv_free([1])
    s.kv_append([], 2)
    k1v1 = [8, 10, 24, 26]
    assert all(m.replay.holds(x) == x for x in k1v1) and {5, 7} <= set(s.idle_blocks("kv/0").tolist())
    assert not {4, 6} & set(s.idle_blocks("kv/0").tolist())
    s.kv_alloc([1])
    s.kv_write_reference([Append(1, 0, 6)], kv_payload, 3)          # K0 V0 again; K1 V1 reached, left part written
    s.kv_write_reference([Append(1, 6, 2)], kv_payload, 4)          # K1 V1 completed
    m.flush()
    steps = list(memory_map.parse(m.text().splitlines())[1])
    assert [(x.step, x.freed, tuple(w.slot for w in x.written)) for x in steps] == [
        (0, (), (7, 9, 23, 25)), (1, (7, 9, 23, 25), ()), (3, tuple(k1v1), (7, 9, 23, 25)), (4, (), tuple(k1v1))]
    replay = Replay(memory_map.parse(m.text().splitlines())[0])
    changes = [replay.apply(x) for x in steps]
    assert changes[0].released == ((7, 1), (9, 1), (23, 1), (25, 1)) and changes[2].released == ((8, 1), (10, 1),
                                                                                                    (24, 1), (26, 1))


def test_each_pass_names_the_step_kv_append_opened_last():
    s, _ = store()
    reg = registration(s)
    s.record_memory_map(registration=reg.digest)
    s.begin_window(bytes(32), reg.digest)
    s.kv_append([], 0)
    for bad in (1, -1):
        with pytest.raises(ValueError, match="kv_append opened last"):
            s.begin_pass(None, step=bad)
    s.begin_pass(None, step=0)
    assert s.end_pass().step == 0
    s.kv_append([], 1)
    with pytest.raises(ValueError, match="above the last pass's 0"):
        s.begin_pass(None, step=0)
    s.begin_pass(None, step=1)
    assert s.end_pass().step == 1


def test_the_map_goes_to_the_provers_file(tmp_path):
    s, _ = store()
    m = s.record_memory_map(tmp_path / "memory-map.jsonl", registration(s).digest)
    steps = run_trace(s, m)
    assert m.text() == (tmp_path / "memory-map.jsonl").read_text() == memory_map.dumps(m.header, steps)


def test_the_map_needs_4096_byte_blocks_registered_first_and_increasing_steps():
    t = Store()
    t.register("w", "hbm", "weights", ArrayBacking(np.zeros(2048, np.uint8)), block_bytes=1024, codec=CODEC,
               receipt=RECEIPT, source="w")
    with pytest.raises(ValueError, match="block_bytes=4096"):
        t.record_memory_map(registration=bytes(64))
    s, _ = store()
    m = s.record_memory_map(registration=registration(s).digest)
    with pytest.raises(RuntimeError, match="already records"):
        s.record_memory_map(registration=bytes(64))
    with pytest.raises(RuntimeError, match="register regions first"):
        s.register("late", "host", "filler", ArrayBacking(np.zeros(BLOCK, np.uint8)), block_bytes=BLOCK, codec=CODEC,
                   receipt=RECEIPT, source="zeros")
    s.kv_alloc([0])
    s.kv_write_reference([Append(0, 0, 4)], kv_payload, 5)
    fill = s.pool.fill.copy()
    with pytest.raises(MemoryMapError, match="strictly increase"):
        s.kv_write_reference([Append(0, 4, 4)], kv_payload, 5)
    assert (s.pool.fill == fill).all() and m.last == 5


PAGE0, PAGE1 = [3, 4, 5, 6, 19, 20, 21, 22], [7, 8, 9, 10, 23, 24, 25, 26]


def deferred(s: Store, lifetime: int = LIFETIME):
    """A map in a session: each append to ``sent`` as (first, digests, payloads), its encode landing when the test
    says (`MemoryMap.encoded`)."""
    sent: list = []
    m = s.record_memory_map(registration=registration(s).digest, lifetime=lifetime,
                            on_append=lambda first, slots, digests, payloads: sent.append((first, digests, payloads)))
    return m, sent


def lines(m) -> list[tuple[int, tuple[int, ...], list[tuple[int, int]]]]:
    return [(x.step, x.freed, [(w.slot, w.index) for w in x.written]) for x in memory_map.parse(m.text().splitlines())[1]]


def test_a_completed_blocks_write_is_recorded_in_the_line_open_when_its_encode_lands():
    """Deferred residency: page 1, written whole at step 0, goes out as one append once the step is done (`complete`),
    the filler its blocks held freed at step 0, where the write reached them.  It is in no record while its coin is
    out (step 0's line closes without it), and the line open when its encode lands, step 1's, writes it at the
    append's indices with its digests.  Encodes land in their appends' order."""
    s, _ = store()
    m, sent = deferred(s)
    s.kv_alloc([1])
    s.kv_write_reference([Append(1, 0, 8)], kv_payload, 0)
    m.complete()
    m.complete()
    assert [(f, len(d)) for f, d, _ in sent] == [(37, 8)] and m.waiting == 8
    assert list(sent[0][1]) == [codec.digest(x) for x in sent[0][2]] == [codec.digest(s.payload(x)) for x in PAGE1]
    s.kv_append([], 1)
    with pytest.raises(RuntimeError, match="not the oldest awaiting its encode"):
        m.encoded(38)
    m.encoded(37)
    s.kv_append([], 2)
    assert lines(m) == [(0, tuple(PAGE1), []), (1, (), list(zip(PAGE1, range(37, 45))))]
    assert m.waiting == 0 and [m.replay.holds(x) for x in PAGE1] == list(range(37, 45)) and m.joined == dict.fromkeys(PAGE1, 0)


def test_a_block_freed_before_its_encode_lands_is_written_when_it_lands_and_freed_a_step_later():
    """Page 1, completed at step 0, is freed after step 1's forward, its last structural reader, before its encode lands
    in step 2's line: its write is recorded at step 2 all the same (its indices are the session's, and the record takes
    them in order), and its free in step 3's line, past that reader (memory-map/v1 would withdraw it, which the map
    doesn't emit yet).  Its slots are dirty meanwhile, so no write's reach frees them again."""
    s, _ = store()
    m, _ = deferred(s)
    s.kv_alloc([1])
    s.kv_write_reference([Append(1, 0, 8)], kv_payload, 0)
    m.complete()
    s.kv_append([], 1)
    s.kv_free([1])
    assert s.pool.dirty[1].all()
    s.kv_append([], 2)
    m.encoded(37)
    s.kv_append([], 3)
    s.kv_append([], 4)
    assert lines(m) == [(0, tuple(PAGE1), []), (2, (), list(zip(PAGE1, range(37, 45)))), (3, tuple(PAGE1), [])]
    assert all(m.replay.holds(x) is None for x in PAGE1) and m.joined == {}


def test_a_re_key_frees_the_old_index_at_once_and_writes_the_new_one_once_its_encode_lands():
    """Under L = 1, page 0, written in window 0, is re-keyed by the line that closes in window 1: that line frees its
    old indices, so window 1's passes hold none of them, and appends it again with the same digests; the line open
    when that append's encode lands writes it at the fresh indices."""
    s, _ = store()
    reg = registration(s)
    m, sent = deferred(s, lifetime=1)
    s.begin_window(bytes(32), reg.digest)
    s.kv_alloc([0])
    s.kv_write_reference([Append(0, 0, 8)], kv_payload, 0)
    m.complete()
    m.encoded(37)
    s.kv_append([], 1)
    s.end_window()
    s.begin_window(bytes(32))
    s.kv_append([], 2)
    assert [f for f, _, _ in sent] == [37, 45] and sent[1][1] == sent[0][1] and m.rekeyed == 8
    m.encoded(45)
    s.kv_append([], 3)
    assert lines(m) == [(0, (), list(zip(PAGE0, range(37, 45)))), (1, tuple(PAGE0), []),
                        (2, (), list(zip(PAGE0, range(45, 53))))]
    assert m.joined == dict.fromkeys(PAGE0, 1)


def test_two_encodes_landing_in_one_line_that_write_one_slot_are_written_a_line_apart():
    """Page 1, completed at step 0 and freed after step 1, is completed again by a new request at step 2, and both
    encodes land in step 2's line (as when a pass discarded at a window's boundary sends the first late): the line
    writes the first, the second waits for the next line, whose write releases the first (in place of its free)."""
    s, _ = store()
    m, sent = deferred(s)
    s.kv_alloc([1])
    s.kv_write_reference([Append(1, 0, 8)], kv_payload, 0)
    m.complete()
    s.kv_append([], 1)
    s.kv_free([1])
    s.kv_alloc([1])
    s.kv_write_reference([Append(1, 0, 8)], lambda li, p, x: kv_payload(li, p, x, 1), 2)
    m.complete()
    m.encoded(37)
    m.encoded(45)
    s.kv_append([], 3)
    s.kv_append([], 4)
    assert lines(m) == [(0, tuple(PAGE1), []), (2, (), list(zip(PAGE1, range(37, 45)))),
                        (3, (), list(zip(PAGE1, range(45, 53))))]
    assert sent[0][1] != sent[1][1] and [m.replay.holds(x) for x in PAGE1] == list(range(45, 53))


def test_a_block_left_part_written_before_the_map_began_is_registered_as_it_is_and_freed_where_a_write_reaches_it():
    """A page a run before the session left part written: the registration takes its bytes as they are, so the map
    begins with it registered, and the session's first write to it frees it there, at the step it reached it, not at
    the step its deferred write is recorded."""
    s, _ = store()
    s.kv_alloc([1])
    s.kv_write_reference([Append(1, 0, 2)], kv_payload, 0)
    assert s.pool.dirty[1, [0, 2]].all()
    m, _ = deferred(s)
    assert not s.pool.dirty.any() and all(m.replay.holds(x) == x for x in PAGE1)
    s.kv_write_reference([Append(1, 2, 6)], kv_payload, 1)
    m.complete()
    s.kv_append([], 2)
    m.encoded(37)
    s.kv_append([], 3)
    assert lines(m) == [(1, tuple(PAGE1), []), (2, (), list(zip(PAGE1, range(37, 45))))]
