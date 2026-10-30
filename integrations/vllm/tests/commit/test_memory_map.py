"""memory-map/v1 from the circuit (`verity_vllm.commit.memory_map`) on a small Llama's request Programs: rows of 1 KiB (4 KV heads
of 128 bf16) in pages of 4 slots, so a plane of a page is one 4 KiB block; request r0 (a 4-row prefill, then 5 decode steps) from
step 0, and r1 (a 4-row prefill, then 1 decode step) from step 1, which ends while r0 runs.  The record is one
`remote_bandwidth.memory_map.Replay` accepts, a complete block is in it exactly while the requirement has its rows live, and the
pages are the ones the engine's slots name.  A warm request (a prefix-cache hit) reads its first page from the request that
computed it: the page is written once and freed after the last reader of either.  GLM-5.3's MLA: the DSA stand-in's rows are its
latent rows, one plane a layer, in MLA's latent page, and 1152-byte latent rows lay out 18 blocks a page of 64 slots, rows
straddling blocks.
"""
from __future__ import annotations

import copy
import hashlib

import pytest
import torch

from tests.program.predict.test_predict import TINY, _phi
from tests.program import test_dsa_derive as DSA
from tests.program.test_dsa_derive import served  # noqa: F401  (the DSA stand-in's fixture)
from verity.core.protocols.remote_bandwidth import codec
from verity.core.protocols.remote_bandwidth.memory_map import Replay, dumps, parse
from verity.ml.circuits.census import step_of
from verity.ml.circuits.defs import Builder, CompositeDefinition, bind
from verity.ml.circuits.program import Program, is_input
from verity.ml.circuits.refs import Affine, Coll, concat
from verity.ml.circuits.requirement import from_census
from verity.ml.circuits.types import Array, Record, Value
from verity_vllm.commit import memory_map as MM
from verity_vllm.program.predict.build import BF16, I32, build_program, decoder
from verity_vllm.program.predict.common import _Peers
from verity.ml.definitions.replay import replayer

WIDE = TINY | {"hidden_size": 512, "num_key_value_heads": 4}
LAYOUT = MM.KvPool(block_size=4)
REGISTRATION = bytes(range(64))
ARRIVE = {"r0": 0, "r1": 1}
#: pages in order of first need: r0's first (step 0), r0's second and r1's first (step 1), r1's second (2), r0's third (5)
TABLES = {"r0": [1, 2, 5], "r1": [3, 4]}


def _payload(r: str, row: MM.Row) -> bytes:
    return b"".join(hashlib.shake_256(repr((r, g)).encode()).digest(1024) for g in row.group)


def decode(prefill: int, positions: int) -> MM.Rows:
    """The rows of a request that prefills `prefill` positions, then decodes one a step to `positions`, its last step reading
    every row: 2 layers of K and V rows of 1 KiB, a row's group naming its (layer, plane, position)."""
    steps = 1 + positions - prefill
    return MM.Rows(tuple(tuple(tuple(MM.Row((("kv", layer, plane, q),), max(0, q - prefill + 1), steps - 1) for q in range(positions))
                               for plane in range(2)) for layer in range(2)), 1024, 2, steps)


def _slot(layer: int, page: int, plane: int) -> int:
    """A block's slot under `LAYOUT` for this session: 6 pages (the null page and 5), a page's K then V block, layer by layer."""
    return layer * 12 + page * 2 + plane


def _slots(reqs, tables, arrive=ARRIVE):
    """vLLM's slot mapping of the rows each request computed at an engine step (none of its cached ones), for pages `tables`."""
    return lambda r, step: [tables[r][q // 4] * 4 + q % 4 for q, x in enumerate(reqs[r].layers[0][0])
                            if q >= reqs[r].cached and arrive[r] + x.born == step]


def _engine(reqs, tables, bs=4, payload=_payload):
    """`block(layer, page, c)` of an engine whose caches hold the rows each request computed at its pages of `bs` slots (unwritten
    slots zero), a page's planes in order."""
    pages: dict[tuple[int, int], bytearray] = {}
    for r, rs in reqs.items():
        plane = bs * rs.row_bytes
        for layer, kv in enumerate(rs.layers):
            for q, page in enumerate(tables[r]):
                buf = pages.setdefault((layer, page), bytearray(len(kv) * plane))
                for k, rows in enumerate(kv):
                    for p in range(max(bs * q, rs.cached), min(bs * q + bs, rs.positions)):
                        row = payload(r, rows[p])
                        at = k * plane + (p % bs) * len(row)
                        buf[at:at + len(row)] = row
    return lambda layer, page, c: bytes(pages[layer, page][c * 4096:(c + 1) * 4096])


@pytest.fixture(scope="module")
def r0() -> Program:
    return build_program(_phi(WIDE), "request", LP=4, T=5).program


@pytest.fixture(scope="module")
def reqs(r0) -> dict[str, MM.Rows]:
    return {"r0": MM.rows(r0), "r1": MM.rows(build_program(_phi(WIDE), "request", LP=4, T=1).program)}


def selected_at(c: dict, values: set[int], step: int) -> dict:
    """Census `c` with each Call of `step` that reads one of `values` selecting it instead, a gather's candidate (the
    census's `selects`, a new entry's `gates` and `bits` a stand-in's, which the requirement doesn't read)."""
    c = copy.deepcopy(c)
    at, reads, sel = step_of(c), c["values"]["reads"], c["selects"]
    pools = {k: set(vs) for k, vs in zip(sel["call"], sel["values"])}
    for x in values:
        for k in [k for k in reads[x] if at(k) == step]:
            reads[x].remove(k)
            pools.setdefault(k, set()).add(x)
    old, ks = dict(zip(sel["call"], zip(sel["gates"], sel["bits"]))), sorted(pools)
    c["selects"] = {"call": ks, "gates": [old.get(k, (1, 0))[0] for k in ks], "bits": [old.get(k, (1, 0))[1] for k in ks],
                    "values": [sorted(pools[k]) for k in ks]}
    return c


def census_values(program: Program, c: dict, row: MM.Row) -> set[int]:
    """The census values a row's leaves are."""
    inputs, v = sum(1 for nd in program.root.body.nodes if is_input(nd.fn)), c["values"]
    return {x for _, j, lo, hi in row.group for x, k in enumerate(v["producer"])
            if k == j + inputs and v["lo"][x] < hi and v["hi"][x] > lo}


def test_rows_are_the_programs_kv(reqs):
    rs = reqs["r0"]
    assert (len(rs.layers), rs.positions, rs.row_bytes, rs.word_bytes, rs.steps) == (2, 9, 1024, 2, 6)
    # the prefill writes positions 0-3 at step 0, decode step t position 3 + t, and the last step reads every row
    for kv in rs.layers:
        for plane in kv:
            assert [x.born for x in plane] == [0, 0, 0, 0, 1, 2, 3, 4, 5] and {x.last for x in plane} == {5}
    groups = [x.group for kv in rs.layers for plane in kv for x in plane]
    assert len(set(groups)) == len(groups) == 2 * 2 * 9


def test_the_record(reqs):
    head, steps = MM.record(reqs, ARRIVE, _payload, REGISTRATION, LAYOUT, TABLES)
    assert (head.blocks, head.registration) == (2 * 6 * 2, REGISTRATION)
    page = {p: sorted(_slot(layer, p, plane) for layer in range(2) for plane in range(2)) for p in range(6)}
    assert [(s.step, list(s.freed), [w.slot for w in s.written]) for s in steps] == [
        (0, [], page[1]),               # r0's prefill fills its first page
        (1, page[2], page[3]),          # r1's fills its first; r0's position 4 reaches its second page's filler
        (2, sorted(page[3] + page[4]), []),  # r1's last step, 2, reads its first last and its position 4 reaches its second
        (4, [], page[2]),               # r0's position 7 completes its second page
        (5, sorted(page[1] + page[2] + page[5]), []),  # r0's last step reads both last; position 8 leaves its third dirty
    ]
    assert [w.index for s in steps for w in s.written] == list(range(24, 36))
    want = codec.digest(b"".join(_payload("r0", x) for x in reqs["r0"].layers[1][1][4:8]))
    assert next(w.digest for w in steps[-2].written if w.slot == _slot(1, 2, 1)) == want
    got_head, got = parse(dumps(head, steps).splitlines())
    rp, got = Replay(got_head), list(got)
    for s in got[:-1]:
        rp.apply(s)
    assert rp.extent == 36 and rp.holds(_slot(0, 3, 0)) is None and rp.holds(_slot(0, 2, 0)) == 32
    assert rp.holds(_slot(0, 5, 0)) == _slot(0, 5, 0) and rp.holds(_slot(0, 4, 0)) is None
    rp.apply(got[-1])
    assert rp.holds(_slot(0, 2, 0)) is None and rp.holds(_slot(0, 1, 0)) is None and rp.holds(_slot(0, 5, 0)) is None


def test_a_complete_block_is_held_while_its_rows_are_live(reqs):
    head, steps = MM.record(reqs, ARRIVE, _payload, REGISTRATION, LAYOUT, TABLES)
    # step t's reads see the blocks resident going into t; its record (frees, then writes) takes effect after them
    by, held = {s.step: s for s in steps}, set()
    for t in range(max(ARRIVE[r] + rs.steps for r, rs in reqs.items())):
        for r, rs in reqs.items():
            a = ARRIVE[r]
            for q, p in enumerate(TABLES[r][: rs.positions // 4]):
                for layer, kv in enumerate(rs.layers):
                    for plane, rows in enumerate(kv):
                        block = rows[4 * q: 4 * q + 4]
                        live = max(x.born for x in block) + a < t <= max(x.last for x in block) + a
                        assert (_slot(layer, p, plane) in held) == live, (t, r, q, layer, plane)
        held -= set(by[t].freed) if t in by else set()
        held |= {w.slot for w in by[t].written} if t in by else set()


def test_a_block_a_gather_may_select_is_freed_after_the_last_step_that_may_select_it(r0, reqs):
    """memory-map/v1 frees a block at its rows' last structural reader, a gather that may select one included: with r0's
    last step, 5, selecting its first page's rows instead of reading them (`selected_at`, a stand-in: no served Program
    gathers its KV), step 4 is their last use, but their `last` and the record are as before, the page freed after step 5
    whichever rows step 5 selected."""
    c = MM.census_of(r0)
    first = {x for kv in reqs["r0"].layers for plane in kv for row in plane[:4] for x in census_values(r0, c, row)}
    picked = selected_at(c, first, 5)
    req = from_census(picked)
    assert first and first <= req.upper[5] and not first & req.uses[5] and first <= req.uses[4]
    assert {req.last[x] for x in first} == {5} and MM.rows(r0, picked) == reqs["r0"]
    record = MM.record({**reqs, "r0": MM.rows(r0, picked)}, ARRIVE, _payload, REGISTRATION, LAYOUT, TABLES)
    assert record == MM.record(reqs, ARRIVE, _payload, REGISTRATION, LAYOUT, TABLES)
    page1 = {_slot(layer, 1, plane) for layer in range(2) for plane in range(2)}
    assert [s.step for s in record[1] if page1 & set(s.freed)] == [5]


def test_the_pages_are_the_ones_the_engine_wrote(reqs):
    observed = {"r0": [7, 3, 9], "r1": [2, 5]}
    assert MM.tables(reqs, ARRIVE, _slots(reqs, observed), 4) == observed
    head, steps = MM.record(reqs, ARRIVE, _payload, REGISTRATION, MM.KvPool(block_size=4, pages=10), observed)
    assert [w.slot for w in steps[0].written] == sorted(layer * 20 + 7 * 2 + plane for layer in range(2) for plane in range(2))
    with pytest.raises(ValueError, match="wrote 2 KV rows at step 1, where its Program writes 4 from position 0"):
        MM.tables(reqs, ARRIVE, lambda r, s: _slots(reqs, observed)(r, s)[2:] if (r, s) == ("r1", 1) else _slots(reqs, observed)(r, s), 4)
    with pytest.raises(ValueError, match="position 0 went to slot 29 .page 7, offset 1."):
        MM.tables(reqs, ARRIVE, lambda r, s: [x + 1 for x in _slots(reqs, observed)(r, s)], 4)
    with pytest.raises(ValueError, match="position 5 went to slot 33 .page 8, offset 1., its page 1 being 3"):
        MM.tables(reqs, ARRIVE, lambda r, s: [33] if (r, s) == ("r0", 2) else _slots(reqs, observed)(r, s), 4)
    with pytest.raises(ValueError, match="took pages"):
        MM.tables(reqs, ARRIVE, _slots(reqs, {"r0": [7, 7, 9], "r1": [2, 5]}), 4)


def test_the_engines_page_layouts(reqs):
    rs = reqs["r0"]
    assert MM.page_of((10, 4, 4, 256), (4096, 256, 1024, 1), 2, rs) == MM.Page("packed", 10, 4, 128)
    assert MM.page_of((10, 2, 4, 4, 128), (4096, 2048, 512, 128, 1), 2, rs) == MM.Page("planes", 10, 4, 128)
    with pytest.raises(ValueError, match="no page memory-map/v1 lays out"):
        MM.page_of((10, 4, 4, 256), (4096, 1024, 256, 1), 2, rs)            # head-major: a block holds part of every slot
    with pytest.raises(ValueError, match="1-byte elements"):
        MM.page_of((10, 4, 4, 256), (4096, 256, 1024, 1), 1, rs)


def test_a_packed_page_holds_k_and_v_in_one_plane(reqs):
    kv = {r: MM.packed(rs) for r, rs in reqs.items()}
    assert (len(kv["r0"].layers[0]), kv["r0"].row_bytes) == (1, 2048)
    assert kv["r0"].layers[1][0][4] == MM.Row(reqs["r0"].layers[1][0][4].group + reqs["r0"].layers[1][1][4].group, 1, 5)
    head, steps = MM.record(kv, ARRIVE, _payload, REGISTRATION, LAYOUT, TABLES)
    # a block is two slots of K and V: r0's second page reaches its first half at step 1 and completes it at 2 (positions 4,
    # 5), its second at 3 and 4; r1's position 4 and r0's 8 each reach a half they leave dirty, at steps 2 and 5
    assert head.blocks == 24 and [(s.step, len(s.freed), len(s.written)) for s in steps] == [
        (0, 0, 4), (1, 2, 4), (2, 4 + 2, 2), (3, 2, 0), (4, 0, 2), (5, 8 + 2, 0)]
    want = codec.digest(b"".join(_payload("r0", x) for x in kv["r0"].layers[0][0][0:2]))
    assert steps[0].written[0].slot == 2 * 1 + 0 and steps[0].written[0].digest == want


def test_the_engine_holds_the_maps_blocks(reqs):
    head, steps = MM.record(reqs, ARRIVE, _payload, REGISTRATION, LAYOUT, TABLES)
    block = _engine(reqs, TABLES)
    assert MM.engine_agrees(reqs, LAYOUT, TABLES, steps, block) == {"compared": 12, "equal": 12}
    bad = lambda layer, page, c: bytes(4096) if (layer, page, c) == (1, 2, 1) else block(layer, page, c)
    assert MM.engine_agrees(reqs, LAYOUT, TABLES, steps, bad) == {"compared": 12, "equal": 11}


def test_a_block_no_step_reads_while_resident_is_in_no_record():
    page = {p: sorted(_slot(layer, p, plane) for layer in range(2) for plane in range(2)) for p in range(3)}
    layout = MM.KvPool(block_size=4, pages=6)
    # the prefill (positions 0-4) completes page 1 and reaches page 2, whose last row (7) the last step, 3, writes: page 2 is
    # never resident while read, so no record writes it, and its filler is freed where the prefill first reaches it
    _, steps = MM.record({"r": decode(5, 8)}, {"r": 0}, _payload, REGISTRATION, layout, {"r": [1, 2]})
    assert [(s.step, list(s.freed), [w.slot for w in s.written]) for s in steps] == [(0, page[2], page[1]), (3, page[1], [])]
    # a prefill-only request's blocks are all in no record: the write reaching their filler frees it
    _, steps = MM.record({"r": decode(8, 8)}, {"r": 0}, _payload, REGISTRATION, layout, {"r": [1, 2]})
    assert [(s.step, list(s.freed), [w.slot for w in s.written]) for s in steps] == [(0, sorted(page[1] + page[2]), [])]


def test_a_page_reused_after_its_request_frees_its_filler_once(reqs):
    head, steps = MM.record(reqs, ARRIVE, _payload, REGISTRATION, LAYOUT, {"r0": [1, 2, 5], "r1": [3, 5]})
    rp = Replay(head)
    for s in steps:
        rp.apply(s)
    # r1's position 4 reaches page 5's filler at step 2; r0's position 8 reaches it again at 5, freed already
    assert [s.step for s in steps if _slot(0, 5, 0) in s.freed] == [2]


#: the session's steps from the engine's step 0 at 64, as a binder that served before numbers them
FIRST = 64
SESSION = {"r0": decode(4, 9), "r1": decode(4, 5)}


def _page(p: int) -> list[int]:
    return sorted(_slot(layer, p, plane) for layer in range(2) for plane in range(2))


def _deferred(*encodes, lifetime: int = 60) -> MM.Deferral:
    """A deferral of ``encodes``, ``(made, landed, first, count)`` on the engine's steps, as the session numbers them."""
    return MM.Deferral(tuple((m + FIRST, None if r is None else r + FIRST, f, c) for m, r, f, c in encodes), FIRST, lifetime)


def _lines(steps):
    return [(s.step, list(s.freed), [(w.slot, w.index) for w in s.written]) for s in steps]


def _replayed(head, steps):
    rp = Replay(head)
    for s in steps:
        rp.apply(s)
    return rp


def test_a_deferred_block_is_written_where_its_encode_lands():
    # each append lands a step after its completion: the filler is freed where a write first reaches it, even at the step
    # completing the block (pages 1 and 3); r1's first page, read last at step 2, where its encode lands, is written there
    # and freed a line later (memory accounting's withdrawn block, until memory-map/v1 records one), and so is r0's second
    d = _deferred((0, 1, 24, 4), (1, 2, 28, 4), (4, 5, 32, 4))
    head, steps = MM.record(SESSION, ARRIVE, _payload, REGISTRATION, LAYOUT, TABLES, d)
    at = lambda p, j: list(zip(_page(p), range(j, j + 4)))  # noqa: E731
    assert _lines(steps) == [(0, _page(1), []), (1, sorted(_page(2) + _page(3)), at(1, 24)), (2, _page(4), at(3, 28)),
                             (3, _page(3), []), (5, sorted(_page(1) + _page(5)), at(2, 32)), (6, _page(2), [])]
    assert _replayed(head, steps).extent == 36
    # landing at once is the map without a deferral
    now = _deferred((0, 0, 24, 4), (1, 1, 28, 4), (4, 4, 32, 4))
    assert MM.record(SESSION, ARRIVE, _payload, REGISTRATION, LAYOUT, TABLES, now) == MM.record(SESSION, ARRIVE, _payload, REGISTRATION,
                                                                                                LAYOUT, TABLES)
    # an encode still out is no write, and its block no free
    _, steps = MM.record(SESSION, ARRIVE, _payload, REGISTRATION, LAYOUT, TABLES, _deferred((0, 1, 24, 4), (1, 2, 28, 4), (4, None, 32, 4)))
    assert _lines(steps)[-2:] == [(3, _page(3), []), (5, sorted(_page(1) + _page(5)), [])]
    # and the circuit's appends after the session's last are never made
    _, steps = MM.record(SESSION, ARRIVE, _payload, REGISTRATION, LAYOUT, TABLES, _deferred((0, 1, 24, 4), (1, 2, 28, 4)))
    assert [w for s in steps for w in s.written if w.index >= 32] == []


def test_two_appends_landing_in_one_line_that_write_one_slot_are_written_a_line_apart():
    # a's page 1, completed at step 0 and read last at 1, lands at 2 with b's, which reuses the page from step 2: the line
    # writes a's, and b's waits a line, whose write releases a's in place of its free
    reqs, arrive, tables = {"a": decode(4, 5), "b": decode(4, 6)}, {"a": 0, "b": 2}, {"a": [1, 2], "b": [1, 3]}
    head, steps = MM.record(reqs, arrive, _payload, REGISTRATION, MM.KvPool(block_size=4, pages=6), tables,
                            _deferred((0, 2, 24, 4), (2, 2, 28, 4)))
    assert _lines(steps) == [(0, _page(1), []), (1, _page(2), []), (2, [], list(zip(_page(1), range(24, 28)))),
                             (3, _page(3), list(zip(_page(1), range(28, 32)))), (4, _page(1), [])]
    assert _replayed(head, steps).holds(_slot(0, 1, 0)) is None


def test_a_re_key_frees_the_old_index_at_once_and_writes_the_new_one_where_its_encode_lands():
    # the session re-keys 4 blocks in the append it makes at step 3: step 2's line frees the oldest live, r0's first page,
    # and step 3's, where the encode lands, writes them again at the fresh indices; step 4's append numbers on from there
    d = _deferred((0, 1, 24, 4), (1, 2, 28, 4), (3, 3, 32, 4), (4, 5, 36, 4), lifetime=2)
    head, steps = MM.record(SESSION, ARRIVE, _payload, REGISTRATION, LAYOUT, TABLES, d)
    at = lambda p, j: list(zip(_page(p), range(j, j + 4)))  # noqa: E731
    assert _lines(steps) == [(0, _page(1), []), (1, sorted(_page(2) + _page(3)), at(1, 24)),
                             (2, sorted(_page(1) + _page(4)), at(3, 28)), (3, _page(3), at(1, 32)),
                             (5, sorted(_page(1) + _page(5)), at(2, 36)), (6, _page(2), [])]
    assert [w.digest for w in steps[1].written] == [w.digest for w in steps[3].written]
    assert _replayed(head, steps).extent == 40


def test_a_deferral_that_is_not_the_circuits_is_refused():
    def record(*encodes, lifetime=60):
        return MM.record(SESSION, ARRIVE, _payload, REGISTRATION, LAYOUT, TABLES, _deferred(*encodes, lifetime=lifetime))
    with pytest.raises(ValueError, match="lifetime cap 1 is refused"):
        record((0, 1, 24, 4), lifetime=1)
    with pytest.raises(ValueError, match="at step 0 is of 3 blocks, and the circuit completed 4"):
        record((0, 1, 24, 3))
    with pytest.raises(ValueError, match="no append of the blocks the circuit completed at steps .1."):
        record((0, 1, 24, 4), (4, 5, 28, 4))
    with pytest.raises(ValueError, match="no append of the store's next blocks from 24"):
        record((0, 1, 25, 4))
    with pytest.raises(ValueError, match="no append of the store's next blocks from 28"):
        record((0, 2, 24, 4), (1, 1, 28, 4))
    with pytest.raises(ValueError, match="re-keys 8 blocks at step 2, and the circuit holds 4 live"):
        record((0, 1, 24, 4), (1, 2, 28, 4), (3, 3, 32, 8), lifetime=2)


def test_refusals(reqs):
    with pytest.raises(ValueError, match="while a request's rows in it are read to step"):
        MM.record(reqs, ARRIVE, _payload, REGISTRATION, LAYOUT, {"r0": [1, 2, 5], "r1": [2, 4]})
    with pytest.raises(ValueError, match="don't hold its 9 positions"):
        MM.record(reqs, ARRIVE, _payload, REGISTRATION, LAYOUT, {"r0": [1, 2], "r1": [3, 4]})
    with pytest.raises(ValueError, match="payload gave"):
        MM.record(reqs, ARRIVE, lambda r, x: bytes(1000), REGISTRATION, LAYOUT, TABLES)
    with pytest.raises(ValueError, match="not whole 4096-byte blocks"):
        MM.record(reqs, ARRIVE, _payload, REGISTRATION, MM.KvPool(block_size=2), TABLES)
    with pytest.raises(ValueError, match="KV differs"):
        MM.record({"r0": reqs["r0"], "t": MM.rows(build_program(_phi(TINY), "request", LP=4, T=1).program)}, {"r0": 0, "t": 0},
                  _payload, REGISTRATION, LAYOUT, {"r0": [1, 2, 3], "t": [4, 5]})


# -- a warm request: a prefix-cache hit on another request's page -------------------------------------------------------------


def warm_program(LP: int, T: int, C: int) -> Program:
    """A warm request's Program as a Build derives one (`build --cached C`, `kv_prefix`): every layer's history opens with the C
    K and V rows of its `prefix` input (bf16 [L, 2, C, row], layer l the l-th to open), and its own rows follow from position C
    (rows C..LP-1 at its first step, then T decodes)."""
    net = decoder(_phi(WIDE), 1, LP + T)
    wrec, rl = Record(*net.leaves()), net.C.KVH * net.C.D
    params = (("prompt", Array(LP, I32)), ("prefix", Array(net.C.NL, Array(2, Array(C, Array(rl, BF16))))), ("weights", wrec))
    B = Builder(params, Value(1))
    model = net.model(B, Coll(wrec, Affine("p", 2, 0, wrec.leaves)), _Peers([], 1, 0))
    prefix = Affine("p", 1, 0, params[1][1].leaves)
    for layer, hist in enumerate(model.hist):
        for kv, rows in enumerate(hist):
            rows += [prefix.slice(((2 * layer + kv) * C + j) * rl, ((2 * layer + kv) * C + j + 1) * rl) for j in range(C)]
    prompt = Affine("p", 0, 0, LP)
    toks = [model.sample(model.forward([Coll(I32, prompt.slice(q, q + 1)) for q in range(C, LP)], C)[-1], 0, LP - 1)]
    for s in range(T):
        toks.append(model.sample(model.forward([Coll(I32, toks[-1].refs)], LP + s)[-1], s + 1, LP + s))
    out = Coll(Array(len(toks), I32), concat([t.refs for t in toks]))
    B.ret_type = out.type
    body = B.finish(out)
    defn = CompositeDefinition("Derived[warm]", 1, ("SRC",), lambda S, _p=params, _r=out.type: (_p, _r), replayer(body), register=False)
    sd = bind(defn, SRC={"class": "warm", "C": C})
    sd.set_body(body)
    return Program(sd)


#: the warm Program's cached prefix as its Build lays it out (``prefix_cache.json``): its `prefix` input, 4 rows of 512 bf16
PREFIX = {"input": 1, "C": 4, "row_leaves": 512, "layers": [f"model.layers.{i}.self_attn.attn" for i in range(2)]}
#: r0 (cold: an 8-row prefill, then 1 decode step) from step 0; r1 (warm: positions 0-3 cached from r0's first page, rows 4-7 at
#: its first step, then 4 decode steps) from step 1
WARM_ARRIVE, WARM_TABLES, PRODUCERS = {"r0": 0, "r1": 1}, {"r0": [1, 2, 3], "r1": [1, 4, 5]}, {"r1": ["r0"]}
WARM_LAYOUT = MM.KvPool(block_size=4, pages=6)


@pytest.fixture(scope="module")
def warm() -> Program:
    return warm_program(8, 4, 4)


@pytest.fixture(scope="module")
def session(warm) -> dict[str, MM.Rows]:
    return {"r0": MM.rows(build_program(_phi(WIDE), "request", LP=8, T=1).program), "r1": MM.rows(warm, prefix=PREFIX)}


def test_a_warm_programs_prefix_rows_are_its_cached_positions(warm, session):
    rs = session["r1"]
    assert (len(rs.layers), rs.positions, rs.cached, rs.row_bytes, rs.steps) == (2, 12, 4, 1024, 5)
    for layer, kv in enumerate(rs.layers):
        for plane, rows in enumerate(kv):
            # positions 0-3 are the prefix input's rows of this layer and plane, read from the first step to the last; the
            # Program computes 4-7 at its first step and 8-11 one a decode step
            assert [x.group for x in rows[:4]] == [((MM.CACHED, 1, o, o + 512),) for o in range((2 * layer + plane) * 4 * 512, (2 * layer + plane + 1) * 4 * 512, 512)]
            assert [x.born for x in rows] == [0] * 8 + [1, 2, 3, 4] and {x.last for x in rows} == {4}
            assert not any(s[0] == MM.CACHED for x in rows[4:] for s in x.group)
    assert MM.rows_of(MM.rows_doc(rs)) == rs and session["r0"].cached == 0
    with pytest.raises(ValueError, match="root node .*'s kc reads an input of the Program"):
        MM.rows(warm)
    with pytest.raises(ValueError, match="isn't row 0 of an input of the cached prefix's 2 x 2 x 2 x 512 leaves"):
        MM.rows(warm, prefix=PREFIX | {"C": 2})
    with pytest.raises(ValueError, match="isn't row 0 of an input"):
        MM.rows(build_program(_phi(WIDE), "request", LP=8, T=1).program, prefix=PREFIX)
    with pytest.raises(ValueError, match="a cached prefix of 4 rows of 256 leaves"):
        MM.rows(warm, prefix=PREFIX | {"row_leaves": 256})


def test_a_warm_requests_cached_pages_are_the_ones_its_producer_wrote(session):
    slots = _slots(session, WARM_TABLES, WARM_ARRIVE)
    assert MM.tables(session, WARM_ARRIVE, slots, 4, PRODUCERS) == WARM_TABLES
    assert slots("r1", 1) == [16, 17, 18, 19], "the engine writes the warm request's rows from position 4 on"
    with pytest.raises(ValueError, match="wrote 8 KV rows at step 1, where its Program writes 4 from position 4"):
        MM.tables(session, WARM_ARRIVE, lambda r, s: [4, 5, 6, 7] + slots(r, s) if (r, s) == ("r1", 1) else slots(r, s), 4, PRODUCERS)
    with pytest.raises(ValueError, match="request r1's Program reads 4 cached rows, and 0 producing requests are named"):
        MM.tables(session, WARM_ARRIVE, slots, 4)
    with pytest.raises(ValueError, match="cached page 0 was computed by r9, no request of the session"):
        MM.tables(session, WARM_ARRIVE, slots, 4, {"r1": ["r9"]})
    with pytest.raises(ValueError, match="cached page 0 was computed by r1, whose Program did not compute it whole"):
        MM.tables(session, WARM_ARRIVE, slots, 4, {"r1": ["r1"]})


def test_a_page_two_requests_share_is_written_and_counted_once(session):
    head, steps = MM.record(session, WARM_ARRIVE, _payload, REGISTRATION, WARM_LAYOUT, WARM_TABLES, None, PRODUCERS)
    page = {p: sorted(_slot(layer, p, plane) for layer in range(2) for plane in range(2)) for p in range(6)}
    assert [(s.step, list(s.freed), [w.slot for w in s.written]) for s in steps] == [
        (0, [], sorted(page[1] + page[2])),     # r0's prefill fills its first two pages
        (1, sorted(page[2] + page[3]), page[4]),  # r0's last step reads page 2 last and reaches page 3; r1 fills page 4 past its prefix
        (2, page[5], []),                       # r1's position 8 reaches page 5's filler (its decodes leave it incomplete till its last)
        (5, sorted(page[1] + page[4]), []),     # r1's last step reads page 1, r0's, and page 4 last
    ]
    # page 1 is r0's block, written once, with r0's rows, held while either request reads it
    assert [w.digest for s in steps for w in s.written if w.slot == _slot(1, 1, 1)] == [
        codec.digest(b"".join(_payload("r0", x) for x in session["r0"].layers[1][1][0:4]))]
    assert sum(len(s.written) for s in steps) == 3 * 2 * 2 and _replayed(head, steps).extent == 24 + 12
    # r0 alone frees page 1 at its own last step: the warm read moves that free and nothing else of r0's
    alone = MM.record({"r0": session["r0"]}, {"r0": 0}, _payload, REGISTRATION, WARM_LAYOUT, {"r0": [1, 2, 3]})[1]
    assert [(s.step, list(s.freed), [w.slot for w in s.written]) for s in alone] == [
        (0, [], sorted(page[1] + page[2])), (1, sorted(page[1] + page[2] + page[3]), [])]
    assert MM.engine_agrees(session, WARM_LAYOUT, WARM_TABLES, steps, _engine(session, WARM_TABLES)) == {"compared": 12, "equal": 12}


def test_a_shared_page_is_refused_where_its_sharing_is_not_the_circuits(session):
    def record(reqs, arrive, tables, producers=PRODUCERS, layout=WARM_LAYOUT):
        return MM.record(reqs, arrive, _payload, REGISTRATION, layout, tables, None, producers)
    # another request writing r0's page 1 while r1 still reads it
    with pytest.raises(ValueError, match="is written at step 3 while a request's rows in it are read to step 5"):
        record(session | {"r2": decode(4, 5)}, WARM_ARRIVE | {"r2": 3}, WARM_TABLES | {"r2": [1, 6]}, layout=MM.KvPool(block_size=4, pages=7))
    # r1 reading page 1 at the step r0 completes it, before its record writes it
    with pytest.raises(ValueError, match="request r1 reads cached slot 2 from step 0, a block its producer r0 completes at step 0"):
        record(session, {"r0": 0, "r1": 0}, WARM_TABLES)
    # r1's prefix named as r2's, whose 3 positions leave page 1 incomplete
    with pytest.raises(ValueError, match="request r1 reads cached slot 2 from step 1, a block its producer r2 never completes there"):
        record(session | {"r2": decode(2, 3)}, WARM_ARRIVE | {"r2": 0}, {"r0": [3, 2, 5], "r1": [1, 4, 5], "r2": [1]}, {"r1": ["r2"]})
    with pytest.raises(ValueError, match="request r1's Program reads 4 cached rows, and 0 producing requests are named"):
        record(session, WARM_ARRIVE, WARM_TABLES, None)


def test_a_cold_session_is_as_before(reqs):
    observed = {"r0": [7, 3, 9], "r1": [2, 5]}
    assert MM.tables(reqs, ARRIVE, _slots(reqs, observed), 4, {}) == observed
    assert MM.record(reqs, ARRIVE, _payload, REGISTRATION, LAYOUT, TABLES, None, {}) == MM.record(reqs, ARRIVE, _payload, REGISTRATION, LAYOUT, TABLES)
    with pytest.raises(ValueError, match="request r1's Program reads 0 cached rows, and 1 producing requests are named"):
        MM.tables(reqs, ARRIVE, _slots(reqs, observed), 4, {"r1": ["r0"]})


# -- GLM-5.3's MLA: one plane of latent rows --------------------------------------------------------------------------------


class Sampled(DSA._Model):
    """The DSA stand-in (`tests/program/test_dsa_derive.py`) with each step closed, as an engine step is, by a greedy sampling event
    over its last token's output."""

    def forward(self, *xs):
        outs, k = [], 0
        for p0, n in self.steps:
            positions = torch.arange(p0, p0 + n, dtype=torch.int64)
            for layer in self.layers:
                outs += list(layer(positions, *xs[k:k + layer.n_inputs]))
                k += layer.n_inputs
            outs.append(torch.argmax(outs[-2][-1], dim=-1))
        return tuple(outs)


def dsa_request(steps=DSA.STEPS, topk: int = DSA.TOPK):
    """The DSA stand-in's request Program over ``steps``: an indexer layer and an IndexShare layer, each step a sampling event's."""
    model = Sampled(list(DSA._model(topk=topk, steps=steps).layers), steps)
    return DSA._derive(model, DSA._inputs(model)).program


@pytest.fixture
def dsa(served):  # noqa: F811
    program = dsa_request()
    return program, MM.rows(program)


def test_a_sparse_mla_layers_rows_are_its_latent_rows(dsa):
    program, rs = dsa
    # an indexer layer and an IndexShare layer, each its own history of one plane: positions 0 and 1 written at the prefill, 2 at
    # the decode, which reads every row
    assert (len(rs.layers), [len(kv) for kv in rs.layers], rs.positions, rs.row_bytes, rs.word_bytes, rs.steps) == (2, [1, 1], 3, 64, 2, 2)
    nodes = program.root.body.nodes
    inputs = sum(1 for nd in nodes if is_input(nd.fn))
    for (plane,) in rs.layers:
        assert [x.born for x in plane] == [0, 0, 1] and {x.last for x in plane} == {1}
        # a row is the normed latent, then the roped key, as vLLM's MLA cache row holds them; the indexer's keys are no row
        assert {tuple((nodes[inputs + j].fn.name, hi - lo) for _, j, lo, hi in x.group) for x in plane} == {
            (("RMSNormTriton", DSA.L), ("RoPEHeadGptJ", DSA.R))}
    assert not {x.group for x in rs.layers[0][0]} & {x.group for x in rs.layers[1][0]}


def test_mlas_cache_is_a_latent_page(dsa, reqs):
    _, rs = dsa
    assert MM.page_of((10, 4, 32), (128, 32, 1), 2, rs) == MM.Page("latent", 10, 4, 32)
    # DSA's indexer cache (bytes, [pages, block_size, 132]) lays out no rows of these: memory accounting has it uncredited until
    # the map packs its rows across blocks
    with pytest.raises(ValueError, match="1-byte elements"):
        MM.page_of((10, 4, 36), (144, 36, 1), 1, rs)
    with pytest.raises(ValueError, match="no page memory-map/v1 lays out 1 plane"):
        MM.page_of((10, 1, 4, 32), (128, 128, 32, 1), 2, rs)          # unsqueezed of its one head
    with pytest.raises(ValueError, match="no page memory-map/v1 lays out 1 plane"):
        MM.page_of((10, 4, 32), (256, 32, 1), 2, rs)                  # a page apart from the next
    with pytest.raises(ValueError, match="no page memory-map/v1 lays out 2 plane"):
        MM.page_of((10, 4, 512), (2048, 512, 1), 2, reqs["r0"])       # K and V rows hold no latent page


def latent(prefill: int, positions: int) -> MM.Rows:
    """GLM-5.3's MLA rows of a request (576 bf16: the 512-wide latent and the 64-wide roped key) that prefills `prefill` positions,
    then decodes one a step to `positions`, its last step reading every row: 2 layers of one plane."""
    steps = 1 + positions - prefill
    return MM.Rows(tuple((tuple(MM.Row((("mla", layer, q),), max(0, q - prefill + 1), steps - 1) for q in range(positions)),)
                         for layer in range(2)), 1152, 2, steps)


def _latent_payload(r: str, row: MM.Row) -> bytes:
    return hashlib.shake_256(repr((r, row.group)).encode()).digest(1152)


def test_a_latent_page_of_64_slots_is_18_blocks_its_rows_straddle():
    rs, tables, layout = {"r": latent(70, 80)}, {"r": [1, 2]}, MM.KvPool(block_size=64, pages=4)
    head, steps = MM.record(rs, {"r": 0}, _latent_payload, REGISTRATION, layout, tables)

    def at(page, *cs):
        return sorted(layer * 4 * 18 + page * 18 + c for layer in range(2) for c in cs)
    # page 2's block c holds slots [c*4096 // 1152, ceil((c+1)*4096 / 1152)): rows 3, 7, 10 and 14 straddle two; position q >= 70
    # is born at step q - 69, so its blocks complete at steps 0, 2, 5 and 9, and block 4 (slots 14-17) stays dirty
    assert head.blocks == 2 * 4 * 18 and [(s.step, list(s.freed), [w.slot for w in s.written]) for s in steps] == [
        (0, at(2, 1), sorted(at(1, *range(18)) + at(2, 0))), (2, at(2, 2), at(2, 1)), (5, at(2, 3), at(2, 2)), (9, at(2, 4), at(2, 3)),
        (10, sorted(at(1, *range(18)) + at(2, 0, 1, 2, 3)), [])]
    rows = rs["r"].layers[0][0]
    want = codec.digest(b"".join(_latent_payload("r", x) for x in rows[64 + 3:64 + 8])[4096 - 3 * 1152:][:4096])
    assert next(w.digest for w in steps[1].written if w.slot == at(2, 1)[0]) == want
    got_head, got = parse(dumps(head, steps).splitlines())
    assert _replayed(got_head, got).extent == head.blocks + 2 * (18 + 4)
    assert MM.engine_agrees(rs, layout, tables, steps, _engine(rs, tables, bs=64, payload=_latent_payload)) == {"compared": 44, "equal": 44}
