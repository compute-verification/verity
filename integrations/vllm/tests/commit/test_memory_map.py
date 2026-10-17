"""memory-map/v1 from the circuit (`verity_vllm.commit.memory_map`) on a small Llama's request Programs: rows of 1 KiB (4 KV heads
of 128 bf16) in pages of 4 slots, so a plane of a page is one 4 KiB block; request r0 (a 4-row prefill, then 5 decode steps) from
step 0, and r1 (a 4-row prefill, then 1 decode step) from step 1, which ends while r0 runs.  The record is one
`remote_bandwidth.memory_map.Replay` accepts, a complete block is in it exactly while the requirement has its rows live, and the
pages are the ones the engine's slots name.
"""
from __future__ import annotations

import hashlib

import pytest

from tests.predict.test_predict import TINY, _phi
from verity.core.protocols.remote_bandwidth import codec
from verity.core.protocols.remote_bandwidth.memory_map import Replay, dumps, parse
from verity_vllm.commit import memory_map as MM
from verity_vllm.predict.build import build_program

WIDE = TINY | {"hidden_size": 512, "num_key_value_heads": 4}
LAYOUT = MM.Layout(block_size=4)
REGISTRATION = bytes(range(64))
ARRIVE = {"r0": 0, "r1": 1}
#: pages in order of first need: r0's first (step 0), r0's second and r1's first (step 1), r1's second (2), r0's third (5)
TABLES = {"r0": [1, 2, 5], "r1": [3, 4]}


def _payload(r: str, row: MM.Row) -> bytes:
    return b"".join(hashlib.shake_256(repr((r, g)).encode()).digest(1024) for g in row.group)


def _slot(layer: int, page: int, plane: int) -> int:
    """A block's slot under `LAYOUT` for this session: 6 pages (the null page and 5), a page's K then V block, layer by layer."""
    return layer * 12 + page * 2 + plane


def _slots(reqs, tables):
    """vLLM's slot mapping of each request's rows at an engine step, for pages `tables`."""
    return lambda r, step: [tables[r][q // 4] * 4 + q % 4 for q, x in enumerate(reqs[r].layers[0][0]) if ARRIVE[r] + x.born == step]


def _engine(reqs, tables, planes=2):
    """`block(layer, page, c)` of an engine whose caches hold each request's rows at its pages (unwritten slots zero), a page's
    planes in order."""
    pages: dict[tuple[int, int], bytearray] = {}
    for r, rs in reqs.items():
        for layer, kv in enumerate(rs.layers):
            for q, page in enumerate(tables[r]):
                buf = pages.setdefault((layer, page), bytearray(8192))
                for plane, rows in enumerate(kv):
                    for p in range(4 * q, min(4 * q + 4, rs.positions)):
                        row = _payload(r, rows[p])
                        at = plane * 4096 + (p % 4) * len(row)
                        buf[at:at + len(row)] = row
    return lambda layer, page, c: bytes(pages[layer, page][c * 4096:(c + 1) * 4096])


@pytest.fixture(scope="module")
def reqs() -> dict[str, MM.Rows]:
    return {"r0": MM.rows(build_program(_phi(WIDE), "request", LP=4, T=5).program),
            "r1": MM.rows(build_program(_phi(WIDE), "request", LP=4, T=1).program)}


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
        (1, [], page[3]),               # and r1's its first
        (3, page[3], []),               # r1's last step is 2
        (4, [], page[2]),               # r0's position 7 completes its second page; its third (position 8) stays dirty
    ]
    assert [w.index for s in steps for w in s.written] == list(range(24, 36))
    want = codec.digest(b"".join(_payload("r0", x) for x in reqs["r0"].layers[1][1][4:8]))
    assert next(w.digest for w in steps[-1].written if w.slot == _slot(1, 2, 1)) == want
    got_head, got = parse(dumps(head, steps).splitlines())
    rp = Replay(got_head)
    for s in got:
        rp.apply(s)
    assert rp.extent == 36 and rp.holds(_slot(0, 3, 0)) is None and rp.holds(_slot(0, 2, 0)) == 32


def test_a_complete_block_is_held_while_its_rows_are_live(reqs):
    head, steps = MM.record(reqs, ARRIVE, _payload, REGISTRATION, LAYOUT, TABLES)
    by, held = {s.step: s for s in steps}, set()
    for t in range(max(ARRIVE[r] + rs.steps for r, rs in reqs.items())):
        held -= set(by[t].freed) if t in by else set()
        for r, rs in reqs.items():
            a = ARRIVE[r]
            for q, p in enumerate(TABLES[r][: rs.positions // 4]):
                for layer, kv in enumerate(rs.layers):
                    for plane, rows in enumerate(kv):
                        live = all(x.born + a < t <= x.last + a for x in rows[4 * q: 4 * q + 4])
                        assert (_slot(layer, p, plane) in held) == live, (t, r, q, layer, plane)
        held |= {w.slot for w in by[t].written} if t in by else set()


def test_the_pages_are_the_ones_the_engine_wrote(reqs):
    observed = {"r0": [7, 3, 9], "r1": [2, 5]}
    assert MM.tables(reqs, ARRIVE, _slots(reqs, observed), 4) == observed
    head, steps = MM.record(reqs, ARRIVE, _payload, REGISTRATION, MM.Layout(block_size=4, pages=10), observed)
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
    # a block is two slots of K and V: r0's second page completes its first half at step 2 (positions 4, 5), its second at 4
    assert head.blocks == 24 and [(s.step, len(s.freed), len(s.written)) for s in steps] == [(0, 0, 4), (1, 0, 4), (2, 0, 2), (3, 4, 0), (4, 0, 2)]
    want = codec.digest(b"".join(_payload("r0", x) for x in kv["r0"].layers[0][0][0:2]))
    assert steps[0].written[0].slot == 2 * 1 + 0 and steps[0].written[0].digest == want


def test_the_engine_holds_the_maps_blocks(reqs):
    head, steps = MM.record(reqs, ARRIVE, _payload, REGISTRATION, LAYOUT, TABLES)
    block = _engine(reqs, TABLES)
    assert MM.engine_agrees(reqs, LAYOUT, TABLES, steps, block) == {"compared": 12, "equal": 12}
    bad = lambda layer, page, c: bytes(4096) if (layer, page, c) == (1, 2, 1) else block(layer, page, c)
    assert MM.engine_agrees(reqs, LAYOUT, TABLES, steps, bad) == {"compared": 12, "equal": 11}


def test_refusals(reqs):
    with pytest.raises(ValueError, match="don't hold its 9 positions"):
        MM.record(reqs, ARRIVE, _payload, REGISTRATION, LAYOUT, {"r0": [1, 2], "r1": [3, 4]})
    with pytest.raises(ValueError, match="payload gave"):
        MM.record(reqs, ARRIVE, lambda r, x: bytes(1000), REGISTRATION, LAYOUT, TABLES)
    with pytest.raises(ValueError, match="not whole 4096-byte blocks"):
        MM.record(reqs, ARRIVE, _payload, REGISTRATION, MM.Layout(block_size=2), TABLES)
    with pytest.raises(ValueError, match="KV differs"):
        MM.record({"r0": reqs["r0"], "t": MM.rows(build_program(_phi(TINY), "request", LP=4, T=1).program)}, {"r0": 0, "t": 0},
                  _payload, REGISTRATION, LAYOUT, {"r0": [1, 2, 3], "t": [4, 5]})
