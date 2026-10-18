"""memory-map/v1 from the circuit (`verity_vllm.commit.memory_map`) on a small Llama's request Programs: rows of 1 KiB (4 KV heads
of 128 bf16) in pages of 4 slots, so a plane of a page is one 4 KiB block; request r0 (a 4-row prefill, then 5 decode steps) from
step 0, and r1 (a 4-row prefill, then 1 decode step) from step 1, which ends while r0 runs.  The record is one
`remote_bandwidth.memory_map.Replay` accepts, and a complete block is in it exactly while the requirement has its rows live.
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


def _payload(r: str, row: MM.Row) -> bytes:
    return hashlib.shake_256(repr((r, row.group)).encode()).digest(1024)


def _slot(layer: int, page: int, plane: int) -> int:
    """A block's slot under `LAYOUT` for this session: 6 pages (the null page and 5), a page's K then V block, layer by layer."""
    return layer * 12 + page * 2 + plane


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
    head, steps = MM.record(reqs, ARRIVE, _payload, REGISTRATION, LAYOUT)
    # pages in order of first need: r0's first (step 0), r0's second and r1's first (step 1), r1's second (2), r0's third (5)
    assert MM.tables_of(reqs, ARRIVE, 4) == {"r0": [1, 2, 5], "r1": [3, 4]}
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
    head, steps = MM.record(reqs, ARRIVE, _payload, REGISTRATION, LAYOUT)
    tables, by, held = MM.tables_of(reqs, ARRIVE, 4), {s.step: s for s in steps}, set()
    for t in range(max(ARRIVE[r] + rs.steps for r, rs in reqs.items())):
        held -= set(by[t].freed) if t in by else set()
        for r, rs in reqs.items():
            a = ARRIVE[r]
            for q, p in enumerate(tables[r][: rs.positions // 4]):
                for layer, kv in enumerate(rs.layers):
                    for plane, rows in enumerate(kv):
                        live = all(x.born + a < t <= x.last + a for x in rows[4 * q: 4 * q + 4])
                        assert (_slot(layer, p, plane) in held) == live, (t, r, q, layer, plane)
        held |= {w.slot for w in by[t].written} if t in by else set()


def test_refusals(reqs):
    with pytest.raises(ValueError, match="don't hold its 9 positions"):
        MM.record(reqs, ARRIVE, _payload, REGISTRATION, LAYOUT, tables={"r0": [1, 2], "r1": [3, 4]})
    with pytest.raises(ValueError, match="payload gave"):
        MM.record(reqs, ARRIVE, lambda r, x: bytes(1000), REGISTRATION, LAYOUT)
    with pytest.raises(ValueError, match="not whole 4096-byte blocks"):
        MM.record(reqs, ARRIVE, _payload, REGISTRATION, MM.Layout(block_size=2))
    with pytest.raises(ValueError, match="KV differs"):
        MM.record({"r0": reqs["r0"], "t": MM.rows(build_program(_phi(TINY), "request", LP=4, T=1).program)}, {"r0": 0, "t": 0},
                  _payload, REGISTRATION, LAYOUT)
