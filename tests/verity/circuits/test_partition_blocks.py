"""`partition_object`'s ``blocks`` on the periods' chain (eight `Gemm_v2{K=16, N=16}` units, a wire whose readers are units
[2, 8), ordered in ranges of two): the level `interface.partition_blocks` builds from the period's members, the field's levels and its
refusals, its digest, and the block `restrict` derives from one of its pairs."""
from __future__ import annotations

import functools
import importlib.util
from pathlib import Path

import pytest

from verity.circuits import partition_object as PO
from verity.circuits import restrict as R
from verity.circuits.codec import descriptor_chunks
from verity.commitments.periods import interface as I

_SPEC = importlib.util.spec_from_file_location("blocks_chain", Path(__file__).parent / "ops/test_boolean_block_circuit.py")
_BB = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BB)


@functools.cache
def chain() -> I.Circuit:
    n, gid = 8, _BB.GID
    wires = (I.Wire(gid, "x", gid, "o0", at=2, radix=(n - 2,), base=0, step=(1,)),)
    return I.Circuit("chain", ((_BB.GEMM, n),), external=(f"{gid}/x",), wires=wires,
                     order=I.ordered({gid: n}, wires, {gid: range(0, n, 2)}))


@functools.cache
def built():
    return chain().program()


def obj(blocks):
    program, query = built()
    return PO.build(program, query, blocks)


@pytest.mark.parametrize("cap,want", [(1, [[u, u + 1] for u in range(8)]), (2, [[0, 2], [2, 4], [4, 6], [6, 8]]),
                                      (8, [[0, 2], [2, 4], [4, 6], [6, 8]])])
def test_the_blocks_are_the_members_cut_at_the_order_and_the_cap(cap, want):
    got = I.partition_blocks(chain(), cap, built())
    assert got == want == [[m.base, m.base + m.n] for m in I.members(chain(), cap, built())]
    o = obj([got])
    assert PO.verify(o, built()[0]).ok and PO.blocks(o) == [tuple(p) for p in want] and PO.levels(o) == [PO.blocks(o)]


def test_the_pairs_cover_the_population_once_in_its_own_order():
    program, query = built()
    population = PO.TemplatePartition(program, query).population
    for cap in (1, 2, 3, 5, 8):
        units = [u for lo, hi in I.partition_blocks(chain(), cap, built()) for u in range(lo, hi)]
        assert units == list(range(population))


def test_without_blocks_the_object_and_its_digest_are_unchanged():
    program, query = built()
    assert PO.build(program, query) == {"format": PO.FORMAT, "program": PO.program_sha512(program), "query": query}
    assert PO.build(program, query, None) == PO.build(program, query)


def test_the_digest_moves_with_the_pairs():
    program, query = built()
    digests = [PO.digest(PO.build(program, query))] + [PO.digest(obj([b])) for b in ([[0, 8]], [[0, 4], [4, 8]],
                                                                                     I.partition_blocks(chain(), 1, built()),
                                                                                     I.partition_blocks(chain(), 2, built()))]
    assert len(set(digests)) == 5 and PO.digest(obj([[[0, 8]], [[0, 8]]])) not in digests


@pytest.mark.parametrize("blocks,code,witness", [
    ([[[0, 2], [3, 8]]], "blocks-gap", [0, [2, 3]]),
    ([[[1, 8]]], "blocks-gap", [0, [0, 1]]),
    ([[[0, 3], [2, 8]]], "blocks-overlap", [0, [[0, 3], [2, 8]]]),
    ([[[2, 8], [0, 2]]], "blocks-unsorted", [0, [[2, 8], [0, 2]]]),
    ([[[0, 8]], [[0, 2], [3, 8]]], "blocks-gap", [1, [2, 3]]),
    ([[[0, 4], [4, 8]], [[0, 2], [2, 6], [6, 8]]], "blocks-refine", [1, [2, 6]]),
    ([[[0, 2], [2, 8]], [[0, 4], [4, 8]]], "blocks-refine", [1, [0, 4]]),
    ([[[0, 8]], [[0, 4]]], "blocks-refine", [1, 8]),
    ([], "blocks-form", []),
    ([[]], "blocks-form", [0, []]),
    ([[0, 8]], "blocks-form", [0, 0]),
    ([[[0, 8]], [[0, 0], [0, 8]]], "blocks-form", [1, [0, 0]]),
    ([[[0, "8"]]], "blocks-form", [0, [0, "8"]]),
    ([[[False, 8]]], "blocks-form", [0, [False, 8]]),
    ([[[0, 2, 8]]], "blocks-form", [0, [0, 2, 8]]),
    ([[(0, 8)]], "blocks-form", [0, (0, 8)]),
    ("0-8", "blocks-form", "0-8"),
])
def test_a_malformed_grouping_is_refused_by_name(blocks, code, witness):
    o = {**obj(None), PO.BLOCKS: blocks}
    v = PO.validate(o)
    assert not v.ok and code in v.codes and witness in v.detail[code]["gates"], v.detail
    assert code in PO.verify(o, built()[0]).codes
    with pytest.raises(ValueError):
        PO.digest(o)


@pytest.mark.parametrize("blocks,code,witness", [([[[0, 9]]], "blocks-range", [0, [9, 8]]),
                                                 ([[[0, 4], [4, 9]]], "blocks-range", [0, [9, 8]]),
                                                 ([[[0, 7]]], "blocks-gap", [0, [7, 8]]),
                                                 ([[[0, 2], [2, 4]]], "blocks-gap", [0, [4, 8]]),
                                                 ([[[0, 9]], [[0, 4], [4, 9]]], "blocks-range", [1, [9, 8]])])
def test_a_grouping_past_or_short_of_the_population_is_refused_by_name(blocks, code, witness):
    o = obj(blocks)
    assert PO.validate(o).ok
    v = PO.verify(o, built()[0])
    assert v.codes == (code,) and witness in v.detail[code]["gates"]


def test_levels_refine_coarsest_first_and_may_repeat():
    for levels in ([[[0, 8]], [[0, 4], [4, 8]], [[0, 2], [2, 4], [4, 6], [6, 8]]], [[[0, 4], [4, 8]]] * 2):
        o = obj(levels)
        assert PO.verify(o, built()[0]).ok and PO.levels(o) == [[tuple(p) for p in lv] for lv in levels]
        with pytest.raises(ValueError, match="blocks-levels"):
            PO.blocks(o)


def test_blocks_on_a_query_other_than_a_template_query_are_refused():
    o = {**PO.build(built()[0], PO.query()), PO.BLOCKS: [[[0, 8]]]}
    assert "blocks-query" in PO.validate(o).codes


def test_restrict_with_a_pair_from_the_field_is_the_members_block():
    program, query = built()
    c = b"".join(descriptor_chunks(program))
    bits = R.Bits(R.program(c))
    o = obj([I.partition_blocks(chain(), 2, built())])
    members = I.members(chain(), 2, built())
    for pair, m in list(zip(PO.blocks(o), members))[:2]:
        assert R.restrict(c, o, [pair], bits).descriptor == R.restrict(c, o, [(m.base, m.base + m.n)], bits).descriptor
