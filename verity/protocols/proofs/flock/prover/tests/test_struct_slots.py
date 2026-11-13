"""`verity_flock.struct_slots`: a structural lowering's MUFU table reads in lookup slots, split by read depth into one ordered
template a net, wired as `read_slots` wires a flat unit's, computes what the flat lowering computes; a Definition with no
slot read is no split."""
from __future__ import annotations

import numpy as np
from verity.circuits.defs import CompositeDefinition
from verity.definitions.ops.fp32 import F32AddFtz
from verity.definitions.ops.mufu import MufuEx2Ftz, MufuRcpFtz
from verity.circuits.refs import array_of
from verity.circuits.types import Array, Value
from verity_flock import struct_lower as SL
from verity_flock import struct_slots as SS
from verity_flock import tmpl as T

F32 = Value(32)


def _chain(B, _S, x, y):
    r = B.call(MufuRcpFtz, B.call(F32AddFtz, B.call(MufuEx2Ftz, x), y))
    return array_of([r, B.call(F32AddFtz, r, x), B.call(MufuEx2Ftz, y)])


CHAIN = CompositeDefinition("StructSlotsChain", 1, (), lambda _S: ((("x", F32), ("y", F32)), Array(3, F32)), _chain,
                            register=False)
ADDS = CompositeDefinition("StructSlotsAdds", 1, (), lambda _S: ((("x", F32), ("y", F32)), Array(2, F32)),
                           lambda B, _S, x, y: array_of([B.call(F32AddFtz, x, y), B.call(F32AddFtz, y, y)]), register=False)


def test_a_read_of_a_reads_value_is_a_later_stage_and_the_split_is_the_flat_lowering():
    """rcp(ex2(x) + y) reads a slot whose index is of depth one: the unit computes both ex2 indices, stage0 the rcp index
    from their words, and the last net the outputs; each net is ordered, and each stage's input port and each slot's index
    has one wire."""
    sp, _tm, _lw = SS.lower(CHAIN)
    assert [SS.net_name(s) for s in range(len(sp.nets))] == ["unit", *(f"stage{k}" for k in range(2))]
    assert all(T.ordered(n)["ok"] for n in sp.nets)
    assert sorted(sp.lookups) == [("ex2", 0), ("ex2", 0), ("rcp", 1)]
    dsts = [tuple(w[:3]) for w in sp.wires]
    slots = {(f"lookup:{t}", i, 0) for t in {t for t, _s in sp.lookups}
             for i in range(sum(1 for t2, _s in sp.lookups if t2 == t))}
    stages = {(SS.net_name(s), 0, p) for s in range(1, len(sp.nets)) for p in range(len(sp.in_widths[s]))}
    assert len(dsts) == len(set(dsts)) and set(dsts) == slots | stages
    assert SS.check(CHAIN, seed=1)["mismatched_bits"] == 0


def test_rows_only_a_later_stage_reads_are_placed_there_when_that_passes_fewer_bits():
    """Placed at their depths, the unit passes on what it computes from x and y for the last net; placed as late as their
    readers allow, the last net computes it from x and y itself. Both splits compute the same outputs, and `lower` keeps the
    one passing fewer bits between its nets."""
    tm, lw, _top = SL.lower(CHAIN, slots=SS.SLOTS)
    sites = {sp.index: sp.sites for sp in lw.order if sp.sites}
    early, late = (SS.split(tm, sites, late=v) for v in (False, True))
    assert SS.boundary(late) < SS.boundary(early) and all(T.ordered(n)["ok"] for n in late.nets)
    x = np.random.default_rng(2).integers(0, 1 << 64, size=(64, 1), dtype=np.uint64)
    assert np.array_equal(SS.evaluate(early, x), SS.evaluate(late, x))
    assert SS.boundary(SS.lower(CHAIN)[0]) == SS.boundary(late)


def test_a_split_is_written_and_read_back(tmp_path):
    sp, _tm, _lw = SS.lower(CHAIN)
    SS.write(sp, tmp_path)
    back = SS.read(tmp_path)
    assert (back.ins, back.outs, back.lookups, back.wires) == (sp.ins, sp.outs, sp.lookups, sp.wires)
    assert [T.write(n) for n in back.nets] == [T.write(n) for n in sp.nets]


def test_a_definition_reading_no_slot_table_is_no_split():
    sp, tm, _lw = SS.lower(ADDS)
    assert sp is None and T.ordered(tm)["ok"]
