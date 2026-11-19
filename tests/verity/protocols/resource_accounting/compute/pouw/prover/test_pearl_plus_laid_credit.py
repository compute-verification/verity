"""Pearl+'s credit per tile over C as its laying lays it (`scheme.laid_tiles`, `scheme.credited_tiles`): C is two tiles of
one credited linear and one unit of no tile, laid under a limit just below the wired core's ANDs, so the laying opens the
core as it does at the served K = 2048; each laid block is its tile's by the gates it holds, and a tile is credited only
when every stratum it has a block in passed."""
from __future__ import annotations

import pytest

from verity.circuits import laying as LY
from verity.circuits import partition_query as PQ
from verity.circuits.codec import descriptor_id
from verity.circuits.count import count
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.ops.rows import Bf16AddRow
from verity.circuits.program import Program
from verity.circuits.refs import Coll, concat, tuple_of
from verity.circuits.types import Array, Tuple
from verity.protocols.resource_accounting.compute.pouw import scheme as PS
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_credited as PC
from verity.protocols.resource_accounting.compute.pouw.prover.circuit.pearl_c_call import ands

SCHEME = "pearl-c-h100-v1-h3"
K = 256


@pytest.fixture(scope="module")
def c():
    ts = PC.tiles(SCHEME, K, 1, 2, tile=(1, 1))
    sig, ret = ts[0].core.signature()
    add = bind(Bf16AddRow, N=1)
    add_sig, add_ret = add.signature()
    params = (*((f"{p}{i}", t) for i in range(len(ts)) for p, t in sig), *((f"{p}_add", t) for p, t in add_sig))

    def body(B, S, *xs):
        it = iter(xs)
        outs = [B.call(t.core, *[next(it) for _ in sig]) for t in ts]
        z = B.call(add, *[next(it) for _ in add_sig])
        return tuple_of(Coll(Array(len(ts), ret), concat([o.refs for o in outs])), z)

    root = CompositeDefinition("PearlPlusLaidCreditC", 1, (), lambda S: (params, Tuple(Array(len(ts), ret), add_ret)),
                               body, register=False)
    program = Program(root, registered=[v for v, _ in params])
    limit = ands(bind(PC.PearlPlusTileCoreWired, **ts[0].core.bindings))
    laid = LY.lay(program, lambda _ins, _outs, a: a < limit)
    strata = PQ.strata(laid, LY.query(laid))
    return ts, program, laid, strata, PS.laid_tiles(program, laid, strata), descriptor_id(add)


@pytest.mark.slow
def test_a_tiles_laid_blocks_hold_all_its_ands_and_it_is_credited_its_macs(c) -> None:
    """Slow: only a laid C of real tiles shows the public rule places every block of a tile, its opened core's included,
    in that tile; one unit a tile (K = 256 under the composer's limit) never exercises it."""
    ts, program, laid, strata, tiles, _add = c
    assert not any(tid.startswith(PC.PearlPlusTileCoreWired.id) for tid, _rs in strata), "the laying opens the core"
    per, whole = count(laid), count(program)
    of = {u: tid for tid, rs in strata for lo, hi in rs for u in range(lo, hi)}
    assert [t.index for t in tiles] == [0, 1]
    for t, tile in zip(tiles, ts):
        assert sum(per[of[b]].ands for b in t.blocks) == whole[descriptor_id(tile.core)].ands
        assert sum(t.kinds.values()) == len(t.blocks) and len(t.kinds) > 6
        assert t.amount == PS.tile_macs(tile.core.bindings) == tile.real * 1 * K
    assert tiles[0].kinds == tiles[1].kinds


@pytest.mark.slow
def test_a_unit_of_another_tile_or_of_none_is_not_counted(c) -> None:
    """Slow: as above: the two tiles' blocks are disjoint, each in its own gates, and the unit of no tile is in neither."""
    _ts, _program, _laid, strata, tiles, add = c
    first, second = (set(t.blocks) for t in tiles)
    (reader,) = [u for tid, rs in strata if tid == add for lo, hi in rs for u in range(lo, hi)]
    assert not first & second and reader not in first | second
    assert first | second | {reader} == {u for _tid, rs in strata for lo, hi in rs for u in range(lo, hi)}
    assert tiles[0].gates[1] <= tiles[1].gates[0]
    assert max(first) < min(second)


@pytest.mark.slow
def test_a_tile_with_one_bad_block_is_not_credited(c) -> None:
    """Slow: as above: one failed block spoils its stratum, and no tile with a block in it is credited, the other tile of
    that stratum included; a failed block of no tile spoils no tile's."""
    _ts, _program, _laid, strata, tiles, add = c
    of = {u: tid for tid, rs in strata for lo, hi in rs for u in range(lo, hi)}
    (reader,) = [u for u, tid in of.items() if tid == add]
    assert PS.credited_tiles(tiles, strata, []) == tiles
    assert PS.credited_tiles(tiles, strata, [reader]) == tiles
    for bad in (tiles[0].blocks[0], tiles[0].blocks[-1], tiles[1].blocks[len(tiles[1].blocks) // 2]):
        assert of[bad] in tiles[0].kinds and PS.credited_tiles(tiles, strata, [bad]) == []


@pytest.mark.slow
def test_strata_whose_blocks_are_not_the_laid_units_are_refused(c) -> None:
    """Slow: as above: the bound rests on a block being one root unit of its stratum's one template."""
    _ts, program, laid, strata, _tiles, _add = c
    (a, ra), (b, rb), *rest = strata
    with pytest.raises(ValueError, match=PS.BLOCK_NOT_UNIT):
        PS.laid_tiles(program, laid, [(a, rb), (b, ra), *rest])
    with pytest.raises(ValueError, match=PS.BLOCK_NOT_UNIT):
        PS.laid_tiles(program, laid, strata[1:])
