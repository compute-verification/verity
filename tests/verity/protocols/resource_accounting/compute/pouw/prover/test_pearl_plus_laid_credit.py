"""Pearl+'s credit over C as its laying lays it (`scheme.laid_tiles`, `scheme.credited_tiles`): C is two tiles, of linears
of one and two columns so that their kinds differ, and one unit of no tile, laid under a limit below the wired core's
ANDs, so the laying opens the core as it does at the served K = 2048; each laid block is its tile's by the gates it holds,
and the tiles are credited all or none, on an audit that drew every stratum holding a tile's block at its count."""
from __future__ import annotations

import pytest

from verity.circuits import laying as LY
from verity.circuits import partition_query as PQ
from verity.circuits.codec import descriptor_id
from verity.circuits.count import count
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.ops.rows import Bf16AddRow
from verity.circuits.program import Program
from verity.circuits.refs import Coll, tuple_of
from verity.circuits.types import Tuple
from verity.protocols.resource_accounting.compute.pouw import scheme as PS
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_credited as PC
from verity.protocols.resource_accounting.compute.pouw.prover.circuit.pearl_c_call import ands

SCHEME = "pearl-c-h100-v1-h3"
K = 256


@pytest.fixture(scope="module")
def c():
    ts = [PC.tiles(SCHEME, K, 1, n, tile=(1, 1))[0] for n in (1, 2)]
    sigs = [t.core.signature() for t in ts]
    add = bind(Bf16AddRow, N=1)
    add_sig, add_ret = add.signature()
    params = (*((f"{p}{i}", t) for i, (sig, _r) in enumerate(sigs) for p, t in sig),
              *((f"{p}_add", t) for p, t in add_sig))

    def body(B, S, *xs):
        it = iter(xs)
        outs = [B.call(t.core, *[next(it) for _ in sig]) for t, (sig, _r) in zip(ts, sigs)]
        z = B.call(add, *[next(it) for _ in add_sig])
        return tuple_of(*(Coll(r, o.refs) for o, (_s, r) in zip(outs, sigs)), z)

    root = CompositeDefinition("PearlPlusLaidCreditC", 1, (), lambda S: (params, Tuple(*(r for _s, r in sigs), add_ret)),
                               body, register=False)
    program = Program(root, registered=[v for v, _ in params])
    limit = min(ands(bind(PC.PearlPlusTileCoreWired, **t.core.bindings)) for t in ts)
    laid = LY.lay(program, lambda _ins, _outs, a: a < limit)
    strata = PQ.strata(laid, LY.query(laid))
    return ts, program, laid, strata, PS.laid_tiles(program, laid, strata), descriptor_id(add)


def _units(strata, tid: str) -> list[int]:
    return [u for t, rs in strata if t == tid for lo, hi in rs for u in range(lo, hi)]


def _audit(strata, k: int = 1) -> tuple[dict[int, bool], dict[str, int]]:
    """Each stratum's first ``k`` blocks drawn and verified, at a registered count of ``k``."""
    return {u: True for tid, _rs in strata for u in _units(strata, tid)[:k]}, {tid: k for tid, _rs in strata}


@pytest.mark.slow
def test_a_tiles_laid_blocks_hold_all_its_ands_and_it_is_credited_its_macs(c) -> None:
    """Slow: only a laid C of real tiles shows the public rule places every block of a tile, its opened core's included,
    in that tile; one unit a tile (K = 256 under the composer's limit) never exercises it. One AND count on both sides."""
    ts, program, laid, strata, tiles, _add = c
    assert not any(tid.startswith(PC.PearlPlusTileCoreWired.id) for tid, _rs in strata), "the laying opens the core"
    per, whole = count(laid), count(program)
    of = {u: tid for tid, rs in strata for lo, hi in rs for u in range(lo, hi)}
    assert [t.index for t in tiles] == [0, 1]
    for t, tile in zip(tiles, ts):
        assert sum(per[of[b]].ands for b in t.blocks) == whole[descriptor_id(tile.core)].ands
        assert sum(t.kinds.values()) == len(t.blocks) and len(t.kinds) > 6
        assert t.amount == PS.tile_macs(tile.core.bindings) == tile.real * 1 * K


@pytest.mark.slow
def test_a_unit_of_another_tile_or_of_none_is_not_counted(c) -> None:
    """Slow: as above: the two tiles' blocks are disjoint, each in its own gates, and the unit of no tile is in neither."""
    _ts, _program, _laid, strata, tiles, add = c
    first, second = (set(t.blocks) for t in tiles)
    (reader,) = _units(strata, add)
    assert not first & second and reader not in first | second
    assert first | second | {reader} == {u for _tid, rs in strata for lo, hi in rs for u in range(lo, hi)}
    assert tiles[0].gates[1] <= tiles[1].gates[0] and max(first) < min(second)


@pytest.mark.slow
def test_the_tiles_are_credited_all_or_none(c) -> None:
    """Slow: as above: a failed block of any stratum holding a tile's block credits no tile, a kind only the second tile
    has included; a failed block of no tile's stratum spoils none."""
    _ts, _program, _laid, strata, tiles, add = c
    verdicts, counts = _audit(strata)
    assert PS.credited_tiles(tiles, strata, verdicts, counts) == tiles
    only = sorted(set(tiles[1].kinds) - set(tiles[0].kinds))
    assert only, "the two tiles differ in their kinds"
    shared = sorted(set(tiles[0].kinds) & set(tiles[1].kinds))
    for kind in (only[0], shared[0]):
        (bad,) = [u for u in verdicts if u in _units(strata, kind)]
        assert PS.credited_tiles(tiles, strata, {**verdicts, bad: False}, counts) == []
    (reader,) = _units(strata, add)
    assert PS.credited_tiles(tiles, strata, {**verdicts, reader: False}, counts) == tiles


@pytest.mark.slow
def test_an_audit_short_of_its_draw_is_refused(c) -> None:
    """Slow: as above: no evidence is no credit. An empty audit, and a stratum holding a tile's block short of its
    registered count, are refused; a stratum of no tile short of its count does not matter."""
    _ts, _program, _laid, strata, tiles, add = c
    verdicts, counts = _audit(strata)
    with pytest.raises(ValueError, match=PS.AUDIT_SHORT):
        PS.credited_tiles(tiles, strata, {}, counts)
    with pytest.raises(ValueError, match=PS.AUDIT_SHORT):
        PS.credited_tiles(tiles, strata, {}, {})
    kind = sorted(tiles[1].kinds)[0]
    with pytest.raises(ValueError, match=PS.AUDIT_SHORT):
        PS.credited_tiles(tiles, strata, verdicts, {**counts, kind: 2})
    with pytest.raises(ValueError, match=PS.AUDIT_SHORT):
        PS.credited_tiles(tiles, strata, {u: v for u, v in verdicts.items() if u not in _units(strata, kind)}, counts)
    (reader,) = _units(strata, add)
    assert PS.credited_tiles(tiles, strata, {u: v for u, v in verdicts.items() if u != reader}, counts) == tiles
    with pytest.raises(ValueError, match=PS.AUDIT_STRAY):
        PS.credited_tiles(tiles, strata, {**verdicts, 10**9: True}, counts)


@pytest.mark.slow
def test_strata_whose_blocks_are_not_the_laid_units_or_a_c_of_no_tile_are_refused(c) -> None:
    """Slow: as above: the bound rests on a block being one root unit of its stratum's one template, and a C with no tile
    has nothing to credit."""
    _ts, program, laid, strata, _tiles, _add = c
    (a, ra), (b, rb), *rest = strata
    with pytest.raises(ValueError, match=PS.BLOCK_NOT_UNIT):
        PS.laid_tiles(program, laid, [(a, rb), (b, ra), *rest])
    with pytest.raises(ValueError, match=PS.BLOCK_NOT_UNIT):
        PS.laid_tiles(program, laid, strata[1:])
    with pytest.raises(ValueError, match=PS.TILE_ABSENT):
        PS.laid_tiles(laid, laid, strata)
