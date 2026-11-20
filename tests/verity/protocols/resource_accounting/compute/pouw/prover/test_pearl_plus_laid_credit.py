"""Pearl+'s credit over C as its laying lays it (`scheme.laid_tiles`, `scheme.credited_tiles`): C is two tiles, of linears
of one and two columns so that their kinds differ, and one unit of no tile, laid under a limit below the wired core's
ANDs, so the laying opens the core as it does at the served K = 2048; each laid block is its tile's by the gates it holds,
and the tiles are credited all or none, on an audit that drew every stratum holding a tile's block at its count."""
from __future__ import annotations

import pytest

from verity.circuits import laying as LY
from verity.circuits import partition_query as PQ
from verity.circuits.codec import definition_digest, definition_entry, descriptor_id, program_digest
from verity.circuits.count import count
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.ops.rows import Bf16AddRow
from verity.circuits.program import Program
from verity.circuits.refs import Coll, tuple_of
from verity.circuits.types import Tuple, Value
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
    laid = LY.lay(program, _fits(ts))
    strata = PQ.strata(laid, LY.query(laid))
    return ts, program, laid, strata, PS.laid_tiles(program, _fits(ts), strata, program_digest(laid)), descriptor_id(add)


def _fits(ts):
    """The fixture's limit, below the wired core's ANDs, so the laying opens the core."""
    limit = min(ands(bind(PC.PearlPlusTileCoreWired, **t.core.bindings)) for t in ts)
    return lambda _ins, _outs, a: a < limit


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
    ts, program, laid, strata, _tiles, _add = c
    (a, ra), (b, rb), *rest = strata
    digest = program_digest(laid)
    with pytest.raises(ValueError, match=PS.BLOCK_NOT_UNIT):
        PS.laid_tiles(program, _fits(ts), [(a, rb), (b, ra), *rest], digest)
    with pytest.raises(ValueError, match=PS.BLOCK_NOT_UNIT):
        PS.laid_tiles(program, _fits(ts), strata[1:], digest)
    with pytest.raises(ValueError, match=PS.TILE_ABSENT):
        PS.laid_tiles(laid, _fits(ts), strata, digest)


@pytest.mark.slow
def test_the_tiles_of_a_c_other_than_the_registered_one_are_refused(c) -> None:
    """Slow: as above: the tiles credited are those of the C the blocks are proved against, the one whose digest was
    registered before any coin; C laid under another limit, or another C's digest, is refused."""
    ts, program, laid, strata, _tiles, _add = c
    with pytest.raises(ValueError, match=PS.C_DIGEST):
        PS.laid_tiles(program, _fits(ts), strata, "00" * 64)
    with pytest.raises(ValueError, match=PS.C_DIGEST):
        PS.laid_tiles(program, lambda _ins, _outs, _a: True, strata, program_digest(laid))


def _forged_c(forged) -> Program:
    """A C of one root unit, ``forged``."""
    sig, ret = forged.signature()
    root = CompositeDefinition("PearlPlusForgedTileC", 1, (), lambda S: (sig, ret),
                               lambda B, S, *xs: B.call(forged, *xs), register=False)
    return Program(root, registered=[v for v, _ in sig])


def _tile_named(body, signature=None) -> CompositeDefinition:
    """A Definition under the tile's name, version and statics, its own ``body`` and ``signature`` (the tile's when
    None)."""
    return CompositeDefinition("PearlPlusTileCoreWords", 5, PC._STATICS, signature or PC.PearlPlusTileCoreWords.signature,
                               body, register=False)


def test_a_unit_with_the_tiles_id_at_a_huge_k_and_a_small_body_is_refused() -> None:
    """A root unit carrying the tile's id at K = 2^20, its body one word passed through, is refused by name before the
    registry builds any tile, so it is never credited its statics' 2^20 · TR · TN MACs."""
    statics = PC.credited_product(SCHEME, 2 ** 20, 1, 1, 1, 1).bindings
    forged = bind(_tile_named(lambda B, S, x: x, lambda S: ((("x", Value(16)),), Value(16))), **statics)
    assert descriptor_id(forged).startswith(PS.TILE + "{") and "K=1048576" in descriptor_id(forged)
    assert PS.scheme_statics(forged) is None
    with pytest.raises(ValueError, match=PS.TILE_FOREIGN):
        PS.laid_tiles(_forged_c(forged), lambda *_: True, [], "")


def _off_scheme() -> dict[str, dict]:
    """The registry's tile at K = 256, one row and one column, each binding right for its own statics and one of them not
    the scheme of record's: RHO 100 (a tile inert-work bound of 1/100 where the scheme's is 1/400), RHOR 400 (a row
    bound of 1/400 where the scheme's at K = 256 is 1/100), DEV sm120 (the sm120 scheme's tile, priced as H100 MACs)."""
    ours = PC.credited_product(SCHEME, K, 1, 1, 1, 1).bindings
    return {"RHO": {**ours, "RHO": 100}, "RHOR": {**ours, "RHOR": 400},
            "DEV": dict(PC.credited_product("pearl-c-sm120-v1-h3", K, 1, 1, 1, 1).bindings)}


def test_every_tile_static_is_read_by_the_credit_rule_pinned_to_the_scheme_or_free() -> None:
    """The tile's statics are exactly the classified ones, a static added to the Definition fails here; the credited
    ones are those `tile_macs` reads, and the scheme of record binds each pinned one."""
    assert set(PC.PearlPlusTileCoreWords.statics) == set(PS.TILE_STATICS)
    assert set(PS.TILE_STATICS.values()) == {"credit", "scheme", "free"}
    credit = {s for s, why in PS.TILE_STATICS.items() if why == "credit"}
    assert PS.tile_macs(dict.fromkeys(credit, 1)) == 1
    assert set(PS.SCHEME_STATICS) <= set(PC.credited_product(SCHEME, K, 1, 1, 1, 1).bindings)


def test_the_scheme_of_record_is_the_lean_credits() -> None:
    """The Python refusal and the Lean credit's price name one scheme: `KERNEL` is the Lean port's `NAME`, which its
    `schemeOf` holds every pass to."""
    from pathlib import Path
    import re

    records = Path(PS.__file__).parent / "verifier/lean/pearlplus/PouwPearlPlus/Records.lean"
    (name,) = re.findall(r'^def NAME : String := "([^"]+)"$', records.read_text(), re.M)
    assert PS.KERNEL == name == SCHEME


@pytest.mark.parametrize("static", ["RHO", "RHOR", "DEV"])
def test_a_registry_tile_off_the_schemes_device_or_bounds_is_refused_by_its_statics(static) -> None:
    """The registry's own tile, its body right for its statics, with RHO, RHOR or DEV not the scheme of record's at its
    shape, is refused by `scheme_statics` naming that static; read from the statics alone, with no body built."""
    tile = bind(PC.PearlPlusTileCoreWords, **_off_scheme()[static])
    why = PS.scheme_statics(tile)
    assert why is not None and (why.startswith(f"{static} = ") or f", {static} = " in why), why


@pytest.mark.slow
@pytest.mark.parametrize("static", ["RHO", "RHOR", "DEV"])
def test_a_c_holding_a_registry_tile_off_the_schemes_statics_is_refused(c, static) -> None:
    """Slow: a C of the registry's tile off the scheme builds that tile's body at its statics (its core's bounds, and for
    sm120 its Gram rows too), which the cheaper test of `scheme_statics` never builds; this one shows `laid_tiles` refuses
    such a C by name, its body being right for its statics, so the refusal does not rest on the body pin."""
    _ts, *_ = c
    tile = bind(PC.PearlPlusTileCoreWords, **_off_scheme()[static])
    assert PS.foreign_tile(tile) is None
    with pytest.raises(ValueError, match=PS.TILE_SCHEME):
        PS.laid_tiles(_forged_c(tile), lambda *_: True, [], "")


@pytest.mark.slow
def test_a_unit_with_the_tiles_id_and_statics_and_another_body_is_refused(c) -> None:
    """Slow: the forged units' bodies call the tile's noise and core, which only the module's built tiles make cheap. A
    unit carrying the tile's id and statics is refused at the reader, not credited its statics' MACs, when its own body
    is the tile's with F_A's and F_B's lines swapped into the core, and when its own body is the tile's and one Definition
    it calls, the wired core under its id and statics, has F_A's and F_B's lines swapped inside."""
    from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_tile as PT

    ts, *_ = c
    real = ts[0].core

    def swapped(B, S, a, b, cert, salt, root, unit, rows, col):
        eb, fa, fb, g = PT.noise(B, S.K, S.TN, S.DEV, salt, root, col)
        return B.call(PC._wired(S), a, b, cert, salt, root, unit, rows, eb, fb, fa, g)

    wired = bind(CompositeDefinition(
        "PearlPlusTileCoreWired", 2, PC._STATICS, PC.PearlPlusTileCoreWired.signature,
        lambda B, S, a, b, cert, salt, root, unit, rows, eb, fa, fb, g:
            PC._wired_body(B, S, a, b, cert, salt, root, unit, rows, eb, fb, fa, g), register=False), **real.bindings)

    def calls_forged(B, S, a, b, cert, salt, root, unit, rows, col):
        eb, fa, fb, g = PT.noise(B, S.K, S.TN, S.DEV, salt, root, col)
        return B.call(wired, a, b, cert, salt, root, unit, rows, eb, fa, fb, g)

    top = bind(_tile_named(swapped), **real.bindings)
    callee = bind(_tile_named(calls_forged), **real.bindings)
    assert descriptor_id(top) == descriptor_id(callee) == descriptor_id(real) and PS.foreign_tile(real) is None
    assert descriptor_id(wired) == descriptor_id(PC._wired(real.S()))
    assert definition_digest(definition_entry(callee)) == definition_digest(definition_entry(real))
    assert "definition digests differ" in PS.foreign_tile(top) and descriptor_id(real) in PS.foreign_tile(top)
    assert descriptor_id(wired) in PS.foreign_tile(callee)
    for forged in (top, callee):
        with pytest.raises(ValueError, match=PS.TILE_FOREIGN):
            PS.laid_tiles(_forged_c(forged), _fits(ts), [], "")
