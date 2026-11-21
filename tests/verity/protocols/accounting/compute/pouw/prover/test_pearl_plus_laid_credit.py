"""Pearl+'s tiles in C as its laying lays it (`scheme.laid_tiles`): C is three tiles, a linear of one column and one of
two whose W is one parameter of two rows, so that their kinds differ, and one unit of no tile, laid under a limit below the
wired core's ANDs, so the laying opens the core as it does at the served K = 2048; each laid block is its tile's by the
gates it holds. The fixture's tiles are a test instance's (K = 256 at 1 × 1, `scheme.test_instance`). The credit itself
is the Lean's (`PouwPearlPlus`, `creditAtAssumption`)."""
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
from verity.circuits.types import Array, Tuple, Value
from verity.protocols.accounting.compute.pouw import scheme as PS
from verity.protocols.accounting.compute.pouw.prover.circuit import pearl_plus_credited as PC
from verity.protocols.accounting.compute.pouw.prover.circuit.pearl_c_call import ands

SCHEME = "pearl-c-h100-v1-h3"
K = 256


def _c_of(ts, w2_rows: int = 2) -> Program:
    """A C of ``ts`` and one residual add: the first tile's ports its own parameters, the others' too but their W
    operands, which are the rows (each tile's column tile) of one parameter ``w2`` of ``w2_rows`` rows."""
    sigs = [t.core.signature() for t in ts]
    add = bind(Bf16AddRow, N=1)
    add_sig, add_ret = add.signature()
    shared = [i > 0 for i in range(len(ts))]
    params = (*((f"{p}{i}", t) for i, (sig, _r) in enumerate(sigs) for p, t in sig if not (shared[i] and p == "b")),
              *((("w2", Array(w2_rows, dict(sigs[-1][0])["b"])),) if any(shared) else ()),
              *((f"{p}_add", t) for p, t in add_sig))

    def body(B, S, *xs):
        held = dict(zip((v for v, _ in params), xs))
        outs = [B.call(t.core, *[held["w2"][t.col] if shared[i] and p == "b" else held[f"{p}{i}"] for p, _ in sig])
                for i, (t, (sig, _r)) in enumerate(zip(ts, sigs))]
        z = B.call(add, *[held[f"{p}_add"] for p, _ in add_sig])
        return tuple_of(*(Coll(r, o.refs) for o, (_s, r) in zip(outs, sigs)), z)

    root = CompositeDefinition("PearlPlusLaidCreditC", 1, (), lambda S: (params, Tuple(*(r for _s, r in sigs), add_ret)),
                               body, register=False)
    return Program(root, registered=[v for v, _ in params])


@pytest.fixture(scope="module")
def c():
    ts = [PC.tiles(SCHEME, K, 1, 1, tile=(1, 1))[0], *PC.tiles(SCHEME, K, 1, 2, tile=(1, 1))]
    add = bind(Bf16AddRow, N=1)
    program = _c_of(ts)
    laid = LY.lay(program, _fits(ts))
    strata = PQ.strata(laid, LY.query(laid))
    return ts, program, laid, strata, PS.laid_tiles(program, _fits(ts), strata, program_digest(laid)), descriptor_id(add)


def _fits(ts):
    """The fixture's limit, below the wired core's ANDs, so the laying opens the core."""
    limit = min(ands(bind(PC.PearlPlusTileCoreWired, **t.core.bindings)) for t in ts)
    return lambda _ins, _outs, a: a < limit


def _units(strata, tid: str) -> list[int]:
    return [u for t, rs in strata if t == tid for lo, hi in rs for u in range(lo, hi)]


@pytest.mark.slow
def test_a_tiles_laid_blocks_hold_all_its_ands_and_it_is_credited_its_macs(c) -> None:
    """Slow: only a laid C of real tiles shows the public rule places every block of a tile, its opened core's included,
    in that tile; one unit a tile (K = 256 under the composer's limit) never exercises it. One AND count on both sides."""
    ts, program, laid, strata, tiles, _add = c
    assert not any(tid.startswith(PC.PearlPlusTileCoreWired.id) for tid, _rs in strata), "the laying opens the core"
    per, whole = count(laid), count(program)
    of = {u: tid for tid, rs in strata for lo, hi in rs for u in range(lo, hi)}
    assert [t.index for t in tiles] == [0, 1, 2]
    for t, tile in zip(tiles, ts):
        assert sum(per[of[b]].ands for b in t.blocks) == whole[descriptor_id(tile.core)].ands
        assert sum(t.kinds.values()) == len(t.blocks) and len(t.kinds) > 6
        assert t.amount == PS.tile_macs(tile.core.bindings) == tile.real * 1 * K
        assert t.test


@pytest.mark.slow
def test_a_unit_of_another_tile_or_of_none_is_not_counted(c) -> None:
    """Slow: as above: the tiles' blocks are disjoint, each in its own gates, and the unit of no tile is in none."""
    _ts, _program, _laid, strata, tiles, add = c
    held = [set(t.blocks) for t in tiles]
    (reader,) = _units(strata, add)
    assert all(not a & b for i, a in enumerate(held) for b in held[i + 1:]) and reader not in set().union(*held)
    assert set().union(*held) | {reader} == {u for _tid, rs in strata for lo, hi in rs for u in range(lo, hi)}
    assert all(s.gates[1] <= t.gates[0] and max(a) < min(b) for s, t, a, b in zip(tiles, tiles[1:], held, held[1:]))


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


@pytest.mark.slow
def test_a_tile_whose_n_is_not_its_linears_columns_is_refused(c) -> None:
    """Slow: the two-column linear's tile is built once by the module's C. Its N (2) held to the columns C gives its
    linear: alone, reading a W of one column tile, it is refused, and beside its own W of two column tiles of which no
    tile reads the second, it is refused too; each by name."""
    ts, *_ = c
    with pytest.raises(ValueError, match=PS.TILE_COLUMNS):
        PS.laid_tiles(_forged_c(ts[1].core), lambda *_: True, [], "")
    with pytest.raises(ValueError, match=PS.TILE_COLUMNS) as e:
        PS.laid_tiles(_c_of([ts[0], ts[1]]), lambda *_: True, [], "")
    assert "column tile 1" in str(e.value)


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


def test_the_served_floor_is_the_lean_credits() -> None:
    """The floor `test_instance` reads (`pearl_c.K_MIN`, `pearl_c.TILE`) is the Lean credit's: `Check.K_MIN` and
    `Check.TILE`, below which its `plan` credits nothing; a tile below it in K or in either side is a test instance's."""
    from pathlib import Path
    import re

    from verity.protocols.accounting.compute.pouw.prover.schemes import pearl_c as C

    check = (Path(PS.__file__).parent / "verifier/lean/pearlplus/PouwPearlPlus/Check.lean").read_text()
    lean = {n: int(v) for n, v in re.findall(r"^def (K_MIN|TILE) : Nat := (\d+)$", check, re.M)}
    assert lean == {"K_MIN": C.K_MIN, "TILE": C.TILE}
    assert PS.test_instance({"K": C.K_MIN, "TM": C.TILE, "TN": C.TILE}) is None
    assert PS.test_instance({"K": C.K_MIN - 1, "TM": C.TILE, "TN": C.TILE})
    assert PS.test_instance({"K": C.K_MIN, "TM": 1, "TN": C.TILE})
    assert PS.test_instance({"K": C.K_MIN, "TM": C.TILE, "TN": C.TILE // 2})


def test_every_tile_static_is_read_by_the_credit_rule_pinned_to_the_scheme_or_free() -> None:
    """The tile's statics are exactly the classified ones, a static added to the Definition fails here; the credited
    ones are those `tile_macs` reads, and the scheme of record binds each pinned one."""
    assert set(PC.PearlPlusTileCoreWords.statics) == set(PS.TILE_STATICS)
    assert set(PS.TILE_STATICS.values()) == {"credit", "scheme", "columns", "free"}
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


def test_the_device_and_bounds_the_scheme_binds_are_the_lean_credits() -> None:
    """The DEV, RHO and RHOR `scheme_statics` holds a tile to (`credited_product` under `KERNEL`) are what the Lean credit
    rule takes at that shape: the tile's bound ρ = `Rule.INERT_BOUND_NUM` / `INERT_BOUND_DEN`, the row's ρ widened by
    `Rule`'s factor at a census K (`Records.Scheme.censusKs`) and ρ elsewhere, and the device whose prices are
    `Records`' (`FADD`, `BF16_MAC`, `FORMING`, `G`)."""
    from fractions import Fraction
    from pathlib import Path
    import re

    from verity.protocols.accounting.compute.pouw.prover.schemes.pearl_c_device import DEVICES

    lean = Path(PS.__file__).parent / "verifier/lean/pearlplus/PouwPearlPlus"
    rule, records = (lean / "Rule.lean").read_text(), (lean / "Records.lean").read_text()

    def nat(text: str, name: str) -> int:
        (v,) = re.findall(rf"^def {name} : Nat := (\d+)(?: \+ (\d+))?$", text, re.M)
        return int(v[0]) + int(v[1] or 0)

    rho = Fraction(nat(rule, "INERT_BOUND_NUM"), nat(rule, "INERT_BOUND_DEN"))
    ((k_census, k_min_gt),) = re.findall(r"censusKs \(s : Scheme\) : List Nat := if (\d+) < s\.kMin then \[(\d+)\]",
                                         records)
    (k_min,) = re.findall(r"^  kMin : Nat := (\d+)$", records, re.M)
    (widen,) = re.findall(r"widen := if s\.censusKs\.contains u\.k then (\d+) else 1", rule)
    census = [int(k_min_gt)] if int(k_census) < int(k_min) else []
    for k in (256, 1024, 2048):
        b = PC.credited_product(SCHEME, k, 1, 1, 1, 1).bindings
        assert Fraction(1, b["RHO"]) == rho, k
        assert Fraction(1, b["RHOR"]) == rho * (int(widen) if k in census else 1), k
        d = DEVICES[b["DEV"]]
        assert (d.prices.fadd, d.prices.bf16_mac, d.prices.forming, d.g) == \
            (nat(records, "FADD"), nat(records, "BF16_MAC"), nat(records, "FORMING"), nat(records, "G")), k


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
    from verity.protocols.accounting.compute.pouw.prover.circuit import pearl_plus_tile as PT

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
