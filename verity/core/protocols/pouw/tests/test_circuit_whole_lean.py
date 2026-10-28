"""The Lean whole-Program table is `pc8`'s `Pc8Check`: its input gates and their widths, its units in `pc8.units`' order,
each unit's computed gates a copy of its template's with every read relocated through the Lean port map, and its
outputs. `lean_layout` restates `WholeCircuit.whole8` (`Security/Definitions/Pouw/PearlC/WholeCircuit.lean`): the input
regions (`eaAt` … `in8`, `inW8`), the blocks (`row8B`, `row8A`, `tileSpec`) and their ports (`portB8`, `portA8`,
`portT8`, with `code8`, `paOut8` and `pbOut8` naming the rows' outputs), and the outputs (`outs8`). Each template's reads
are its own `CallGraph`'s, which `test_circuit_row_lean.py` and `test_circuit_tile_lean.py` tie to the Lean unit tables;
this test ties the wiring between them, so it and the pinned descriptor digests are what tie `tileCheck8`'s circuit to
the Program. No primitive is evaluated.

A `CallGraph` folds wiring into sorted read sets, so it can't tell a port map from one permuted within a gate's reads
(16 P_A words read by one peel step); `bound_ports` reads each member's operands in the flat layout instead, where a
member's gates are its template's in order, and finds each parameter leaf's producer exactly."""
from __future__ import annotations

import pytest

from verity.ml.circuits import Program
from verity.ml.circuits.codec import program_digest
from verity.ml.circuits.cut import CallGraph, call_scope

pc8 = pytest.importorskip("verity.ml.protocols.pouw.circuit.pc8", reason="the Program is `verity.ml.protocols.pouw.circuit.pc8`'s")

#: `program_digest` of the `Pc8Check` Programs the Lean layout was checked against; a change to any needs the Lean re-read
DIGESTS = {
    ("sm120", 2, 128, 3, (1, 1)): "e8c2f1dc2089cb3a260faa982dc4e2300294b75a97519045398c0198684bf8cc1b63dea4140a2cf8ad9e205c245e4aae293f3b69781468940866225ca345c267",
    ("sm120", 4, 128, 4, (2, 2)): "0c543396657700464262e4797817d5ec7420669f52d0cc3a5e6d775a32d5b16788ec5dd57df82d2d2b258577511120e986c906a70c9899f0bb50644b08601303",
    ("h100", 2, 256, 2, (1, 2)): "d8112b40ed2b64d509ea1640cc7f7c10792909d228a7193c9e67d9d37d9aaa868c232658600a0b3358f5cf813e55d6550c7d8b6eb1b04ada14bc00200be6c298",
}


# ---- `RowCircuit`'s offsets ------------------------------------------------------------------------------------------------

def row_in(k: int, b: bool) -> int:
    """`rowIn K b`."""
    return 65 * k + 32 + (1024 if b else 0)


def post8_at(k: int, b: int) -> int:
    """`post8At K b`, through `fmaAt` … `code8At`."""
    n8 = k // 8
    o = b + 2 * (k - 1) + n8
    o += min(16, n8) + min(8, n8) + min(4, n8) + min(2, n8) + min(1, n8)   # rhoAt
    o += 2 + 6 + n8 + 1 + 3 * k + 8 + 96 + 32                           # al8At, sq8At, th8At, dead8At, ok8At, ep8At, ew8At, code8At
    return o + 3 * k


def code8(k: int, b: int, t: int) -> int:
    """`code8 K b t`: `code8At K b + 3·t + 2`."""
    return post8_at(k, b) - 3 * k + 3 * t + 2


def pa_out8(k: int, w: int) -> int:
    """`paOut8 K w`."""
    nw, b = k // 128, row_in(k, False)
    return post8_at(k, b) + (32 * (5 * nw + 1) + w if w < 32 else (w - 32) * (5 * nw + 1) + 5 * nw)


def pb_out8(k: int, w: int) -> int:
    """`pbOut8 K w`."""
    nw, b = k // 128, row_in(k, True)
    return post8_at(k, b) + (w * (5 * nw + 35) + 5 * nw + 34 if w < 32 else 32 * (5 * nw + 35) + 2 * (w - 32) + 1)


def row_n(k: int, b: bool) -> int:
    """`row8A_N`, `row8B_N`: the row tables' gate counts."""
    nw = k // 128
    return post8_at(k, row_in(k, b)) + (32 * (5 * nw + 35) + 65 if b else 32 * (5 * nw + 1) + 33)


def tile_in(k: int, tm: int, tn: int) -> int:
    """`tileIn K TM TN`."""
    return (tm + tn) * k + 64 * (tm + tn)


def block_len(k: int) -> int:
    """`blockLen K`."""
    return 5 * (k // 128) + 4


# ---- `WholeCircuit.whole8` ---------------------------------------------------------------------------------------------------

def lean_layout(m: int, k: int, n: int, tm: int, tn: int):
    """`whole8`'s input count and widths, its blocks as (role, instances, template inputs, computed gates, port), and its
    outputs in `Pc8Check`'s return order."""
    ea, b = m * k, m * (k + 32)
    eb, fa = b + n * k, (m + n) * (k + 32)
    fb, g = fa + 32 * k, fa + 64 * k
    nin = g + 1024
    widths = [32] * ea + [8] * (b - ea) + [32] * (eb - b) + [8] * (g - eb) + [32] * 1024
    lb, la = row_n(k, True) - row_in(k, True), row_n(k, False) - row_in(k, False)
    lt = tm * tn * block_len(k)

    def port_a(i, a):
        if a < k:
            return i * k + a
        if a < k + 32:
            return ea + 32 * i + (a - k)
        return fa + (a - (k + 32)) if a < 33 * k + 32 else fb + (a - (33 * k + 32))

    def port_b(j, a):
        if a < k:
            return b + j * k + a
        if a < k + 32:
            return eb + 32 * j + (a - k)
        if a < 33 * k + 32:
            return fb + (a - (k + 32))
        return fa + (a - (33 * k + 32)) if a < 65 * k + 32 else g + (a - (65 * k + 32))

    row_b = lambda j, a: nin + j * lb + (a - row_in(k, True))                  # noqa: E731
    row_a = lambda i, a: nin + n * lb + i * la + (a - row_in(k, False))        # noqa: E731
    tiles = nin + n * lb + m * la

    def port_t(e, a):
        rs, cs = e // (n // tn), e % (n // tn)
        if a < tm * k:
            return row_a(rs * tm + a // k, code8(k, row_in(k, False), a % k))
        if a < (tm + tn) * k:
            a -= tm * k
            return row_b(cs * tn + a // k, code8(k, row_in(k, True), a % k))
        a -= (tm + tn) * k
        if a < 64 * tm:
            return row_a(rs * tm + a // 64, pa_out8(k, a % 64))
        a -= 64 * tm
        return row_b(cs * tn + a // 64, pb_out8(k, a % 64))

    blocks = [("row_b", n, row_in(k, True), lb, port_b), ("row_a", m, row_in(k, False), la, port_a),
              ("tile", (m // tm) * (n // tn), tile_in(k, tm, tn), lt, port_t)]

    def entry(i, j, last):
        e, ti, tj = i // tm * (n // tn) + j // tn, i % tm, j % tn
        r = (ti * tn + tj) * block_len(k) + (block_len(k) - 1 if last else 5 * (k // 128) - 1)
        return tiles + e * lt + r

    nw = k // 128
    outs = ([entry(i, j, False) for i in range(m) for j in range(n)] + [entry(i, j, True) for i in range(m) for j in range(n)]
            + [row_a(i, post8_at(k, row_in(k, False)) + 32 * (5 * nw + 1) + 32) for i in range(m)]
            + [row_b(j, post8_at(k, row_in(k, True)) + 32 * (5 * nw + 35) + 64) for j in range(n)])
    return nin, widths, blocks, outs


def reads_of(g: CallGraph, base: int) -> list[set[int]]:
    """Each computing gate's reads as gate numbers: an input leaf by its id, a computing gate `c` as `base + c`."""
    reads = [set() for _ in range(g.n)]
    for s, d in zip(g.src, g.dst):
        reads[d].add(base + s)
    for ix, d in g.input_reads:
        reads[d].add(g.input_ids[ix])
    return reads


def bound_ports(fn, unit, template, tin: int) -> dict[int, int]:
    """Member `unit` of `fn`'s root body against its template's one-Call Program, gate by gate in the flat layout: the same
    primitive, every operand inside the template its own gate at the member's offset, and every parameter leaf `a` read
    from one producer. Returns each `a`'s producer, as a flat gate of `fn`'s one-Call Program."""
    c, call = call_scope(fn)
    ct, callt = call_scope(template)
    assert callt.offset == tin
    nd = fn.body.nodes[unit.node]
    base = call.offset + fn.body.offsets[unit.node] + unit.member * nd.fn.gates
    bound: dict[int, int] = {}
    for r in range(template.gates):
        gt, gw = ct.gate(tin + r), c.gate(base + r)
        assert gw.prim.id == gt.prim.id and len(gw.operands) == len(gt.operands), (unit, r)
        for ot, ow in zip(gt.operands, gw.operands):
            if ot >= tin:
                assert ow == base + (ot - tin), (unit, r, ot, ow)
            else:
                assert bound.setdefault(ot, ow) == ow, (unit, r, ot)
    return bound


CASES = [("sm120", 2, 128, 3, (1, 1)), ("sm120", 4, 128, 4, (2, 2)), ("h100", 2, 256, 2, (1, 2))]


def check_whole(fn, unit_cut, units, tp, layout) -> None:
    """`fn`, a Program of unit templates `tp` (by role) with `unit_cut` and `units` its scheme's, against a Lean
    layout (`lean_layout`'s four parts)."""
    g, owner, verdict = unit_cut(fn)
    assert verdict.ok, verdict
    nin, widths, blocks, outs = layout
    assert sorted(g.input_ids) == list(range(nin))
    c, call = call_scope(fn)
    assert call.offset == nin
    assert [c.gate(r).prim.ret.w for r in range(nin)] == widths
    whole = reads_of(g, nin)
    names = {v: s for s, v in g.prims.items()}
    compact = {j: nin + i for i, j in enumerate(g.keep)}
    us = units(fn)
    assert [u.role for u in us] == [r for r, cnt, *_ in blocks for _ in range(cnt)]
    at, unit = 0, 0
    for role, cnt, tin, ln, port in blocks:
        tg = CallGraph(tp[role])
        assert sorted(tg.input_ids) == list(range(tin))
        assert tg.n == ln
        tnames = {v: s for s, v in tg.prims.items()}
        local = reads_of(tg, tin)
        o = nin + at
        for q in range(cnt):
            reloc = lambda a: port(q, a) if a < tin else o + q * ln + (a - tin)   # noqa: E731
            for r in range(ln):
                gc = at + q * ln + r
                assert owner[gc] == unit + q
                assert g.width[gc] == tg.width[r]
                assert names[g.prim[gc]] == tnames[tg.prim[r]]
                assert whole[gc] == {reloc(a) for a in local[r]}, (role, q, r)
            bound = bound_ports(fn, us[unit + q], tp[role], tin)
            assert sorted(bound) == list(range(tin))
            assert all(port(q, a) == (b if b < nin else compact[b - nin]) for a, b in bound.items()), (role, q)
        at += cnt * ln
        unit += cnt
    assert at == g.n
    assert [nin + x for x in g.outputs] == outs


def check_layout(dev: str, m: int, k: int, n: int, tile: tuple[int, int]) -> None:
    check_whole(pc8.check(pc8.Shape(m, k, n, tile, dev)), pc8.unit_cut, pc8.units, pc8.templates(k, tile, dev),
                lean_layout(m, k, n, *tile))


@pytest.mark.parametrize("dev,m,k,n,tile", CASES)
def test_the_lean_whole_table_is_pc8_check(dev, m, k, n, tile):
    check_layout(dev, m, k, n, tile)


@pytest.mark.parametrize("dev,m,k,n,tile", CASES)
def test_the_checked_programs_are_pinned(dev, m, k, n, tile):
    assert program_digest(Program(pc8.check(pc8.Shape(m, k, n, tile, dev)))) == DIGESTS[dev, m, k, n, tile]


@pytest.mark.slow
def test_the_lean_whole_table_at_the_deployed_tile():
    """Catches a wiring the small tiles don't exercise, such as one gated on the tile being `pc8.TILE`: one 64 x 64 tile
    over 64 rows of A and B at K = 128 (about 300k gates)."""
    assert pc8.TILE == (64, 64)
    check_layout("sm120", 64, 128, 64, (64, 64))


@pytest.mark.slow
def test_the_lean_whole_table_at_decodes_tile():
    """Catches a wiring gated on `TM = 64` that the small tiles don't exercise: decode's 32 x 64 tile, the audit grid's
    at m = 32 and so the tile `tileCheck8` reads there (`tileSide`), over 32 rows of A and 64 of B at K = 128."""
    check_layout("sm120", 32, 128, 64, (32, 64))
