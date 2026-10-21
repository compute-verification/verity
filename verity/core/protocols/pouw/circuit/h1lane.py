"""A served H1 leaf proved as its lanes and a combine (PRF-134, option A): the drawn unit stays the R × C leaf with its one
XOR receipt, and its proof is one statement a lane plus one that combines them, each small enough to prove alone.

* **A lane** of the leaf's H1 tile (`h1tile`) is lane (g, q), LANE = g Q + q, G = R / 4 and Q = C / 8: rows ``g + G s``
  (s < 4) and columns ``2 q + 2 Q (s >> 1) + (s & 1)`` (s < 8), at 32 × 32 rows g + 8 s and columns 2 q + 8 (s >> 1) + (s & 1).
  `PouwH1LaneZy_v1{TILE, LANE}` is ``(a: its 4 A rows, b: its 8 B rows, zy: the leaf's whole record, seeds: the leaf's
  [seed_j, σ_R]) -> {partial, ok}``. TILE is the leaf's tile (`h1tile.PouwH1Tile` v1 or v3 on its full R × C grid,
  `h1zy.TILES`), which fixes R, C and K. The lane takes its rows' and columns' noise blocks from seed_j at their offsets in
  the R × C layout (`h1tile.Layout.row`, `col`), and its fold keys
  at block LANE of σ_R's stream, so its sets and its fold are the tile's for that lane, gate for gate.
* **The partial** is the lane's share of CP-W: after set s, word 2 s is the XOR of the lane's chains' y (`PouwXinFinal` of
  its state's first half) and word 2 s + 1 of z, 2 (KS + m) words. The tile's receipt is the XOR over its lanes of their
  partials (`h1tile._sets`), so the receipt check is the combine's.
* **ok** is `h1zy.PouwZyCheck{4, 8}` on the lane's 32 cells: their record z and y, picked from the leaf's record at the
  lanes' public cell indices (`cells`), against the lane's z (v1: each cell's exact product; v3: its one-stream z from the
  lane's O' and D words with its wrap correction and dither peel) and its rows' s_a and columns' s_w. Each cell of the
  record is one lane's, so the 32 lanes check every cell once.
* **The combine** `PouwH1Combine_v1{N, W}` is ``(p: N rows of W u32, receipt: W u32) -> ok``, 1 exactly when the XOR of the
  rows is the receipt word for word (`segment.PouwEq32`, `h1zy.PouwAndBit`).
* **The seeds** are the leaf's one `seeds.PouwUnitSeeds` (`_v2`, on bits) on its salt, position and roots, a unit of its own:
  a position seeds one unit (`ml.circuits.ports`, ``E_COIN_REUSE``), and the lanes share the leaf's seed_j and σ_R, as the
  tile's rows and columns do. No lane reads the coin or the position.
* **The leaf** `PouwH1Leaf_v1{TILE}` is the leaf's Program (`program`): ``(a: R rows, b: C rows, zy, receipt, salt,
  position, roots)``, its seeds, its lanes in order, each on its rows of ``a`` and ``b`` and the seeds, then the combine on
  their partials and the leaf's committed ``receipt``; it returns each lane's {partial, ok} and the combine's ok. Its
  `query` makes the seeds, each lane and the combine a unit (unit 0 the seeds, units 1 .. N the lanes, N + 1 the combine),
  one template each, so one statement each. The seed pair and each lane's partial reach their readers as wires between
  units of two statements: the registered route carries them as values produced after the draw (`one_stage.produced`),
  the producer's output row and the reader's input row opening one root (`PRODUCED_SEEDS`, `PRODUCED`, `PARTIAL_PORT`).

`lane_reference` is a lane on words, ``(partial, ok)``, and `combine_reference` the combine.
"""
from __future__ import annotations

from typing import Any, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits import ports as _ports
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll, Explicit, array_of, concat
from verity.ml.circuits.types import Array, Record, Tuple

from verity.core.protocols.sampled_proofs.one_stage import partition as P

from . import h1tile as H
from . import h1zy as Z
from . import seeds as SD

SG = H.SG
BIT, U32 = BL.BIT, H.U32
M32 = (1 << 32) - 1
#: a lane's rows and columns
LANE_R, LANE_C = 4, 8
#: the leaf's seed pair [seed_j, σ_R] on bits, the seeds unit's result and every lane's ``seeds``
SEEDS = BL.bit_type(SD.SEED_PAIR)
#: the produced values of a leaf, by its result field or template: its one seed pair, and its partials, one row a lane
PRODUCED_SEEDS, PRODUCED = "PouwUnitSeeds_v2", "partial"
#: the lane statement's output row of its partial, and the combine's input row of lane l's (`partial_port`)
PARTIAL_PORT = "o0"


def partial_port(lane: int) -> str:
    """The combine statement's input row of lane ``lane``'s partial: its parameter ``p[lane]``'s port."""
    return f"p{int(lane)}"


# -- the tile's shape ---------------------------------------------------------------------------------------------------------

def _tile(TILE: Any) -> tuple[int, int, int]:
    """(R, C, K) of an H1 tile bound to its full grid."""
    (_, a), (_, b) = TILE.signature()[0][:2]
    return a.n, b.n, a.elem.n


def _refuse_tile(b: dict) -> str | None:
    t = b.get("TILE")
    if getattr(getattr(t, "defn", None), "id", None) not in Z.TILES:
        return f"TILE is an H1 tile ({', '.join(Z.TILES)}) bound to its full grid, not {getattr(t, 'id', t)!r}"
    r, c = t.bindings["R"], t.bindings["C"]
    if tuple(t.bindings["PAIRS"]) != tuple((i, j) for i in range(r) for j in range(c)):
        return f"TILE's z is its full {r} × {c} grid in row-major order, the record's cells"
    return None


def _refuse_lane(b: dict) -> str | None:
    why = _refuse_tile(b)
    if why:
        return why
    lanes = H.Layout(*_tile(b["TILE"])).lanes
    ln = b.get("LANE")
    if not (isinstance(ln, int) and not isinstance(ln, bool) and 0 <= ln < lanes):
        return f"LANE is one of the tile's {lanes} lanes, not {ln!r}"
    return None


def _stream(TILE: Any) -> bool:
    return TILE.defn.id == "PouwH1Tile_v3"


def layout(TILE: Any) -> H.Layout:
    return H.Layout(*_tile(TILE))


def cells(lay: H.Layout, lane: int) -> list[int]:
    """Lane ``lane``'s 32 cells as indices of the R × C record (row-major), in the order of its 4 × 8 sub-tile (row slot
    major): the cells its ok checks."""
    rows, cols = lay.rows(lane), lay.cols(lane)
    return [rows[sr] * lay.c + cols[sc] for sr in range(LANE_R) for sc in range(LANE_C)]


# -- the lane in gates --------------------------------------------------------------------------------------------------------

def _blocks(B: Any) -> Any:
    """``at(seed, first, n)``: blocks ``first .. first + n - 1`` of the seed's noise stream, each counter a constant of the
    body (`seeds.noise_blocks` at an offset)."""
    bit: dict[int, Any] = {}

    def const(v: int) -> Any:
        if v not in bit:
            bit[v] = B.call(BL.constant(v)).refs[0]
        return bit[v]

    def at(seed: Coll, first: int, n: int) -> list[Coll]:
        return [B.call(SD.PouwNoiseBlockBits, seed,
                       Coll(SD.POS_BITS, Explicit(tuple(const(v) for v in SD.position_leaves(t, True)))))
                for t in range(first, first + n)]

    return at


def _bits(blocks: list[Coll], n: int) -> Coll:
    return Coll(Array(n, BIT), concat([b.refs for b in blocks]).slice(0, n))


def _codes(rows: Coll, n: int) -> list[Coll]:
    return [rows[i].field("codes") for i in range(n)]


def _scales(rows: Coll, n: int) -> Coll:
    return Coll(Array(n, Z.F32), concat([rows[i].field("scale").refs for i in range(n)]))


def _lane_signature(S: Any) -> tuple:
    r, c, k = _tile(S.TILE)
    return ((("a", Array(LANE_R, Z.ROW(k))), ("b", Array(LANE_C, Z.ROW(k))), ("zy", Z.RECORD(r, c)), ("seeds", SEEDS)),
            Record(("partial", Array(2 * H.Layout(r, c, k).sets, U32)), ("ok", BIT)))


def _lane_body(B: Any, S: Any, a: Coll, b: Coll, zy: Coll, seeds: Coll) -> Coll:
    lay = layout(S.TILE)
    k, ks, ln = lay.k, lay.ks, S.LANE
    rows, cols = lay.rows(ln), lay.cols(ln)
    seed_j, sigma_r = seeds[0], seeds[1]
    _blocks_at = _blocks(B)
    ac, bc = _codes(a, LANE_R), _codes(b, LANE_C)
    X, A, XIN = [], [], []
    for s in range(LANE_R):
        nz = _bits(_blocks_at(seed_j, lay.row(rows[s]), lay.rb), lay.row_bits)
        X.append(B.batch(SG.PouwI8AddBit, ac[s], nz[0:k], axes=(0, 0)))
        XIN.append((ac[s], nz[0:k]))
        A.append(H._digits(B, X[-1], [nz[k + 32 * d:k + 32 * d + 32] for d in range(lay.m - 1)], lay))
    W, U, NU, WIN = [], [], [], []
    for s in range(LANE_C):
        nz = _bits(_blocks_at(seed_j, lay.col(cols[s]), lay.cb), lay.col_bits)
        U.append(B.batch(H.PouwU129, Coll(Array(32, H.I8), nz[k:k + H.U_BITS].refs), axes=(0,)))
        NU.append(B.batch(SG.PouwI8Neg, U[-1], axes=(0,)))
        wc = B.batch(SG.PouwI8AddBit, bc[s], nz[0:k], axes=(0, 0))
        ur = Coll(Array(k, H.I8), concat([U[-1].refs] * ks))
        W.append(B.batch(SG.PouwI8Add, wc, ur, axes=(0, 0)))
        WIN.append((bc[s], nz[0:k], ur))
    rs, cs = [SG.row_slot(e) for e in range(32)], [SG.col_slot(e) for e in range(32)]
    T, prev = [], None
    for s in range(ks):
        xs = [X[i][32 * s:32 * s + 32] for i in range(LANE_R)]
        ws = [W[j][32 * s:32 * s + 32] for j in range(LANE_C)]
        prev = [B.call(SG.PouwSegDot0, xs[rs[e]], ws[cs[e]]) if prev is None else
                B.call(SG.PouwSegDot, prev[e], xs[rs[e]], ws[cs[e]]) for e in range(32)]
        T.append(prev)
    op = [B.call(SG.PouwSegDot, T[-1][e], NU[cs[e]], A[rs[e]][0]) for e in range(32)]
    D = [[B.call(SG.PouwSegDot0, U[cs[e]], A[rs[e]][d]) for e in range(32)] for d in range(1, lay.m)]
    state = Coll(Array(H.KEY_BITS, BIT), _blocks_at(sigma_r, lay.lane(ln), 1)[0].refs)
    partial = []
    for p in [*T, op, *D]:
        state = B.call(SG.PouwFx1Absorb, state, array_of(p))
        partial += [B.call(SG.PouwXinFinal, state[0:256]).refs, B.call(SG.PouwXinFinal, state[256:512]).refs]
    entry = {(rs[e], cs[e]): e for e in range(32)}
    order = [entry[(sr, sc)] for sr in range(LANE_R) for sc in range(LANE_C)]
    if _stream(S.TILE):
        f = H.ONE_STREAM[lay.m]
        wb = [B.batch(H.PouwWrapBits, *WIN[j], axes=(0, 0, 0)) for j in range(LANE_C)]
        xa = [B.batch(H.PouwXWrap, *XIN[i], axes=(0, 0)) for i in range(LANE_R)]
        z = []
        for e in order:
            i, j = rs[e], cs[e]
            acc = B.call(f, op[e], *(d[e] for d in D))
            for s in range(ks):
                acc = B.call(H.PouwWrapSeg, acc, X[i][32 * s:32 * s + 32], wb[j][32 * s:32 * s + 32])
            for s in range(ks):
                seg = slice(32 * s, 32 * s + 32)
                acc = B.call(H.PouwPeelSeg, acc, X[i][seg], XIN[i][1][seg], xa[i][seg], WIN[j][0][seg], WIN[j][1][seg])
            z.append(acc.refs)
    else:
        exact = S.TILE.bindings["EXACT"]
        z = [B.call(exact, ac[rs[e]], bc[cs[e]]).refs for e in order]
    mine = cells(lay, ln)
    zr, yr = zy.field("z"), zy.field("y")
    sub = Coll(Z.RECORD(LANE_R, LANE_C), concat([zr[t].refs for t in mine] + [yr[t].refs for t in mine]))
    ok = B.call(bind(Z.PouwZyCheck, R=LANE_R, C=LANE_C), sub, Coll(Array(32, Z.S32), concat(z)), _scales(a, LANE_R),
                _scales(b, LANE_C))
    return Coll(_lane_signature(S)[1], concat(partial + [ok.refs]))


def _lane_ports(S: Any) -> Any:
    return _ports.Ports(ret={"partial": _ports.Out(sink=True)})


PouwH1LaneZy = CompositeDefinition(
    "PouwH1LaneZy", 1, ("TILE", "LANE"), _lane_signature, _lane_body, conformance="protocol-defined", refuse=_refuse_lane,
    register=SG is not None, ports=_lane_ports,
    doc="Lane LANE of the H1 tile TILE with its cells' z/y record checked (`circuit.h1lane`): its 4 A rows and 8 B rows, the "
        "leaf's whole record and the leaf's [seed_j, σ_R], to {partial: the lane's CP-W share, its XinFinal of each state half "
        "after each set, whose XOR over the lanes is TILE's receipt; ok: PouwZyCheck{4, 8} on its 32 cells of the record}.")


def lane(tile: Any, ln: int) -> Any:
    """Lane ``ln`` of ``tile`` (an H1 tile bound to its full grid, `h1window.h1_tile`'s)."""
    return bind(PouwH1LaneZy, TILE=tile, LANE=int(ln))


# -- the combine --------------------------------------------------------------------------------------------------------------

def _refuse_nw(b: dict) -> str | None:
    n, w = b.get("N"), b.get("W")
    if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (n, w)):
        return f"N and W are positive integers, not {n!r} and {w!r}"
    return None


def _combine_signature(S: Any) -> tuple:
    return (("p", Array(S.N, Array(S.W, U32))), ("receipt", Array(S.W, U32))), BIT


def _combine_body(B: Any, S: Any, p: Coll, receipt: Coll) -> Coll:
    from .boolean import PouwXor32
    acc = p[0]
    for ln in range(1, S.N):
        acc = B.batch(PouwXor32, acc, p[ln], axes=(0, 0))
    eq = B.batch(SG.PouwEq32, acc, receipt, axes=(0, 0))
    return Z._and_all(B, [eq[w] for w in range(S.W)])


def _combine_ports(S: Any) -> Any:
    return _ports.Ports(params={"p": _ports.In(receipt=True)}, ret=_ports.Out(sink=True))


PouwH1Combine = CompositeDefinition(
    "PouwH1Combine", 1, ("N", "W"), _combine_signature, _combine_body, conformance="protocol-defined", refuse=_refuse_nw,
    ports=_combine_ports,
    doc="1 exactly when the XOR of the N rows p (W u32 each) is the receipt, word for word (`circuit.h1lane`): a leaf's "
        "lanes' partials against its committed receipt.")


def combine(lay: H.Layout) -> Any:
    """The combine of ``lay``'s lanes: N its lanes, W its receipt's words."""
    return bind(PouwH1Combine, N=lay.lanes, W=2 * lay.sets)


# -- the leaf -----------------------------------------------------------------------------------------------------------------

#: the leaf's parameters, in order
LEAF_PARAMS = ("a", "b", "zy", "receipt", "salt", "position", "roots")


def _leaf_signature(S: Any) -> tuple:
    r, c, k = _tile(S.TILE)
    lay = H.Layout(r, c, k)
    params = S.TILE.signature()[0]
    lanes = [lane(S.TILE, ln).signature()[1] for ln in range(lay.lanes)]
    return ((("a", Array(r, Z.ROW(k))), ("b", Array(c, Z.ROW(k))), ("zy", Z.RECORD(r, c)),
             ("receipt", Array(2 * lay.sets, U32)), *params[2:]),
            Tuple(*lanes, BIT))


def _pick(rows: Coll, which: Sequence[int]) -> Coll:
    return Coll(Array(len(which), rows[0].type), concat([rows[i].refs for i in which]))


def _leaf_body(B: Any, S: Any, a: Coll, b: Coll, zy: Coll, receipt: Coll, salt: Coll, position: Coll, roots: Coll) -> Coll:
    lay = layout(S.TILE)
    seeds = B.call(SD.PouwUnitSeedsBits, salt, position, roots)
    outs = [B.call(lane(S.TILE, ln), _pick(a, lay.rows(ln)), _pick(b, lay.cols(ln)), zy, seeds) for ln in range(lay.lanes)]
    ok = B.call(combine(lay), array_of([o.field("partial") for o in outs]), receipt)
    return Coll(_leaf_signature(S)[1], concat([o.refs for o in outs] + [ok.refs]))


def _leaf_ports(S: Any) -> Any:
    width = lane(S.TILE, 0).signature()[1]
    pw = width.fields[0][1].leaves
    n = layout(S.TILE).lanes
    sinks = tuple((ln * width.leaves, ln * width.leaves + pw, _ports.Out(sink=True)) for ln in range(n))
    return _ports.Ports(params={"salt": _ports.In(kind="coin"), "position": _ports.In(kind="position")},
                        ret=_ports.Leaves(sinks + ((n * width.leaves, n * width.leaves + 1, _ports.Out(sink=True)),)))


PouwH1Leaf = CompositeDefinition(
    "PouwH1Leaf", 1, ("TILE",), _leaf_signature, _leaf_body, conformance="protocol-defined", refuse=_refuse_tile,
    register=False, ports=_leaf_ports,
    doc="One H1 leaf as its lanes and a combine (`circuit.h1lane`): each lane of TILE on its rows of a and b, the leaf's "
        "record, salt, position and roots, then PouwH1Combine on the lanes' partials and the leaf's committed receipt; "
        "each lane's {partial, ok}, then the combine's ok.")


def program(tile: Any) -> Program:
    """The leaf's Program over ``tile`` (an H1 tile bound to its full grid)."""
    return Program(bind(PouwH1Leaf, TILE=tile))


def templates(tile: Any) -> list[Any]:
    """The leaf's templates, in unit order: its seeds, its lanes, then the combine."""
    lay = layout(tile)
    return [bind(SD.PouwUnitSeedsBits)] + [lane(tile, ln) for ln in range(lay.lanes)] + [combine(lay)]


def query(tile: Any) -> dict[str, Any]:
    """The leaf's partition query: the seeds, each lane and the combine one unit (unit 0 the seeds, units 1 .. N the lanes
    in order, N + 1 the combine)."""
    return P.template_instances_query(templates(tile))


# -- on words -----------------------------------------------------------------------------------------------------------------

def lane_noise(seed_j: bytes, lay: H.Layout, ln: int) -> tuple[dict, dict]:
    """Lane ``ln``'s rows' and columns' noise from seed_j, by tile row and column (`h1tile.noise`'s, at their offsets)."""
    rows, cols, k = {}, {}, lay.k
    for i in lay.rows(ln):
        s = H._stream(seed_j, lay.row(i), lay.row_bits)
        rows[i] = (s[:k], [s[k + 32 * d:k + 32 * d + 32] for d in range(lay.m - 1)])
    for j in lay.cols(ln):
        s = H._stream(seed_j, lay.col(j), lay.col_bits)
        cols[j] = (s[:k], [H.u129(H._value(s[k + 8 * t:k + 8 * t + 8])) for t in range(32)])
    return rows, cols


def lane_reference(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], sa: Sequence[int], sw: Sequence[int],
                   zy: tuple[Sequence[int], Sequence[int]], salt: bytes, position: int, roots: tuple[bytes, bytes],
                   lay: H.Layout, ln: int) -> tuple[list[int], int]:
    """Lane ``ln`` on words: (partial, ok) for its 4 A rows and 8 B rows (s8 encodings, row and column slot order), their
    scales (f32 encodings), the leaf's whole record ``zy`` (z s32 words, y bf16 encodings, R × C row-major), the leaf's
    salt, position and roots. z is the exact product (v1's and v3's are both A Bᵀ)."""
    rows_i, cols_j = lay.rows(ln), lay.cols(ln)
    A_, B_ = dict(zip(rows_i, a)), dict(zip(cols_j, b))
    rows, cols = lane_noise(SD.seed(salt, position, *roots), lay, ln)
    sets = H.lane_sets(A_, B_, rows, cols, lay, ln)
    st = list(H.fold_keys(SD.receipt_seed(salt, position, *roots), H.Layout(lay.r, lay.c, lay.k))[ln])
    partial = []
    for p in sets:
        st = SG.absorb(H.FOLD, st, p)
        for h in range(2):
            v = 0
            for w in st[8 * h:8 * h + 8]:
                v ^= w
            partial.append(v & M32)
    mine = cells(lay, ln)
    z = [SG.seg_dot(0, a[sr], b[sc]) for sr in range(LANE_R) for sc in range(LANE_C)]
    ok = Z.check([zy[0][t] for t in mine], [zy[1][t] for t in mine], z, sa, sw)
    return partial, ok


def combine_reference(partials: Sequence[Sequence[int]], receipt: Sequence[int]) -> int:
    """The combine on words: 1 exactly when the XOR of ``partials`` is ``receipt``."""
    acc = [0] * len(receipt)
    for p in partials:
        acc = [x ^ (y & M32) for x, y in zip(acc, p)]
    return int(len(partials) > 0 and all(len(p) == len(receipt) for p in partials)
               and acc == [w & M32 for w in receipt])


__all__ = ["LANE_C", "LANE_R", "LEAF_PARAMS", "PARTIAL_PORT", "PRODUCED", "PouwH1Combine",
           "PouwH1LaneZy", "PouwH1Leaf", "cells", "combine", "combine_reference", "lane", "lane_noise", "lane_reference",
           "layout", "partial_port", "program", "query", "templates"]
