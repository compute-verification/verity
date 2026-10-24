"""H1's z/y records checked in the tile unit (RT4-INT-33): the served `verity_int8::scaled_mm_zy` commits each R × C
unit's record (#3053's z table, kernels' #3018), and `PouwH1TileZy` reads its unit's record as an input and checks both
halves: `Accounting.Served.ValueChecked`'s product and dequant are checked by the unit, under Served's checked-record form
(pending), rather than taken on the fused kernel's word.

* **The record** of a unit is its exact s32 z_u = A_u B_uᵀ over the whole K, its cells row-major, then
  y_u = dq(z_u, s_a, s_w) in bf16, row-major, little-endian (`int8_exact.zy_records`; 6,144 bytes at the served 32 × 32).
  As the unit's ``zy`` it is one bit row of 48 R C bits, `RECORD`'s leaves: z's R C words, then y's. It is a run input,
  committed in its call's z table under round s's root before round s's coin (`h1window.table`), not a registered value
  in the declared circuit's sense, which is committed before coin 0.
* **Types.** A record's z words are s32 (`S32`, two's complement, [-2^31, 2^31 - 1]) and a row's codes s8 (`S8`): an
  extractor that read z as u32 would see z mod 2^32, and the product of a negative dot would fail. The circuit's types are
  bits either way; the names say how each word reads.
* **The dequant** is the served order, `int8_exact.dequantize` and the fused kernel's epilogue:
  y = RN_bf16(RN32(RN32(RN32(z) · s_a) · s_w)), the activation scale first, with no bias add. PoUW's `NcpDequant` (and
  `reference.dequant`) is another function: bf16(RN32(RN32(s · t) · RN32(z))) + bias multiplies the scales together first,
  which f32 rounding doesn't commute with, and its bf16 bias add maps -0 to +0. So it is a Definition of its own,
  `PouwZyDequant_v2`, circuits' three Boolean pieces in that order (`S32ToF32Rn_v2`, two `F32Mul_v4`, `F2fpBf16_v2`, the
  epilogue of `ScaledMmS8Coordinate_v2`), its word view `PouwZyDequant_v1` on their words (`dequant`). Every NaN is 0x7FFF
  (``cvt.rn.bf16.f32``); torch's CPU cast gives 0x7FC0, the one encoding where the CPU oracle and the GPU differ.
* **The scales.** A unit's A row is its K codes and then its f32 s_a, little-endian (#3053's A table leaf, `ROW`), and its
  B row the K weight codes and then their f32 s_w: each scale is committed with its row, so a unit reads the A row's scale
  at the A row's position and the weight row's at the weight row's (`Accounting.Served.DequantOf`: s_a with the A row, s_w
  registered).
* **The unit** `PouwH1TileZy_v1{TILE}` is ``(a: R rows, b: C rows, zy, salt, position, roots) -> {receipt, ok}``: TILE, the
  H1 tile (`h1tile.PouwH1Tile` v1 or v3 over the full grid), on the rows' codes gives z and the receipt; `PouwZyCheck_v1`
  holds the record to them, ``ok`` 1 exactly when every cell's record z is z and its record y is `PouwZyDequant_v2` of z at
  its row's s_a and its column's s_w. Every cell is dequantized, a padded row's too: an empty slot's y is dq(0, +0, s_w),
  which is +0 only for a finite positive s_w (0x8000 for a negative one, 0x7FFF for an infinite or NaN one), so no cell's
  y is taken to be zero. The z and y the unit computes are inner words: the served values are the record's, committed
  before the coin, and a proof's outputs are committed after it, so the unit returns the receipt and one bit.
* **ok** is a check word, no served value (no ``outZ``, ``outY`` or ``served`` role; Served's ``check`` class, pending), read
  through a row of known content: the verifier holds it to 1 under a public salt (h1_prove's `ok_salt`), so a proof the
  verifier accepts is one whose unit computed 1, and a record that is not the unit's product and its dequant has no
  accepted proof. The ok rows are the verifier's own registered value ``ok`` (h1_prove's `register_ok`, #3124: row u the
  word 1 under unit u's public salt), and each unit's ok port ``o1`` reads it at the unit's row through ``--registered``,
  so Lean opens each ok row against that root rather than reading ok off the public file alone (proofs'
  ``CheckedSessionSound``, #3108, binds ok only through a port of ``own``). A lane's and a combine's ok take the same
  form: each statement's ok port (``o0``, its one output) reads the verifier's own ``ok``, one row of the word 1 a
  statement under its public salt, beside the ``partials`` ports.
* **A leaf's lanes** (option A: the served path keeps one receipt a 32 × 32 leaf, and a drawn leaf is proved as its 32
  lanes and their combine). Option A is a proof strategy, not a new Served form (RT4-LOGIC-30): a drawn leaf is one Cd
  unit and meets Served as `PouwH1TileZy_v1` does, and BRIDGE2's linked realization carries the 33 statements to it.
  CP-W's receipt is the XOR over the leaf's lanes of their partials (`h1tile.fold`). The partials are proof-layer values,
  not Cd inputs: the prover commits them after the draw under hiding leaves, like a proof's own outputs, one row per lane
  (``partials``, row l lane l; proofs' service piece, `one_stage.produced`), and the 33 statements open that one
  commitment as registered ports of the verifier's own, lane l's ``partials`` at row l and the combine's ``p0`` .. ``p31``
  at rows 0 .. 31 (#3116's `pouw.leaf_lanes.reads`; each statement's ok is its ``o0``), since a port ``own`` doesn't name
  is free in #3108's relation (RT4-LOGIC-29(c)). A partial is 2 (KS + m) u32 (8,320 bits at K = 4096), so it is an
  input, never an output, and a row of 8 (KS + m) bytes. The combine reads only those rows and the receipt, and C-Flock
  proves a statement only with a row that compresses before its last block, so `combine` is staged only at K >= 384
  (`LEAST_K`: 112-byte rows, whose padding spills into a second block); `lane` takes any positive K. Both stage only a
  leaf whose R is a multiple of 16, since a lane reads the R admit bits as one row of whole 16-bit words.
  `PouwH1LaneZy_v1{TILE}` is lane l of TILE's leaf, ``(a: its 4 A rows
  by row slot, b: its 8 B rows by column slot, admit: the leaf's R admit bits, zy: the leaf's whole record, partials: its
  row, salt, position, roots, lane) -> {ok}``. Its noise is the leaf's seed_j stream at its rows' and columns' blocks
  (`h1tile.Layout`) and its keys block l of σ_R's, so its chain's words are the lane's partial as the served receipt XORs
  them (kernels' ``lane_partials``, #3095), each masked by v4's gate, the admit bits' OR (`PouwOrBit`, `PouwMask32`): a
  dead leaf's partials are zero. ok is 1 exactly when its partials row is those words and its 32 cells of the record
  (`lane_cells`: by row slot, then column slot) pass `PouwZyCheck_v1{4, 8}` at its rows' z (TILE's v1 or v3 form) and
  scales, a dead leaf's cells too (z = 0, y = dq(0, +0, s_w)). The index l is a public input, one 16-bit word (`LANE`:
  C-Flock's public rows are whole 16-bit words): each noise block's counter and each of the lane's cells is picked from
  the leaf's by a `PouwMuxBit` tree over its low `lane_bits` bits, so one Definition at each K is every lane of a leaf,
  and ok is 0 at an l past the leaf's lanes (any higher bit set: their `PouwOrBit`s select a 0 by a `PouwMuxBit`, 16 -
  `lane_bits` ANDs). ok is a sink (`Out(sink=True)`): a Definition with a coin and a
  position port that returns no sink is a seed, whose body no port rule follows (`ports._coin_role`), and the lane, a
  unit, would be one. `PouwH1Combine_v1{TILE}` reads the 32 ``partials`` rows and the leaf's committed ``receipt`` and
  returns ok, 1 exactly when the rows' XOR is the receipt word by word, so the 33 oks hold the receipt as the tile unit's
  receipt output does. The XOR is free and the equality is 64 (KS + m) - 1 ANDs (8,319 at K = 4096, 895 at K = 384).

`records` gives a unit's honest record on words (`scaled_mm_zy`'s for its rows and scales), `reference` the unit on words,
``(receipt, ok)``, `lane_partial` a lane's honest partials row, and `lane_reference` and `combine_reference` a lane's and the
combine's ok. `statement_rows` is a lane's and the combine's input rows as `class_statement.unit_lowering` takes them, and
`statement_ports` their C-Flock ports, each input's and the output's bits in its port's row, which Lean states as
`PouwWindow.Lanes` (with each port's registered row, and each unit input wire's port and bit) for proofs' decoders.
"""
from __future__ import annotations

from typing import Any, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits import ports as _ports
from verity.ml.circuits.boolean.forms import NOT, and_reduce
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind, composite
from verity.ml.circuits.ops import fp32 as F
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.refs import Coll, Explicit, concat
from verity.ml.circuits.types import Array, Record, Value

from . import h1tile as H
from . import seeds as SD

BIT = BL.BIT
#: an s8 code, an s32 product, an f32 scale and a bf16 output, each its bits
S8, S32, F32, BF16 = Array(8, BIT), Array(32, BIT), Array(32, BIT), Array(16, BIT)
M32 = (1 << 32) - 1


def ROW(k: int) -> Record:
    """An A or B row of the unit: K s8 codes, then the row's f32 scale (8 K + 32 bits, the codes' bytes then the scale's,
    little-endian)."""
    return Record(("codes", Array(int(k), S8)), ("scale", F32))


def RECORD(r: int, c: int) -> Record:
    """A unit's z/y record: its R C cells' s32 z, row-major, then their bf16 y (48 R C bits)."""
    return Record(("z", Array(int(r) * int(c), S32)), ("y", Array(int(r) * int(c), BF16)))


# -- the dequant --------------------------------------------------------------------------------------------------------------

def dequant(z: int, sa: int, sw: int) -> int:
    """RN_bf16(RN32(RN32(RN32(z) · sa) · sw)) on encodings (z s32, sa and sw f32, the result bf16), circuits' word evaluators:
    `S32ToF32Rn_v1`, `F32Mul_v2` twice, `F2fpBf16_v1`."""
    return P.F2fpBf16.evaluate(F.F32MulV2.evaluate(F.F32MulV2.evaluate(P.S32ToF32Rn.evaluate(z & M32), sa & M32), sw & M32))


@composite("PouwZyDequant", 1, [], lambda S: ((("z", P.S32), ("sa", P.F32), ("sw", P.F32)), P.BF16),
           conformance="protocol-defined: the served epilogue, int8_exact.dequantize and scaled_mm_zy's y, "
                       "RN_bf16(RN32(RN32(RN32(z) * s_a) * s_w)), the activation scale first, no bias; every NaN 0x7FFF")
def PouwZyDequantWord(B, S, z, sa, sw):
    """`F2fpBf16_v1(F32Mul_v2(F32Mul_v2(S32ToF32Rn_v1(z), sa), sw))`."""
    return B.call(P.F2fpBf16, B.call(F.F32MulV2, B.call(F.F32MulV2, B.call(P.S32ToF32Rn, z), sa), sw))


@composite("PouwZyDequant", 2, [], lambda S: ((("z", S32), ("sa", F32), ("sw", F32)), BF16),
           conformance="protocol-defined: PouwZyDequant_v1 on bits")
def PouwZyDequant(B, S, z, sa, sw):
    """`F2fpBf16_v2(F32Mul_v4(F32Mul_v4(S32ToF32Rn_v2(z), sa), sw))`: `PouwZyDequant_v1` on bits, the epilogue of
    `ScaledMmS8Coordinate_v2`."""
    return B.call(BG.F2fpBf16, B.call(EW.F32MulV4, B.call(EW.F32MulV4, B.call(BG.S32ToF32Rn, z), sa), sw))


PouwZyDequant.word = PouwZyDequantWord  # type: ignore[attr-defined]


# -- the check ----------------------------------------------------------------------------------------------------------------

def zy_cell(zr: int, yr: int, z: int, y: int) -> int:
    """1 exactly when a record's cell (zr, yr) is (z, y)."""
    return int((zr & M32) == (z & M32) and (yr & 0xFFFF) == (y & 0xFFFF))


PouwZyCellWord = PrimitiveDefinition("PouwZyCell", 1, (("zr", Value(32)), ("yr", Value(16)), ("z", Value(32)), ("y", Value(16))),
                                     Value(1), zy_cell, conformance="protocol-defined")
PouwZyCell = definition("PouwZyCell", 2,
                        lambda C_, zr, yr, z, y: [and_reduce(C_, [NOT(p ^ q) for p, q in zip(list(zr) + list(yr),
                                                                                             list(z) + list(y))])],
                        (("zr", S32), ("yr", BF16), ("z", S32), ("y", BF16)), BIT, word=PouwZyCellWord,
                        conformance="protocol-defined", doc="zr = z and yr = y, on bits: PouwZyCell_v1")
PouwAndBitWord = PrimitiveDefinition("PouwAndBit", 1, (("a", Value(1)), ("b", Value(1))), Value(1),
                                     lambda a, b: a & b & 1, conformance="protocol-defined")
PouwAndBit = definition("PouwAndBit", 2, lambda C_, a, b: [C_.AND(a[0], b[0])], (("a", BIT), ("b", BIT)), BIT,
                        word=PouwAndBitWord, conformance="protocol-defined", doc="a AND b on bits: PouwAndBit_v1")


def _check_signature(S: Any) -> tuple:
    n = S.R * S.C
    return (("zy", RECORD(S.R, S.C)), ("z", Array(n, S32)), ("sa", Array(S.R, F32)), ("sw", Array(S.C, F32))), BIT


def _refuse_rc(b: dict) -> str | None:
    r, c = b.get("R"), b.get("C")
    if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (r, c)):
        return f"R and C are positive integers, not {r!r} and {c!r}"
    return None


def _and_all(B: Any, bits: list) -> Coll:
    while len(bits) > 1:
        bits = [B.call(PouwAndBit, bits[i], bits[i + 1]) if i + 1 < len(bits) else bits[i] for i in range(0, len(bits), 2)]
    return bits[0]


@composite("PouwZyCheck", 1, ["R", "C"], _check_signature, conformance="protocol-defined", refuse=_refuse_rc)
def PouwZyCheck(B, S, zy, z, sa, sw):
    """1 exactly when each cell (i, j) of an R × C unit, row-major, has record z = z and record y =
    `PouwZyDequant_v2`(z, sa[i], sw[j]): one dequant and one `PouwZyCell` a cell, batched, then their AND (`PouwAndBit`)."""
    n = S.R * S.C
    sa_c = Coll(Array(n, F32), concat([sa[i].refs for i in range(S.R) for _j in range(S.C)]))
    sw_c = Coll(Array(n, F32), concat([sw[j].refs for _i in range(S.R) for j in range(S.C)]))
    y = B.batch(PouwZyDequant, z, sa_c, sw_c, axes=(0, 0, 0))
    ok = B.batch(PouwZyCell, zy.field("z"), zy.field("y"), z, y, axes=(0, 0, 0, 0))
    return _and_all(B, [ok[t] for t in range(n)])


# -- the unit -----------------------------------------------------------------------------------------------------------------

#: the tile units the record check takes: (a, b, salt, position, roots) -> {z, receipt}
TILES = ("PouwH1Tile_v1", "PouwH1Tile_v3")


def _tile(S: Any) -> tuple[int, int, int]:
    (_, a), (_, b) = S.TILE.signature()[0][:2]
    return a.n, b.n, a.elem.n


def _refuse_tile(b: dict) -> str | None:
    t = b.get("TILE")
    if getattr(getattr(t, "defn", None), "id", None) not in TILES:
        return f"TILE is an H1 tile ({', '.join(TILES)}) bound to its full grid, not {getattr(t, 'id', t)!r}"
    r, c = t.bindings["R"], t.bindings["C"]
    if tuple(t.bindings["PAIRS"]) != tuple((i, j) for i in range(r) for j in range(c)):
        return f"TILE's z is its full {r} × {c} grid in row-major order, the record's cells"
    return None


def _signature(S: Any) -> tuple:
    r, c, k = _tile(S)
    params, ret = S.TILE.signature()
    return ((("a", Array(r, ROW(k))), ("b", Array(c, ROW(k))), ("zy", RECORD(r, c)), *params[2:]),
            Record(("receipt", ret.fields[1][1]), ("ok", BIT)))


def _rows(a: Coll, n: int, field: str) -> Coll:
    return Coll(Array(n, a[0].field(field).type), concat([a[i].field(field).refs for i in range(n)]))


def _body(B: Any, S: Any, a: Coll, b: Coll, zy: Coll, salt: Coll, position: Coll, roots: Coll) -> Coll:
    r, c, _k = _tile(S)
    out = B.call(S.TILE, _rows(a, r, "codes"), _rows(b, c, "codes"), salt, position, roots)
    ok = B.call(bind(PouwZyCheck, R=r, C=c), zy, out.field("z"), _rows(a, r, "scale"), _rows(b, c, "scale"))
    return Coll(_signature(S)[1], concat([out.field("receipt").refs, ok.refs]))


def _unit_ports(S: Any) -> Any:
    return _ports.Ports(params={"salt": _ports.In(kind="coin"), "position": _ports.In(kind="position")},
                        ret={"receipt": _ports.Out(sink=True)})


PouwH1TileZy = CompositeDefinition(
    "PouwH1TileZy", 1, ("TILE",), _signature, _body, conformance="protocol-defined", refuse=_refuse_tile,
    register=H.SG is not None, ports=_unit_ports,
    doc="The H1 tile with its z/y record checked (`circuit.h1zy`): TILE on the A and B rows' codes, the salt, position and "
        "roots gives z and the receipt; the unit returns the receipt and ok, 1 exactly when the record's z is z and its y "
        "is PouwZyDequant_v2 of z at each cell's row scale s_a and column scale s_w, each the f32 after its row's codes.")


def unit(tile: Any) -> Any:
    """`PouwH1TileZy` over ``tile`` (an H1 tile unit bound to its full grid, `h1window.tile_unit`'s)."""
    return bind(PouwH1TileZy, TILE=tile)


# -- a leaf's lanes and their combine ------------------------------------------------------------------------------------------

PouwMuxBitWord = PrimitiveDefinition("PouwMuxBit", 1, (("s", Value(1)), ("a", Value(1)), ("b", Value(1))), Value(1),
                                     lambda s, a, b: (b if s & 1 else a) & 1, conformance="protocol-defined")
PouwMuxBit = definition("PouwMuxBit", 2, lambda C_, s, a, b: [a[0] ^ C_.AND(s[0], a[0] ^ b[0])],
                        (("s", BIT), ("a", BIT), ("b", BIT)), BIT, word=PouwMuxBitWord, conformance="protocol-defined",
                        doc="b where s is 1, else a, on bits: PouwMuxBit_v1")


#: a lane's public index: one 16-bit word, since C-Flock's public rows are whole 16-bit words
LANE = Array(16, BIT)


def lane_bits(r: int, c: int) -> int:
    """The bits of a public lane index of an R × C leaf, whose (R / 4) (C / 8) lanes are a power of two: the low bits of
    its `LANE` word."""
    return ((r // 4) * (c // 8)).bit_length() - 1


def lane_cells(lay: H.Layout, ln: int) -> list[int]:
    """Lane ln's cells as indices of its leaf's record (row-major), by its row slot then its column slot."""
    return [i * lay.c + j for i in lay.rows(ln) for j in lay.cols(ln)]


def _refuse_lane(b: dict) -> str | None:
    why = _refuse_tile(b)
    if why:
        return why
    r, c = b["TILE"].bindings["R"], b["TILE"].bindings["C"]
    k = b["TILE"].signature()[0][0][1].elem.n
    if k < 64:
        return f"K = {k} is under 64: H1's lanes need KS = K / 32 >= 2 (a digit word)"
    g, q = r // 4, c // 8
    if r % 4 or c % 8 or g * q < 2 or g & (g - 1) or q & (q - 1) or g * q > 1 << LANE.n:
        return (f"a leaf's lanes are a public index of whole bits: R / 4 and C / 8 powers of two, at least two lanes and at "
                f"most 2^{LANE.n}, not {r} × {c}")
    return None


def _lane_layout(S: Any) -> H.Layout:
    r, c, k = _tile(S)
    return H.Layout(r, c, k)


def _words(lay: H.Layout) -> Array:
    """A lane's partial, or the leaf's receipt: 2 (KS + m) u32."""
    return Array(2 * lay.sets, H.U32)


def _lane_signature(S: Any) -> tuple:
    lay = _lane_layout(S)
    params, _ret = S.TILE.signature()
    return ((("a", Array(4, ROW(lay.k))), ("b", Array(8, ROW(lay.k))), ("admit", Array(lay.r, BIT)),
             ("zy", RECORD(lay.r, lay.c)), ("partials", _words(lay)), *params[2:],
             ("lane", LANE)), Record(("ok", BIT)))


def _pick(B: Any, sel: list, cands: list) -> Coll:
    """cands[sum_b sel[b] 2^b]: a `PouwMuxBit` tree over the selector's bits, least significant first."""
    for s in sel:
        cands = [B.batch(PouwMuxBit, s, cands[i], cands[i + 1], axes=(None, 0, 0)) for i in range(0, len(cands), 2)]
    return cands[0]


def _noise(B: Any, seed: Coll, sel: list, firsts: list[int], n: int, const: Any) -> Coll:
    """The first n bits of seed's stream from block firsts[sel]: each block's u64 counter picked from the constants."""
    blocks = []
    for t in range(-(-n // SD.BLOCK_BITS)):
        cands = [Coll(Array(64, BIT), Explicit(tuple(const(v) for v in SD.position_leaves(f + t, True)))) for f in firsts]
        blocks.append(B.call(SD.PouwNoiseBlockBits, seed, Coll(SD.POS_BITS, _pick(B, sel, cands).refs)))
    return Coll(Array(n, BIT), concat([blk.refs for blk in blocks]).slice(0, n))


def _lane_body(B: Any, S: Any, a: Coll, b: Coll, admit: Coll, zy: Coll, partials: Coll, salt: Coll, position: Coll,
               roots: Coll, lane: Coll) -> Coll:
    lay = _lane_layout(S)
    G, Q = lay.r // 4, lay.c // 8
    sel = [lane[t] for t in range(lane_bits(lay.r, lay.c))]
    qb, bit = Q.bit_length() - 1, {}

    def const(v: int) -> Any:
        if v not in bit:
            bit[v] = B.call(BL.constant(v)).refs[0]
        return bit[v]

    seed_j, sigma_r = SD.unit_seeds_call(B, salt, position, roots, True)
    codes = [a[s].field("codes") for s in range(4)], [b[t].field("codes") for t in range(8)]
    X, A, XIN = [], [], []
    for s in range(4):
        nz = _noise(B, seed_j, sel[qb:], [lay.row(g + G * s) for g in range(G)], lay.row_bits, const)
        x, d, xin = H._row(B, codes[0][s], nz, lay)
        X.append(x)
        A.append(d)
        XIN.append(xin)
    W, U, NU, WIN = [], [], [], []
    for t in range(8):
        nz = _noise(B, seed_j, sel[:qb], [lay.col(2 * q + 2 * Q * (t >> 1) + (t & 1)) for q in range(Q)], lay.col_bits, const)
        u, nu, w, win = H._col(B, codes[1][t], nz, lay)
        U.append(u)
        NU.append(nu)
        W.append(w)
        WIN.append(win)
    key = _noise(B, sigma_r, sel, [lay.lane(ln) for ln in range(lay.lanes)], H.KEY_BITS, const)
    cp, words = H._chain(B, lay, X, W, U, NU, A, key)
    cells = [(s, t) for s in range(4) for t in range(8)]
    if S.TILE.defn.id == "PouwH1Tile_v3":
        entry = {(H.SG.row_slot(e), H.SG.col_slot(e)): e for e in range(32)}
        wb = [B.batch(H.PouwWrapBits, *WIN[t], axes=(0, 0, 0)) for t in range(8)]
        xa = [B.batch(H.PouwXWrap, *XIN[s], axes=(0, 0)) for s in range(4)]
        z = [H._cell_z(B, lay, words[entry[c]], X[c[0]], XIN[c[0]], xa[c[0]], WIN[c[1]], wb[c[1]]).refs for c in cells]
    else:
        z = [B.call(S.TILE.bindings["EXACT"], codes[0][s], codes[1][t]).refs for s, t in cells]
    zf, yf = zy.field("z"), zy.field("y")
    picked = []
    for at in range(32):
        cands = [Coll(Array(48, BIT), concat([zf[u].refs, yf[u].refs])) for u in (lane_cells(lay, ln)[at]
                                                                                    for ln in range(lay.lanes))]
        picked.append(_pick(B, sel, cands))
    rec = Coll(RECORD(4, 8), concat([p[0:32].refs for p in picked] + [p[32:48].refs for p in picked]))
    ok = B.call(bind(PouwZyCheck, R=4, C=8), rec, Coll(Array(32, S32), concat(z)), _rows(a, 4, "scale"),
                _rows(b, 8, "scale"))
    live = [admit[i] for i in range(lay.r)]
    while len(live) > 1:
        live = [B.call(H.PouwOrBit, live[i], live[i + 1]) if i + 1 < len(live) else live[i] for i in range(0, len(live), 2)]
    chain = B.batch(H.PouwMask32, Coll(_words(lay), concat([w.refs for pair in cp for w in pair])), live[0], axes=(0, None))
    eq = B.batch(H.SG.PouwEq32, partials, chain, axes=(0, 0))
    high = [lane[t] for t in range(len(sel), LANE.n)]
    while len(high) > 1:
        high = [B.call(H.PouwOrBit, high[i], high[i + 1]) if i + 1 < len(high) else high[i] for i in range(0, len(high), 2)]
    checked = _and_all(B, [ok] + [eq[w] for w in range(2 * lay.sets)])
    return Coll(_lane_signature(S)[1], B.call(PouwMuxBit, high[0], checked, Coll(BIT, Explicit((const(0),)))).refs)


def _lane_ports(S: Any) -> Any:
    return _ports.Ports(params={"salt": _ports.In(kind="coin"), "position": _ports.In(kind="position")},
                        ret={"ok": _ports.Out(sink=True)})


PouwH1LaneZy = CompositeDefinition(
    "PouwH1LaneZy", 1, ("TILE",), _lane_signature, _lane_body, conformance="protocol-defined", refuse=_refuse_lane,
    register=H.SG is not None, ports=_lane_ports,
    doc="One lane of TILE's R × C leaf (`circuit.h1zy`): the lane's 4 A rows and 8 B rows (by row and column slot), the "
        "leaf's R admit bits, its whole z/y record, the lane's committed partial, the leaf's salt, position and roots, and "
        "the public lane index, a 16-bit word. Its noise is the leaf's seed_j stream at its rows' and columns' blocks and its "
        "keys block lane of σ_R's, so its chain's 2 (KS + m) words are the lane's part of the receipt, each AND the admit "
        "bits' OR. ok is 1 exactly when the index is one of the leaf's lanes, the partial is those words and its 32 cells "
        "of the record are its rows' product z (TILE's v1 or v3 form) and PouwZyDequant_v2 of z at each cell's row and "
        "column scale.")


def _combine_signature(S: Any) -> tuple:
    lay = _lane_layout(S)
    return (("partials", Array(lay.lanes, _words(lay))), ("receipt", _words(lay))), Record(("ok", BIT))


def _combine_body(B: Any, S: Any, partials: Coll, receipt: Coll) -> Coll:
    lay = _lane_layout(S)
    xor = Coll(_words(lay), concat([H._xor(B, [partials[ln][w] for ln in range(lay.lanes)]).refs
                                    for w in range(2 * lay.sets)]))
    eq = B.batch(H.SG.PouwEq32, xor, receipt, axes=(0, 0))
    return Coll(_combine_signature(S)[1], _and_all(B, [eq[w] for w in range(2 * lay.sets)]).refs)


PouwH1Combine = CompositeDefinition(
    "PouwH1Combine", 1, ("TILE",), _combine_signature, _combine_body, conformance="protocol-defined", refuse=_refuse_lane,
    register=H.SG is not None,
    doc="The receipt check of TILE's leaf over its lanes' committed partials (`circuit.h1zy`): ok is 1 exactly when, word "
        "by word, the XOR over the lanes in lane order of each partial's word (PouwXor32_v2) is the leaf's committed "
        "receipt's.")


#: the least K at which a leaf's combine is proved (`combine`); a lane is proved at any positive K
LEAST_K = 384


def _leaf(tile: Any) -> int:
    """``tile``'s K, if its leaf's lanes and combine are staged; else ValueError. A leaf has a positive K (`leaf_lanes`'
    word order), and a lane reads its R admit bits as one row, which C-Flock takes only as whole 16-bit words
    (`class_statement.unit_lowering`), so its R is a multiple of 16 (the served leaf's is 32)."""
    a = tile.signature()[0][0][1]
    r, k = int(a.n), int(a.elem.n)
    if k < 1:
        raise ValueError(f"K = {k}: a leaf's lanes and combine are staged only at a positive K")
    if r % 16:
        raise ValueError(f"R = {r}: a lane reads the leaf's R admit bits as one row, and C-Flock takes only a row of whole "
                         f"16-bit words (class_statement.unit_lowering): a leaf's lanes and combine are staged only at an R "
                         f"that is a multiple of 16, as the served 32 × 32 leaf's is")
    return k


def lane(tile: Any) -> Any:
    """`PouwH1LaneZy` over ``tile`` (the leaf's H1 tile bound to its full grid, `h1window.h1_tile`'s): the lane a prover
    stages, at any K and R `_leaf` takes."""
    _leaf(tile)
    return bind(PouwH1LaneZy, TILE=tile)


def combine(tile: Any) -> Any:
    """`PouwH1Combine` over ``tile``, the leaf's H1 tile, at an R `_leaf` takes and a K of at least `LEAST_K`: the combine
    reads only the partials and the receipt, 8 (KS + m)-byte rows, and C-Flock proves a statement only with a row that
    compresses before its last block (``no row has a compression before its last block``, `verity_flock.circuit`'s and
    `Flock.HmRow`'s), a row of at least 112 bytes, whose 17 bytes of SHA-512 padding spill past its first block."""
    k = _leaf(tile)
    n = 8 * H.Layout(4, 8, k).sets
    if n + 17 <= 128:
        raise ValueError(f"K = {k}: the combine reads only a lane's partials and the receipt, {n}-byte rows, but C-Flock "
                         f"proves a statement only with a row that compresses before its last block, at least 112 bytes: "
                         f"K = {LEAST_K} is the least K the combine is proved at")
    return bind(PouwH1Combine, TILE=tile)


# -- the statements' ports ------------------------------------------------------------------------------------------------------

#: a lane's public row ``pub``: the leaf's salt, position and roots and the lane's index, in that order (#3116's
#: `pouw.leaf_lanes.PUBLIC`)
PUBLIC = ("salt", "position", "roots", "lane")


def _fields(path: str, t: Any, bit: int) -> list[tuple[str, int, int]]:
    """``t`` from bit ``bit`` of a row as its fields ``(path, first bit, bits)``: a record by its fields, anything else whole."""
    out = []
    for p, ft in ([(f"{path}.{n}", ft) for n, ft in t.fields] if isinstance(t, Record) else [(path, t)]):
        out.append((p, bit, sum(ft.leaf_widths())))
        bit += out[-1][2]
    return out


def statement_rows(d: Any) -> list[tuple[str, list[str]]]:
    """The input rows of a leaf's lane or combine statement, ``d`` `PouwH1LaneZy` or `PouwH1Combine` bound to its TILE, in
    order, each ``(port, params)`` as `class_statement.unit_lowering` takes them (``a[0]`` row 0 of ``a``): a lane's ``a0``
    .. ``a3`` (its A rows by row slot), ``b0`` .. ``b7``, ``admit``, ``zy``, ``partials`` and ``pub`` (`PUBLIC`); the
    combine's ``p0`` .. ``p<lanes - 1>`` (its partials rows) and ``receipt``; as #3116's `pouw.leaf_lanes` reads them."""
    ty = dict(d.signature()[0])
    if d.defn is PouwH1LaneZy:
        rows = ([(f"{n}{i}", [f"{n}[{i}]"]) for n in ("a", "b") for i in range(ty[n].n)]
                + [(n, [n]) for n in ("admit", "zy", "partials")] + [("pub", list(PUBLIC))])
    elif d.defn is PouwH1Combine:
        rows = [(f"p{x}", [f"partials[{x}]"]) for x in range(ty["partials"].n)] + [("receipt", ["receipt"])]
    else:
        raise ValueError(f"a leaf's statement is PouwH1LaneZy_v1 or PouwH1Combine_v1, not {d.defn.id}")
    if {p.partition("[")[0] for _port, ps in rows for p in ps} != set(ty):
        raise AssertionError(f"{d.defn.id}: the ports {rows} are not its parameters {list(ty)}")
    return rows


def statement_ports(d: Any) -> list[tuple[str, list[tuple[str, int, int]]]]:
    """The C-Flock ports of a leaf's lane or combine statement (``d`` as `statement_rows` takes it, at any K), in order:
    each input row of `statement_rows` with its fields ``(path, first bit, bits)`` in its bit row, the Definition's inputs
    it holds, their leaves concatenated in signature order (`class_statement.unit_lowering`; ``a[0].codes`` the codes of
    row 0 of ``a``), and last the output port ``o0``, ``ok``. Lean states it as `PouwWindow.Lanes`."""
    params, ret = d.signature()
    ty = dict(params)
    out = []
    for port, ps in statement_rows(d):
        fields: list[tuple[str, int, int]] = []
        for p in ps:
            name, _, row = p.partition("[")
            fields += _fields(p, ty[name].elem if row else ty[name], fields[-1][1] + fields[-1][2] if fields else 0)
        out.append((port, fields))
    return out + [("o0", _fields("ok", ret.fields[0][1], 0))]


# -- on words -----------------------------------------------------------------------------------------------------------------

def _sx(v: int, bits: int) -> int:
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def records(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], sa: Sequence[int],
            sw: Sequence[int]) -> tuple[list[int], list[int]]:
    """A unit's honest record on words, what `scaled_mm_zy` writes for its rows: z, each cell's exact A Bᵀ (s32, its
    two's-complement word), and y, each cell's `dequant` at its row's s_a and its column's s_w (bf16 encodings), both
    row-major. ``a`` and ``b`` are s8 code rows (encodings), ``sa`` and ``sw`` f32 encodings."""
    z = [sum(_sx(x, 8) * _sx(w, 8) for x, w in zip(ai, bj)) & M32 for ai in a for bj in b]
    return z, [dequant(z[i * len(b) + j], sa[i], sw[j]) for i in range(len(a)) for j in range(len(b))]


def check(zr: Sequence[int], yr: Sequence[int], z: Sequence[int], sa: Sequence[int], sw: Sequence[int]) -> int:
    """`PouwZyCheck` on words: 1 exactly when record (zr, yr) is z and its dequant, cell by cell."""
    c = len(sw)
    return int(len(zr) == len(yr) == len(z) == len(sa) * c and all(
        zy_cell(zr[t], yr[t], z[t], dequant(z[t], sa[t // c], sw[t % c])) for t in range(len(z))))


def reference(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], sa: Sequence[int], sw: Sequence[int],
              zy: tuple[Sequence[int], Sequence[int]], salt: bytes, position: int, roots: tuple[bytes, bytes],
              stream: bool = False) -> tuple[list[int], int]:
    """The unit on words: (receipt, ok) for A's rows and B's rows (s8 encodings), their scales (f32 encodings), the record
    ``zy`` (z s32 words, y bf16 encodings, row-major), the salt, position and roots; ``stream``: over v3 (`h1tile`)."""
    z, receipt = H.reference(a, b, salt, position, roots, stream=stream)
    return receipt, check(zy[0], zy[1], z, sa, sw)


def lane_partial(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], admit: Sequence[int], salt: bytes, position: int,
                 roots: tuple[bytes, bytes], lane: int, tile: tuple[int, int]) -> list[int]:
    """Lane ``lane``'s honest partial on words, the row the prover commits for it, from its 4 A rows and 8 B rows by slot
    (s8 encodings) and the leaf's R admit bits: `h1tile.lane_partials`' for the lane, zero words for a dead leaf."""
    lay = H.Layout(int(tile[0]), int(tile[1]), len(a[0]))
    if isinstance(admit, (bool, int)) or len(admit) != lay.r:
        raise ValueError(f"admit is the leaf's {lay.r} admit bits, not {admit!r}")
    if not any(g & 1 for g in admit):
        return [0] * (2 * lay.sets)
    rows, cols = H.noise(SD.seed(salt, position, *roots), lay)
    af: list = [None] * lay.r
    bf: list = [None] * lay.c
    for s, i in enumerate(lay.rows(lane)):
        af[i] = a[s]
    for t, j in enumerate(lay.cols(lane)):
        bf[j] = b[t]
    keys = H.fold_keys(SD.receipt_seed(salt, position, *roots), lay)
    return H.fold(keys[lane], H.lane_sets(af, bf, rows, cols, lay, lane))


def lane_reference(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], sa: Sequence[int], sw: Sequence[int],
                   admit: Sequence[int], zy: tuple[Sequence[int], Sequence[int]], partial: Sequence[int], salt: bytes,
                   position: int, roots: tuple[bytes, bytes], lane: int, tile: tuple[int, int]) -> int:
    """`PouwH1LaneZy` on words: ok of lane ``lane`` of an R × C ``tile``'s leaf, from the lane's 4 A rows and 8 B rows by
    slot (s8 encodings) and their scales, the leaf's R admit bits, its record ``zy`` (all R C cells), the lane's committed
    ``partial``, and the leaf's salt, position and roots: 1 exactly when ``lane`` is one of the leaf's lanes, the partial
    is `lane_partial`'s and the lane's cells of the record pass `check`; ``lane`` a `LANE` word."""
    lay = H.Layout(int(tile[0]), int(tile[1]), len(a[0]))
    if not 0 <= lane < 1 << LANE.n:
        raise ValueError(f"a lane index is a {LANE.n}-bit word, not {lane}")
    if lane >= lay.lanes:
        return 0
    want = lane_partial(a, b, admit, salt, position, roots, lane, tile)
    z = [H.SG.seg_dot(0, a[s], b[t]) for s in range(4) for t in range(8)]
    cells = lane_cells(lay, lane)
    return int([v & M32 for v in partial] == want and check([zy[0][u] for u in cells], [zy[1][u] for u in cells], z, sa, sw))


def combine_reference(partials: Sequence[Sequence[int]], receipt: Sequence[int]) -> int:
    """`PouwH1Combine` on words: 1 exactly when the lanes' partials XORed word by word are ``receipt``."""
    out = [0] * len(receipt)
    for p in partials:
        out = [x ^ (v & M32) for x, v in zip(out, p, strict=True)]
    return int(out == [v & M32 for v in receipt])


__all__ = ["BF16", "F32", "LANE", "LEAST_K", "PUBLIC", "S32", "S8", "PouwAndBit", "PouwH1Combine", "PouwH1LaneZy",
           "PouwH1TileZy", "PouwMuxBit", "PouwZyCell", "PouwZyCheck", "PouwZyDequant", "PouwZyDequantWord", "RECORD", "ROW",
           "TILES", "check", "combine", "combine_reference", "dequant", "lane", "lane_bits", "lane_cells", "lane_partial",
           "lane_reference", "records", "reference", "statement_ports", "statement_rows", "unit", "zy_cell"]
