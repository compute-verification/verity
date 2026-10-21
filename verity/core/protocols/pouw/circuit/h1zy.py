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
  accepted proof.

`records` gives a unit's honest record on words (`scaled_mm_zy`'s for its rows and scales), and `reference` the unit on
words, ``(receipt, ok)``.
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
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array, Record, Value

from . import h1tile as H

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


__all__ = ["BF16", "F32", "S32", "S8", "PouwAndBit", "PouwH1TileZy", "PouwZyCell", "PouwZyCheck", "PouwZyDequant", "PouwZyDequantWord",
           "RECORD", "ROW", "TILES", "check", "dequant", "records", "reference", "unit", "zy_cell"]
