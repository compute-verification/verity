"""H1's z/y records checked in the tile unit: the served `verity_int8::scaled_mm_zy` commits each R × C
unit's record (its call's z table, `h1window.TABLES`), and `PouwH1TileZy` reads its unit's record as an input and checks both
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
* **The scales.** A unit's A row is its K codes and then its f32 s_a, little-endian (its A table leaf, `ROW`), and its
  B row the K weight codes and then their f32 s_w: each scale is committed with its row, so a unit reads the A row's scale
  at the A row's position and the weight row's at the weight row's (`Accounting.Served.DequantOf`: s_a with the A row, s_w
  registered).
* **The unit** `PouwH1TileZy_v1{TILE}` is ``(a: R rows, b: C rows, zy, salt, position, roots) -> {receipt, ok}``: TILE, the
  H1 tile (`h1tile.PouwH1Tile` v1 or v3 over the full grid; or J10-R's v5 or v6, `J10R_TILES`, whose ``call`` and
  ``blocks`` the unit takes after ``roots``), on the rows' codes gives z and the receipt; `PouwZyCheck_v1`
  holds the record to them, ``ok`` 1 exactly when every cell's record z is z and its record y is `PouwZyDequant_v2` of z at
  its row's s_a and its column's s_w. Every cell is dequantized, a padded row's too: an empty slot's y is dq(0, +0, s_w),
  which is +0 only for a finite positive s_w (0x8000 for a negative one, 0x7FFF for an infinite or NaN one), so no cell's
  y is taken to be zero. The z and y the unit computes are inner words: the served values are the record's, committed
  before the coin, and a proof's outputs are committed after it, so the unit returns the receipt and one bit.
* **ok** is a check word, no served value (no ``outZ``, ``outY`` or ``served`` role; Served's ``check`` class, pending), read
  through a row of known content: the verifier holds it to 1 under a public salt (h1_prove's `ok_salt`), so a proof the
  verifier accepts is one whose unit computed 1, and a record that is not the unit's product and its dequant has no
  accepted proof. The ok rows are the verifier's own registered value ``ok`` (h1_prove's `register_ok`: row u the
  word 1 under unit u's public salt), and each unit's ok port ``o1`` reads it at the unit's row through ``--registered``,
  so Lean opens each ok row against that root rather than reading ok off the public file alone (``CheckedSessionSound``,
  `Security/Proofs/Accounting/TS/CheckedSound.lean`, binds ok only through a port of ``own``). A lane's and a combine's ok take the same
  form: each statement's ok port (``o0``, its one output) reads the verifier's own ``ok``, one row of the word 1 a
  statement under its public salt, beside the ``partials`` ports.
* **A leaf's lanes** (option A: the served path keeps one receipt a 32 × 32 leaf, and a drawn leaf is proved as its 32
  lanes and their combine). Option A is a proof strategy, not a new Served form: a drawn leaf is one Cd
  unit and meets Served as `PouwH1TileZy_v1` does, and `Accounting.Guarantees.TS.LeafLanesSound`'s linked realization carries the 33 statements to it.
  CP-W's receipt is the XOR over the leaf's lanes of their partials (`h1tile.fold`). The partials are proof-layer values,
  not Cd inputs: the prover commits them after the draw under hiding leaves, like a proof's own outputs, one row per lane
  (``partials``, row l lane l; proofs' service piece, `one_stage.produced`), and the 33 statements open that one
  commitment as registered ports of the verifier's own, lane l's ``partials`` at row l and the combine's ``p0`` .. ``p31``
  at rows 0 .. 31 (`pouw.leaf_lanes.reads`; each statement's ok is its ``o0``), since a port ``own`` doesn't name
  is free in ``CheckedSessionSound``'s relation. A partial is 2 (KS + m) u32 (8,320 bits at K = 4096), so it is an
  input, never an output, and a row of 8 (KS + m) bytes. The combine reads only those rows and the receipt, and C-Flock
  proves a statement only with a row that compresses before its last block, so `combine` is staged only at K >= 384
  (`LEAST_K`: 112-byte rows, whose padding spills into a second block); `lane` takes any positive K. Both stage only a
  leaf whose R is a multiple of 16, since a lane reads the R admit bits as one row of whole 16-bit words.
  `PouwH1LaneZy_v1{TILE}` is lane l of TILE's leaf, ``(a: its 4 A rows
  by row slot, b: its 8 B rows by column slot, admit: the leaf's R admit bits, zy: the leaf's whole record, partials: its
  row, salt, position, roots, lane) -> {ok}``. Its noise is the leaf's seed_j stream at its rows' and columns' blocks
  (`h1tile.Layout`) and its keys block l of σ_R's, so its chain's words are the lane's partial as the served receipt XORs
  them (the kernels' ``lane_partials``), each masked by v4's gate, the admit bits' OR (`PouwOrBit`, `PouwMask32`): a
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
* **A lane in chunks.** A lane past K = `CHUNK_MAX` (4096, whose lane fills C-Flock's 2^27 block) outgrows the block, so
  it is proved as `chunks`(K) = n statements along K (`lane_statements`): `PouwH1LaneZyChunk_v1{TILE, CHUNKS = n,
  CHUNK = c}` over entries [c K_c, (c + 1) K_c), K_c = K / n a multiple of `CHUNK_ALIGN` (6144: 2 × 3072, 12288: 3 × 4096,
  14336: 4 × 3584, 16384: 4 × 4096). A chunk reads its 4 A and 8 B rows at its entries, each with the row's scale
  (`ROW`(K_c)), the leaf's salt, position and roots and the index. A table at such a K commits each row as those n
  chunk rows, row r's chunk c its leaf n r + c (`chunk_leaves`, `chunk_words`). Its noise is the whole lane's (`h1tile.Layout` at K):
  its entries' dither and coin bits, and U at offset K of each column's stream, so every chunk derives the same U. Its
  chain is the whole lane's, continued: chunk c > 0 reads ``carry_in``, what chunk c - 1 hands on (`CARRY`: the fx1
  state, each entry's prefix sum T, each row slot's residue sums, each cell's z, the chain's words so far, unmasked, and
  the rows' scales), and chunk c < n - 1 holds its committed ``carry_out`` to what it computed. Chunk 0 starts the chain
  from block l of σ_R's stream and z by `GemmAccumulatorS8_v2{K_c}`, TILE's EXACT at K_c; a later chunk continues z by
  EXACT's own DOT steps from the carried z, so the last chunk's z is EXACT{K}'s bit for bit. The last chunk alone takes
  the digit coins, absorbs O' and D_1 .. D_(m-1), and reads ``admit``, ``zy`` and ``partials``, with the whole lane's
  checks: its words masked by the admit gate are the partials row, and its cells pass `PouwZyCheck_v1{4, 8}`. Each chunk
  past the first holds its rows' scales to the carried ones, so every bit of each chunk row and carry row is read and a
  token row's chunks agree on its scale. The carries are proof-layer values, committed after the draw like the partials,
  never outputs; the partials and the combine are the whole lane's, so such a leaf is 32 n + 1 statements. Over v1 only:
  v3's z corrects O' and the digits over every segment of K.
* **The unit on chunk rows.** A window at such a K binds `PouwH1TileZyChunks_v1{UNIT, CHUNKS = n}` (`window_unit`), UNIT
  the K's `PouwH1TileZy_v1` over `PouwH1Tile_v1`: the same parameters (``a``, ``b``, ``zy``, ``salt``, ``position``,
  ``roots``) and outputs (``receipt``, ``ok``), its ``a`` the n R chunk rows of its R A rows (`ROW`(K_c), row i's chunk c
  at n i + c, as its table commits them) and its ``b`` the n C of its B rows. It is UNIT on the whole rows, each its
  chunks' codes in turn at chunk 0's scale, and its ok is UNIT's AND every later chunk's scale equal to chunk 0's (as a
  lane's chunks hold each row's scale to the carried one), (n - 1)(R + C) `PouwEq32`s more. Its drawn leaf is UNIT's
  leaf, proved as above, so a chunked K is refused over v3 and v4. Over J10-R's tiles (`J10R_TILES`, v5 and v6) the
  unit on chunk rows is `PouwH1TileZyChunks_v2{UNIT, CHUNKS = n}`: v1's body with UNIT's ``call`` and ``blocks`` after
  ``roots``, passed on as they are. UNIT runs on the whole rows at the whole K, so its noise is the call-scale stream's
  rows and columns from ``blocks`` (`h1tile.j10r_blocks` at the call's slot and the whole K, `Layout`(R, C, K)'s RB and
  CB), `h1tile.j10r_reference`'s word for word: chunking cuts the committed rows, never the noise stream.
* **A J10-R leaf's lanes.** A drawn per-call unit over `PouwH1Tile_v5` is proved as v1's leaf is, its 32 lanes and their
  combine, a lane past `CHUNK_MAX` in `chunks`(K) chunks: `PouwH1LaneZy_v2{TILE}` and `PouwH1LaneZyChunk_v2{TILE, CHUNKS,
  CHUNK}` are v1's lane and lane chunk with the unit's ``call`` and ``blocks`` after ``roots`` (`PUBLIC_J10R`), and
  `PouwH1Combine_v2{TILE}` is v1's combine that also folds the leaf's 32 n statements' oks into its own (an AND tree):
  it reads ``oks``, each lane's and lane chunk's ok row (`OK_WORD`, the word 1 when it computed 1) in statement order
  (`leaf_statements`), and its ok is 1 exactly when v1's is and every ok row is 1. So a J10-R leaf has one output bit
  however many chunks its lanes take, and the other statements' oks are committed values like the partials (each
  statement's ``o0`` opens its row of them), not outputs. Their noise is v5's: seed_c from one
  `seeds.PouwCallSeeds` (every chunk; a chunk past the first leaves its σ_R unread), each block's counter the leaf's row or
  column offset, picked over the index as v1's is, plus ``blocks[0]`` or ``blocks[1]`` (`seeds.PouwAdd64`), so row i's
  blocks are blocks[0] + i RB + t and column j's blocks[1] + j CB + t, `h1tile.j10r_noise`'s; their keys are v1's, block
  l of σ_R's at the unit's own position. So the lanes' partials XOR to `PouwH1Tile_v5`'s receipt, and the rest is v1's.
  Over v5 only: v6's one-stream z, as v3's, has no lane in chunks, and every served J10-R K is past 4096.

`records` gives a unit's honest record on words (`scaled_mm_zy`'s for its rows and scales), `reference` the unit on words,
``(receipt, ok)``, and `chunks_reference` the unit on chunk rows (each with ``j10r``: over J10-R), `lane_partial` a lane's honest partials row, and `lane_reference` and `combine_reference` a lane's and the
combine's ok; `lane_chunk_carries` a lane's honest carries in chunks, and `lane_chunk_reference` a chunk's ok (each a
lane's, with ``j10r``: over J10-R's v5).
`statement_rows` is a lane's, a chunk's and the combine's input rows as `class_statement.unit_lowering` takes them, and
`statement_ports` their C-Flock ports, each input's and the output's bits in its port's row, which Lean states as
`PouwWindow.Lanes` (with each port's registered row, and each unit input wire's port and bit) for proofs' decoders: v1's
lane and combine, and J10-R's lane, lane chunks and combine (`chunkPorts`, `combine2Ports`); not v1's lane chunks.
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
from verity.ml.circuits.refs import Coll, Explicit, array_of, concat
from verity.ml.circuits.types import Array, Record, Value

from verity.core.protocols.pouw.weights import CHUNK_ALIGN, CHUNK_MAX, chunks

from . import h1tile as H
from . import seeds as SD

SG = H.SG
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
#: the J10-R tile units the unit takes too: (a, b, salt, position, roots, call, blocks) -> {z, receipt}
J10R_TILES = ("PouwH1Tile_v5", "PouwH1Tile_v6")


def _tile(S: Any) -> tuple[int, int, int]:
    (_, a), (_, b) = S.TILE.signature()[0][:2]
    return a.n, b.n, a.elem.n


def _refuse_tile(b: dict, tiles: tuple[str, ...] = TILES) -> str | None:
    t = b.get("TILE")
    if getattr(getattr(t, "defn", None), "id", None) not in tiles:
        return f"TILE is an H1 tile ({', '.join(tiles)}) bound to its full grid, not {getattr(t, 'id', t)!r}"
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


def _body(B: Any, S: Any, a: Coll, b: Coll, zy: Coll, *public: Coll) -> Coll:
    r, c, _k = _tile(S)
    out = B.call(S.TILE, _rows(a, r, "codes"), _rows(b, c, "codes"), *public)
    ok = B.call(bind(PouwZyCheck, R=r, C=c), zy, out.field("z"), _rows(a, r, "scale"), _rows(b, c, "scale"))
    return Coll(_signature(S)[1], concat([out.field("receipt").refs, ok.refs]))


def _unit_ports(S: Any) -> Any:
    return _ports.Ports(params={"salt": _ports.In(kind="coin"), "position": _ports.In(kind="position")},
                        ret={"receipt": _ports.Out(sink=True)})


PouwH1TileZy = CompositeDefinition(
    "PouwH1TileZy", 1, ("TILE",), _signature, _body, conformance="protocol-defined",
    refuse=lambda b: _refuse_tile(b, TILES + J10R_TILES), register=H.SG is not None, ports=_unit_ports,
    doc="The H1 tile with its z/y record checked (`circuit.h1zy`): TILE on the A and B rows' codes, the salt, position and "
        "roots (a J10-R TILE's then its call and blocks) gives z and the receipt; the unit returns the receipt and ok, 1 "
        "exactly when the record's z is z and its y is PouwZyDequant_v2 of z at each cell's row scale s_a and column scale "
        "s_w, each the f32 after its row's codes.")


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


def _refuse_lane(b: dict, tiles: tuple[str, ...] = TILES) -> str | None:
    why = _refuse_tile(b, tiles)
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


def _noise(B: Any, seed: Coll, sel: list, firsts: list[int], n: int, const: Any, at: int = 0,
           base: Coll | None = None) -> Coll:
    """Bits [at, at + n) of seed's stream from block firsts[sel]: each block's u64 counter picked from the constants, and
    with ``base`` that counter plus ``base`` (`seeds.PouwAdd64`, J10-R's counters from a unit's public ``blocks``)."""
    t0 = at // SD.BLOCK_BITS
    blocks = []
    for t in range(t0, -(-(at + n) // SD.BLOCK_BITS)):
        cands = [Coll(Array(64, BIT), Explicit(tuple(const(v) for v in SD.position_leaves(f + t, True)))) for f in firsts]
        counter = Coll(SD.POS_BITS, _pick(B, sel, cands).refs)
        if base is not None:
            counter = B.call(SD.PouwAdd64Bits, base, counter)
        blocks.append(B.call(SD.PouwNoiseBlockBits, seed, counter))
    lo = at - t0 * SD.BLOCK_BITS
    return Coll(Array(n, BIT), concat([blk.refs for blk in blocks]).slice(lo, lo + n))


def _lane_seeds(B: Any, S: Any, v: dict, keys: bool) -> tuple[Coll, Coll | None, tuple[Coll, Coll] | None, int]:
    """A lane statement's noise seed, σ_R (with ``keys``), the bases its counters add to (J10-R's ``blocks``, else None)
    and what each column's first block (`h1tile.Layout.col`) less is its counter: seed_j and today's constant counters,
    or under J10-R seed_c from one `seeds.PouwCallSeeds`, row i's blocks from ``blocks[0] + i RB`` and column j's from
    ``blocks[1] + j CB``, as `h1tile`'s v5 derives them. A J10-R chunk past the first takes seed_c from the same call and
    leaves its σ_R unread."""
    lay = _lane_layout(S)
    if S.TILE.defn.id in J10R_TILES:
        seed_c, sigma_r = SD.call_seeds_call(B, v["salt"], v["position"], v["call"], v["roots"], True)
        return seed_c, sigma_r, (v["blocks"][0], v["blocks"][1]), lay.col(0)
    if keys:
        return (*SD.unit_seeds_call(B, v["salt"], v["position"], v["roots"], True), None, 0)
    return SD.seed_call(B, v["salt"], v["position"], v["roots"], True), None, None, 0


def _lane_body(B: Any, S: Any, *args: Coll) -> Coll:
    v = dict(zip([p for p, _t in _lane_signature(S)[0]], args, strict=True))
    a, b, admit, zy, partials, lane = v["a"], v["b"], v["admit"], v["zy"], v["partials"], v["lane"]
    lay = _lane_layout(S)
    G, Q = lay.r // 4, lay.c // 8
    sel = [lane[t] for t in range(lane_bits(lay.r, lay.c))]
    qb, const = Q.bit_length() - 1, _consts(B)
    seed_j, sigma_r, base, col0 = _lane_seeds(B, S, v, True)
    rb, cb = base or (None, None)
    codes = [a[s].field("codes") for s in range(4)], [b[t].field("codes") for t in range(8)]
    X, A, XIN = [], [], []
    for s in range(4):
        nz = _noise(B, seed_j, sel[qb:], [lay.row(g + G * s) for g in range(G)], lay.row_bits, const, base=rb)
        x, d, xin = H._row(B, codes[0][s], nz, lay)
        X.append(x)
        A.append(d)
        XIN.append(xin)
    W, U, NU, WIN = [], [], [], []
    for t in range(8):
        nz = _noise(B, seed_j, sel[:qb], [lay.col(2 * q + 2 * Q * (t >> 1) + (t & 1)) - col0 for q in range(Q)],
                    lay.col_bits, const, base=cb)
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
    ok = _cells_ok(B, lay, sel, zy, z, a, b)
    chain = _gated(B, lay, admit, [w.refs for pair in cp for w in pair])
    eq = B.batch(H.SG.PouwEq32, partials, chain, axes=(0, 0))
    return Coll(_lane_signature(S)[1], _lane_ok(B, lane, sel, [ok] + [eq[w] for w in range(2 * lay.sets)], const).refs)


def _or_all(B: Any, bits: list) -> Coll:
    while len(bits) > 1:
        bits = [B.call(H.PouwOrBit, bits[i], bits[i + 1]) if i + 1 < len(bits) else bits[i] for i in range(0, len(bits), 2)]
    return bits[0]


def _cells_ok(B: Any, lay: H.Layout, sel: list, zy: Coll, z: list, a: Coll, b: Coll) -> Coll:
    """`PouwZyCheck_v1{4, 8}` of the lane's 32 cells of the leaf's record, each picked over the index's low bits, at the
    lane's z (refs, cell by cell) and its rows' scales."""
    zf, yf = zy.field("z"), zy.field("y")
    picked = []
    for at in range(32):
        cands = [Coll(Array(48, BIT), concat([zf[u].refs, yf[u].refs])) for u in (lane_cells(lay, ln)[at]
                                                                                    for ln in range(lay.lanes))]
        picked.append(_pick(B, sel, cands))
    rec = Coll(RECORD(4, 8), concat([p[0:32].refs for p in picked] + [p[32:48].refs for p in picked]))
    return B.call(bind(PouwZyCheck, R=4, C=8), rec, Coll(Array(32, S32), concat(z)), _rows(a, 4, "scale"),
                  _rows(b, 8, "scale"))


def _gated(B: Any, lay: H.Layout, admit: Coll, words: list) -> Coll:
    """The chain's words (refs) each AND the admit bits' OR."""
    live = _or_all(B, [admit[i] for i in range(lay.r)])
    return B.batch(H.PouwMask32, Coll(_words(lay), concat(words)), live, axes=(0, None))


def _lane_ok(B: Any, lane: Coll, sel: list, checks: list, const: Any) -> Coll:
    """The AND of ``checks`` where the index's bits past ``sel`` are all 0, else 0."""
    high = _or_all(B, [lane[t] for t in range(len(sel), LANE.n)])
    return B.call(PouwMuxBit, high, _and_all(B, checks), Coll(BIT, Explicit((const(0),))))


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

#: the J10-R tile a leaf's lanes are proved over: v5's exact dots (v6's one-stream z, as v3's, has no lane in chunks)
J10R_LANE_TILES = ("PouwH1Tile_v5",)


def _refuse_lane_j10r(b: dict) -> str | None:
    t = getattr(getattr(b.get("TILE"), "defn", None), "id", None)
    if t in TILES:
        return f"PouwH1LaneZy_v2 is over J10-R's PouwH1Tile_v5, not {t}: {t}'s lane is PouwH1LaneZy_v1"
    return _refuse_lane(b, J10R_LANE_TILES)


PouwH1LaneZy2 = CompositeDefinition(
    "PouwH1LaneZy", 2, ("TILE",), _lane_signature, _lane_body, conformance="protocol-defined", refuse=_refuse_lane_j10r,
    register=H.SG is not None, ports=_lane_ports,
    doc="v1's lane over J10-R (`circuit.h1zy`): one lane of TILE's R × C leaf, TILE PouwH1Tile_v5, with v1's parameters "
        "and the unit's call and blocks after its roots, before the lane index. Its noise is seed_c's (PouwCallSeeds_v2 "
        "at the call's first position) at its rows' and columns' blocks, each counter the leaf's row or column offset "
        "picked over the index plus blocks[0] or blocks[1] (PouwAdd64_v2), and its keys block lane of σ_R's at the unit's "
        "own position, so its chain's words are the lane's part of PouwH1Tile_v5's receipt. ok is v1's.")


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
    "PouwH1Combine", 1, ("TILE",), _combine_signature, _combine_body, conformance="protocol-defined",
    refuse=_refuse_lane, register=H.SG is not None,
    doc="The receipt check of TILE's leaf over its lanes' committed partials (`circuit.h1zy`): ok is 1 exactly when, word "
        "by word, the XOR over the lanes in lane order of each partial's word (PouwXor32_v2) is the leaf's committed "
        "receipt's. It reads only the leaf's layout, so it is one Definition over v1's and v3's lanes.")

#: a lane or lane chunk statement's ok row, as the combine reads it: one 16-bit word, the ok in bit 0 and the rest 0
OK_WORD = Array(16, BIT)


def leaf_statements(lay: H.Layout) -> int:
    """The lane and lane chunk statements of a leaf, the combine's ``oks``: each lane's `chunks`(K)."""
    return lay.lanes * chunks(lay.k)


def _combine2_signature(S: Any) -> tuple:
    lay = _lane_layout(S)
    params, ret = _combine_signature(S)
    return (*params, ("oks", Array(leaf_statements(lay), OK_WORD))), ret


def _combine2_body(B: Any, S: Any, partials: Coll, receipt: Coll, oks: Coll) -> Coll:
    receipt_ok = _combine_body(B, S, partials, receipt).field("ok")
    n = leaf_statements(_lane_layout(S))
    high = _or_all(B, [oks[i][t] for i in range(n) for t in range(1, OK_WORD.n)])
    folded = _and_all(B, [receipt_ok] + [oks[i][0] for i in range(n)])
    return Coll(_combine2_signature(S)[1], B.call(PouwMuxBit, high, folded, Coll(BIT, Explicit((_consts(B)(0),)))).refs)


PouwH1Combine2 = CompositeDefinition(
    "PouwH1Combine", 2, ("TILE",), _combine2_signature, _combine2_body, conformance="protocol-defined",
    refuse=lambda b: _refuse_lane(b, J10R_LANE_TILES), register=H.SG is not None,
    doc="v1's receipt check folding the leaf's statements' oks (`circuit.h1zy`), TILE J10-R's PouwH1Tile_v5: v1's "
        "partials and receipt, then oks, each lane's and lane chunk's ok row in statement order (lane by lane, a lane's "
        "chunks in chain order), a 16-bit word. ok is 1 exactly when the partials' XOR is the receipt and every ok row is "
        "the word 1, so it is the leaf's one ok, an AND tree over its statements', and theirs are committed values, never "
        "outputs.")


#: the least K at which a leaf's combine is proved (`combine`); a lane is proved at any K the Definitions bind (64 <= K)
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
    stages, at any K and R `_leaf` takes; v1 over PouwH1Tile_v1 and _v3, v2 over J10-R's PouwH1Tile_v5."""
    _leaf(tile)
    return bind(PouwH1LaneZy2 if tile.defn.id in J10R_TILES else PouwH1LaneZy, TILE=tile)


def combine(tile: Any) -> Any:
    """`PouwH1Combine` over ``tile``, the leaf's H1 tile (v1 over PouwH1Tile_v1 and _v3, v2, which folds the leaf's
    statements' oks, over J10-R's PouwH1Tile_v5), at an R `_leaf` takes and a K of at least `LEAST_K`: the combine
    reads only the partials and the receipt, 8 (KS + m)-byte rows, and C-Flock proves a statement only with a row that
    compresses before its last block (``no row has a compression before its last block``, `verity_flock.circuit`'s and
    `Flock.HmRow`'s), a row of at least 112 bytes, whose 17 bytes of SHA-512 padding spill past its first block."""
    k = _leaf(tile)
    n = 8 * H.Layout(4, 8, k).sets
    if n + 17 <= 128:
        raise ValueError(f"K = {k}: the combine reads only a lane's partials and the receipt, {n}-byte rows, but C-Flock "
                         f"proves a statement only with a row that compresses before its last block, at least 112 bytes: "
                         f"K = {LEAST_K} is the least K the combine is proved at")
    return bind(PouwH1Combine2 if tile.defn.id in J10R_TILES else PouwH1Combine, TILE=tile)


# -- a lane in chunks along K ---------------------------------------------------------------------------------------------------

def chunk_leaves(row: bytes, k: int) -> list[bytes]:
    """A registered A or B row at contraction ``k`` (`ROW`(k)'s bytes: its k codes, then its f32 scale little-endian) as the
    `chunks`(k) = n leaves a table commits it as: leaf c is the codes [c K_c, (c + 1) K_c), K_c = k / n, then the row's
    scale again (`ROW`(K_c), what `PouwH1LaneZyChunk` reads), so row r's chunk c is the table's leaf n r + c and each
    chunk's codes stay contiguous. At k <= `CHUNK_MAX` the one leaf is the row. ValueError on an odd k or a row that isn't
    k + 4 bytes."""
    k, row = int(k), bytes(row)
    if k % 2 or len(row) != k + 4:
        raise ValueError(f"a row at K = {k} is its K codes and its scale's 4 bytes, whole u16 words, not {len(row)} bytes")
    n = chunks(k)
    kc = k // n
    return [row[c * kc:(c + 1) * kc] + row[k:] for c in range(n)]


def chunk_words(rows: Sequence[bytes], k: int) -> list[list[int]]:
    """A table's leaves over ``rows`` at contraction ``k`` (each `ROW`(k)'s bytes): each row's `chunk_leaves` in turn, each
    leaf its K_c / 2 + 2 u16 words, its bytes little-endian in pairs (``hm96-sha512/row/v2`` rows, as
    `registered.Registered.commit` takes them)."""
    return [[leaf[i] | leaf[i + 1] << 8 for i in range(0, len(leaf), 2)] for row in rows for leaf in chunk_leaves(row, k)]


def CARRY(lay: H.Layout, n: int, c: int) -> Record:
    """What chunk c of a lane in n chunks hands chunk c + 1: after its last segment, its fx1 state (16 u32), each entry's
    prefix sum T (32 u32), each row slot's residue sums (4 × 32 u32), each cell's z (32 s32, by row slot then column slot),
    every chain word so far, unmasked (2 (c + 1) KS_c u32, KS_c = K / 32 n), and its rows' scales (4 and 8 f32):
    880 + 8 (c + 1) KS_c bytes."""
    ks = lay.k // n // 32
    return Record(("state", Array(16, H.U32)), ("t", Array(32, H.U32)), ("res", Array(4, Array(32, H.U32))),
                  ("z", Array(32, S32)), ("words", Array(2 * (c + 1) * ks, H.U32)), ("sa", Array(4, F32)),
                  ("sw", Array(8, F32)))


def _chunk(S: Any) -> tuple[H.Layout, int, int, int]:
    lay = _lane_layout(S)
    return lay, S.CHUNKS, S.CHUNK, lay.k // S.CHUNKS


def _chunk_signature(S: Any) -> tuple:
    lay, n, c, kc = _chunk(S)
    params, _ret = S.TILE.signature()
    ps = [("a", Array(4, ROW(kc))), ("b", Array(8, ROW(kc)))] + ([("carry_in", CARRY(lay, n, c - 1))] if c else [])
    ps += ([("admit", Array(lay.r, BIT)), ("zy", RECORD(lay.r, lay.c)), ("partials", _words(lay))] if c == n - 1 else
           [("carry_out", CARRY(lay, n, c))])
    return (*ps, *params[2:], ("lane", LANE)), Record(("ok", BIT))


def _refuse_chunk(b: dict, tile: str = "PouwH1Tile_v1") -> str | None:
    why = _refuse_lane(b, TILES + J10R_TILES)
    if why:
        return why
    t, n, c = b["TILE"], b.get("CHUNKS"), b.get("CHUNK")
    if t.defn.id != tile:
        return (f"this lane in chunks is over {tile}, not {t.defn.id} (PouwH1LaneZyChunk_v1 is over PouwH1Tile_v1 and _v2 "
                "over J10-R's PouwH1Tile_v5): v3's and v6's one-stream z starts from O' and the digits, which only the last "
                "chunk has, and corrects them over every segment of K, so the earlier chunks' corrections would be a "
                "carried sum the last adds, an s32 add the tile has no Definition for")
    if getattr(t.bindings["EXACT"], "defn", None) is not BG.GemmAccumulatorS8:
        return "TILE's EXACT is GemmAccumulatorS8_v2, whose DOT steps a chunk continues from the carried z"
    if not (isinstance(n, int) and not isinstance(n, bool) and n >= 2):
        return f"CHUNKS is an integer of at least 2 (one chunk is the whole lane, PouwH1LaneZy_v1), not {n!r}"
    if not (isinstance(c, int) and not isinstance(c, bool) and 0 <= c < n):
        return f"CHUNK is a chunk's index in [0, {n}), not {c!r}"
    k = t.signature()[0][0][1].elem.n
    if k % (32 * n):
        return f"K = {k} is {n} chunks of whole 32-entry segments only where 32 {n} divides it"
    return None


def _consts(B: Any) -> Any:
    bit: dict = {}

    def const(v: int) -> Any:
        if v not in bit:
            bit[v] = B.call(BL.constant(v)).refs[0]
        return bit[v]
    return const


def _chunk_body(B: Any, S: Any, *args: Coll) -> Coll:
    lay, n, c, kc = _chunk(S)
    v = dict(zip([p for p, _t in _chunk_signature(S)[0]], args, strict=True))
    a, b, lane, cin = v["a"], v["b"], v["lane"], v.get("carry_in")
    last, ks, lo, m = c == n - 1, kc // 32, c * kc, lay.m
    G, Q = lay.r // 4, lay.c // 8
    sel = [lane[t] for t in range(lane_bits(lay.r, lay.c))]
    qb, const = Q.bit_length() - 1, _consts(B)
    seed_j, sigma_r, base, col0 = _lane_seeds(B, S, v, not c)
    rb, cb = base or (None, None)
    codes = [a[s].field("codes") for s in range(4)], [b[t].field("codes") for t in range(8)]
    X, Y, A = [], [], []
    for s in range(4):
        nz = _noise(B, seed_j, sel[qb:], [lay.row(g + G * s) for g in range(G)], kc + (32 * (m - 1) if last else 0), const,
                    lo, rb)
        x = B.batch(SG.PouwI8AddBit, codes[0][s], nz[0:kc], axes=(0, 0))
        y = cin.field("res")[s] if c else B.batch(SG.PouwSx8, x[0:32], axes=(0,))
        for seg in range(0 if c else 1, ks):
            y = B.batch(SG.PouwResAdd, y, x[32 * seg:32 * seg + 32], axes=(0, 0))
        X.append(x)
        Y.append(y)
        if last:
            A.append(H._sum_digits(B, y, [nz[kc + 32 * d:kc + 32 * d + 32] for d in range(m - 1)], m))
    W, U, NU = [], [], []
    for t in range(8):
        firsts = [lay.col(2 * q + 2 * Q * (t >> 1) + (t & 1)) - col0 for q in range(Q)]
        if last:
            nz = _noise(B, seed_j, sel[:qb], firsts, kc + H.U_BITS, const, lo, cb)
            cz, uz = nz[0:kc], nz[kc:kc + H.U_BITS]
        else:
            cz = _noise(B, seed_j, sel[:qb], firsts, kc, const, lo, cb)
            uz = _noise(B, seed_j, sel[:qb], firsts, H.U_BITS, const, lay.k, cb)
        u = B.batch(H.PouwU129, Coll(Array(32, H.I8), uz.refs), axes=(0,))
        if last:
            NU.append(B.batch(SG.PouwI8Neg, u, axes=(0,)))
        wc = B.batch(SG.PouwI8AddBit, codes[1][t], cz, axes=(0, 0))
        W.append(B.batch(SG.PouwI8Add, wc, Coll(Array(kc, H.I8), concat([u.refs] * ks)), axes=(0, 0)))
        U.append(u)
    rs, cs = [SG.row_slot(e) for e in range(32)], [SG.col_slot(e) for e in range(32)]
    prev = [cin.field("t")[e] for e in range(32)] if c else None
    sets = []
    for seg in range(ks):
        xs = [x[32 * seg:32 * seg + 32] for x in X]
        ws = [w[32 * seg:32 * seg + 32] for w in W]
        prev = [B.call(SG.PouwSegDot0, xs[rs[e]], ws[cs[e]]) if prev is None else
                B.call(SG.PouwSegDot, prev[e], xs[rs[e]], ws[cs[e]]) for e in range(32)]
        sets.append(prev)
    if last:
        sets.append([B.call(SG.PouwSegDot, prev[e], NU[cs[e]], A[rs[e]][0]) for e in range(32)])
        sets += [[B.call(SG.PouwSegDot0, U[cs[e]], A[rs[e]][d]) for e in range(32)] for d in range(1, m)]
    if c:
        state, words = Coll(Array(H.KEY_BITS, BIT), cin.field("state").refs), [cin.field("words").refs]
    else:
        key = _noise(B, sigma_r, sel, [lay.lane(ln) for ln in range(lay.lanes)], H.KEY_BITS, const)
        state, words = Coll(Array(H.KEY_BITS, BIT), key.refs), []
    for p in sets:
        state = B.call(SG.PouwFx1Absorb, state, array_of(p))
        words += [B.call(SG.PouwXinFinal, state[0:256]).refs, B.call(SG.PouwXinFinal, state[256:512]).refs]
    cells = [(s, t) for s in range(4) for t in range(8)]
    exact = S.TILE.bindings["EXACT"]
    if c:
        z = [cin.field("z")[u] for u in range(32)]
        for seg in range(ks):
            z = [B.call(exact.bindings["DOT"], z[u], codes[0][s][32 * seg:32 * seg + 32], codes[1][t][32 * seg:32 * seg + 32])
                 for u, (s, t) in enumerate(cells)]
    else:
        acc = bind(exact.defn, K=kc, DOT=exact.bindings["DOT"])
        z = [B.call(acc, codes[0][s], codes[1][t]) for s, t in cells]
    scales = concat([_rows(a, 4, "scale").refs, _rows(b, 8, "scale").refs])
    checks = []
    if c:
        carried = concat([cin.field("sa").refs, cin.field("sw").refs])
        eq = B.batch(SG.PouwEq32, Coll(Array(12, H.U32), scales), Coll(Array(12, H.U32), carried), axes=(0, 0))
        checks += [eq[w] for w in range(12)]
    if last:
        checks.append(_cells_ok(B, lay, sel, v["zy"], [zz.refs for zz in z], a, b))
        eq = B.batch(SG.PouwEq32, v["partials"], _gated(B, lay, v["admit"], words), axes=(0, 0))
        checks += [eq[w] for w in range(2 * lay.sets)]
    else:
        out = v["carry_out"]
        nw = sum(out.type.leaf_widths()) // 32
        mine = concat([state.refs, *[e.refs for e in prev], *[y.refs for y in Y], *[zz.refs for zz in z], *words, scales])
        eq = B.batch(SG.PouwEq32, Coll(Array(nw, H.U32), out.refs), Coll(Array(nw, H.U32), mine), axes=(0, 0))
        checks += [eq[w] for w in range(nw)]
    return Coll(_chunk_signature(S)[1], _lane_ok(B, lane, sel, checks, const).refs)


PouwH1LaneZyChunk = CompositeDefinition(
    "PouwH1LaneZyChunk", 1, ("TILE", "CHUNKS", "CHUNK"), _chunk_signature, _chunk_body, conformance="protocol-defined",
    refuse=_refuse_chunk, register=H.SG is not None, ports=_lane_ports,
    doc="Chunk CHUNK of CHUNKS of one lane of TILE's R × C leaf along K (`circuit.h1zy`): the lane's 4 A rows and 8 B rows "
        "at the chunk's K / CHUNKS entries with each row's scale, what the chunk before hands on (CARRY, past the first), "
        "the leaf's salt, position and roots and the public lane index; the last chunk also the leaf's R admit bits, its "
        "z/y record and the lane's committed partial, any other what it hands on. Its noise is the whole lane's at the "
        "chunk's entries and U's, its chain the whole lane's continued from the carried fx1 state, prefix sums and "
        "residue sums, its z TILE's EXACT continued from the carried z. ok is 1 exactly when the index is one of the "
        "leaf's lanes, the rows' scales are the carried ones (past the first), and the last chunk's checks are the whole "
        "lane's, any other's carry_out what it computed.")

PouwH1LaneZyChunk2 = CompositeDefinition(
    "PouwH1LaneZyChunk", 2, ("TILE", "CHUNKS", "CHUNK"), _chunk_signature, _chunk_body, conformance="protocol-defined",
    refuse=lambda b: _refuse_chunk(b, "PouwH1Tile_v5"), register=H.SG is not None, ports=_lane_ports,
    doc="v1's lane chunk over J10-R (`circuit.h1zy`): chunk CHUNK of CHUNKS of one lane of TILE's R × C leaf, TILE "
        "PouwH1Tile_v5 over GemmAccumulatorS8_v2, with v1's parameters and the unit's call and blocks after its roots, "
        "before the lane index. Its noise is the whole lane's under J10-R, seed_c's (PouwCallSeeds_v2 at the call's first "
        "position, every chunk) at the chunk's entries and U's, each counter the leaf's row or column offset picked over "
        "the index plus blocks[0] or blocks[1] (PouwAdd64_v2); chunk 0's keys are block lane of σ_R's at the unit's own "
        "position. Its carries and ok are v1's, so the last chunk's partial is the lane's part of PouwH1Tile_v5's receipt.")


def lane_chunk(tile: Any, n: int, c: int) -> Any:
    """`PouwH1LaneZyChunk` over ``tile`` (the leaf's H1 tile, as `lane` takes it): chunk ``c`` of the lane in ``n``, v1
    over PouwH1Tile_v1 and v2 over J10-R's PouwH1Tile_v5."""
    _leaf(tile)
    d = PouwH1LaneZyChunk2 if tile.defn.id in J10R_TILES else PouwH1LaneZyChunk
    return bind(d, TILE=tile, CHUNKS=int(n), CHUNK=int(c))


def lane_statements(tile: Any) -> list[Any]:
    """The statements a lane of ``tile``'s leaf is proved as, in chain order: the whole lane (`lane`) at K <= `CHUNK_MAX`,
    else its `chunks`(K) chunks (`lane_chunk`)."""
    n = chunks(_leaf(tile))
    return [lane(tile)] if n == 1 else [lane_chunk(tile, n, c) for c in range(n)]


# -- the unit on chunk rows -------------------------------------------------------------------------------------------------------

def _refuse_tile_chunks(b: dict) -> str | None:
    u, n = b.get("UNIT"), b.get("CHUNKS")
    if getattr(u, "defn", None) is not PouwH1TileZy:
        return f"UNIT is h1zy's PouwH1TileZy_v1 over an H1 tile, not {getattr(u, 'id', u)!r}"
    t = u.bindings["TILE"]
    if t.defn.id != "PouwH1Tile_v1":
        return (f"a unit on chunk rows is over PouwH1Tile_v1, not {t.defn.id}: its drawn leaf is proved as lanes in chunks "
                "(PouwH1LaneZyChunk_v1), and v3's and v4's one-stream z starts from O' and the digits, which only a lane's "
                "last chunk has")
    if getattr(t.bindings["EXACT"], "defn", None) is not BG.GemmAccumulatorS8:
        return "TILE's EXACT is GemmAccumulatorS8_v2, whose DOT steps a lane's chunks continue from the carried z"
    if not (isinstance(n, int) and not isinstance(n, bool) and n >= 2):
        return f"CHUNKS is an integer of at least 2 (one chunk is the unit itself, PouwH1TileZy_v1), not {n!r}"
    k = t.signature()[0][0][1].elem.n
    if k % (32 * n):
        return f"K = {k} is {n} chunks of whole 32-entry segments only where 32 {n} divides it"
    return None


def _tile_chunks(S: Any) -> tuple[int, int, int, int]:
    """A unit on chunk rows' R, C, K and n."""
    r, c, k = _tile(S.UNIT.S())
    return r, c, k, S.CHUNKS


def _tile_chunks_signature(S: Any) -> tuple:
    r, c, k, n = _tile_chunks(S)
    params, ret = S.UNIT.signature()
    return (("a", Array(n * r, ROW(k // n))), ("b", Array(n * c, ROW(k // n))), *params[2:]), ret


def _whole_rows(rows: Coll, count: int, n: int, k: int) -> Coll:
    """``count`` whole rows (`ROW`(k)) from their chunk rows, row i's chunk c at n i + c: its chunks' codes in turn, then
    chunk 0's scale."""
    refs = []
    for i in range(count):
        refs += [rows[n * i + c].field("codes").refs for c in range(n)] + [rows[n * i].field("scale").refs]
    return Coll(Array(count, ROW(k)), concat(refs))


def _tile_chunks_body(B: Any, S: Any, a: Coll, b: Coll, zy: Coll, *public: Coll) -> Coll:
    r, c, k, n = _tile_chunks(S)
    out = B.call(S.UNIT, _whole_rows(a, r, n, k), _whole_rows(b, c, n, k), zy, *public)
    later = [(rows, i, ch) for rows, m in ((a, r), (b, c)) for i in range(m) for ch in range(1, n)]
    w = len(later)
    eq = B.batch(SG.PouwEq32, Coll(Array(w, H.U32), concat([rows[n * i + ch].field("scale").refs for rows, i, ch in later])),
                 Coll(Array(w, H.U32), concat([rows[n * i].field("scale").refs for rows, i, _ch in later])), axes=(0, 0))
    ok = _and_all(B, [out.field("ok")] + [eq[x] for x in range(w)])
    return Coll(_tile_chunks_signature(S)[1], concat([out.field("receipt").refs, ok.refs]))


PouwH1TileZyChunks = CompositeDefinition(
    "PouwH1TileZyChunks", 1, ("UNIT", "CHUNKS"), _tile_chunks_signature, _tile_chunks_body, conformance="protocol-defined",
    refuse=_refuse_tile_chunks, register=H.SG is not None, ports=_unit_ports,
    doc="UNIT (h1zy's PouwH1TileZy_v1 over an R × C PouwH1Tile_v1 at K) on its rows as CHUNKS = n chunk rows each "
        "(`circuit.h1zy`): its n R A rows and n C B rows, row i's chunk c at n i + c its K / n codes from c K / n and the "
        "row's f32 scale, then UNIT's record, salt, position and roots. Each whole row is its chunks' codes in turn and "
        "chunk 0's scale; the unit returns UNIT's receipt on them, and ok, UNIT's ok AND every later chunk's scale equal "
        "to chunk 0's.")


def _refuse_tile_chunks_j10r(b: dict) -> str | None:
    u, n = b.get("UNIT"), b.get("CHUNKS")
    if getattr(u, "defn", None) is not PouwH1TileZy:
        return f"UNIT is h1zy's PouwH1TileZy_v1 over a J10-R tile, not {getattr(u, 'id', u)!r}"
    t = u.bindings["TILE"]
    if t.defn.id not in J10R_TILES:
        return (f"PouwH1TileZyChunks_v2 is over J10-R's tiles ({', '.join(J10R_TILES)}), not {t.defn.id}: "
                "PouwH1Tile_v1's unit on chunk rows is PouwH1TileZyChunks_v1")
    if not (isinstance(n, int) and not isinstance(n, bool) and n >= 2):
        return f"CHUNKS is an integer of at least 2 (one chunk is the unit itself, PouwH1TileZy_v1), not {n!r}"
    k = t.signature()[0][0][1].elem.n
    if k % (32 * n):
        return f"K = {k} is {n} chunks of whole 32-entry segments only where 32 {n} divides it"
    return None


PouwH1TileZyChunks2 = CompositeDefinition(
    "PouwH1TileZyChunks", 2, ("UNIT", "CHUNKS"), _tile_chunks_signature, _tile_chunks_body, conformance="protocol-defined",
    refuse=_refuse_tile_chunks_j10r, register=H.SG is not None, ports=_unit_ports,
    doc="v1's unit on chunk rows over J10-R (`circuit.h1zy`): UNIT h1zy's PouwH1TileZy_v1 over an R × C PouwH1Tile_v5 or "
        "_v6 at K, on its n R A and n C B chunk rows (row i's chunk c at n i + c, its K / n codes from c K / n and the "
        "row's f32 scale), then UNIT's record, salt, position, roots, call and blocks. Each whole row is its chunks' codes "
        "in turn and chunk 0's scale, so UNIT's noise is the call-scale stream's rows and columns at K from blocks; the "
        "unit returns UNIT's receipt on them, and ok, UNIT's ok AND every later chunk's scale equal to chunk 0's.")


def unit_chunks(tile: Any, n: int) -> Any:
    """The zy unit `unit`(``tile``) on ``n`` chunk rows a row: `PouwH1TileZyChunks_v1` over it, or v2 over a J10-R tile
    (`J10R_TILES`)."""
    d = PouwH1TileZyChunks2 if getattr(getattr(tile, "defn", None), "id", None) in J10R_TILES else PouwH1TileZyChunks
    return bind(d, UNIT=unit(tile), CHUNKS=int(n))


#: the units on chunk rows: v1 over PouwH1Tile_v1, v2 over J10-R's tiles
CHUNKED = (PouwH1TileZyChunks, PouwH1TileZyChunks2)


def window_unit(tile: Any) -> Any:
    """The tile unit a window binds at ``tile``'s K, whose rows are the ones a table at K commits (`chunk_leaves`): `unit`
    at `chunks`(K) = 1, else `unit_chunks` on its n = `chunks`(K) chunk rows. ValueError at a K `chunks` refuses, and past
    `CHUNK_MAX` on any tile but PouwH1Tile_v1 over GemmAccumulatorS8_v2 and J10-R's (`J10R_TILES`)."""
    n = chunks(tile.signature()[0][0][1].elem.n)
    return unit(tile) if n == 1 else unit_chunks(tile, n)


# -- the statements' ports ------------------------------------------------------------------------------------------------------

#: a lane's public row ``pub``: the leaf's salt, position and roots and the lane's index, in that order
#: (`pouw.leaf_lanes.PUBLIC`)
PUBLIC = ("salt", "position", "roots", "lane")
#: a J10-R lane's (`PouwH1LaneZy_v2`, `PouwH1LaneZyChunk_v2`): the unit's call and blocks after its roots
PUBLIC_J10R = ("salt", "position", "roots", "call", "blocks", "lane")


def _fields(path: str, t: Any, bit: int) -> list[tuple[str, int, int]]:
    """``t`` from bit ``bit`` of a row as its fields ``(path, first bit, bits)``: a record by its fields, anything else whole."""
    out = []
    for p, ft in ([(f"{path}.{n}", ft) for n, ft in t.fields] if isinstance(t, Record) else [(path, t)]):
        out.append((p, bit, sum(ft.leaf_widths())))
        bit += out[-1][2]
    return out


def statement_rows(d: Any) -> list[tuple[str, list[str]]]:
    """The input rows of a leaf's lane, lane chunk or combine statement, ``d`` `PouwH1LaneZy`, `PouwH1LaneZyChunk` or
    `PouwH1Combine` bound to its TILE, in order, each ``(port, params)`` as `class_statement.unit_lowering` takes them
    (``a[0]`` row 0 of ``a``): a lane's ``a0`` .. ``a3`` (its A rows by row slot), ``b0`` .. ``b7``, ``admit``, ``zy``,
    ``partials`` and ``pub`` (`PUBLIC`; a J10-R lane's `PUBLIC_J10R`); a chunk's the same rows of its entries, then those
    of ``carry_in``, ``admit``, ``zy``, ``partials`` and ``carry_out`` it reads, and ``pub``; the combine's ``p0`` ..
    ``p<lanes - 1>`` (its partials rows) and ``receipt``, and v2's ``ok0`` .. (its ``oks`` rows, statement i's at
    ``ok<i>``); as `pouw.leaf_lanes` reads them."""
    ty = dict(d.signature()[0])
    pub = [("pub", list(PUBLIC_J10R if d.defn in (PouwH1LaneZy2, PouwH1LaneZyChunk2) else PUBLIC))]
    if d.defn in (PouwH1LaneZy, PouwH1LaneZy2):
        rows = ([(f"{n}{i}", [f"{n}[{i}]"]) for n in ("a", "b") for i in range(ty[n].n)]
                + [(n, [n]) for n in ("admit", "zy", "partials")] + pub)
    elif d.defn in (PouwH1LaneZyChunk, PouwH1LaneZyChunk2):
        rows = ([(f"{n}{i}", [f"{n}[{i}]"]) for n in ("a", "b") for i in range(ty[n].n)]
                + [(n, [n]) for n in ("carry_in", "admit", "zy", "partials", "carry_out") if n in ty] + pub)
    elif d.defn in (PouwH1Combine, PouwH1Combine2):
        rows = [(f"p{x}", [f"partials[{x}]"]) for x in range(ty["partials"].n)] + [("receipt", ["receipt"])]
        rows += [(f"ok{i}", [f"oks[{i}]"]) for i in range(ty["oks"].n)] if "oks" in ty else []
    else:
        raise ValueError(f"a leaf's statement is PouwH1LaneZy, PouwH1LaneZyChunk or PouwH1Combine (v1 or v2), not "
                         f"{d.defn.id}")
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
              stream: bool = False, j10r: tuple[int, tuple[int, int]] | None = None) -> tuple[list[int], int]:
    """The unit on words: (receipt, ok) for A's rows and B's rows (s8 encodings), their scales (f32 encodings), the record
    ``zy`` (z s32 words, y bf16 encodings, row-major), the salt, position and roots; ``stream``: over v3 (`h1tile`), and
    with ``j10r`` ``(call, blocks)`` over J10-R's v5 (v6 with ``stream``, `h1tile.j10r_reference`)."""
    if j10r is None:
        z, receipt = H.reference(a, b, salt, position, roots, stream=stream)
    else:
        z, receipt = H.j10r_reference(a, b, salt, position, roots, j10r[0], j10r[1], stream=stream)
    return receipt, check(zy[0], zy[1], z, sa, sw)


def chunks_reference(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], sa: Sequence[int], sw: Sequence[int],
                     zy: tuple[Sequence[int], Sequence[int]], salt: bytes, position: int, roots: tuple[bytes, bytes],
                     n: int, stream: bool = False,
                     j10r: tuple[int, tuple[int, int]] | None = None) -> tuple[list[int], int]:
    """`PouwH1TileZyChunks` on words: (receipt, ok) for its n R A and n C B chunk rows (s8 encodings, row i's chunk c at
    n i + c) and their scales (f32 encodings, one a chunk row), the record ``zy``, the salt, position and roots: `reference`
    on the whole rows (each its chunks' codes in turn, at chunk 0's scale), its ok AND every chunk's scale its row's chunk
    0's. With ``j10r`` ``(call, blocks)`` it is v2's over J10-R (``stream``: over v6), whose blocks are the call-scale
    stream's at the whole K (`h1tile.j10r_blocks`)."""
    def whole(rows: Sequence[Sequence[int]]) -> list[list[int]]:
        return [[v for c in range(n) for v in rows[n * i + c]] for i in range(len(rows) // n)]

    receipt, ok = reference(whole(a), whole(b), sa[::n], sw[::n], zy, salt, position, roots, stream, j10r)
    same = all(s[n * i + c] & M32 == s[n * i] & M32 for s in (sa, sw) for i in range(len(s) // n) for c in range(n))
    return receipt, int(ok and same)


#: a J10-R unit's ``(call, blocks)`` (`h1tile.j10r_reference`'s), or None for today's noise
J10R = tuple[int, tuple[int, int]] | None


def _leaf_noise(salt: bytes, position: int, roots: tuple[bytes, bytes], lay: H.Layout, j10r: J10R) -> tuple[list, list]:
    if j10r is None:
        return H.noise(SD.seed(salt, position, *roots), lay)
    return H.j10r_noise(SD.call_seed(salt, j10r[0], *roots), lay, j10r[1])


def lane_partial(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], admit: Sequence[int], salt: bytes, position: int,
                 roots: tuple[bytes, bytes], lane: int, tile: tuple[int, int], j10r: J10R = None) -> list[int]:
    """Lane ``lane``'s honest partial on words, the row the prover commits for it, from its 4 A rows and 8 B rows by slot
    (s8 encodings) and the leaf's R admit bits: `h1tile.lane_partials`' for the lane, zero words for a dead leaf. With
    ``j10r`` ``(call, blocks)`` the leaf is J10-R's v5, its noise `h1tile.j10r_noise`'s."""
    lay = H.Layout(int(tile[0]), int(tile[1]), len(a[0]))
    if isinstance(admit, (bool, int)) or len(admit) != lay.r:
        raise ValueError(f"admit is the leaf's {lay.r} admit bits, not {admit!r}")
    if not any(g & 1 for g in admit):
        return [0] * (2 * lay.sets)
    rows, cols = _leaf_noise(salt, position, roots, lay, j10r)
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
                   position: int, roots: tuple[bytes, bytes], lane: int, tile: tuple[int, int], j10r: J10R = None) -> int:
    """`PouwH1LaneZy` on words: ok of lane ``lane`` of an R × C ``tile``'s leaf, from the lane's 4 A rows and 8 B rows by
    slot (s8 encodings) and their scales, the leaf's R admit bits, its record ``zy`` (all R C cells), the lane's committed
    ``partial``, and the leaf's salt, position and roots: 1 exactly when ``lane`` is one of the leaf's lanes, the partial
    is `lane_partial`'s and the lane's cells of the record pass `check`; ``lane`` a `LANE` word. With ``j10r``
    ``(call, blocks)`` it is `PouwH1LaneZy_v2`'s, over J10-R's v5."""
    lay = H.Layout(int(tile[0]), int(tile[1]), len(a[0]))
    if not 0 <= lane < 1 << LANE.n:
        raise ValueError(f"a lane index is a {LANE.n}-bit word, not {lane}")
    if lane >= lay.lanes:
        return 0
    want = lane_partial(a, b, admit, salt, position, roots, lane, tile, j10r)
    z = [H.SG.seg_dot(0, a[s], b[t]) for s in range(4) for t in range(8)]
    cells = lane_cells(lay, lane)
    return int([v & M32 for v in partial] == want and check([zy[0][u] for u in cells], [zy[1][u] for u in cells], z, sa, sw))


def combine_reference(partials: Sequence[Sequence[int]], receipt: Sequence[int],
                      oks: Sequence[int] | None = None) -> int:
    """`PouwH1Combine` on words: 1 exactly when the lanes' partials XORed word by word are ``receipt``; with ``oks`` (v2,
    the statements' ok rows' u16 words) and every one of them 1."""
    out = [0] * len(receipt)
    for p in partials:
        out = [x ^ (v & M32) for x, v in zip(out, p, strict=True)]
    return int(out == [v & M32 for v in receipt] and all(int(w) & 0xFFFF == 1 for w in oks or ()))


def carry_words(carry: dict) -> list[int]:
    """A carry (`CARRY`'s fields as a dict of words: ``state``, ``t``, ``res`` 4 rows of 32, ``z``, ``words``, ``sa``,
    ``sw``) as its row's u32 words, field by field."""
    return [v & M32 for v in (*carry["state"], *carry["t"], *(w for r in carry["res"] for w in r), *carry["z"],
                              *carry["words"], *carry["sa"], *carry["sw"])]


def _chunk_step(lay: H.Layout, n: int, c: int, a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], sa: Sequence[int],
                sw: Sequence[int], rows: list, cols: list, key: Sequence[int] | None, ln: int, cin: dict | None) -> dict:
    """Chunk c of n of lane ln on words, from its rows' entries of the chunk (``a`` 4 and ``b`` 8, s8 encodings) and their
    scales, the leaf's noise (`h1tile.noise` at the whole K), the lane's fold keys (chunk 0) or the carry in: the carry
    out, whose ``words`` after the last chunk are the lane's whole chain, unmasked."""
    kc = lay.k // n
    lo, ks, last = c * kc, kc // 32, c == n - 1
    R, Cs = lay.rows(ln), lay.cols(ln)
    rs, cs = [SG.row_slot(e) for e in range(32)], [SG.col_slot(e) for e in range(32)]
    dot = P.BlackwellS8ImmaDot32.evaluate
    X = [[(a[s][e] + rows[R[s]][0][lo + e]) & 0xFF for e in range(kc)] for s in range(4)]
    U = [cols[Cs[t]][1] for t in range(8)]
    W = [[(b[t][e] + cols[Cs[t]][0][lo + e] + U[t][e % 32]) & 0xFF for e in range(kc)] for t in range(8)]
    T = list(cin["t"]) if cin else None
    Y = [list(r) for r in cin["res"]] if cin else [[0] * 32 for _ in range(4)]
    z = list(cin["z"]) if cin else [0] * 32
    sets = []
    for seg in range(ks):
        g = slice(32 * seg, 32 * seg + 32)
        T = [SG.seg_dot(0 if T is None else T[e], X[rs[e]][g], W[cs[e]][g]) for e in range(32)]
        sets.append(T)
        Y = [[(Y[s][r] + _sx(X[s][g][r], 8)) & M32 for r in range(32)] for s in range(4)]
        z = [dot(z[u], *a[s][g], *b[t][g]) for u, (s, t) in enumerate((s, t) for s in range(4) for t in range(8))]
    if last:
        D = SG.sum_digits([[_sx(v, 32) for v in y] for y in Y], [[rows[i][1][d] for i in R] for d in range(lay.m - 1)],
                          lay.m)
        neg = [[(-v) & 0xFF for v in u] for u in U]
        sets.append([SG.seg_dot(T[e], neg[cs[e]], D[0][rs[e]]) for e in range(32)])
        sets += [[SG.seg_dot(0, U[cs[e]], D[d][rs[e]]) for e in range(32)] for d in range(1, lay.m)]
    st, words = (list(cin["state"]), list(cin["words"])) if cin else (list(key), [])
    for p in sets:
        st = SG.absorb(H.FOLD, st, p)
        words += [H._xor_words(st[0:8]), H._xor_words(st[8:16])]
    return {"state": st, "t": T, "res": Y, "z": z, "words": words, "sa": [v & M32 for v in sa], "sw": [v & M32 for v in sw]}


def _chunk_noise(salt: bytes, position: int, roots: tuple[bytes, bytes], lay: H.Layout, ln: int, j10r: J10R) -> tuple:
    rows, cols = _leaf_noise(salt, position, roots, lay, j10r)
    return rows, cols, H.fold_keys(SD.receipt_seed(salt, position, *roots), lay)[ln]


def lane_chunk_carries(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], sa: Sequence[int], sw: Sequence[int],
                       salt: bytes, position: int, roots: tuple[bytes, bytes], lane: int, tile: tuple[int, int],
                       n: int, j10r: J10R = None) -> list[dict]:
    """Lane ``lane``'s chain in ``n`` chunks on words, from its 4 A rows and 8 B rows by slot over the whole K (s8
    encodings) and their scales: each chunk's carry out (`CARRY`, as `carry_words` takes it), the honest ``carry_out`` of
    chunks 0 .. n - 2 and the ``carry_in`` of chunks 1 .. n - 1; the last's ``words`` are the lane's chain, unmasked, and
    its ``z`` the lane's cells' exact products. With ``j10r`` ``(call, blocks)``: over J10-R's v5, at the whole K."""
    lay = H.Layout(int(tile[0]), int(tile[1]), len(a[0]))
    rows, cols, key = _chunk_noise(salt, position, roots, lay, lane, j10r)
    kc, out, cin = lay.k // n, [], None
    for c in range(n):
        cin = _chunk_step(lay, n, c, [r[c * kc:c * kc + kc] for r in a], [r[c * kc:c * kc + kc] for r in b], sa, sw, rows,
                          cols, key, lane, cin)
        out.append(cin)
    return out


def lane_chunk_reference(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], sa: Sequence[int], sw: Sequence[int],
                         admit: Sequence[int] | None, zy: tuple[Sequence[int], Sequence[int]] | None,
                         partial: Sequence[int] | None, carry_in: dict | None, carry_out: dict | None, salt: bytes,
                         position: int, roots: tuple[bytes, bytes], lane: int, tile: tuple[int, int], n: int, c: int,
                         j10r: J10R = None) -> int:
    """`PouwH1LaneZyChunk` on words: ok of chunk ``c`` of ``n`` of lane ``lane`` of an R × C ``tile``'s leaf, from the lane's
    4 A rows and 8 B rows by slot at the chunk's entries (s8 encodings) and the rows' scales, ``carry_in`` (past chunk 0),
    the last chunk's ``admit``, ``zy`` and ``partial`` as `lane_reference` takes them, any other's ``carry_out`` (each a
    carry as `carry_words` takes it, the others None), and the leaf's salt, position and roots: 1 exactly when ``lane`` is
    one of the leaf's lanes, the scales are ``carry_in``'s, and the last chunk's partial is the chain's words (zero for a
    dead leaf) and its cells pass `check`, any other's ``carry_out`` what it computed. With ``j10r`` ``(call, blocks)`` it
    is `PouwH1LaneZyChunk_v2`'s, over J10-R's v5 (``blocks`` at the whole K)."""
    lay = H.Layout(int(tile[0]), int(tile[1]), len(a[0]) * n)
    if not 0 <= lane < 1 << LANE.n:
        raise ValueError(f"a lane index is a {LANE.n}-bit word, not {lane}")
    if lane >= lay.lanes:
        return 0
    last = c == n - 1
    rows, cols, key = _chunk_noise(salt, position, roots, lay, lane, j10r)
    got = _chunk_step(lay, n, c, a, b, sa, sw, rows, cols, None if c else key, lane, carry_in if c else None)
    ok = not c or got["sa"] + got["sw"] == [v & M32 for v in (*carry_in["sa"], *carry_in["sw"])]
    if not last:
        return int(ok and carry_words(carry_out) == carry_words(got))
    want = got["words"] if any(g & 1 for g in admit) else [0] * len(got["words"])
    cells = lane_cells(lay, lane)
    return int(ok and [v & M32 for v in partial] == want
               and check([zy[0][u] for u in cells], [zy[1][u] for u in cells], got["z"], sa, sw))


__all__ = ["BF16", "CARRY", "CHUNKED", "CHUNK_ALIGN", "CHUNK_MAX", "F32", "J10R", "J10R_LANE_TILES", "LANE", "LEAST_K",
           "OK_WORD", "PUBLIC", "PUBLIC_J10R", "S32", "S8",
           "PouwAndBit", "PouwH1Combine", "PouwH1Combine2", "PouwH1LaneZy", "PouwH1LaneZy2", "PouwH1LaneZyChunk",
           "PouwH1LaneZyChunk2", "PouwH1TileZy", "PouwH1TileZyChunks", "PouwH1TileZyChunks2", "PouwMuxBit",
           "PouwZyCell", "PouwZyCheck", "PouwZyDequant", "PouwZyDequantWord", "RECORD", "ROW", "J10R_TILES", "TILES",
           "carry_words",
           "check", "chunk_leaves", "chunk_words", "chunks", "chunks_reference", "combine", "combine_reference", "dequant",
           "lane", "lane_bits", "lane_cells", "lane_chunk", "lane_chunk_carries", "lane_chunk_reference", "lane_partial",
           "lane_reference", "lane_statements", "leaf_statements", "records", "reference", "statement_ports",
           "statement_rows", "unit", "unit_chunks", "window_unit", "zy_cell"]
