/-!
# A drawn H1 leaf's 33 statements, port by port (option A)

Under option A (PRF-134) a drawn leaf is proved as its lanes, `h1zy`'s `PouwH1LaneZy_v1{TILE}` (lane `l` by its public
index), and their combine `PouwH1Combine_v1{TILE}`, each a C-Flock statement of one unit. This states, for a leaf's tile
and contraction (`Tile`), which port of each statement holds each of the Definition's inputs and its output, at which bits
of the port's row (`Field`), and where the verifier opens that row (`Source`): the port of `h1zy.statement_ports`, with
each port's row as #3116's `pouw.leaf_lanes` reads it (`reads`, `known`). `tests/test_lanes.py` holds it to the first and
to `h1tile.TileShape`'s rows and columns, at the served 32 × 32 leaf at K = 448, 4096 and 14336 and at the sidecar's 8 × 16
leaf at K = 64.

* **A port's row** is a bit row (`hm96-sha512/row/v2`) of 16-bit words, bit `t` of the row bit `t % 16` of word `t / 16`.
  Its fields are the inputs it holds, their leaves concatenated in signature order, least significant bit first
  (`class_statement.unit_lowering`). An input row is exactly its fields' bits, which C-Flock takes only as whole words
  (`unit_lowering`, `Flock.ProvedScope.v2Row`); the output port `o0` is `ok`, bit 0, zero-padded to one word.
* **The fields**, by their path in the signature (`a[0].codes`: the codes of the lane's A row at row slot 0):
  * an A or B row (`h1zy.ROW`): `.codes`, code `k` the s8 at bits `[8k, 8k + 8)`, then `.scale`, its f32;
  * the leaf's record (`h1zy.RECORD`): `zy.z`, cell `u = i C + j`'s s32 at bits `[32u, 32u + 32)` of it, then `zy.y`,
    its bf16 at `[16u, 16u + 16)` of it;
  * a partial (`partials`, the combine's `partials[x]`) or the receipt: `Tile.nw` u32 words, word `w` at bits
    `[32w, 32w + 32)` (16-bit words `2w` and `2w + 1`), in #3095's set order (`leaf_lanes.word_order`);
  * `admit`: the leaf's `R` admit bits, row `i`'s at bit `i`;
  * `salt`, `position` and `roots` (`seeds.types`): a digest's 16 and a position's 2 big-endian u32 halves, then `roots`'
    two digests, each half least significant bit first (`seeds.leaves`);
  * `lane`: the index, one 16-bit word (`h1zy.LANE`).
* **Where each row is opened** (`Source`): a registered value's row (`read`: lane `l`'s A row at slot `s` is row
  `Tile.rowOf l s` of `a`, its B row at slot `t` row `Tile.colOf l t` of `b`, its partial row `l` of `partials`; the
  combine's `p<x>` row `x`), or the one row of a value the verifier registers itself at its known content (`known`: lane
  `l`'s `pub` and `admit`, and statement `i`'s `o0`, value `ok<i>`, the row 1; statement `i` is lane `i`, then the
  combine). STAND-IN: `a`, `b`, `zy` and `receipt` are the leaf's own rows, as `leaf_lanes` reads them; in a served window
  they are the window's table rows (`circuit.h1window`).
* **The unit's input wires** (`wires?`, `wireOf?`): the input ports' rows one after another in port order, wire `w` the
  unit's input column `w`, so port `p`'s row bit `b` is wire `b` past the bits of the ports before it
  (`class_statement.unit_lowering`, whose input leaf `j` is row word `j`: META `leaves_in` the identity). The test holds it
  to the lowered circuits' own port records (`unit_lowering`'s rows and input columns, `UnitStager`'s META `ports`,
  `row_schemas` and `leaves_in`), and C-Flock's refusals to `wires?`'s.

Code: Lean core only, never `Security/`, so proofs' decoders (`Security/Proofs/Accounting/TS/Session/Lanes`) import it
from `pouw_window` through `security_properties`.
-/

namespace PouwWindow.Lanes

/-- H1's digit count `m` at `KS` segments (`segment.n_digits`): the least `m ≥ 1` with `-128^m ≤ -128 KS`, that is
`KS ≤ 128^(m-1)` (which gives its second condition, `127 KS / 128^(m-1) ≤ 127`). -/
def nDigits (ks : Nat) : Nat := go ks 1 1
where
  /-- `p = 128^(m-1)`; the fuel bounds the search, since `m ≤ KS + 1`. -/
  go : Nat → Nat → Nat → Nat
    | 0, m, _ => m
    | fuel + 1, m, p => if ks ≤ p then m else go fuel (m + 1) (128 * p)

/-- A drawn leaf's tile: `r × c` cells at contraction `k` (`h1tile.TileShape`), tiled by lanes of 4 rows and 8 columns. -/
structure Tile where
  r : Nat
  c : Nat
  k : Nat
  deriving Repr, DecidableEq

namespace Tile

def ks (T : Tile) : Nat := T.k / 32

def m (T : Tile) : Nat := nDigits T.ks

/-- The sets a lane absorbs: `KS` prefix sets, `O′`, then `D_1 .. D_(m-1)`. -/
def sets (T : Tile) : Nat := T.ks + T.m

/-- A partial's, or the receipt's, u32 words: `2 (KS + m)`. -/
def nw (T : Tile) : Nat := 2 * T.sets

/-- The leaf's lanes are `G × Q`, `G = R / 4` by row and `Q = C / 8` by column. -/
def g (T : Tile) : Nat := T.r / 4

def q (T : Tile) : Nat := T.c / 8

def lanes (T : Tile) : Nat := T.g * T.q

/-- Lane `l`'s row at row slot `s < 4` (`h1tile.TileShape.rows`): `⌊l / Q⌋ + G s`. -/
def rowOf (T : Tile) (l s : Nat) : Nat := l / T.q + T.g * s

/-- Lane `l`'s column at column slot `t < 8` (`h1tile.TileShape.cols`): `2 (l mod Q) + 2 Q ⌊t / 2⌋ + t mod 2`. -/
def colOf (T : Tile) (l t : Nat) : Nat := 2 * (l % T.q) + 2 * T.q * (t / 2) + t % 2

private def pow2 (n : Nat) : Bool := n != 0 && n &&& (n - 1) == 0

/-- `h1zy`'s refusal of a leaf's tile (`_refuse_lane`), and its s8 accumulator's (`gemm._refuse_k`) of a `K` past whole
32-wide steps or at or past `S8_K_LIMIT = 2^17`, from which a step can clamp and `z` is not the exact dot: `K` at least
64 (`KS ≥ 2`, a digit word, as `h1tile`'s scheme asks), `R / 4` and `C / 8` powers of two, at least 2 lanes and at most
`2^16`. -/
def check (T : Tile) : Except String Unit := do
  if T.k % 32 != 0 then throw s!"K = {T.k} is not a multiple of the 32-wide step"
  if T.k < 64 then throw s!"K = {T.k} is under 64: H1's lanes need KS = K / 32 >= 2 (a digit word)"
  if 2 ^ 17 ≤ T.k then throw s!"K = {T.k} is at or past 2^17, from which the s8 accumulator can clamp"
  if T.r % 4 != 0 || T.c % 8 != 0 || T.lanes < 2 || !pow2 T.g || !pow2 T.q || T.lanes > 2 ^ 16 then
    throw s!"a leaf's lanes are a public index of whole bits: R / 4 and C / 8 powers of two, at least two lanes and at \
      most 2^16, not {T.r} × {T.c}"

end Tile

/-- One input of the Definition, or its output, in a port's row: its path in the signature, its first bit and its bits. -/
structure Field where
  path : String
  bit : Nat
  bits : Nat
  deriving Repr, DecidableEq

/-- The field's first 16-bit word of the row. -/
def Field.word (f : Field) : Nat := f.bit / 16

/-- The 16-bit words of the row the field's bits touch, from `word`. -/
def Field.words (f : Field) : Nat := (f.bit + f.bits + 15) / 16 - f.bit / 16

/-- Where the verifier opens a port's row. -/
inductive Source where
  /-- row `row` of the registered value `value` -/
  | read (value : String) (row : Nat)
  /-- the one row of `value`, which the verifier registers itself at its known content -/
  | known (value : String)
  deriving Repr, DecidableEq

/-- A statement's port: its name, where its row is opened, its fields from bit 0, and whether it is the output. -/
structure Port where
  name : String
  source : Source
  fields : List Field
  output : Bool
  deriving Repr

/-- The port's bits: its fields'. -/
def Port.bits (p : Port) : Nat := (p.fields.map Field.bits).sum

/-- The port's row in 16-bit words. -/
def Port.words (p : Port) : Nat := (p.bits + 15) / 16

/-- Fields from bit `b`, each `(path, bits)`, one after another. -/
def fieldsFrom : Nat → List (String × Nat) → List Field
  | _, [] => []
  | b, (p, n) :: rest => ⟨p, b, n⟩ :: fieldsFrom (b + n) rest

/-- An A or B row's fields (`h1zy.ROW`): its `K` s8 codes, then its f32 scale. -/
def rowFields (path : String) (k : Nat) : List Field := fieldsFrom 0 [(path ++ ".codes", 8 * k), (path ++ ".scale", 32)]

/-- A digest's bits (`seeds.DIGEST_BITS`): 16 u32 halves. -/
def digestBits : Nat := 512

/-- A position's bits (`seeds.POS_BITS`): 2 u32 halves. -/
def positionBits : Nat := 64

/-- The lane index's bits (`h1zy.LANE`): one 16-bit word. -/
def laneBits : Nat := 16

/-- Statement `i`'s output port: `ok`, opened at the verifier's own value `ok<i>`. -/
def okPort (i : Nat) : Port := ⟨"o0", .known s!"ok{i}", fieldsFrom 0 [("ok", 1)], true⟩

/-- **Lane `l`'s ports**, in order: `a0 .. a3` its A rows by row slot, `b0 .. b7` its B rows by column slot, `admit`,
`zy`, `partials` (row `l`), `pub` (`h1zy.PUBLIC`: salt, position, roots, index), then `o0`. -/
def lanePorts (T : Tile) (l : Nat) : List Port :=
  (List.range 4).map (fun s => ⟨s!"a{s}", .read "a" (T.rowOf l s), rowFields s!"a[{s}]" T.k, false⟩) ++
  (List.range 8).map (fun t => ⟨s!"b{t}", .read "b" (T.colOf l t), rowFields s!"b[{t}]" T.k, false⟩) ++
  [⟨"admit", .known "admit", fieldsFrom 0 [("admit", T.r)], false⟩,
   ⟨"zy", .read "zy" 0, fieldsFrom 0 [("zy.z", 32 * T.r * T.c), ("zy.y", 16 * T.r * T.c)], false⟩,
   ⟨"partials", .read "partials" l, fieldsFrom 0 [("partials", 32 * T.nw)], false⟩,
   ⟨"pub", .known s!"pub{l}", fieldsFrom 0 [("salt", digestBits), ("position", positionBits),
     ("roots", 2 * digestBits), ("lane", laneBits)], false⟩,
   okPort l]

/-- **The combine's ports**, in order: `p<x>` row `x` of `partials` for each lane `x`, the leaf's `receipt`, then `o0`
(statement `lanes`'s). -/
def combinePorts (T : Tile) : List Port :=
  (List.range T.lanes).map (fun x => ⟨s!"p{x}", .read "partials" x, fieldsFrom 0 [(s!"partials[{x}]", 32 * T.nw)],
    false⟩) ++
  [⟨"receipt", .read "receipt" 0, fieldsFrom 0 [("receipt", 32 * T.nw)], false⟩, okPort T.lanes]

/-- Lane `l`'s ports, where the tile is a leaf's and `l` one of its lanes. -/
def lane? (T : Tile) (l : Nat) : Except String (List Port) := do
  T.check
  if T.lanes ≤ l then throw s!"an {T.r} × {T.c} leaf has {T.lanes} lanes, not lane {l}"
  pure (lanePorts T l)

/-- The combine's ports, where the tile is a leaf's. -/
def combine? (T : Tile) : Except String (List Port) := do
  T.check
  pure (combinePorts T)

/-! ## A J10-R leaf

A drawn unit over J10-R's `PouwH1Tile_v5` is proved as `h1zy`'s v2 statements: each lane in `n` chunks along K
(`PouwH1LaneZyChunk_v2`, `n = weights.chunks K`, `PouwCheck.Chunks.chunks`, which the caller passes; at `n = 1` the whole
lane, `PouwH1LaneZy_v2`), then `PouwH1Combine_v2`, which folds the other statements' oks into its own. Statement `i < lanes n`
is lane `i / n`'s chunk `i % n`; `leaf_lanes`' J10-R leaf reads them so:

* chunk `c` of lane `l` reads its A and B rows' chunk `c` (`h1zy.ROW` at `K / n`), row `n i + c` of a table that commits
  row `i` as `n` chunk rows (`h1zy.chunk_leaves`); `carry_in` (`c > 0`) row `l` of `carry<c - 1>` and `carry_out`
  (`c < n - 1`) row `l` of `carry<c>` (`h1zy.CARRY`); its last chunk `admit`, `zy` and `partials` (row `l`) as v1's lane;
  `pub` (`h1zy.PUBLIC_J10R`: the unit's `call` and `blocks` after its roots); and its output `o0` is row `n l + c` of
  `oks`, a committed value (`okRead`), never a known row;
* the combine reads v1's `p<x>` and `receipt`, then `ok<i>` row `i` of `oks` (`h1zy.OK_WORD`, 16 bits) for each
  statement `i < lanes n`, and its `o0` is the known row 1 of `ok<lanes n>`, the leaf's one known ok. -/

/-- A J10-R lane's `call` and each of its two `blocks` (`seeds.POS_BITS`): a u64's 2 u32 halves. -/
def callBits : Nat := 64

/-- A J10-R lane's public row (`h1zy.PUBLIC_J10R`): salt, position, roots, call, blocks, index. -/
def pubJ10r (l : Nat) : Port := ⟨"pub", .known s!"pub{l}", fieldsFrom 0 [("salt", digestBits), ("position", positionBits),
  ("roots", 2 * digestBits), ("call", callBits), ("blocks", 2 * callBits), ("lane", laneBits)], false⟩

/-- Statement `i`'s output port under J10-R: row `i` of the committed `oks`. -/
def okRead (i : Nat) : Port := ⟨"o0", .read "oks" i, fieldsFrom 0 [("ok", 1)], true⟩

/-- What chunk `c` of a lane in `n` hands chunk `c + 1` (`h1zy.CARRY`): its fx1 state (16 u32), each entry's prefix sum
(32 u32), each row slot's residue sums (4 × 32 u32), each cell's z (32 s32), the chain's words so far (`2 (c + 1) KS_c`
u32, `KS_c = K / 32 n`) and its rows' scales (4 and 8 f32). -/
def carryFields (path : String) (T : Tile) (n c : Nat) : List Field :=
  fieldsFrom 0 [(path ++ ".state", 512), (path ++ ".t", 1024), (path ++ ".res", 4096), (path ++ ".z", 1024),
    (path ++ ".words", 64 * (c + 1) * (T.k / (32 * n))), (path ++ ".sa", 128), (path ++ ".sw", 256)]

/-- **Chunk `c` of J10-R lane `l` in `n`'s ports**, in order (`h1zy.statement_rows`): `a0 .. a3`, `b0 .. b7` at the chunk's
`K / n` entries, `carry_in` past the first chunk, then the last chunk's `admit`, `zy` and `partials` or any other's
`carry_out`, `pub`, then `o0`. At `n = 1` the whole lane's. -/
def chunkPorts (T : Tile) (n l c : Nat) : List Port :=
  (List.range 4).map (fun s => ⟨s!"a{s}", .read "a" (n * T.rowOf l s + c), rowFields s!"a[{s}]" (T.k / n), false⟩) ++
  (List.range 8).map (fun t => ⟨s!"b{t}", .read "b" (n * T.colOf l t + c), rowFields s!"b[{t}]" (T.k / n), false⟩) ++
  (if c = 0 then [] else [⟨"carry_in", .read s!"carry{c - 1}" l, carryFields "carry_in" T n (c - 1), false⟩]) ++
  (if c + 1 = n then
    [⟨"admit", .known "admit", fieldsFrom 0 [("admit", T.r)], false⟩,
     ⟨"zy", .read "zy" 0, fieldsFrom 0 [("zy.z", 32 * T.r * T.c), ("zy.y", 16 * T.r * T.c)], false⟩,
     ⟨"partials", .read "partials" l, fieldsFrom 0 [("partials", 32 * T.nw)], false⟩]
   else [⟨"carry_out", .read s!"carry{c}" l, carryFields "carry_out" T n c, false⟩]) ++
  [pubJ10r l, okRead (n * l + c)]

/-- **The J10-R combine's ports**, in order: v1's `p<x>` and `receipt`, then `ok<i>` row `i` of `oks` for each of the
`lanes n` lane and chunk statements, then `o0` (statement `lanes n`'s). -/
def combine2Ports (T : Tile) (n : Nat) : List Port :=
  (combinePorts T).dropLast ++
  (List.range (T.lanes * n)).map (fun i => ⟨s!"ok{i}", .read "oks" i, fieldsFrom 0 [(s!"oks[{i}]", 16)], false⟩) ++
  [okPort (T.lanes * n)]

/-- `h1zy`'s refusal of a J10-R lane in `n` chunks (`_refuse_chunk`): the tile's (`Tile.check`), and `K` whole 32-entry
segments in each of `n ≥ 1` chunks. -/
def checkChunks (T : Tile) (n : Nat) : Except String Unit := do
  T.check
  if n = 0 then throw "a lane is at least one chunk"
  if T.k % (32 * n) != 0 then throw s!"K = {T.k} is {n} chunks of whole 32-entry segments only where 32 {n} divides it"

/-- Chunk `c` of J10-R lane `l` in `n`'s ports, where the tile is a leaf's, `l` one of its lanes and `c` one of `n`. -/
def chunk? (T : Tile) (n l c : Nat) : Except String (List Port) := do
  checkChunks T n
  if T.lanes ≤ l then throw s!"an {T.r} × {T.c} leaf has {T.lanes} lanes, not lane {l}"
  if n ≤ c then throw s!"a lane in {n} chunks has chunks 0 .. {n - 1}, not chunk {c}"
  pure (chunkPorts T n l c)

/-- The J10-R combine's ports over lanes in `n` chunks, where the tile is a leaf's. -/
def combine2? (T : Tile) (n : Nat) : Except String (List Port) := do
  checkChunks T n
  pure (combine2Ports T n)

/-- The port holding the input or output at `path`, and its field there. -/
def fieldOf? (ps : List Port) (path : String) : Option (Port × Field) :=
  ps.findSome? fun p => (p.fields.find? (·.path == path)).map (p, ·)

/-- The field holding bit `b` of the port's row, if one does. -/
def Port.fieldAt? (p : Port) (b : Nat) : Option Field :=
  p.fields.find? fun f => f.bit ≤ b ∧ b < f.bit + f.bits

/-- **The unit's input wires**: each input port with its first wire, in `class_statement.unit_lowering`'s leaf order. The
input ports' rows come one after another, each from its bit 0, and wire `w` is the unit's input column `w`: input leaf
`w / 16` (the u16 input `in<w / 16>`, which META `leaves_in` places on row word `w / 16` of the input rows in order, the
identity) at its bit `w % 16`. So port `p`'s row bit `b` is wire `first + b`, and its 16-bit word `j`'s bit `t` is wire
`first + 16 j + t`. The output `o0` holds no input wire. -/
def inputWires (ps : List Port) : List (Port × Nat) := go (ps.filter fun p => !p.output) 0
where
  go : List Port → Nat → List (Port × Nat)
    | [], _ => []
    | p :: rest, w => (p, w) :: go rest (w + p.bits)

/-- The unit's input wires, where every input row is whole 16-bit words, which `unit_lowering` takes only. -/
def wires? (ps : List Port) : Except String (List (Port × Nat)) := do
  for p in ps do
    if !p.output && p.bits % 16 != 0 then throw s!"row {p.name} is {p.bits} bits, not whole u16 words"
  pure (inputWires ps)

/-- The port input wire `w` is in and the bit of the port's row it holds; none past the input rows' bits (the unit's input
group's zero padding). -/
def wireOf? (ps : List Port) (w : Nat) : Option (Port × Nat) :=
  (inputWires ps).findSome? fun (p, lo) => if lo ≤ w ∧ w < lo + p.bits then some (p, w - lo) else none

end PouwWindow.Lanes
