/-
The statement forms the soundness proof covers (PROTOCOL.md §16.15). `verify` refuses every other: soundness is one theorem
over every session the verifier accepts, so a form no theorem covers is refused, never accepted. Widening the proof to
another form is a change to one rule here, with the proof that the headlines cover it.
-/
import Flock.HmRow

namespace Flock.ProvedScope

/-- The input columns of a group list's ports, one per bit. -/
def portBits (gs : Array Group) : List Nat := (portCols gs).toList.flatMap fun q => List.range' q.1 q.2

/-- Which of the columns below `n` are port bits (`portBits`), as an array: a unit row's reads are each one lookup. -/
def portMask (gs : Array Group) (n : Nat) : Array Bool :=
  (portBits gs).foldl (fun m x => m.setIfInBounds x true) (Array.replicate n false)

/-- Every input column a computed row of `net` reads is a bit of one of its input ports. -/
def portReadsB (net : Net) : Bool :=
  let n := min (net.inWords * WORD) net.constPos
  let mask := portMask net.inGroups n
  (List.range (net.constPos - n)).all fun i =>
    ((net.a.row (n + i)).toList ++ (net.b.row (n + i)).toList).all fun x => decide (x < n → mask.getD x false = true)

/-- Every row is an unsegmented `sha512/row/v1` row: `16 · words` bits in at least one chunk. -/
def v1Ports (c : Circuit) : Bool :=
  c.ports.all fun p => !p.v2 && p.segs.isNone && p.bits == 16 * p.words && 0 < p.chunks

/-- The circuits the headlines cover: the unit reads only its leaf group, returns the outputs, has at most `2^32` rows and
reads only port bits among its inputs, and every row is a `sha512/row/v1` row. -/
def scopeOk (c : Circuit) : Bool :=
  c.unit.inGroups.size == 1 && c.outNet == c.unitNet && decide (c.unit.unitLog ≤ 32) && portReadsB c.unit && v1Ports c

/-- A `sha512/row/v2` row the hidden-output headlines cover: unsegmented, `16 · words` bits, its bytes filling its
compressions' blocks and ending, with SHA-512's 17 bytes of padding, in its `hm96` slot's block. -/
def v2Row (p : Port) : Bool :=
  p.v2 && p.segs.isNone && p.bits == 16 * p.words && decide (128 * p.chunks ≤ 2 * p.words) &&
    decide (2 * p.words + 17 ≤ 128 * (p.chunks + 1))

/-- A `sha512/row-seg/v1` row the hidden-output headlines cover: segmented as `HmRow.segShape` cuts its bits,
`16 · words` bits, its compressions its segments' and then its outer message's, and each segment's bytes ending, with
SHA-512's 17 bytes of padding, in its last compression's block. -/
def segRow (p : Port) : Bool :=
  let s := (p.bits + HmRow.ROW_SEG_BITS - 1) / HmRow.ROW_SEG_BITS
  p.segs == some (HmRow.segShape p.bits).1 && p.bits == 16 * p.words &&
    p.chunks == (((HmRow.segShape p.bits).1.toList.take s).map Prod.snd).sum + ((64 * s + 17 + 127) / 128 - 1) &&
    (List.range s).all fun j =>
      let m := min (p.bits - HmRow.ROW_SEG_BITS * j) HmRow.ROW_SEG_BITS
      decide (128 * (((m + 7) / 8 + 17 + 127) / 128 - 1) ≤ (m + 7) / 8)

/-- Every row is a `v2Row` or a `segRow`, of any mix, and each output row is a `v2Row` holding one output port. -/
def bitRows (c : Circuit) : Bool :=
  c.ports.all (fun p => v2Row p || segRow p) && c.outRowPorts.all (· == 1) &&
    (List.range c.outRowPorts.size).all fun j => v2Row (c.ports.getD (HmOut.nIn c + j) default)

/-- The circuits the hidden-output headlines cover: `scopeOk`'s conditions on the unit, and every row a
`sha512/row/v1` row (`v1Ports`) or every row a `v2Row` or a `segRow`, each output row a `v2Row` of one output port
(`bitRows`). -/
def scopeOkH (c : Circuit) : Bool :=
  c.unit.inGroups.size == 1 && c.outNet == c.unitNet && decide (c.unit.unitLog ≤ 32) && portReadsB c.unit &&
    (v1Ports c || bitRows c)

/-- Which part of `scopeOk` (`scopeOkH` for a statement with hidden outputs) a circuit fails, for the refusal. -/
def scopeWhy (hidden : Bool) (c : Circuit) : String :=
  if c.unit.inGroups.size != 1 then s!"a unit reading {c.unit.inGroups.size} input groups"
  else if c.outNet != c.unitNet then "a circuit whose outputs a tail stage computes (META out_net)"
  else if c.unit.unitLog > 32 then s!"a unit of 2^{c.unit.unitLog} rows"
  else if !portReadsB c.unit then "a unit row reading an input column that is not a port bit"
  else if hidden then "rows that are neither all sha512/row/v1 rows of 16 · words bits in at least one chunk nor all \
    unsegmented sha512/row/v2 rows of 16 · words bits ending in their hm96 block or sha512/row-seg/v1 rows of \
    16 · words bits whose segments end in their last block, the output rows unsegmented, one output port each"
  else "a row that is not a sha512/row/v1 row of 16 · words bits in at least one chunk"

/-- The circuit as `Stmt.setupTables` parses it: `HmOut`'s for a statement with hidden outputs, else `HmRow`'s. -/
def circuitOf (tags : Tags) (file : ByteArray) (tables : List (String × ByteArray)) : V Circuit :=
  if tags.hiddenOutputs then HmOut.parse tags file tables else HmRow.parse tags file tables

/-- The public file as the headlines read it: `HmOut`'s for a statement with hidden outputs, else `HmRow`'s. -/
def publicOf (tags : Tags) (c : Circuit) (file : ByteArray) : V Public :=
  if tags.hiddenOutputs then HmOut.loadPublic tags c file else HmRow.loadPublic tags c file

/-- The forms of `verify`'s statement the proof covers, each with what a statement outside it is: an hm96-sha512 row
statement (`Stmt.setupTables`, the setup the proof models), untyped, with retained rounds and `hm96-sha512/v1` leaves, at
most one table without `--zk`, and no public input ports (the proof's setup takes no `--public-inputs`). -/
def statementRules (tags : Tags) (zk : Bool) (count : Nat) (publicInputs : Bool) : List (Bool × String) :=
  [(tags.hm96Rows, s!"{tags.name}, a statement before hm96-sha512 rows"),
   (!tags.typed, s!"{tags.name}, a typed statement"),
   (tags.retainRounds, s!"{tags.name}, a statement without retained rounds"),
   (tags.merkleLeaf == "hm96-sha512/v1", s!"{tags.name}, a statement with {tags.merkleLeaf} leaves"),
   (zk || decide (count ≤ 1), s!"{count} tables in a session without --zk"),
   (!publicInputs, "public input ports (--public-inputs)")]

/-- The first rule that fails, by what it refuses. -/
def firstFailure (rules : List (Bool × String)) : Option String := (rules.find? fun r => !r.1).map (·.2)

def refusal (why : String) : String := s!"refused: {why}: a statement form the soundness proof does not cover"

/-- Where a shared-row public file (header `shared_rows`) is a form the proof covers: a `--zk` session of a statement
with hidden outputs whose verifier holds its own registered reads (`--registered F`), the setting of
`EndToEndRegistered` (an untyped statement, which `statementRules` requires of every form). -/
def sharedRowsOk (tags : Tags) (zk registered : Bool) : Bool := zk && tags.hiddenOutputs && registered

/-- **`verify`'s statement is one the soundness proof covers**: every rule of `statementRules`, and on the circuit and public
file as the proof reads them, a shared-row public file only where `sharedRowsOk`, and `scopeOk` (`scopeOkH` for a
statement with hidden outputs, whose headlines also cover `sha512/row/v2` and `sha512/row-seg/v1` rows). `tags` are the
session's (under `--zk`, `Zk.tags`'s), `count` its tables, `publicInputs` whether the verifier holds its own public
input words and `registered` whether it holds its own registered reads. When this check moves from `scopeOk` to a format
check (`formatOk`), a tail must stay refused (`c.outNet == c.unitNet`): `formatOk` puts no condition on a tail's stage
nets, and the session headline's `keyProg` reads the unit nets only. -/
def check (tags : Tags) (zk : Bool) (circuitFile publicFile : ByteArray) (tables : List (String × ByteArray)) (count : Nat)
    (publicInputs registered : Bool) : Except String Unit :=
  match firstFailure (statementRules tags zk count publicInputs) with
  | some why => .error (refusal why)
  | none => match circuitOf tags circuitFile tables with
    | .error e => .error (refusal s!"a circuit that does not parse ({e})")
    | .ok c => match publicOf tags c publicFile with
      | .error e => .error (refusal s!"a public file that does not load ({e})")
      | .ok pub =>
        if pub.tables.isSome && !sharedRowsOk tags zk registered then
          .error (refusal "a shared-row public file (header shared_rows)")
        else if (if tags.hiddenOutputs then scopeOkH c else scopeOk c) then .ok ()
        else .error (refusal (scopeWhy tags.hiddenOutputs c))

/-- What a draw's law is, when no soundness claim draws by it: the claims draw by the stratified law (with or without
`--zk`) and, under `--zk`, by the subset law, so a bernoulli draw, a work draw (with or without a closure) and a subset
draw without `--zk` are outside them. -/
def lawWhy (zk : Bool) (d : Draw.UnitDraw) : Option String :=
  match d.law with
  | .stratified .. => none
  | .subset _ => if zk then none else some "a subset draw without --zk"
  | .bernoulli .. => some "a bernoulli draw"
  | .work .. => some (if d.closure.isEmpty then "a work draw" else "a work draw with a closure")

/-- **The session's draw is by a law a soundness claim draws by** (PROTOCOL.md §16.15): `draw` is the record's
(`recordDraw`), none for a session without one. Every claim is about a drawn session, so a session without a draw is
refused (D12); so is a draw that does not parse. -/
def drawLaw (zk : Bool) : Option Lean.Json → Except String Unit
  | none => .error "refused: a session without a draw: every soundness claim is about a drawn session (D12)"
  | some j => match Draw.UnitDraw.ofJson j with
    | .error e => .error s!"refused: a draw that does not parse ({e})"
    | .ok d => match lawWhy zk d with
      | some why => .error s!"refused: {why}: no soundness claim draws by this law"
      | none => .ok ()

end Flock.ProvedScope
