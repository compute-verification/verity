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

/-- Every row is a `sha512/row/v1` row: `16 · words` bits in at least one chunk. -/
def v1Ports (c : Circuit) : Bool :=
  c.ports.all fun p => !p.v2 && p.bits == 16 * p.words && 0 < p.chunks

/-- The circuits the headlines cover: the unit reads only its leaf group, returns the outputs, has at most `2^32` rows and
reads only port bits among its inputs, and every row is a `sha512/row/v1` row. -/
def scopeOk (c : Circuit) : Bool :=
  c.unit.inGroups.size == 1 && c.outNet == c.unitNet && decide (c.unit.unitLog ≤ 32) && portReadsB c.unit && v1Ports c

/-- Which part of `scopeOk` a circuit fails, for the refusal. -/
def scopeWhy (c : Circuit) : String :=
  if c.unit.inGroups.size != 1 then s!"a unit reading {c.unit.inGroups.size} input groups"
  else if c.outNet != c.unitNet then "a circuit whose outputs a tail stage computes (META out_net)"
  else if c.unit.unitLog > 32 then s!"a unit of 2^{c.unit.unitLog} rows"
  else if !portReadsB c.unit then "a unit row reading an input column that is not a port bit"
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

/-- **`verify`'s statement is one the soundness proof covers**: every rule of `statementRules`, and on the circuit and public
file as the proof reads them, no shared-row public file and `scopeOk`. `tags` are the session's (under `--zk`,
`Zk.tags`'s), `count` its tables and `publicInputs` whether the verifier holds its own public input words. -/
def check (tags : Tags) (zk : Bool) (circuitFile publicFile : ByteArray) (tables : List (String × ByteArray)) (count : Nat)
    (publicInputs : Bool) : Except String Unit :=
  match firstFailure (statementRules tags zk count publicInputs) with
  | some why => .error (refusal why)
  | none => match circuitOf tags circuitFile tables with
    | .error e => .error (refusal s!"a circuit that does not parse ({e})")
    | .ok c => match publicOf tags c publicFile with
      | .error e => .error (refusal s!"a public file that does not load ({e})")
      | .ok pub =>
        if pub.tables.isSome then .error (refusal "a shared-row public file (header shared_rows)")
        else if scopeOk c then .ok ()
        else .error (refusal (scopeWhy c))

end Flock.ProvedScope
