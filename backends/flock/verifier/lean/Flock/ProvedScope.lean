/-
The statement forms the soundness proof covers (PROTOCOL.md §16.15, §16.16). `verify` refuses every other: soundness is one theorem
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

/-- Which of the indices below `n` are in `xs`, as an array. -/
def maskOf (xs : List Nat) (n : Nat) : Array Bool :=
  xs.foldl (fun m x => m.setIfInBounds x true) (Array.replicate n false)

/-- A canonical row (PROTOCOL.md §16.16): a `sha512/row/v2` bit row, unsegmented. -/
def rowOk (p : Port) : Bool := p.v2 && p.segs.isNone

/-- Every output row holds exactly the bits of its run of output ports (`HmOut.outGroup`): no zero-padded output row. -/
def outRowsExact (c : Circuit) : Bool :=
  (List.range c.outRowPorts.size).all fun j =>
    (c.ports.getD (HmOut.nIn c + j) default).bits == 16 * (HmOut.outGroup c j).2.2.2

/-- Every unit's every cut input (a port past the leaf group's) is a leaf cut, a row word: none is wired from another
net's output. The parse gives each cut input exactly one source, a wire or a leaf cut. -/
def cutsCovered (c : Circuit) : Bool :=
  let n0 := (c.unit.inGroups.getD 0 default).bits.size
  let np := (portCols c.unit.inGroups).size
  let m := maskOf (c.leafCuts.toList.filterMap fun e => if e.2.1 < np then some (e.1 * np + e.2.1) else none) (c.upv * np)
  (List.range c.upv).all fun k => (List.range (np - n0)).all fun i => m.getD (k * np + (n0 + i)) false

/-- **The canonical I/O format** (PROTOCOL.md §16.16), as the soundness proof's `Canonical.formatOk` reads it: every row
canonical, output rows exact, every unit input a row bit or the zero, computed unit rows reading port bits only, at most
`2^32` unit rows, and the VUs per block a power of two. -/
def formatOk (c : Circuit) : Bool :=
  c.ports.all rowOk && outRowsExact c && cutsCovered c && portReadsB c.unit && decide (c.unit.unitLog ≤ 32) &&
    2 ^ c.g.log2 == c.g

/-- The canonical circuits the canonical headline covers: under `--zk`, a statement with hidden outputs, the unit
returning the outputs (a tail stays refused: no claim binds a tail's outputs to its inputs), in the canonical format. -/
def canonicalOk (tags : Tags) (zk : Bool) (c : Circuit) : Bool :=
  zk && tags.hiddenOutputs && c.outNet == c.unitNet && formatOk c

/-- Which part of `canonicalOk` a circuit fails under `--zk` with hidden outputs, for the refusal. -/
def canonicalWhy (c : Circuit) : String :=
  if c.outNet != c.unitNet then "a circuit whose outputs a tail stage computes (META out_net)"
  else if !c.ports.all rowOk then "a row that is not an unsegmented sha512/row/v2 row, in a circuit with sha512/row/v2 rows"
  else if !outRowsExact c then "an output row that is not exactly its output ports' bits"
  else if !cutsCovered c then "a unit input wired from another net's output, not a row bit"
  else if !portReadsB c.unit then "a unit row reading an input column that is not a port bit"
  else if c.unit.unitLog > 32 then s!"a unit of 2^{c.unit.unitLog} rows"
  else "a vus_per_block that is not a power of two"

/-- Why `check` refuses a circuit outside `scopeOk` and `canonicalOk`: `canonicalWhy` for a circuit with a
`sha512/row/v2` row under `--zk` with hidden outputs, where the canonical form is covered, else `scopeWhy`. -/
def circuitWhy (tags : Tags) (zk : Bool) (c : Circuit) : String :=
  if zk && tags.hiddenOutputs && c.ports.any (·.v2) then canonicalWhy c else scopeWhy c

/-- The circuit as `Stmt.setupTables` parses it: `HmOut`'s for a statement with hidden outputs, else `HmRow`'s. -/
def circuitOf (tags : Tags) (file : ByteArray) (tables : List (String × ByteArray)) : V Circuit :=
  if tags.hiddenOutputs then HmOut.parse tags file tables else HmRow.parse tags file tables

/-- The public file as the headlines read it: `HmOut`'s for a statement with hidden outputs, else `HmRow`'s; with the
verifier's own public input words (`publicInputs`), `HmIn`'s at the circuit's public input ports, as `Stmt.setupHidden`
loads it, every instance's public words at its end. -/
def publicOf (tags : Tags) (c : Circuit) (file : ByteArray) (publicInputs : Bool := false) : V Public :=
  if publicInputs then HmIn.parse c >>= fun pi => HmIn.loadPublic tags c pi file
  else if tags.hiddenOutputs then HmOut.loadPublic tags c file else HmRow.loadPublic tags c file

/-- The forms of `verify`'s statement the proof covers, each with what a statement outside it is: an hm96-sha512 row
statement (`Stmt.setupTables`, the setup the proof models), untyped, with retained rounds and `hm96-sha512/v1` leaves, and
at most one table without `--zk`. -/
def statementRules (tags : Tags) (zk : Bool) (count : Nat) : List (Bool × String) :=
  [(tags.hm96Rows, s!"{tags.name}, a statement before hm96-sha512 rows"),
   (!tags.typed, s!"{tags.name}, a typed statement"),
   (tags.retainRounds, s!"{tags.name}, a statement without retained rounds"),
   (tags.merkleLeaf == "hm96-sha512/v1", s!"{tags.name}, a statement with {tags.merkleLeaf} leaves"),
   (zk || decide (count ≤ 1), s!"{count} tables in a session without --zk")]

/-- The first rule that fails, by what it refuses. -/
def firstFailure (rules : List (Bool × String)) : Option String := (rules.find? fun r => !r.1).map (·.2)

def refusal (why : String) : String := s!"refused: {why}: a statement form the soundness proof does not cover"

/-- **`verify`'s statement is one the soundness proof covers**: every rule of `statementRules`, and on the circuit and public
file as the proof reads them, public input words only at a circuit `canonicalOk` passes, no shared-row public file, and
`scopeOk` or `canonicalOk`. `tags` are the session's (under `--zk`, `Zk.tags`'s), `count` its tables and `publicInputs`
whether the verifier holds its own public input words (`--public-inputs F`): the claim at them is the canonical
headline's, so at any other circuit they are refused. `canonicalOk` keeps a tail refused (`c.outNet == c.unitNet`):
`formatOk` puts no condition on a tail's stage nets, and the session headline's `keyProg` reads the unit nets only. -/
def check (tags : Tags) (zk : Bool) (circuitFile publicFile : ByteArray) (tables : List (String × ByteArray)) (count : Nat)
    (publicInputs : Bool) : Except String Unit :=
  match firstFailure (statementRules tags zk count) with
  | some why => .error (refusal why)
  | none => match circuitOf tags circuitFile tables with
    | .error e => .error (refusal s!"a circuit that does not parse ({e})")
    | .ok c =>
      if publicInputs && !canonicalOk tags zk c then .error (refusal "public input ports (--public-inputs)")
      else match publicOf tags c publicFile publicInputs with
      | .error e => .error (refusal s!"a public file that does not load ({e})")
      | .ok pub =>
        if pub.tables.isSome then .error (refusal "a shared-row public file (header shared_rows)")
        else if scopeOk c || canonicalOk tags zk c then .ok ()
        else .error (refusal (circuitWhy tags zk c))

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

/-- Whether `check` passes `verify`'s statement by `canonicalOk` alone: its circuit is outside `scopeOk`. -/
def canonicalAt (tags : Tags) (zk : Bool) (circuitFile : ByteArray) (tables : List (String × ByteArray)) : Bool :=
  match circuitOf tags circuitFile tables with
  | .ok c => !scopeOk c && canonicalOk tags zk c
  | .error _ => false

/-- **A canonical statement's draw is by the subset law** (PROTOCOL.md §16.16): the canonical claim draws by the live
subset draw alone, so at a statement `check` passes by `canonicalOk` alone (`canonical`) a stratified draw is refused. -/
def canonicalLaw (canonical : Bool) : Option Lean.Json → Except String Unit
  | some j => match Draw.UnitDraw.ofJson j with
    | .ok d => match canonical, d.law with
      | true, .stratified .. => .error "refused: a stratified draw at a canonical statement: no soundness claim draws by this law"
      | _, _ => .ok ()
    | .error _ => .ok ()
  | none => .ok ()

end Flock.ProvedScope
