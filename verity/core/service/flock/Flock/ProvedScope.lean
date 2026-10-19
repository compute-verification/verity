/-
The statement forms the soundness proof covers (PROTOCOL.md §16.15). `verify` refuses every other: soundness is one theorem
over every session the verifier accepts, so a form no theorem covers is refused, never accepted. Widening the proof to
another form is a change to one rule here, with the proof that the headlines cover it. With hidden outputs that is the
general instance form (`scopeOkI`, once `wiresAcyclic` passed); with public outputs, the unit's form (`scopeOk`).
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

/-- **The circuits the hidden-output headlines cover at the template instance's program** (`instProg`): `scopeOkH`'s
conditions but the first two, so the unit may read any number of input groups and any stage may return the outputs. Its
headlines also need `wiresAcyclic`, which `check` requires first wherever the wires feed the unit or the outputs
(`wiresFeed`); where they don't, the circuit passes `scopeOkH` and `scopeOkH`'s headlines cover it. -/
def scopeOkI (c : Circuit) : Bool :=
  decide (c.unit.unitLog ≤ 32) && portReadsB c.unit && (v1Ports c || bitRows c)

/-- `f` over row `i`'s entries in order: `(m.row i).foldl f init`, without building the row. -/
@[inline] def foldRow {β : Type} (m : Sparse) (i : Nat) (f : β → Nat → β) (init : β) : β :=
  let s := m.start.getD i 0
  Nat.fold (m.len.getD i 0) (fun t _ acc => f acc (rd32 m.data (s + 4 * t))) init

/-- What `net`'s output ports read: for each, the input ports whose bits it reads, ascending; and whether its computed rows
and output ports read, among the input columns, only port bits, each row only earlier rows. A column's reads are a mask,
bit `p + 1` for input port `p` and bit `0` for an input column no port carries or a row read out of order; a computed
row's are the union of the columns' it reads, the constant reads nothing, and a column past it, which no row computes, is
read out of order. A lookup slot, which the verifier builds from its pinned table, is taken whole: each output port reads
every input port. -/
def outReads (net : Net) : Array (Array Nat) × Bool :=
  let ins := portCols net.inGroups
  let outs := portCols net.outGroups
  if net.lookup.isSome then (outs.map fun _ => Array.range ins.size, true)
  else
    let n := min (net.inWords * WORD) net.constPos
    let inMask := (Array.range ins.size).foldl (fun m p =>
      let q := ins.getD p default
      let bit := 1 <<< (p + 1)
      (Array.range q.2).foldl (fun m x => m.setIfInBounds (q.1 + x) bit) m) (Array.replicate n 1)
    let col := fun (rows : Array Nat) (x : Nat) =>
      if x < n then inMask.getD x 1 else if x < net.constPos then rows.getD (x - n) 1 else if x == net.constPos then 0
      else 1
    -- most reads add nothing to the mask so far: keep it then, rather than build an equal one
    let union := fun (m y : Nat) => if y == 0 || y == m then m else if m == 0 then y else m ||| y
    let rows := Nat.fold (net.constPos - n) (fun i _ rows =>
      let m := foldRow net.b (n + i) (fun m x => union m (col rows x))
        (foldRow net.a (n + i) (fun m x => union m (col rows x)) 0)
      rows.push m) #[]
    let masks := outs.map fun q => (Array.range q.2).foldl (fun m x => union m (col rows (q.1 + x))) 0
    (masks.map fun m => (Array.range ins.size).filter fun p => m.testBit (p + 1),
      rows.all (· % 2 == 0) && masks.all (· % 2 == 0))

/-- One pass over the wires: each wire whose destination `dst` is not yet filled fills it when its source is `ready`.
Returns the filled ports and whether the pass filled any. -/
def fillPass (ready : Array Bool → Wire → Bool) (dst : Wire → Option Nat) (ws : Array Wire) (av : Array Bool) :
    Array Bool × Bool :=
  ws.foldl (fun (av, ch) w => match dst w with
    | some d => if !av.getD d true && ready av w then (av.set! d true, true) else (av, ch)
    | none => (av, ch)) (av, false)

/-- `fillPass` until a pass fills nothing, at most `fuel` passes. -/
def fillLoop (pass : Array Bool → Array Bool × Bool) : Nat → Array Bool → Array Bool
  | 0, av => av
  | fuel + 1, av => match pass av with
    | (av', true) => fillLoop pass fuel av'
    | (av', false) => av'

/-- **Every slot computes from the leaves, in some order of the wires**: outside the rows' `sha512x3` and `hm96` slots,
starting from the units' leaf ports and the leaf cuts, the wires fill every input port of every slot, each wire once every
input port its source port reads in its own slot (`outReads`) is filled; and no row or output port of a unit or stage
reads an input column that is not a port bit. Each pass fills at least one port, so `wires.size + 1` passes reach the
fixpoint. Then every slot's rows evaluate in the order the ports filled, each reading only computed columns: the wires
have no cycle through the rows. -/
def wiresAcyclic (c : Circuit) : Bool :=
  let rowNet := fun (n : Nat) => n == c.shaNet || n == c.hmNet
  let nPorts := c.nets.map fun e => (portCols e.2.inGroups).size
  let slots := fun (n : Nat) => if rowNet n then 0 else c.perVu.getD n 0
  let base := (Array.range c.nets.size).foldl (fun b n => b.push (b.back?.getD 0 + slots n * nPorts.getD n 0)) #[0]
  let node := fun (n i p : Nat) => base.getD n 0 + i * nPorts.getD n 0 + p
  let inSlot := fun (n i p : Nat) => decide (i < slots n) && decide (p < nPorts.getD n 0)
  let reads := (Array.range c.nets.size).map fun n =>
    if slots n == 0 then (#[], true) else outReads (c.nets.getD n default).2
  let g0 := (c.unit.inGroups.getD 0 default).bits.size
  let av := (Array.range c.upv).foldl (fun av u => (Array.range g0).foldl (fun av p =>
    if inSlot c.unitNet u p then av.set! (node c.unitNet u p) true else av) av)
    (Array.replicate (base.getD c.nets.size 0) false)
  let av := c.leafCuts.foldl (fun av e =>
    if inSlot c.unitNet e.1 e.2.1 then av.set! (node c.unitNet e.1 e.2.1) true else av) av
  let ready := fun (av : Array Bool) (w : Wire) =>
    decide (w.src.2.1 < slots w.src.1) && match (reads.getD w.src.1 (#[], false)).1[w.src.2.2]? with
      | some ps => ps.all fun q => av.getD (node w.src.1 w.src.2.1 q) false
      | none => false
  let dst := fun (w : Wire) =>
    if inSlot w.dst.1 w.dst.2.1 w.dst.2.2 then some (node w.dst.1 w.dst.2.1 w.dst.2.2) else none
  (fillLoop (fillPass ready dst c.wires) (c.wires.size + 1) av).all id &&
    (Array.range c.nets.size).all fun n => slots n == 0 || (reads.getD n (#[], false)).2

/-- Where the wires' order matters to what the unit computes: the unit reads input groups past its leaf group (cut inputs
a wire may fill), or a tail stage returns the outputs. -/
def wiresFeed (c : Circuit) : Bool := c.unit.inGroups.size != 1 || c.outNet != c.unitNet

def wiresWhy : String :=
  "a slot no order of the wires computes from the leaves: a cycle through wires, an input port no wire fills, or a read \
    of an input column that is not a port bit"

/-- Which part of `scopeOk` (`scopeOkI` for a statement with hidden outputs) a circuit fails, for the refusal. -/
def scopeWhy (hidden : Bool) (c : Circuit) : String :=
  if !hidden && c.unit.inGroups.size != 1 then s!"a unit reading {c.unit.inGroups.size} input groups"
  else if !hidden && c.outNet != c.unitNet then "a circuit whose outputs a tail stage computes (META out_net)"
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

/-- **The most tables a `--zk` session may have**, `2^16`: ZK's tape premise holds only over a bounded table count. -/
def zkTableCap : Nat := 65536

/-- **The proof's Merkle leaves the headlines cover**: `hm96-sha512/v1` in every form, and plain SHA-512 leaves
(`flock-leaf/sha512-unsalted`, a recursive session's inner statement, §16.16) on a statement with hidden outputs without
`--zk`. Without `--zk` a leaf only binds its column, which either leaf does (a second column under one leaf is a SHA-512
collision); `--zk`'s hiding needs `hm96`'s salts. -/
def leafOk (tags : Tags) (zk : Bool) : Bool :=
  tags.merkleLeaf == "hm96-sha512/v1" ||
    (!zk && tags.hiddenOutputs && tags.merkleLeaf == "flock-leaf/sha512-unsalted")

/-- The forms of `verify`'s statement the proof covers, each with what a statement outside it is: an hm96-sha512 row
statement (`Stmt.setupTables`, the setup the proof models), untyped, with retained rounds and leaves `leafOk` covers, at
most one table without `--zk` and `zkTableCap` with it, and no public input ports (the proof's setup takes no
`--public-inputs`). -/
def statementRules (tags : Tags) (zk : Bool) (count : Nat) (publicInputs : Bool) : List (Bool × String) :=
  [(tags.hm96Rows, s!"{tags.name}, a statement before hm96-sha512 rows"),
   (!tags.typed, s!"{tags.name}, a typed statement"),
   (tags.retainRounds, s!"{tags.name}, a statement without retained rounds"),
   (leafOk tags zk, s!"{tags.name}, a statement with {tags.merkleLeaf} leaves{if zk then " under --zk" else ""}"),
   (decide (count ≤ if zk then zkTableCap else 1),
    if zk then s!"{count} tables in a --zk session, more than {zkTableCap}"
    else s!"{count} tables in a session without --zk"),
   (!publicInputs, "public input ports (--public-inputs)")]

/-- The first rule that fails, by what it refuses. -/
def firstFailure (rules : List (Bool × String)) : Option String := (rules.find? fun r => !r.1).map (·.2)

def refusal (why : String) : String := s!"refused: {why}: a statement form the soundness proof does not cover"

/-- The least `m` the proof covers: `25` under `--zk`, else `22`. -/
def mMin (zk : Bool) : Nat := if zk then 25 else 22

/-- **The tables' sizes the proof covers**: every table's `m` (`ms`, each table's `Setup.m`) at least `mMin`. A table's
`m` is its drawn statement's, which only the setup computes, so `verify` checks this on the session's tables right after
`check`, not on the statement's files. Each verifier's own schedule refuses a smaller `m` before it (`fast100` below `22`,
`Zk.schedule` at `m ≤ 24`), so this refuses no session `verify` would otherwise accept. -/
def sizesOk (zk : Bool) (ms : Array Nat) : Except String Unit :=
  match ms.find? (· < mMin zk) with
  | some m => .error (refusal s!"m = {m}, below {mMin zk}{if zk then " under --zk" else " without --zk"}")
  | none => .ok ()

/-- Where a shared-row public file (header `shared_rows`) is a form the proof covers: a `--zk` session of a statement
with hidden outputs whose verifier holds its own registered reads (`--registered F`), the setting of
`EndToEndRegistered` (an untyped statement, which `statementRules` requires of every form). -/
def sharedRowsOk (tags : Tags) (zk registered : Bool) : Bool := zk && tags.hiddenOutputs && registered

/-- **`verify`'s statement is one the soundness proof covers**: every rule of `statementRules`, and on the circuit and public
file as the proof reads them, a shared-row public file only where `sharedRowsOk`, wherever the wires feed the unit or the
outputs (`wiresFeed`) an order of the wires that computes every slot (`wiresAcyclic`), and `scopeOk` (`scopeOkI` for a
statement with hidden outputs). `tags` are the session's (under `--zk`, `Zk.tags`'s), `count` its tables, `publicInputs`
whether the verifier holds its own public input words and `registered` whether it holds its own registered reads. The
tables' sizes are `sizesOk`'s, which `verify` runs after this on the tables the session set up.

With hidden outputs this is the general instance form: one unit, repeated, reading any number of committed input groups,
and any number of stages the verifier evaluates, each reading wires from earlier outputs, the outputs returned by any of
them. Its headlines are at the template instance's program (`instProg`), which needs `wiresAcyclic`. A circuit the wires
don't feed passes `scopeOkH` and is also covered at the unit's program (`keyProg`). With public outputs `scopeOk` keeps
`wiresFeed` false: the public-output headlines read the outputs off the unit, so the unit reads only its leaf group and
returns the outputs. -/
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
        else if wiresFeed c && !wiresAcyclic c then .error (refusal wiresWhy)
        else if (if tags.hiddenOutputs then scopeOkI c else scopeOk c) then .ok ()
        else .error (refusal (scopeWhy tags.hiddenOutputs c))

/-- `check` on its circuit as `circuitOf` returns it (`parsed`), which `verify` parses once for this and the setup
(`check_eq`). -/
def checkOf (parsed : V Circuit) (tags : Tags) (zk : Bool) (publicFile : ByteArray) (count : Nat)
    (publicInputs registered : Bool) : Except String Unit :=
  match firstFailure (statementRules tags zk count publicInputs) with
  | some why => .error (refusal why)
  | none => match parsed with
    | .error e => .error (refusal s!"a circuit that does not parse ({e})")
    | .ok c => match publicOf tags c publicFile with
      | .error e => .error (refusal s!"a public file that does not load ({e})")
      | .ok pub =>
        if pub.tables.isSome && !sharedRowsOk tags zk registered then
          .error (refusal "a shared-row public file (header shared_rows)")
        else if wiresFeed c && !wiresAcyclic c then .error (refusal wiresWhy)
        else if (if tags.hiddenOutputs then scopeOkI c else scopeOk c) then .ok ()
        else .error (refusal (scopeWhy tags.hiddenOutputs c))

theorem check_eq (tags : Tags) (zk : Bool) (circuitFile publicFile : ByteArray) (tables : List (String × ByteArray))
    (count : Nat) (publicInputs registered : Bool) :
    check tags zk circuitFile publicFile tables count publicInputs registered
      = checkOf (circuitOf tags circuitFile tables) tags zk publicFile count publicInputs registered :=
  rfl

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
