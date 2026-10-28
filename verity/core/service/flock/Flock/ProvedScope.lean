/-
The statement forms the soundness proof covers (PROTOCOL.md §16.15). `verify` refuses every other: soundness is one theorem
over every session the verifier accepts, so a form no theorem covers is refused, never accepted. Widening the proof to
another form is a change to one rule here, with the proof that the headlines cover it. With hidden outputs that is the
general instance form (`scopeOkI`, once `wiresAcyclic` passed); with public outputs, the unit's form (`scopeOk`).
-/
import Flock.HmRow
import Flock.JsonParse

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

/-- No output port is pinned (META `pinned_outputs`, `HmRow.pinnedCount`): a form no soundness theorem covers yet. -/
def unpinned (c : Circuit) : Bool := HmRow.pinnedCount c.metaJ == 0

def pinnedWhy : String := HmRow.pinnedWhy

/-- Every row is an unsegmented `sha512/row/v1` row: `16 · words` bits in at least one chunk. -/
def v1Rows (c : Circuit) : Bool :=
  c.ports.all fun p => !p.v2 && p.segs.isNone && p.bits == 16 * p.words && 0 < p.chunks

/-- `v1Rows`, and no pinned output (`unpinned`). -/
def v1Ports (c : Circuit) : Bool := v1Rows c && unpinned c

/-- The circuits the headlines cover: the unit reads only its leaf group, returns the outputs, has at most `2^32` rows and
reads only port bits among its inputs, and every row is a `sha512/row/v1` row. -/
def scopeOk (c : Circuit) : Bool :=
  c.unit.inGroups.size == 1 && c.outNet == c.unitNet && decide (c.unit.unitLog ≤ 32) && portReadsB c.unit && v1Ports c

/-- A `sha512/row/v2` row the hidden-output headlines cover: unsegmented, `16 · words` bits, its bytes and SHA-512's 17
bytes of padding filling its compressions' blocks and ending in its `hm96` slot's block. The last compression's block
may hold the bytes' end and the padding's start (`2 · words mod 128 ≥ 112`), the `hm96` block then only padding. -/
def v2Row (p : Port) : Bool :=
  p.v2 && p.segs.isNone && p.bits == 16 * p.words && decide (128 * p.chunks ≤ 2 * p.words + 16) &&
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
def bitRowsU (c : Circuit) : Bool :=
  c.ports.all (fun p => v2Row p || segRow p) && c.outRowPorts.all (· == 1) &&
    (List.range c.outRowPorts.size).all fun j => v2Row (c.ports.getD (HmOut.nIn c + j) default)

/-- `bitRowsU`, and no pinned output (`unpinned`). -/
def bitRows (c : Circuit) : Bool := bitRowsU c && unpinned c

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
  else if v1Rows c || (hidden && bitRowsU c) then pinnedWhy
  else if hidden then "rows that are neither all sha512/row/v1 rows of 16 · words bits in at least one chunk nor all \
    unsegmented sha512/row/v2 rows of 16 · words bits ending in their hm96 block or sha512/row-seg/v1 rows of \
    16 · words bits whose segments end in their last block, the output rows unsegmented, one output port each"
  else "a row that is not a sha512/row/v1 row of 16 · words bits in at least one chunk"

/-- **A templated circuit the hidden-output headlines cover** (its statement circuit is the flat
circuit its template unrolls to, `HmOut.unrolled`), checked on the template's description, never on the rows it unrolls
to: a `--zk` session of one table at hidden outputs without the verifier's own registered reads, whose unrolled unit
(`Tmpl.unroll`'s net: the root's columns, the template's ports) reads only its leaf group and returns the outputs, so the
wires feed neither (`wiresFeed`), at most `2^32` rows, and `scopeOkI`'s rows. `TmplUnit.read` passed `Tmpl.unrollOk`,
whose `Tmpl.ordered` makes the unrolled unit read only port bits among its inputs (`portReadsB`). -/
def scopeOkT (hidden zk registered : Bool) (count : Nat) (c : Circuit) (tu : TmplUnit) : Bool :=
  hidden && zk && !registered && decide (count ≤ 1) && tu.t.inGroups.size == 1 && c.outNet == c.unitNet &&
    decide (max 7 (log2ceil (tu.t.defs.getD tu.t.root default).cols) ≤ 32) && (v1Ports c || bitRows c)

/-- Which part of `scopeOkT` a templated circuit fails, for the refusal. -/
def scopeWhyT (hidden zk registered : Bool) (count : Nat) (c : Circuit) (tu : TmplUnit) : String :=
  if !hidden then "a templated statement without hidden outputs"
  else if !zk then "a templated statement without --zk"
  else if registered then "a templated statement with the verifier's registered reads (--registered)"
  else if count > 1 then s!"a templated statement in {count} tables"
  else if tu.t.inGroups.size != 1 then s!"a templated unit reading {tu.t.inGroups.size} input groups"
  else if c.outNet != c.unitNet then "a templated circuit whose outputs a tail stage computes (META out_net)"
  else if max 7 (log2ceil (tu.t.defs.getD tu.t.root default).cols) > 32 then "a templated unit of more than 2^32 rows"
  else scopeWhy true c

/-- The circuit as `Stmt.setupTables` parses it: for a statement with hidden outputs the one its session commits
(`HmOut.parseCommit`: a templated statement's carries its template, `Circuit.tmpl`), else `HmRow`'s. -/
def circuitOf (tags : Tags) (file : ByteArray) (tables : List (String × ByteArray)) : V Circuit :=
  if tags.hiddenOutputs then HmOut.parseCommit tags file tables else HmRow.parse tags file tables

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

/-- Where a shared-row public file (header `shared_rows`) is a form the proof covers: a statement with hidden outputs and
`hm96-sha512/v1` leaves whose verifier holds its own registered reads (`--registered F`), with or without `--zk`: under
`--zk` the setting of `EndToEndRegistered` (`FailClosed.SoundZHJR`), without it `FailClosed.SoundExecHR`. A statement with
plain SHA-512 leaves (a recursive session's inner statement, whose registered reads `Registered.checkDescription` reads)
keeps its own rows. `zk` is unread: `--zk` needs `hm96-sha512/v1` leaves (`leafOk`), so the rule is the same in both. -/
def sharedRowsOk (tags : Tags) (_zk registered : Bool) : Bool :=
  tags.hiddenOutputs && registered && tags.merkleLeaf == "hm96-sha512/v1"

/-- What `check` requires of the circuit and public file it read: a shared-row public file only where `sharedRowsOk`; for a
templated circuit (`Circuit.tmpl`) `scopeOkT`, on its description; for any other, wherever the wires feed the unit or the
outputs (`wiresFeed`) an order of the wires that computes every slot (`wiresAcyclic`), and `scopeOk` (`scopeOkI` for a
statement with hidden outputs). -/
def circuitRules (tags : Tags) (zk : Bool) (count : Nat) (registered : Bool) (c : Circuit) (pub : Public) :
    Except String Unit :=
  if pub.tables.isSome && !sharedRowsOk tags zk registered then
    .error (refusal "a shared-row public file (header shared_rows)")
  else match c.tmpl with
    | some tu =>
      if scopeOkT tags.hiddenOutputs zk registered count c tu then .ok ()
      else .error (refusal (scopeWhyT tags.hiddenOutputs zk registered count c tu))
    | none =>
      if wiresFeed c && !wiresAcyclic c then .error (refusal wiresWhy)
      else if (if tags.hiddenOutputs then scopeOkI c else scopeOk c) then .ok ()
      else .error (refusal (scopeWhy tags.hiddenOutputs c))

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
returns the outputs. A templated statement's circuit (`HmOut.parseCommit`'s, carrying its template) is checked on its
description (`scopeOkT`): its statement circuit, the flat circuit it unrolls to, is one `scopeOkH` covers at one `--zk`
table. -/
def check (tags : Tags) (zk : Bool) (circuitFile publicFile : ByteArray) (tables : List (String × ByteArray)) (count : Nat)
    (publicInputs registered : Bool) : Except String Unit :=
  match firstFailure (statementRules tags zk count publicInputs) with
  | some why => .error (refusal why)
  | none => match circuitOf tags circuitFile tables with
    | .error e => .error (refusal s!"a circuit that does not parse ({e})")
    | .ok c => match publicOf tags c publicFile with
      | .error e => .error (refusal s!"a public file that does not load ({e})")
      | .ok pub => circuitRules tags zk count registered c pub

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
      | .ok pub => circuitRules tags zk count registered c pub

theorem check_eq (tags : Tags) (zk : Bool) (circuitFile publicFile : ByteArray) (tables : List (String × ByteArray))
    (count : Nat) (publicInputs registered : Bool) :
    check tags zk circuitFile publicFile tables count publicInputs registered
      = checkOf (circuitOf tags circuitFile tables) tags zk publicFile count publicInputs registered :=
  rfl

theorem circuitOf_hidden {tags : Tags} (hh : tags.hiddenOutputs = true) (file : ByteArray)
    (tables : List (String × ByteArray)) : circuitOf tags file tables = HmOut.parseCommit tags file tables := by
  simp only [circuitOf, hh, ↓reduceIte]

/-- At hidden outputs, the circuit `HmOut.parse` reads (a flat statement's: it refuses a templated one's tags) is the one
`circuitOf` reads. -/
theorem circuitOf_of_parse {tags : Tags} {file : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (hh : tags.hiddenOutputs = true) (h : HmOut.parse tags file tables = .ok c) : circuitOf tags file tables = .ok c := by
  rw [circuitOf_hidden hh, HmOut.parseCommit, if_neg (by simp [HmOut.parse_isTmpl h])]
  exact h

/-- At hidden outputs and a flat statement's tags, the circuit `circuitOf` reads is `HmOut.parse`'s. -/
theorem parse_of_circuitOf {tags : Tags} {file : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (hh : tags.hiddenOutputs = true) (hi : tags.isTmpl = false) (h : circuitOf tags file tables = .ok c) :
    HmOut.parse tags file tables = .ok c := by
  rwa [circuitOf_hidden hh, HmOut.parseCommit, if_neg (by simp [hi])] at h

/-- What `--plain-tables` refuses of a statement: hidden outputs with leaves other than `hm96-sha512/v1` (the plain proof
at several tables reads hm96 leaves; `check` at one table also passes plain SHA-512 leaves at hidden outputs) and more
than `Tags.MAX_TABLES` tables, what `Stmt.setupTables` sets up. -/
def plainTablesWhy (tags : Tags) (count : Nat) : Option String :=
  if tags.hiddenOutputs && tags.merkleLeaf != "hm96-sha512/v1" then
    some s!"{tags.name}, a statement with hidden outputs and {tags.merkleLeaf} leaves under --plain-tables"
  else if count ≤ Tags.MAX_TABLES then none
  else some s!"{count} tables in a session under --plain-tables, more than {Tags.MAX_TABLES}"

/-- The refusal of a session verified without the verifier's own program (`--program`), which `verify` reaches only when
every check has passed. -/
def NO_PROGRAM : String :=
  "refused: a session without the verifier's program (--program): a verifier that cannot read the Program cannot know what the statement computes, nor which of its values are registered"

/-- `program_sha512` of the verifier's own `--program` bytes: SHA-512 of the canonical descriptor without `annotations`
(`TemplateQuery.Desc`'s, the digest a partition object and META name); none for bytes that are not a JSON descriptor. -/
def programDigest (b : ByteArray) : Option String := do
  let j ← (JsonParse.parse (← String.fromUTF8? b)).toOption
  let bare := match j with
    | .obj kvs => Lean.Json.mkObj (kvs.toArray.filter (·.1 != "annotations")).toList
    | x => x
  return hexOf (Sha512.hash (← (canon bare).toOption).toUTF8)

/-- META's `program_sha512`, the Program the circuit binds; none when META names none. -/
def metaProgram (c : Circuit) : Option String := (c.metaJ.getObjValAs? String "program_sha512").toOption

/-- What `--plain-tables` refuses of the verifier's own program against the circuit: a session without it
(`NO_PROGRAM`), bytes that are not a JSON descriptor, a circuit whose META binds no program, and a program whose digest
is not META's `program_sha512`. -/
def programWhyP (c : Circuit) : Option ByteArray → Option String
  | none => some NO_PROGRAM
  | some b => match programDigest b, metaProgram c with
    | none, _ => some "refused: the verifier's program (--program) is not a JSON descriptor"
    | some _, none => some "refused: a circuit whose META binds no program_sha512 under --plain-tables: the verifier cannot hold its program to the circuit"
    | some h, some m => if h = m then none
      else some s!"refused: the verifier's program (--program, program_sha512 {h}) is not the circuit's (META program_sha512 {m})"

/-- **The verifier's own program is the circuit's** (`--plain-tables`): `programWhyP` on the circuit `verify` parsed. -/
def programP (parsed : V Circuit) (program : Option ByteArray) : Except String Unit :=
  match parsed with
  | .error e => .error (refusal s!"a circuit that does not parse ({e})")
  | .ok c => match programWhyP c program with
    | some why => .error why
    | none => .ok ()

/-- **`verify --plain-tables`'s statement is one the soundness proof covers** (PROTOCOL.md §16.15): a plain (no `--zk`)
session of public or hidden outputs at any number of tables up to `Tags.MAX_TABLES`, at hidden outputs with
`hm96-sha512/v1` leaves, every condition `check` puts on a one-table plain session, and the verifier's own program is
the one META binds (`programP`). The one-table plain form keeps `check` without the flag. -/
def checkP (tags : Tags) (circuitFile publicFile : ByteArray) (tables : List (String × ByteArray)) (count : Nat)
    (program : Option ByteArray) (publicInputs registered : Bool) : Except String Unit :=
  match plainTablesWhy tags count with
  | some why => .error (refusal why)
  | none => match check tags false circuitFile publicFile tables 1 publicInputs registered with
    | .error e => .error e
    | .ok () => programP (circuitOf tags circuitFile tables) program

/-- `checkP` on its circuit as `circuitOf` returns it (`checkP_eq`). -/
def checkPOf (parsed : V Circuit) (tags : Tags) (publicFile : ByteArray) (count : Nat) (program : Option ByteArray)
    (publicInputs registered : Bool) : Except String Unit :=
  match plainTablesWhy tags count with
  | some why => .error (refusal why)
  | none => match checkOf parsed tags false publicFile 1 publicInputs registered with
    | .error e => .error e
    | .ok () => programP parsed program

theorem checkP_eq (tags : Tags) (circuitFile publicFile : ByteArray) (tables : List (String × ByteArray))
    (count : Nat) (program : Option ByteArray) (publicInputs registered : Bool) :
    checkP tags circuitFile publicFile tables count program publicInputs registered
      = checkPOf (circuitOf tags circuitFile tables) tags publicFile count program publicInputs registered :=
  rfl

/-- **What an accepted `checkP` establishes**: the form (at most `Tags.MAX_TABLES` tables, hm96 leaves at hidden outputs),
`check` at one table, and the verifier's own program held to the circuit's. -/
theorem checkP_ok {tags : Tags} {circuitFile publicFile : ByteArray} {tables : List (String × ByteArray)} {count : Nat}
    {program : Option ByteArray} {publicInputs registered : Bool}
    (h : checkP tags circuitFile publicFile tables count program publicInputs registered = .ok ()) :
    plainTablesWhy tags count = none ∧ check tags false circuitFile publicFile tables 1 publicInputs registered = .ok () ∧
      programP (circuitOf tags circuitFile tables) program = .ok () := by
  unfold checkP at h
  cases hw : plainTablesWhy tags count with
  | some w => simp [hw] at h
  | none =>
    simp only [hw] at h
    cases hc : check tags false circuitFile publicFile tables 1 publicInputs registered with
    | error e => simp [hc] at h
    | ok u =>
      cases u
      simp only [hc] at h
      exact ⟨rfl, rfl, h⟩

/-- **`programWhyP` passes only the circuit's program**: the verifier holds bytes `b`, META binds a program, and `b`'s
digest is META's `program_sha512`. -/
theorem programWhyP_none {c : Circuit} {program : Option ByteArray} (h : programWhyP c program = none) :
    ∃ b, program = some b ∧ (metaProgram c).isSome = true ∧ programDigest b = metaProgram c := by
  cases program with
  | none => simp [programWhyP] at h
  | some b =>
    refine ⟨b, rfl, ?_⟩
    simp only [programWhyP] at h
    cases hd : programDigest b with
    | none => simp [hd] at h
    | some d =>
      cases hm : metaProgram c with
      | none => simp [hd, hm] at h
      | some m =>
        simp only [hd, hm] at h
        by_cases he : d = m
        · subst he; exact ⟨rfl, rfl⟩
        · simp [he] at h

/-- **An accepted `programP` holds the verifier's program to META's**: the circuit parsed, and `programWhyP` passes. -/
theorem programP_ok {parsed : V Circuit} {program : Option ByteArray} (h : programP parsed program = .ok ()) :
    ∃ c, parsed = .ok c ∧ programWhyP c program = none := by
  cases parsed with
  | error e => simp [programP] at h
  | ok c =>
    refine ⟨c, rfl, ?_⟩
    simp only [programP] at h
    cases hw : programWhyP c program with
    | none => rfl
    | some w => simp [hw] at h

/-- What a draw's law is, when no soundness claim draws by it at the session's form (PROTOCOL.md §16.15): the one rule
for which law each form draws by. Without `--plain-tables`: the stratified law at every form `check` passes, each claim at
one table (`SoundExec`, `SoundExecH`, `SoundZ`, `SoundZH`); the subset law under `--zk` (`SoundZJ`, `SoundZHJ`, `SoundZHJR`)
and, without it, exactly where a plain claim draws by it: at hidden outputs, at the one table `check` admits there
(`SoundExecH` at its subset draw design). Under `--plain-tables` (`plainTables`, `SoundExecJ`), the subset law alone. So a
bernoulli draw, a work draw (with or without a closure), a subset draw without `--zk` or `--plain-tables` at public outputs,
and a stratified draw under `--plain-tables` are outside them. -/
def lawWhy (zk : Bool) (tags : Tags) (plainTables : Bool) (d : Draw.UnitDraw) : Option String :=
  match d.law with
  | .stratified .. => if plainTables then some "a stratified draw under --plain-tables" else none
  | .subset _ => if zk || tags.hiddenOutputs || plainTables then none else some "a subset draw without --zk"
  | .bernoulli .. => some "a bernoulli draw"
  | .work .. => some (if d.closure.isEmpty then "a work draw" else "a work draw with a closure")

/-- **The session's draw is by a law a soundness claim draws by at its form** (`lawWhy`, PROTOCOL.md §16.15): `draw` is
the record's (`recordDraw`), none for a session without one. Every claim is about a drawn session, so a session without
a draw is refused (D12); so is a draw that does not parse. -/
def drawLaw (zk : Bool) (tags : Tags) (plainTables : Bool) : Option Lean.Json → Except String Unit
  | none => .error "refused: a session without a draw: every soundness claim is about a drawn session (D12)"
  | some j => match Draw.UnitDraw.ofJson j with
    | .error e => .error s!"refused: a draw that does not parse ({e})"
    | .ok d => match lawWhy zk tags plainTables d with
      | some why => .error s!"refused: {why}: no soundness claim draws by this law"
      | none => .ok ()

/-- **The verifier holds its own program**: a session verified without it (`--program`) is refused (`NO_PROGRAM`),
whatever else it passes. -/
def programHeld (program : Option ByteArray) : Except String Unit :=
  if program.isSome then .ok () else .error NO_PROGRAM

theorem programHeld_ok {program : Option ByteArray} (h : programHeld program = .ok ()) : program.isSome = true := by
  unfold programHeld at h
  split at h
  · assumption
  · cases h

end Flock.ProvedScope
