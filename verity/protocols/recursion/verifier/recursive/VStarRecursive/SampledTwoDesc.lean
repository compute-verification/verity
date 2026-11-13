/-
The two-stage form's descriptor check (`descOk`): what the auditor's copies of a two-stage registration must satisfy
before a run is read against them, decided on the form's params, its blobs and the run of record's input
(`Spot.Run.Input`, its interior `Spot.Run.Interior`), once per template and never per unit, request or bit:

* the first stage: the boundary registration `Run.run` hands `Select.select` (`firstOf`), its replay layout
  (`Slots.layout`) and the design `Select.select` samples on it (`firstLaw`), well formed on the layout's strata
  (`designOk`);
* the second stage's cells and counts, as `Run.run` computes them (`cellsOf`), a count of at least `1` and a cell for
  each template;
* each template's reference statement: its member's (`memberOf`) circuit and public files on the blobs, the pinned
  tables, and its partition object and registered reads as `flock-verify verify --zk` reads their text (`refOf`),
  set up as `verifyCmd` sets a statement up (`setupAt`), in Flock's masked scope (`zkOk`), of a length each field's
  encoding holds (`fits`), reading as a member's (`templateOk`), fitting the two-stage scope (`memberOk`) and in the
  scope of Flock's hidden-output proof (`ProvedScope.scopeOkH`);
* the registered wiring, on the templates' names, the outer block's links and wires and the reference reads: each name
  one template's (`namesOnce`), each wire's writer before its reader (`writersFirst`), each row a template's units
  read source bits at a port a link or a wire names (`sourceRowsNamed`), each wired source bit with a gate to go to
  (`wiresMoveRows`, on the design's rows and the cells), each wire's reader port like its writer's output port
  (`wiresAlike`), no wire at a row a template writes that no link names (`writtenRowsUnwired`), each such row some
  wire's writer port (`writtenRowsRead`), each reference read no link or wire names a port of the interior
  registration (`refReadsOwn`), an interior registration that registers no value, each template's circuit naming its
  ports once (`portsOnce`) and every returned output bit on a computed row (`outsOk`), each read off one unit's net.

Nothing calls this module yet. Its definitions have the shapes of the two-stage verifier's own (`SpotFlock.Code`), over
the components the form holds in place of the auditor's held copies, and a proof (`SpotFlock.HeldOfDesc`) reads each
of those as this one.
-/
module

public import VStarRecursive.SampledTwo
public import Spot.Bits

@[expose] public section

namespace VStar.Recursive.SampledTwoDesc

open Lean (Json)
open Flock

/-! ## The first stage, the cells and the requests -/

/-- **The boundary registration the first stage samples on**, as `Run.run` hands `Select.select`: the verifier's copies
with the sampling design and work table the rule expands over the record's population, the resolved rule and the closure map. -/
structure First where
  verifier : Spot.Registration.Verifier
  rule : Spot.Select.Rule
  closure : Option (Array (Int × Array Int))

/-- `Run.run`'s boundary registration at the run of record's input; none where a step of it is refused. -/
def firstOf (i : Spot.Run.Input) : Option First := do
  let v := i.params.verifier
  let n ← Spot.nat? (Spot.getD i.record "population")
  let rule ← i.rule.over n (Spot.Select.strataSizes v.program v.query)
  let law ← (rule.lawOver n).toOption
  let workTable ← (i.rule.tableOver v.program v.query v.workTable).toOption
  let resolved ← (Spot.Select.resolve i.rule v.program v.query).toOption
  pure ⟨{ v with law := law, workTable := workTable }, resolved, i.closure⟩

/-- **The second stage's cells and counts**, as `Run.run` computes them: each template's proof blocks in a replay block
and the two-stage rule's count in each; none under a one-stage rule or where either is refused. -/
def cellsOf (i : Spot.Run.Input) (it : Spot.Run.Interior) : Option (Array Nat × Array Nat) := do
  let ns ← (Spot.Select.replayCells it.replay it.params.verifier.query).toOption
  match i.rule with
  | .twoStage _ _ k => do
    let ids ← (Spot.Select.templateIds it.params.verifier.query).toOption
    let ks ← (k.over ids ns).toOption
    pure (ns, ks)
  | _ => none

/-- **The requests of a boundary layout**: its proof blocks that are a unit's. -/
def requests (l : Spot.Slots.SlotOrder) : Nat := (l.blocks.filter fun b => b.unit.isSome).size

/-- **The first stage's law**, as `Select.select` samples on the boundary registration: none for the `All` rule, else
the sampling design it derives over the program's population, or the one the rule names where it derives none. -/
def firstLaw (rule : Spot.Select.Rule) (d : Spot.Program) (query : Json) (table : Option Json) :
    V (Option Draw.Law) := do
  let n ← Spot.Partition.units d query
  let named ← rule.lawOver n
  let derived ← Spot.Draw.deriveLaw d query named table
  match rule with
  | .all => pure none
  | _ => do
    let parsed ← Spot.Draw.flockLaw ((derived.map (·.toJson)).getD named) n
    pure (some (derived.getD parsed))

/-- **A sampling design is well formed over strata of sizes `ns`**; `none`, the `All` rule's, always is. -/
def lawOk (ns : List Nat) : Option Draw.Law → Bool
  | none => true
  | some (.subset K) => 1 ≤ K && K ≤ ns.sum
  | some (.bernoulli num den) => 0 < den && num ≤ den
  | some (.stratified K _) => 1 ≤ K && K ≤ ns.sum && ns.all (1 ≤ ·)
  | some (.work _ ws) => (ws.toList.map (·.work)).length == ns.length &&
      (ws.toList.map (·.floor)).length == ns.length && (ws.toList.map (·.floor)).all (1 ≤ ·) &&
      0 < ((ns.zip (ws.toList.map (·.work))).map fun p => p.1 * p.2).sum && ns.all (1 ≤ ·)

/-- **The first stage's design on the replay layout's strata**: its two rules, the first stage's law and the input
class's `All`, one a class of `l`, each well formed over its class's strata sizes in slots. -/
def designOk (law : Option Draw.Law) (l : Spot.Slots.SlotOrder) : Bool :=
  [law, none].length == l.classes.size &&
    ([law, none].zip (l.classes.toList.map fun c => c.toList.map (·.size))).all fun p => lawOk p.2 p.1

/-! ## The templates' reference statements -/

/-- **Template `t`'s member**: the first of the interior audit's members whose circuit is the one the params name for
template `t`. -/
def memberOf (p : SampledTwo.Params) (t : Nat) : Option Spot.OneStageZk.Member :=
  (p.circuits[t]?).bind fun c => p.members.find? (·.own.circuit == c.2.1)

/-- A file read as JSON, as `flock-verify verify` reads its inputs: UTF-8, then `JsonParse.parse`. -/
def jsonOf (what : String) (b : ByteArray) : Except String Json :=
  match String.fromUTF8? b with
  | none => .error s!"{what}: not UTF-8"
  | some t => (Flock.JsonParse.parse t).mapError fun e => s!"{what}: not JSON: {e}"

/-- `verifyCmd`'s refusal of a public header that names an identifier where it holds none of its own. -/
def noIdentifier (st : Flock.Stmt) : V Unit :=
  match st.pub.header.getObjVal? "identifier" with
  | .error _ => pure ()
  | .ok _ => throw "identifier: the public header names an identifier; pass the verifier's own (--identifier F)"

/-- **`verify --zk`'s setup** of a statement of one table at no sample, with no recursive input and no public input
words: `--zk`'s tags `tz` and coins, the statement's tags `tags`, the files and the verifier's own partition object,
program and registered reads. -/
def setupAt (tz tags : Tags) (circuit pub : ByteArray) (coins : Coins) (tables : List (String × ByteArray))
    (partition : Option Json) (program : Option ByteArray) (registered : Option (Array Registered.Port)) :
    V (SessionSpec × Array Flock.Stmt) := do
  if !tz.hm96Rows then
    throw s!"{tz.name}: refused: a statement without hm96-sha512 rows has no draw, and every claim is about a drawn session"
  RecSetup.checkInner none none (pure (none, none))
  let (spec, sts) ← Flock.Stmt.setupTables tz circuit pub coins 1 tables none partition program none none none none
  for st in sts do
    checkRegistered st registered
    noIdentifier st
  Registered.checkFull registered none
  RecSetup.checks none none tags tables sts circuit partition program registered none none none
  sts.forM Zk.shapeOk
  sts.forM (RecCoins.check · registered (RecCoins.Source.atC3 none none none) (RecSetup.c3 none none))
  sts.forM (RecVStar.checkPubRows · registered none none)
  pure (spec, sts)

/-- **A template's reference statement, read and set up**: its statement's tags, `--zk`'s tags, its circuit and public
files, the pinned tables, the text of its partition object and registered reads, as read back, its program, and the
session and tables its setup gives. -/
structure Ref where
  tags : Tags
  tz : Tags
  circuit : ByteArray
  pub : ByteArray
  tables : List (String × ByteArray)
  partitionB : ByteArray
  partition : Json
  program : ByteArray
  registeredB : Option ByteArray
  registered : Option (Array Registered.Port)
  spec : SessionSpec
  sts : Array Flock.Stmt

/-- **Template `t`'s reference statement**: its member's statement under `--zk` on the blobs, at no sample, with its
member's public file and registered reads, read and set up as `flock-verify verify --zk` reads and sets it up. -/
def refOf (p : SampledTwo.Params) (bs : Core.Blobs) (t : Nat) : V Ref := do
  let some m := memberOf p t | throw s!"template {t}: no member's circuit is the one the params name"
  let circuit ← bs.get m.own.circuit "circuit"
  let pub ← bs.get m.own.pub "public file"
  let some tags := Tags.byName? Spot.OneStageZk.STATEMENT | throw "unknown statement"
  if tags.plainLeaves then throw s!"verify --zk: refused: {tags.name}, a statement with plain SHA-512 leaves"
  let partitionB := (RecordText.printT m.own.partition).toUTF8
  let partition ← jsonOf "partition" partitionB
  let registeredB := m.own.registered.map fun j => (RecordText.printT j).toUTF8
  let registered ← registeredB.mapM fun b => do Registered.ofJson (← jsonOf "registered reads" b)
  let tables := Spot.OneStageZk.tablesOf bs
  let tz := Zk.tags tags (Zk.maskLogOf circuit) 1
  let (spec, sts) ← setupAt tz tags circuit pub (.tree (Zk.coinSpec (tags.tableNames 1))) tables (some partition)
    (some m.own.program) registered
  pure { tags, tz, circuit, pub, tables, partitionB, partition, program := m.own.program, registeredB, registered,
         spec, sts }

/-- **Every field of a reference statement fits its encoding's length word**: each file, table and text under
`2^64` bytes. -/
def fits (r : Ref) : Bool :=
  decide (Spot.OneStageZk.STATEMENT.toUTF8.size < 2 ^ 64) && decide (r.circuit.size < 2 ^ 64) &&
    decide (r.pub.size < 2 ^ 64) && decide (r.tables.length < 2 ^ 64) &&
    r.tables.all (fun q => decide (q.1.toUTF8.size < 2 ^ 64) && decide (q.2.size < 2 ^ 64)) &&
    decide (r.partitionB.size < 2 ^ 64) && decide (r.program.size < 2 ^ 64) &&
    r.registeredB.all fun b => decide (b.size < 2 ^ 64)

/-- **A session's statement is in Flock's masked scope**: it sets up with `1` to `200` tables at at most two link
points, each at an `m` from `25` to `35` and of a shape in `Stmt.InRange`. -/
def zkOk (spec : SessionSpec) (sts : Array Flock.Stmt) : Bool :=
  0 < sts.size && sts.size ≤ 200 && spec.points ≤ 2 &&
    sts.toList.all fun st =>
      25 ≤ st.m && st.m ≤ 35 && st.c.kLog ≤ 27 && st.regions.size ≤ 1024 && spec.mPts ≤ 64

/-- **The statement circuit** of a reference statement: the circuit the scope check parses, a templated statement's
unrolled; `default` where it doesn't parse. -/
def Ref.stmt (r : Ref) : Circuit :=
  HmOut.unrolled (match ProvedScope.circuitOf r.tz r.circuit r.tables with
    | .ok c => c
    | .error _ => default)

/-- **A reference statement reads as a member's**: hidden outputs and untyped tags, a circuit file whose descriptor
reads, a statement circuit whose unit has at most `2^32` rows, one table, and each port a `sha512/row/v2` row. -/
def templateOk (r : Ref) : Bool :=
  r.tz.hiddenOutputs && !r.tz.typed &&
    (match Desc.parseStmt r.tz r.circuit r.tables with
      | .ok _ => true
      | .error _ => false) &&
    decide (r.stmt.unit.unitLog ≤ 32) && (r.sts.toList.map (·.m)).length == 1 && r.stmt.ports.all ProvedScope.v2Row

/-- **The templates' reference statements**, in the params' template order. -/
def refsOf (p : SampledTwo.Params) (bs : Core.Blobs) : Option (List Ref) :=
  (List.range p.circuits.size).mapM fun t => (refOf p bs t).toOption

/-- **The templates' circuits**: each reference statement's statement circuit, in the params' template order. A
statement circuit is a parse of its circuit file, so `descOk` reads this list once and every clause reads a template's
circuit off it (`circAt`): a parse a template, whatever the count of reads. -/
def circsOf (rs : List Ref) : List Circuit := rs.map Ref.stmt

/-- **Template `t`'s circuit** among the templates' circuits `cs`: the first template's past them. -/
def circAt (cs : List Circuit) (t : Nat) : Circuit := cs.getD t (cs.headD default)

/-! ## A member's size -/

/-- A unit's source gates: `upv` copies of the unit's input bits, the bits before its constant row. -/
def unitIns (c : Circuit) : Nat := c.upv * min (c.unit.inWords * WORD) c.unit.constPos

/-- The returned output bits of one unit: sixteen for each returned output leaf of each of its `upv` instances. -/
def outBits (c : Circuit) : Nat := ((List.range c.upv).map fun k => 16 * (c.leavesOut.getD k #[]).size).sum

/-- **A member fits the two-stage scope**: `unitLog ≤ 32`, an instance a unit at least, and a unit's inputs and
returned output bits, the most one unit's proof block commits, at most `UNIT_CLASS_MAX`, whatever its count of units. -/
def memberOk (c : Circuit) : Bool :=
  decide (c.unit.unitLog ≤ 32) && decide (0 < c.upv) &&
    decide (unitIns c + outBits c ≤ Spot.OneStageZk.UNIT_CLASS_MAX)

/-! ## The registered wiring -/

/-- The link that names template `t`'s session port `port`, as the second stage finds it. -/
def linkOf (links : Option (Array Spot.SecondStage.Link)) (t port : String) : Option Spot.SecondStage.Link :=
  (links.getD #[]).find? fun l => l.template == t && l.port == port

/-- The wire that names template `t`'s session port `port`, as the second stage finds it. -/
def wireOf (wires : Option (Array Spot.SecondStage.Wire)) (t port : String) : Option Spot.SecondStage.Wire :=
  (wires.getD #[]).find? fun w => w.template == t && w.port == port

/-- The first of `T` templates named `name`. -/
def writerOf (T : Nat) (names : List String) (name : String) : Option Nat :=
  (List.range T).find? fun t => names.getD t "" == name

/-- A template name's place: the first template of the name, `T` where none has it. -/
def posOf (T : Nat) (names : List String) (name : String) : Nat := (writerOf T names name).getD T

/-- The first port of circuit `c` named `name`; port `0` where none is. -/
def portNamed (c : Circuit) (name : String) : Nat :=
  ((List.range c.ports.size).find? fun q => c.ports[q]!.name == name).getD 0

/-- **Each name is one template's.** -/
def namesOnce (T : Nat) (names : List String) : Bool := decide ((List.range T).map fun t => names.getD t "").Nodup

/-- **Each wire's writer's part is before its reader's.** -/
def writersFirst (T : Nat) (names : List String) (wires : Option (Array Spot.SecondStage.Wire)) : Bool :=
  (wires.getD #[]).all fun w => decide (posOf T names w.writer < posOf T names w.template)

/-- Wire `w`'s port at reader `t` is like its writer's output port: as many words, and the same row schema; true where
the wire's writer names no template. -/
def wireAlike (T : Nat) (names : List String) (circ : Nat → Circuit) (w : Spot.SecondStage.Wire) (t : Nat) : Bool :=
  match writerOf T names w.writer with
  | some tw =>
    ((circ t).ports[portNamed (circ t) w.port]!).words == ((circ tw).ports[portNamed (circ tw) w.wport]!).words &&
      PortRoots.schemaOf (circ t) (portNamed (circ t) w.port) ==
        PortRoots.schemaOf (circ tw) (portNamed (circ tw) w.wport)
  | none => true

/-- **Each wire's reader port is like its writer's output port**, at every template of the wire's reader's name. -/
def wiresAlike (T : Nat) (names : List String) (wires : Option (Array Spot.SecondStage.Wire)) (circ : Nat → Circuit) :
    Bool :=
  (wires.getD #[]).all fun w =>
    (List.range T).all fun t => !(w.template == names.getD t "") || wireAlike T names circ w t

/-- **Each reference read at a port no link or wire names reads a port of the interior registration** (`ports`). -/
def refReadsOwn (T : Nat) (names : List String) (links : Option (Array Spot.SecondStage.Link))
    (wires : Option (Array Spot.SecondStage.Wire)) (refReads : Nat → Json) (ports : List String) : Bool :=
  (List.range T).all fun t =>
    (Spot.entries (Spot.getD (refReads t) "ports")).all fun pe =>
      (linkOf links (names.getD t "") pe.1).isSome || (wireOf wires (names.getD t "") pe.1).isSome ||
        match Spot.getD pe.2 "value" with
        | .str v => ports.contains v
        | _ => false

/-- **The rows a unit of circuit `c` reads source bits at**: each input bit's row, where it has one. -/
def sourceRows (c : Circuit) : List String :=
  let ix := Spot.Bits.Index.of c
  (List.range ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn).filterMap fun b =>
    (Spot.Bits.srcPos c 0 (Spot.Bits.srcKey c ix b)).map fun x => x.row

/-- **The rows a unit of circuit `c` writes**: its output row ports' names. -/
def writtenRows (c : Circuit) : List String :=
  (List.range c.outRowPorts.size).map fun o => (c.ports.getD (Flock.HmOut.nIn c + o) default).name

/-- **Every returned output bit of circuit `c`'s unit sits on a computed row** of the unit net: past its input columns
and before its constant. -/
def outsComputed (c : Circuit) : Bool :=
  let R := Spot.Bits.Rows.ofNet c.unit
  (List.range c.upv).all fun k => (List.range (c.leavesOut.getD k #[]).size).all fun j =>
    (List.range 16).all fun t =>
      decide (R.nIn ≤ Spot.Bits.outCol c j t) && decide (Spot.Bits.outCol c j t < R.nIn + R.nComp)

/-- **Every row a template's units read source bits at is a port a link or a wire names.** -/
def sourceRowsNamed (T : Nat) (names : List String) (links : Option (Array Spot.SecondStage.Link))
    (wires : Option (Array Spot.SecondStage.Wire)) (circ : Nat → Circuit) : Bool :=
  (List.range T).all fun t =>
    (sourceRows (circ t)).all fun row =>
      (linkOf links (names.getD t "") row).isSome || (wireOf wires (names.getD t "") row).isSome

/-- **No wire names a row a template's units write that no link names.** -/
def writtenRowsUnwired (T : Nat) (names : List String) (links : Option (Array Spot.SecondStage.Link))
    (wires : Option (Array Spot.SecondStage.Wire)) (circ : Nat → Circuit) : Bool :=
  (List.range T).all fun t =>
    (writtenRows (circ t)).all fun row =>
      (linkOf links (names.getD t "") row).isSome || (wireOf wires (names.getD t "") row).isNone

/-- **Every row a template's units write that no link names is some wire's writer port.** -/
def writtenRowsRead (T : Nat) (names : List String) (links : Option (Array Spot.SecondStage.Link))
    (wires : Option (Array Spot.SecondStage.Wire)) (circ : Nat → Circuit) : Bool :=
  (List.range T).all fun t =>
    (writtenRows (circ t)).all fun row =>
      (linkOf links (names.getD t "") row).isSome ||
        (wires.getD #[]).any fun w => w.writer == names.getD t "" && w.wport == row

/-- **Each template's circuit names its ports once.** -/
def portsOnce (T : Nat) (circ : Nat → Circuit) : Bool :=
  (List.range T).all fun t => decide ((circ t).ports.toList.map (·.name)).Nodup

/-- **Every returned output bit of each template's unit sits on a computed row.** -/
def outsOk (T : Nat) (circ : Nat → Circuit) : Bool :=
  (List.range T).all fun t => outsComputed (circ t)

/-! ## Every wired source bit has a gate to go to

Decided once per template, on the rows a unit of the template reads source bits at, the design's rows (`rows`), the
cells (`ns`) and each template's count of units (`units`). -/

/-- **Template `t`'s units**, of `T` templates: its cell's `ns[t]` a request, over `r` requests; none past them. -/
def unitsAt (r : Nat) (ns : Array Nat) (T t : Nat) : Nat := if t < T then r * ns.getD t 0 else 0

/-- The rows template `t`'s units read at its session port `port`, as the second stage finds them. -/
def rowsOf (rows : Array Spot.SecondStage.Rows) (t port : String) : Option Spot.SecondStage.Rows :=
  rows.find? fun r => r.template == t && r.port == port

/-- **The row and the bit of it that each input column of one unit of circuit `c` reads**: `sourceRows`, with each
one's bit. -/
def sourceCells (c : Circuit) : List (String × Nat) :=
  let ix := Spot.Bits.Index.of c
  (List.range ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn).filterMap fun b =>
    (Spot.Bits.srcPos c 0 (Spot.Bits.srcKey c ix b)).map fun x => (x.row, x.bit)

/-- Bit `bit` of the row named `wport` is one a unit of circuit `c` writes: `wport` names one of its output row
ports and the bit is inside that output port's words, sixteen bits a word. -/
def outBit (c : Circuit) (wport : String) (bit : Nat) : Bool :=
  (List.range c.outPorts.size).any fun o =>
    (c.ports.getD (Flock.HmOut.nIn c + o) default).name == wport &&
      decide (bit < 16 * (c.outPorts.getD o default).2)

/-- Wire `w`'s writer writes bit `bit` of its output row: its writer names a template whose unit writes that bit of
the wire's writer port. -/
def wireBit (T : Nat) (names : List String) (circ : Nat → Circuit) (w : Spot.SecondStage.Wire) (bit : Nat) : Bool :=
  match writerOf T names w.writer with
  | some tw => outBit (circ tw) w.wport bit
  | none => false

/-- Bit `bit` of template `t`'s row `row` has a writer's bit to go to: where a wire names the row and no link does,
the wire's writer writes that bit; true at every other row. -/
def bitGoes (T : Nat) (names : List String) (links : Option (Array Spot.SecondStage.Link))
    (wires : Option (Array Spot.SecondStage.Wire)) (circ : Nat → Circuit) (t : Nat) (row : String) (bit : Nat) :
    Bool :=
  match linkOf links (names.getD t "") row, wireOf wires (names.getD t "") row with
  | none, some w => wireBit T names circ w bit
  | _, _ => true

/-- Every unit of template `t` has a unit of template `tw` to read wire `w` at, at rows `r`: no link names the
writer's port; the reader's cell is not empty and its units are whole cells; the writer has as many cells of units;
the rows a replay block holds are no more than the writer's cell; the rows' runs hold the reader's cell, each position
once; and each run's last row is one of the rows a replay block holds. -/
def cellGoes (names : List String) (links : Option (Array Spot.SecondStage.Link)) (ns : Array Nat)
    (units : Nat → Nat) (t tw : Nat) (w : Spot.SecondStage.Wire) (r : Spot.SecondStage.Rows) : Bool :=
  (linkOf links (names.getD tw "") w.wport).isNone && decide (0 < ns.getD t 0) &&
    decide (units t % ns.getD t 0 = 0) &&
    decide (units t / ns.getD t 0 * ns.getD tw 0 ≤ units tw) &&
    decide (r.per ≤ ns.getD tw 0) && Spot.SecondStage.fits r.runs (ns.getD t 0) &&
    r.runs.all fun x => decide (x.row + (x.count - 1) * x.step < r.per)

/-- Every unit of template `t` has a writer's unit to read wire `w` at, at its row `row`: the row has rows, the
wire's writer names a template, and each unit of `t` has a unit of it (`cellGoes`). -/
def wireGoes (T : Nat) (names : List String) (links : Option (Array Spot.SecondStage.Link))
    (rows : Array Spot.SecondStage.Rows) (ns : Array Nat) (units : Nat → Nat) (t : Nat) (row : String)
    (w : Spot.SecondStage.Wire) : Bool :=
  match rowsOf rows (names.getD t "") row, writerOf T names w.writer with
  | some r, some tw => cellGoes names links ns units t tw w r
  | _, _ => false

/-- Every unit of template `t` has a writer's unit to read its row `row` at: where a wire names the row and no link
does, `wireGoes`; true at every other row. -/
def rowGoes (T : Nat) (names : List String) (links : Option (Array Spot.SecondStage.Link))
    (wires : Option (Array Spot.SecondStage.Wire)) (rows : Array Spot.SecondStage.Rows) (ns : Array Nat)
    (units : Nat → Nat) (t : Nat) (row : String) : Bool :=
  match linkOf links (names.getD t "") row, wireOf wires (names.getD t "") row with
  | none, some w => wireGoes T names links rows ns units t row w
  | _, _ => true

/-- **Every wired source bit has a gate to go to**: for each template, each row its units read source bits at has a
writer's unit for each of its units (`rowGoes`), and each bit one unit reads there is one the writer's unit writes
(`bitGoes`). -/
def wiresMoveRows (T : Nat) (names : List String) (links : Option (Array Spot.SecondStage.Link))
    (wires : Option (Array Spot.SecondStage.Wire)) (rows : Array Spot.SecondStage.Rows) (circ : Nat → Circuit)
    (ns : Array Nat) (units : Nat → Nat) : Bool :=
  (List.range T).all fun t =>
    ((sourceRows (circ t)).all fun row => rowGoes T names links wires rows ns units t row) &&
      (sourceCells (circ t)).all fun rb => bitGoes T names links wires circ t rb.1 rb.2

/-! ## The check -/

/-- **The registered wiring's checks decided per template**, at `T` templates named `names`, the links and wires, the
design's rows, the templates' circuits, the cells, each template's count of units, the reference reads and the
interior registration's verifier `vi`. -/
def wiringOk (T : Nat) (names : List String) (links : Option (Array Spot.SecondStage.Link))
    (wires : Option (Array Spot.SecondStage.Wire)) (rows : Array Spot.SecondStage.Rows) (circ : Nat → Circuit)
    (ns : Array Nat) (units : Nat → Nat) (refReads : Nat → Json) (vi : Spot.Registration.Verifier) : Bool :=
  namesOnce T names && writersFirst T names wires && sourceRowsNamed T names links wires circ &&
    wiresMoveRows T names links wires rows circ ns units && wiresAlike T names wires circ &&
    writtenRowsUnwired T names links wires circ && writtenRowsRead T names links wires circ &&
    refReadsOwn T names links wires refReads (vi.ports.map (·.1)).toList &&
    (vi.registered.map (·.1)).toList.isEmpty && portsOnce T circ && outsOk T circ

/-- **The two-stage descriptor check** (module docstring), at the form's params `p`, its blobs `bs`, and the run of
record's input `i` and interior `it`. Its bound on a unit (`memberOk`) is stricter than `Spot.Partition`'s, which counts
a unit's returned output bits alone: this one counts its input bits too, so it refuses a unit whose returned bits fit a
size class and whose inputs with them pass `UNIT_CLASS_MAX`. A unit of 2,048 input bits returning 48 counts 2,096 here,
and one of 4,096 returning 32 counts 4,128. What it refuses beyond `Spot.Partition` costs completeness alone. -/
def descOk (p : SampledTwo.Params) (bs : Core.Blobs) (i : Spot.Run.Input) (it : Spot.Run.Interior) : Bool :=
  match firstOf i, cellsOf i it, refsOf p bs with
  | some f, some (ns, ks), some rs =>
    match (Spot.Slots.layout f.verifier.program f.verifier.query).toOption,
        (firstLaw f.rule f.verifier.program f.verifier.query f.verifier.workTable).toOption with
    | some l, some law =>
      let T := it.circuits.size
      let names := (it.circuits.map (·.1)).toList
      let links := Spot.SecondStage.replayLinks i.params.verifier.program.prog it.replay.prog.programFn i.record
      let wires := Spot.SecondStage.replayWires i.params.verifier.program.prog it.replay.prog.programFn
      let rows := Spot.SecondStage.replayRows i.params.verifier.program.prog it.replay.prog.programFn
      let refReads := fun t => (it.circuits.map (·.2.2.2)).toList.getD t .null
      let cs := circsOf rs
      p.circuits.size == T && !rs.isEmpty &&
        rs.all (fun r => fits r && zkOk r.spec r.sts && templateOk r && ProvedScope.scopeOkH r.stmt) &&
        (List.range T).all (fun t => memberOk (circAt cs t)) && designOk law l &&
        ks.size == T && ks.toList.all (1 ≤ ·) && ns.size == T &&
        wiringOk T names links wires rows (circAt cs) ns (unitsAt (requests l) ns T) refReads it.params.verifier
    | _, _ => false
  | _, _, _ => false

end VStar.Recursive.SampledTwoDesc
