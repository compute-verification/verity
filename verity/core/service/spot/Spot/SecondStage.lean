/-
**Round 2's statement check**: a two-stage run's second-stage sessions held, each by its statement, to the interior
registration and to the second draw's cells (`check`). `Run.run` adds it to the interior registration's audit record as
`checks.statements` (`attach`), and `service.Stream` runs its mirror (`second_stage`), held to it by difftest
(`second_statements`).

* **The parts and their cells** are the verifier's own: one part per stratum of the interior program under the check's
  query (`Partition.strata`, which `population-order` holds to the population's units, template by template), each one
  range of `Q·n_t` slots from `off_t`, `Q` the replay sample's size (`parts`). Template `t`'s cell in the sample's
  `i`-th replay block `S[i]` is its `n_t` slots from `off_t + i·n_t` (`Part.span`). One template's cells are
  `Select.secondLaw`'s strata.
* **A statement** (`Stmt`) is one second-stage session's, as its files state it: the template its circuit is an
  instance of (its stratum's name), its public file's header, its session record's sample (`unit_draw`), and the
  registered reads it was verified against (`--registered`), null for none.
* **Its checks**, each a code (`stmtCodes`, `stmtFailures`):
  - `second-statement`: it is not one;
  - `second-template`: its template is none of the parts', or an earlier statement's;
  - `second-verified`: no accepted verdict of record names its header, its registered reads and its sample: the
    SHA-512 of the canonical JSON of the public file's header and of the `--registered` file it checked the session
    against (`public_header_sha512`, `registered_sha512`, none for none) and the sample it checked it at (`unit_draw`;
    `Checked`, `verifiedOf`). So the checks below read only a header, reads and a sample that were verified;
  - `second-tables`: an accepted verdict of record naming its header, registered reads and sample set its session up as
    other than one Flock table (`session_tables`, `tablesOk`): round 2 reads one session of one table for each part, so
    a session split across tables (Flock's `tableDraw`) is refused, as one split across statements is
    (`second-template`);
  - `second-header`: its header is not the reference header the auditor's params give its template (`Own.refs`, the
    registered reference statement's, fixed before the run) placed in the run (`atRun`), both with the run's fields
    blanked (`RUN_FIELDS`: `instances`, `units.indices`, `frame_v3.roots` and `frame_v3.domain_ids`, which the next two
    checks hold to the run; `units.blocks`, the indices cut into blocks, which the session's verifier holds to them;
    and `registered`, whose reads `second-read` holds), or the params give its template none.
    Placed in the run, the reference's `set` is the session's input set, `served/<run>/m<k>` (`setName`: `<run>` the
    interior record's `window.run`, `k` its template's position among the parts, `posOf`), and its
    `frame_v3.bindings` are each of its `frame_v3.schemas` ports' bindings in the run (`portBinding`: the window-domain
    rule over the interior record's program, partition and that port's leaves, a port a wire names its writer's output
    port's, `heldPort`): the verifier computes both, so the prover chooses neither;
  - `second-population`: its header's `units.indices` are not its part's slots in order, or its `instances` not their
    count;
  - `second-roots`: its scheme is not the interior record's, or its `frame_v3` roots and `domain_ids`, its registered
    reads aside, are not the record's for its template's ports (port `p` of template `t` is the record's `t/p` when the
    record names its ports by template, else `p`), each of which it roots or reads;
  - `second-sample`: its sample is not the verifier's own, the second draw's units in its part as a subset of its slots
    (`ownDraw`);
  - `second-read`: a port's registered read (`readOk`) is not, for a port a link names, the boundary port it names,
    each slot reading only its unit's row of its cell's replay block's (`S[i]·r + ρ(j)`, `r` the port's rows a block
    and `ρ` its units' rows, the port's `Rows`, `j` the slot's unit in its cell, the port's leaves the boundary
    population's `r` each); for a port a wire names, its writer's output port of the interior record, each slot of
    replay block `S[i]` reading only its writer unit's row there (`i·n_w + ρ(j)`, the port's leaves the writer's
    `Q·n_w` slots'); else its own port of the interior record, each slot `u` reading
    only its own rows (`[(u - off_t)·r, (u - off_t + 1)·r)`, `r` the port's leaves over the part's `Q·n_t` slots), or
    a value either registration registers; each at its root, domain and leaves. Or a port a link or a wire names is
    not read.
  - `second-refs`: its header shares rows (`shared_rows`: tables whose rows its instances read by the refs in its public
    file's body) at a port it doesn't read registered (`refsOk`). At a port it reads registered, the session's verifier
    holds each instance's ref to the row at the position the read gives its unit (`Flock.Registered.check`), and
    `second-read` holds the read to the slot's own rows or its cell's replay block's; at any other it only bounds a ref
    by its table, so a ref could point a drawn unit at another unit's rows.
  - `second-values`: its registered reads are not its template's reference statement's (`Own.refReads`, the
    registered reads the auditor's params give beside its header, fixed before the run and taken as given), both with
    each port's run fields blanked (`blankReads`), or the params give its template none. The run's fields are those
    `second-read` holds to the run: every port's `root`, `domain` and `leaves` (`READ_RUN_FIELDS`), and the rows
    (`reads`, `reads_from`: `READ_ROW_FIELDS`) of a port a link names or one reading a port of the interior record (its
    own, or at a port a wire names its writer's), whose rows the run places (`runPlacesRows`). So the ports it reads
    registered, the value each reads, and the rows at
    which it reads a value a registration registers are fixed before the run: `second-read` alone lets a port no link
    names read any value either registration registers, at any rows.
* **The links** (`Link`) are the verifier's own, derived from the replay block's registered Definition and never
  supplied (`replayLinks`). A template is a composite every node calling which is one of the body's, its callers, its
  units in a replay block theirs in body order, a batch's members one each (`callersOf`). One whose callers each read
  one aligned row of one of the block's parameters, of the template's parameter's width, reads that boundary port
  (`slice`), and one whose callers' whole outputs the block returns writes its output's, or, where it returns several
  fields, the field they are in (`o<j>` for field `j`). Every caller reads a linked port alike, of one value.
* **The wires** (`Wire`) are the verifier's own too, from the same Definition (`replayWires`): a template whose
  parameter is, at each caller, one field of one instance of another template's caller reads that template's output
  port `o<f>` for field `f`, its writer's, and the interior commits that value once, at its writer's port (no copy
  tree). So a wired port reads its writer's output value: `second-read` holds it to that port at its writer's rows, and
  `atRun` binds it by that port (`heldPort`). A body that reads a node any other way, an argument gathering several
  nodes' outputs among them, makes no wires.
* **The rows** (`Rows`) of each port a link or a wire names are the verifier's own too (`replayRows`): unit `j` of a
  replay block reads the row its caller names, a link's the row of the parameter its slice is, a wire's its writer's
  unit, summarized once per Definition as runs (`Run`) and never unrolled, so each of several callers reads its own
  row, never another unit's.
* **The interior record's order**: `second-interior-order` when the replay sample the interior record names
  (`window.replay_sample`, which the proof server writes) is not the verifier's own, `S`, in its order
  (`interiorOrderOk`). The record then states that its blocks are `S`'s, block `S[i]`'s cell `off_t + i·n_t`, rather
  than leaving it to convention. It is checked before any statement is read, and fails the check whatever else does.
* **Over the statements**: `second-cells` when the interior program's strata make no parts, `second-links` when the
  replay block's Definition makes no links (`linksOf`) or one names a template of no part or one whose units its rows
  don't place (`placed`), `second-wires` when it makes no wires (`wiresOf`) or one names a template of no part, one
  whose units its rows don't place, or a writer of other units a block than its rows, `second-statements` when no
  statements are given, `second-order` for each statement for a part other than the one at its position (round 2 reads
  the statements by position, the `i`-th the `i`-th part's, in the parts' order, the interior program's strata's),
  `second-cover` for each part no statement is for (every part has one, drawn by the second draw or not), and
  `second-reported` when the `{value: root}` the verifiers of record report the sessions read (`Record.sessionRegistered`)
  is not the statements' (`statementReads`). A passing check (`check_ok`) holds the interior record to the replay
  sample in its order (`interiorOrderOk_sample`), the statements to the parts, one each by position, every header to
  its template's reference placed in the run but for the run's fields (`check_ok_header`), every port a header shares
  rows at to a registered read (`check_ok_refs`), every statement's registered reads to its template's reference
  statement's but for the run's fields, a registered value's rows among what they hold (`check_ok_values`), every
  session's sample to the second draw, cell by cell, and every read it reports to a statement's.
* **The form**: once every check passes, the run is refused last (`two-stage-form`) unless `FORM` allows it; while it
  doesn't, no run passes (`check_refused`).
-/
module

public import Flock.FV3
public import Flock.Program
public import Spot.Record
meta import Flock.FV3
meta import Flock.Program
meta import Spot.Record

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.SecondStage

open Lean (Json)

/-- **The switch**: whether the verifier accepts a two-stage run whose every check of round 2 passes. Off, it refuses
each one last (`two-stage-form`): the two-stage guarantee (`Flock.SecurityProofs.SampledKSTwo`) takes its sessions'
openings (`SessionOpensTwo`) as a fact of the code, and SPOT[Flock] derives that fact only from its layout's `Embeds` and
`FrameIs`, both open. -/
def FORM : Bool := false

def FORM_WHY : String :=
  "refused: a two-stage run, its second-stage sessions over an interior registration: a statement form the soundness " ++
  "proof does not cover"

def CELLS_WHY : String :=
  "the interior program's strata are not one range each, of a positive multiple of the replay sample's size"

def STATEMENTS_WHY : String :=
  "a two-stage outcome names its second-stage sessions' statements: each one's template, header, draw and registered reads"

def REPORTED_WHY : String :=
  "the registered reads the verifiers of record report are not the statements' registered reads, value by value"

def LINKS_WHY : String :=
  "the replay block's Definition reads a parameter, or returns, other than through templates of the interior's " ++
  "parts, each caller reading one aligned row of one parameter alone, or returned whole within one field, alike at " ++
  "every caller, or a link's rows don't place its part's units"

def WIRES_WHY : String :=
  "the replay block's Definition reads a node other than as a template's parameter that is one field of one " ++
  "instance of another template, alike at every caller, or a wire names a template of no part, or its rows don't " ++
  "place its part's units or its writer's"

def INTERIOR_ORDER_WHY : String :=
  "the interior record's replay sample (window.replay_sample) is not the verifier's own, in its order"

/-- A template's part of the interior population: `Q·n` slots from `lo`, `n` in each sampled replay block. -/
structure Part where
  template : String
  lo : Nat
  n : Nat
  deriving Repr, Inhabited, DecidableEq

/-- Slot `u` is one of part `p`'s, in a sample of `q` replay blocks. -/
def Part.holds (q : Nat) (p : Part) (u : Nat) : Bool := decide (p.lo ≤ u) && decide (u < p.lo + q * p.n)

/-- Slot `u`'s place in the sample: the index of its cell's replay block. -/
def Part.span (p : Part) (u : Nat) : Nat := (u - p.lo) / p.n

/-- Slot `u`'s unit in its cell: its place among the `n` slots there. -/
def Part.unit (p : Part) (u : Nat) : Nat := (u - p.lo) % p.n

/-- A stratum's part in a sample of `q` replay blocks: its one range, of a positive multiple of `q` slots. -/
def partOf (q : Nat) (s : String × Array (Nat × Nat)) : Option Part :=
  match s.2.toList with
  | [(lo, hi)] => if 0 < q ∧ lo < hi ∧ (hi - lo) % q = 0 then some ⟨s.1, lo, (hi - lo) / q⟩ else none
  | _ => none

/-- **The parts**: one per stratum, none when a stratum is not such a range. -/
def parts (q : Nat) (strata : Array (String × Array (Nat × Nat))) : Option (Array Part) := strata.mapM (partOf q)

/-- One second-stage session's statement (module docstring). -/
structure Stmt where
  template : String
  header : Json
  draw : Json
  registered : Json

def STMT_KEYS : Array String := #["draw", "header", "registered", "template"]

/-- A statement from its JSON: exactly `STMT_KEYS`, a string, two objects and an object or null. -/
def Stmt.ofJson? (j : Json) : Option Stmt :=
  if keys? j != some STMT_KEYS then none else
  match getD j "template", getD j "header", getD j "draw", getD j "registered" with
  | .str t, .obj h, .obj d, .null => some ⟨t, .obj h, .obj d, .null⟩
  | .str t, .obj h, .obj d, .obj r => some ⟨t, .obj h, .obj d, .obj r⟩
  | _, _, _, _ => none

/-- What an accepted verdict of record names of the session it checked (`flock-verify verify`'s verdict): the digests
of the public file's header and of the `--registered` file it checked the session against (`public_header_sha512`,
`registered_sha512`, null for none), the sample it checked it at (`unit_draw`) and the number of tables it set the
session up as (`session_tables`). -/
structure Checked where
  header : String
  registered : Json
  draw : Json
  tables : Json

/-- What each accepted verdict of record among `verdicts` that names a header names (`Checked`). -/
def verifiedOf (verdicts : Array (String × Json)) : Array Checked :=
  verdicts.filterMap fun (_, v) =>
    match getD v "public_header_sha512" with
    | .str h =>
      if truthy (getD v "of_record") && jeq (getD v "accepted") (.bool true) then
        some ⟨h, getD v "registered_sha512", getD v "unit_draw", getD v "session_tables"⟩
      else none
    | _ => none

/-- One table: the `session_tables` a verdict of record names of a session round 2 reads. -/
def ONE_TABLE : Json := Lean.toJson (1 : Nat)

/-- What a verdict of record that checked statement `s`'s session, as round 2 reads one, names of it (`Checked`): its
header's digest, its registered reads' (null for none), its sample and one table; none when one isn't canonical JSON. -/
def Stmt.checked (s : Stmt) : Option Checked := do
  let h ← (canonSha512 s.header).toOption
  let r ← match s.registered with
    | .null => pure Json.null
    | r => (canonSha512 r).toOption.map Json.str
  pure ⟨h, r, s.draw, ONE_TABLE⟩

/-- A link of the two-stage design: template `template`'s session port `port` reads the boundary registration's port
`value`. -/
structure Link where
  template : String
  port : String
  value : String
  deriving Repr, Inhabited, DecidableEq

def Link.toJson (l : Link) : Json :=
  Json.mkObj [("port", Json.str l.port), ("template", Json.str l.template), ("value", Json.str l.value)]

/-- Links from their JSON: a list of objects of exactly `template`, `port` and `value`, each a string. -/
def linksOf? : Json → Option (Array Link)
  | .arr ls => ls.mapM fun l =>
    if keys? l != some #["port", "template", "value"] then none else
    match getD l "template", getD l "port", getD l "value" with
    | .str t, .str p, .str v => some ⟨t, p, v⟩
    | _, _, _ => none
  | _ => none

/-- A wire of the two-stage design: template `template`'s session port `port` reads template `writer`'s output port
`wport`, one value the interior commits once, at the writer's port. -/
structure Wire where
  template : String
  port : String
  writer : String
  wport : String
  deriving Repr, Inhabited, DecidableEq

def Wire.toJson (w : Wire) : Json :=
  Json.mkObj [("port", Json.str w.port), ("template", Json.str w.template), ("wport", Json.str w.wport),
    ("writer", Json.str w.writer)]

/-- Wires from their JSON: a list of objects of exactly `template`, `port`, `writer` and `wport`, each a string. -/
def wiresOf? : Json → Option (Array Wire)
  | .arr ws => ws.mapM fun w =>
    if keys? w != some #["port", "template", "wport", "writer"] then none else
    match getD w "template", getD w "port", getD w "writer", getD w "wport" with
    | .str t, .str p, .str x, .str o => some ⟨t, p, x, o⟩
    | _, _, _, _ => none
  | _ => none

/-- A run of units' rows: `count` units from unit `unit`, the first reading row `row`, each next `step` rows on. -/
structure Run where
  unit : Nat
  count : Nat
  row : Nat
  step : Nat
  deriving Repr, Inhabited, DecidableEq

/-- The rows template `template`'s units read at its session port `port`, one a link or a wire names: of the value's
`per` rows a replay block, the boundary port's (a link) or its writer's units' (a wire), unit `j` reads the one its
`runs` give it (`rowAt`). -/
structure Rows where
  template : String
  port : String
  per : Nat
  runs : Array Run
  deriving Repr, Inhabited, DecidableEq

def Rows.toJson (r : Rows) : Json :=
  Json.mkObj [("per", Lean.toJson r.per), ("port", Json.str r.port),
    ("runs", Json.arr (r.runs.map fun x => Json.arr #[Lean.toJson x.unit, Lean.toJson x.count, Lean.toJson x.row,
      Lean.toJson x.step])), ("template", Json.str r.template)]

/-- Rows from their JSON: a list of objects of exactly `template`, `port`, `per` and `runs`, two strings, a natural
number and a list of `[unit, count, row, step]`, each a natural number. -/
def rowsOf? : Json → Option (Array Rows)
  | .arr rs => rs.mapM fun r => do
    guard (keys? r == some #["per", "port", "runs", "template"])
    let .str t := getD r "template" | none
    let .str p := getD r "port" | none
    let per ← nat? (getD r "per")
    let .arr xs := getD r "runs" | none
    let runs ← xs.mapM fun x => (match x with
      | .arr #[u, c, w, s] => do pure (Run.mk (← nat? u) (← nat? c) (← nat? w) (← nat? s))
      | _ => none : Option Run)
    pure (Rows.mk t p per runs)
  | _ => none

/-- What the verifier holds of the run: the replay sample `S`, the parts (`parts`), the interior and boundary
registrations' records, the second draw's units, the design's links (`replayLinks`, none when the replay block's
Definition makes none), the `{value: root}` the verifiers of record report the sessions read
(`Record.sessionRegistered`), what they name of each session they accepted (`verifiedOf`), the reference header of each
template's statement and the registered reads it reads (`refReads`, null for none), as the auditor's params give them
(`Run.templateCircuits`), taken as given: nothing here verifies the reference statement against them, the design's
wires (`replayWires`, none when the Definition makes none), and the rows of the ports they and the links name
(`replayRows`). -/
structure Own where
  sample : Array Nat
  parts : Option (Array Part)
  interior : Json
  boundary : Json
  drawn : Array Nat
  links : Option (Array Link) := some #[]
  reported : Option (Array (String × Json)) := none
  verified : Array Checked := #[]
  refs : Array (String × Json) := #[]
  refReads : Array (String × Json) := #[]
  wires : Option (Array Wire) := some #[]
  rows : Array Rows := #[]

/-- The record names its ports by template, `<template>/<port>`. -/
def byTemplate (rec : Json) : Bool := (Record.byPort rec).any fun (q, _) => q.contains '/'

/-- The interior record's port for port `p` of template `t`'s circuit. -/
def portOf (named : Bool) (t p : String) : String := if named then t ++ "/" ++ p else p

/-- The interior record's ports of template `t`: those under `t/` when it names its ports by template, else all. -/
def templatePorts (named : Bool) (t : String) (rec : Json) : Array String :=
  ((Record.byPort rec).map (·.1)).filter fun q => !named || q.startsWith (t ++ "/")

/-- The output ports `o<j>` of a circuit returning `t`, one per field: a tuple's or a record's fields, else one. -/
def outputs : Flock.Program.Ty → Nat
  | .tuple fs => fs.size
  | .record fs => fs.size
  | _ => 1

/-- The fields of a circuit returning `t`, one per output port (`outputs`): a tuple's or a record's, else `t`. -/
def fields : Flock.Program.Ty → Array Flock.Program.Ty
  | .tuple fs => fs
  | .record fs => fs.map (·.2)
  | t => #[t]

/-- What part `q` of a body's reference reads: parameters (`true`) and nodes (`false`), by index. -/
def partSources : Flock.Program.Part → List (Bool × Nat)
  | .affine p i _ _ => [(p, i)]
  | .strided p i _ _ _ _ _ => [(p, i)]
  | .explicit items => items.toList.map fun r => (r.p, r.i)

/-- What reference `s` reads: its parts' (`partSources`). -/
def sources (s : Flock.Program.Seq) : List (Bool × Nat) := s.parts.toList.flatMap partSources

/-- How many nodes of all of `defs`' bodies call each definition, by index. -/
def callCounts (defs : Array Flock.Program.Defn) : Array Nat :=
  defs.foldl (fun acc d => d.body.nodes.foldl (fun acc x => acc.modify x.fn (· + 1)) acc) (Array.replicate defs.size 0)

/-- A replay block's body's templates and their units (`callersOf`). -/
structure Callers where
  /-- each node's template, none for a node no template's -/
  template : Array (Option Flock.Program.Defn)
  /-- each node's first unit among its definition's -/
  first : Array Nat
  /-- each definition's units in the body -/
  units : Array Nat

/-- Replay block `D`'s body's templates and their units, once over `defs` and `D`'s body: each node's template, the
composite definition it calls or batches when every node of `defs` that calls it is one of `D`'s body's; each node's
first unit among its definition's in a replay block, its callers before it in body order holding the units before it,
a batch its members' (`Node.members`), never walked; and each definition's units. -/
def callersOf (defs : Array Flock.Program.Defn) (D : Flock.Program.Defn) : Callers := Id.run do
  let all := callCounts defs
  let here := D.body.nodes.foldl (fun acc x => acc.modify x.fn (· + 1)) (Array.replicate defs.size 0)
  let mut units := Array.replicate defs.size 0
  let mut template := #[]
  let mut first := #[]
  for x in D.body.nodes do
    first := first.push (units.getD x.fn 0)
    units := units.modify x.fn (· + x.members)
    template := template.push (match defs[x.fn]? with
      | some c =>
        if (x.form == .call || x.form == .batch) && !c.prim && all.getD x.fn 0 == here.getD x.fn 0 then some c
        else none
      | none => none)
  return ⟨template, first, units⟩

/-- Node `k`'s template (`callersOf`). -/
def Callers.of (cs : Callers) (k : Nat) : Option Flock.Program.Defn := cs.template.getD k none

/-- Runs `rs` with run `r` after them: merged into the last when `r` is one unit right after it continuing its rows
(the last one unit, at a row no later than `r`'s, or `r`'s row the last's next), else appended. -/
def pushRun (rs : Array Run) (r : Run) : Array Run :=
  match rs.back? with
  | some l =>
    if r.count == 1 && l.unit + l.count == r.unit then
      if l.count == 1 && l.row ≤ r.row then rs.pop.push { l with count := 2, step := r.row - l.row }
      else if 1 < l.count && r.row == l.row + l.count * l.step then rs.pop.push { l with count := l.count + 1 }
      else rs.push r
    else rs.push r
  | none => rs.push r

/-- Unit `j`'s row under runs `rs`, in order of their first units: the run holding it (`unit ≤ j < unit + count`),
found by bisection over their first units, gives `row + (j - unit)·step`; none for a unit no run holds. -/
def rowAt (rs : Array Run) (j : Nat) : Option Nat := Id.run do
  let mut lo := 0
  let mut hi := rs.size
  for _ in [0:rs.size] do
    if hi - lo ≤ 1 then break
    let mid := (lo + hi) / 2
    if rs[mid]!.unit ≤ j then lo := mid else hi := mid
  match rs[lo]? with
  | some r => if r.unit ≤ j && j < r.unit + r.count then some (r.row + (j - r.unit) * r.step) else none
  | none => none

/-- Runs `rs` hold units `0` to `n - 1`, each once, in order: the first from unit `0`, each next from the last's end,
none empty, the last ending at `n`. -/
def fits (rs : Array Run) (n : Nat) : Bool :=
  rs.foldl (fun (e : Option Nat) r => e.bind fun e => if r.unit == e && 0 < r.count then some (e + r.count) else none)
    (some 0) == some n

/-- One caller's share of a link or a wire `edge`: its template's index `fn`, the value's rows a replay block `per`,
and its units' rows `run`. -/
structure Share (α : Type) where
  edge : α
  fn : Nat
  per : Nat
  run : Run

/-- Shares `xs` gathered one per template port (`port`), in order of first appearance, each port's runs in order
(`pushRun`), with its value's rows a block; none when two of a port name different values (`same`) or rows a block,
or a port's runs don't hold each of its template's `units` once, in order (`fits`). -/
def gather {α : Type} (port : α → String × String) (same : α → α → Bool) (units : Array Nat) (xs : List (Share α)) :
    Option (Array (α × Nat × Array Run)) := do
  let g ← xs.foldlM (fun (acc : Array (Share α × Array Run)) x =>
    match acc.findIdx? (fun (y, _) => port y.edge == port x.edge) with
    | some j => do
      let (y, rs) ← acc[j]?
      guard (same y.edge x.edge && y.per == x.per)
      pure (acc.set! j (y, pushRun rs x.run))
    | none => pure (acc.push (x, #[x.run]))) #[]
  guard (g.all fun (y, rs) => fits rs (units.getD y.fn 0))
  pure (g.map fun (y, rs) => (y.edge, y.per, rs))

/-- The row of its collection's rows node `x`'s argument `a` reads, `W` leaves a member, and each next member's step,
when the argument is one aligned slice of `W` leaves a member of one collection, its one affine part: the same slice
for every member (a call's argument, or a batch's shared one, step `0`), or a member's from the first, outer axis
mapped (step `1`); with its space and index. None for any other argument. -/
def slice (x : Flock.Program.Node) (a : Flock.Program.Arg) (W : Nat) : Option (Bool × Nat × Nat × Nat) := do
  guard (0 < W)
  let [.affine p i base c] := a.seq.parts.toList | none
  guard (base % W == 0)
  match a.dims with
  | none => guard (c == W); pure (p, i, base / W, 0)
  | some (_, L) => guard (x.form == .batch && L == W && c == x.members * W); pure (p, i, base / W, 1)

/-- The links node `k` of replay block `D`'s body makes, each its share (`Share`): for each argument that reads one of
`D`'s parameters, its template's parameter there reads that one, the parameter's rows a block those of the template
parameter's width, its units the rows its slice is (`slice`). None when the argument reads anything else too, or
anything but such a slice, or `k` is no template's. -/
def nodeLinks (cs : Callers) (D : Flock.Program.Defn) (k : Nat) (x : Flock.Program.Node) :
    Option (List (Share Link)) :=
  (List.range x.args.size).filterMapM fun a => do
    let arg ← x.args[a]?
    let src := sources arg.seq
    match src.find? (·.1) with
    | none => pure none
    | some (_, i) =>
      guard (src.all (· == (true, i)))
      let c ← cs.of k
      let (port, ty) ← c.params[a]?
      let (name, pty) ← D.params[i]?
      let (_, _, row, step) ← slice x arg ty.leaves
      guard (pty.leaves % ty.leaves == 0)
      pure (some ⟨⟨c.key, port, name⟩, x.fn, pty.leaves / ty.leaves, ⟨cs.first.getD k 0, x.members, row, step⟩⟩)

/-- The field of a value of type `t` from leaf `off`, of `c` leaves: its index among `fields t`; none for none. -/
def fieldAt (t : Flock.Program.Ty) (off c : Nat) : Option Nat :=
  ((fields t).zipIdx.foldl (fun (acc : Nat × Option Nat) (f, j) =>
    (acc.1 + f.leaves, acc.2 <|> if acc.1 == off && f.leaves == c then some j else none)) (0, none)).2

/-- The field of a value of type `t` holding its leaves `[s, s + c)`: its index and first leaf; none for none. -/
def fieldOf (t : Flock.Program.Ty) (s c : Nat) : Option (Nat × Nat) :=
  ((fields t).zipIdx.foldl (fun (acc : Nat × Option (Nat × Nat)) (f, j) =>
    (acc.1 + f.leaves, acc.2 <|> if acc.1 ≤ s && s + c ≤ acc.1 + f.leaves then some (j, acc.1) else none))
    (0, none)).2

/-- The links replay block `D`'s return makes, each its share (`Share`): each part of its return, which holds all of
its leaves, node `k`'s whole output, `k` a template's of one output, within one field `j` of the return at a multiple
of that output's leaves from the field's first, makes the template's `o0` `D`'s `o<j>`, the field's rows a block those
of the output's width, and `k`'s units the field's rows from there on, one each. None when a part is anything else. -/
def returnLinks (cs : Callers) (D : Flock.Program.Defn) : Option (List (Share Link)) := do
  guard (D.body.ret.count == D.ret.leaves)
  (List.range D.body.ret.parts.size).mapM fun q => do
    let part ← D.body.ret.parts[q]?
    let s ← D.body.ret.starts[q]?
    let .affine false k 0 c := part | none
    let x ← D.body.nodes[k]?
    let t ← cs.of k
    let w := t.ret.leaves
    guard (outputs t.ret == 1 && 0 < w && c == x.out.leaves && c == x.members * w)
    let (j, off) ← fieldOf D.ret s c
    let f ← (fields D.ret)[j]?
    guard ((s - off) % w == 0 && f.leaves % w == 0)
    pure ⟨⟨t.key, "o0", "o" ++ toString j⟩, x.fn, f.leaves / w,
      ⟨cs.first.getD k 0, x.members, (s - off) / w, if x.members == 1 then 0 else 1⟩⟩

/-- **The design's links** with their rows a block and runs, from replay block `defs[d]`'s registered Definition, its
parameters and its body: those its nodes make (`nodeLinks`) and its return (`returnLinks`), one per template port
(`gather`), each value its own port's name. None when its body reads a parameter, or returns, any other way, or a
template's callers read a linked port unlike each other. -/
def linkRuns (defs : Array Flock.Program.Defn) (d : Nat) : Option (Array (Link × Nat × Array Run)) := do
  let D ← defs[d]?
  guard (!D.prim)
  let cs := callersOf defs D
  let ins ← (List.range D.body.nodes.size).mapM fun k => do nodeLinks cs D k (← D.body.nodes[k]?)
  let outs ← returnLinks cs D
  gather (fun l => (l.template, l.port)) (fun a b => a.value == b.value) cs.units (ins.flatten ++ outs)

/-- **The design's links** (`linkRuns`). -/
def linksOf (defs : Array Flock.Program.Defn) (d : Nat) : Option (Array Link) := (linkRuns defs d).map (·.map (·.1))

/-- The links of the replay block program `P` defines under descriptor id `key` (`linksOf`), each value the boundary
registration's port for it, as record `rec` names its ports (`portOf` under `key`). -/
def replayLinks (P : Flock.Program.Prog) (key : String) (rec : Json) : Option (Array Link) := do
  let d ← P.index[key]?
  let ls ← linksOf P.defs d
  pure (ls.map fun l => { l with value := portOf (byTemplate rec) key l.value })

/-- The link that names template `t`'s session port `port`. -/
def linkOf (own : Own) (t port : String) : Option Link :=
  (own.links.getD #[]).find? fun l => l.template == t && l.port == port

/-- The wires node `k` of replay block `D`'s body makes, each its share (`Share`): for each argument that reads a node,
its template's parameter there reads that node's output field `f`, one of its template's (`fieldAt`), the node one
instance and the argument the field whole, the same for each of `k`'s members: the writer's units a block its rows,
`k`'s units reading the node's. None when such an argument reads anything else too, or part of a field, several
nodes, a node no template's, or `k` is no template's. An argument that reads only `D`'s parameters is a link's
(`nodeLinks`). -/
def nodeWires (cs : Callers) (D : Flock.Program.Defn) (k : Nat) (x : Flock.Program.Node) :
    Option (List (Share Wire)) :=
  (List.range x.args.size).filterMapM fun a => do
    let arg ← x.args[a]?
    if (sources arg.seq).all (·.1) then return none
    let [.affine false k' off c] := arg.seq.parts.toList | none
    let w ← D.body.nodes[k']?
    let tw ← cs.of k'
    let t ← cs.of k
    let (port, ty) ← t.params[a]?
    guard (arg.dims.isNone && w.members == 1 && 0 < c && c == ty.leaves && w.out.leaves == tw.ret.leaves)
    let f ← fieldAt tw.ret off c
    pure (some ⟨⟨t.key, port, tw.key, "o" ++ toString f⟩, x.fn, cs.units.getD w.fn 0,
      ⟨cs.first.getD k 0, x.members, cs.first.getD k' 0, 0⟩⟩)

/-- **The design's wires** with their rows a block and runs, from replay block `defs[d]`'s registered Definition:
those its nodes make (`nodeWires`), one per template port (`gather`). None when its body reads a node any other way,
or a template's callers read a wired port unlike each other. -/
def wireRuns (defs : Array Flock.Program.Defn) (d : Nat) : Option (Array (Wire × Nat × Array Run)) := do
  let D ← defs[d]?
  guard (!D.prim)
  let cs := callersOf defs D
  let ws ← (List.range D.body.nodes.size).mapM fun k => do nodeWires cs D k (← D.body.nodes[k]?)
  gather (fun w => (w.template, w.port)) (fun a b => a.writer == b.writer && a.wport == b.wport) cs.units ws.flatten

/-- **The design's wires** (`wireRuns`). -/
def wiresOf (defs : Array Flock.Program.Defn) (d : Nat) : Option (Array Wire) := (wireRuns defs d).map (·.map (·.1))

/-- The wires of the replay block program `P` defines under descriptor id `key` (`wiresOf`). -/
def replayWires (P : Flock.Program.Prog) (key : String) : Option (Array Wire) := do
  wiresOf P.defs (← P.index[key]?)

/-- **The design's rows**: those of the ports the links and the wires of the replay block program `P` defines under
descriptor id `key` name (`linkRuns`, `wireRuns`), none for either it makes none of. -/
def replayRows (P : Flock.Program.Prog) (key : String) : Array Rows :=
  match P.index[key]? with
  | some d =>
    ((linkRuns P.defs d).getD #[]).map (fun (l, per, rs) => ⟨l.template, l.port, per, rs⟩) ++
      ((wireRuns P.defs d).getD #[]).map fun (w, per, rs) => ⟨w.template, w.port, per, rs⟩
  | none => #[]

/-- The rows of template `t`'s session port `port` (`Own.rows`). -/
def rowsOf (own : Own) (t port : String) : Option Rows := own.rows.find? fun r => r.template == t && r.port == port

/-- Part `p`'s units are placed at its template's session port `port`: its rows' runs hold the `n` units a block of
`p` (`fits`). -/
def placed (own : Own) (p : Part) (port : String) : Bool :=
  match rowsOf own p.template port with
  | some r => fits r.runs p.n
  | none => false

/-- The links are derived, each for a template one of parts `ps` is of, whose units its rows place (`placed`). -/
def linksOk (ps : Array Part) (own : Own) : Bool :=
  match own.links with
  | some ls => ls.all fun l => ps.any fun p => p.template == l.template && placed own p l.port
  | none => false

/-- The wires are derived, each between templates parts `ps` are of, its reader's units placed by its rows (`placed`),
whose rows a block are its writer's units a block. -/
def wiresOk (ps : Array Part) (own : Own) : Bool :=
  match own.wires with
  | some ws => ws.all fun w =>
    ps.any (fun p => p.template == w.template && placed own p w.port) &&
      ps.any fun p => p.template == w.writer && (rowsOf own w.template w.port).any (·.per == p.n)
  | none => false

/-- The wire that names template `t`'s session port `port`. -/
def wireOf (own : Own) (t port : String) : Option Wire :=
  (own.wires.getD #[]).find? fun w => w.template == t && w.port == port

/-- Wire `w`'s writer's part. -/
def writerOf (own : Own) (w : Wire) : Option Part := (own.parts.getD #[]).find? (·.template == w.writer)

/-- The interior record's port that holds template `t`'s session port `p`: at a port a wire names, its writer's output
port, the one tree the interior commits that value in; else its own (`portOf`). -/
def heldPort (own : Own) (named : Bool) (t p : String) : String :=
  match wireOf own t p with
  | some w => portOf named w.writer w.wport
  | none => portOf named t p

/-- **The run's fields** of a statement's header, each a path of keys: the only ones a header may differ from its
template's reference placed in the run in, which other checks hold instead: `instances` and `units.indices`
(`second-population`), `units.blocks`, the indices cut into blocks of `vus_per_block` (the session's verifier, which
refuses a header whose `units` aren't `Flock.HmRow.unitsEntry` of its indices), `frame_v3.roots` and
`frame_v3.domain_ids` (`second-roots`), and `registered`, the reads' values, positions and paths (`second-read`, which
holds each read to its root, domain and rows, and `second-values`, which holds the value each port reads, and the rows
at which it reads a value a registration registers, to the reference's, through the `--registered` file the session's
verifier holds the header's to). A reference staged at other slots differs from the header only in the indices and
their blocks. -/
def RUN_FIELDS : List (List String) :=
  [["instances"], ["units", "indices"], ["units", "blocks"], ["frame_v3", "roots"], ["frame_v3", "domain_ids"],
    ["registered"]]

/-- Object `j` with key `k` set to `v`; anything else unchanged. -/
def setKey (j : Json) (k : String) (v : Json) : Json :=
  match j with
  | .obj _ => j.setObjVal! k v
  | _ => j

/-- `j` with the field at key path `path` set to `v` where its parent is an object, each key before the last followed
only to an object; anything else unchanged. -/
def putAt (v : Json) : Json → List String → Json
  | j, [] => j
  | j, [k] => setKey j k v
  | j, k :: ks => match field? j k with
    | some (.obj kvs) => setKey j k (putAt v (.obj kvs) ks)
    | _ => j

/-- `j` with the field at key path `path` null (`putAt`). -/
def blankAt (j : Json) (path : List String) : Json := putAt .null j path

/-- Header `h` with the run's fields (`RUN_FIELDS`) blanked, null. -/
def blankRun (h : Json) : Json := RUN_FIELDS.foldl blankAt h

/-- The reference header the auditor's params give template `t` (`Own.refs`). -/
def refOf (own : Own) (t : String) : Option Json := (own.refs.find? (·.1 == t)).map (·.2)

/-- The run the interior record names (`window.run`). -/
def runOf (rec : Json) : Option String :=
  match getD (getD rec "window") "run" with
  | .str r => some r
  | _ => none

/-- The input set of the `k`-th part's session in run `run`, as the served path names it: `served/<run>/m<k>`, the
interior's `k`-th member being the `k`-th part. -/
def setName (run : String) (k : Nat) : String := "served/" ++ run ++ "/m" ++ toString k

/-- The window-domain rule's tag (`verity.ml.commitments.window.WINDOW_DOMAIN`). -/
def WINDOW_DOMAIN : String := "verity/one-stage/served-domain/v1"

/-- The binding, in run `run`, of the interior record `rec`'s port `port` of row schema `schema`: the window-domain rule
(`verity.ml.commitments.window`), the identity digest (`tagged_sha512` of the canonical JSON) of the record's program
and partition, the port, its schema, the leaves the record gives it and the run. None when the record names no such
port, or no program or partition. -/
def portBinding (rec : Json) (port schema run : String) : Option String := do
  let .str program := getD rec "program" | none
  let .str partition := getD rec "partition" | none
  let (_, e) ← (Record.byPort rec).find? (·.1 == port)
  let leaves ← nat? (getD e "leaves")
  let m ← (Flock.canonUtf8 (Json.mkObj [("program", .str program), ("partition", .str partition), ("port", .str port),
    ("schema", .str schema), ("leaves", Lean.toJson leaves), ("run", .str run)])).toOption
  pure (Flock.hexOf (Flock.FV3.Sha512.taggedSha512 WINDOW_DOMAIN m.toUTF8))

/-- Template `t`'s reference header `r` placed in the run, for the session of the `k`-th part: its `set`
the run's input set (`setName`) and its `frame_v3.bindings` the binding of each port of its `frame_v3.schemas`
(`portBinding`, the interior record's port that holds it, `heldPort`: a wired port's its writer's), which the verifier
computes, so the prover chooses neither. None when the interior record names no run, or a port the verifier can't
bind. -/
def atRun (own : Own) (named : Bool) (k : Nat) (t : String) (r : Json) : Option Json := do
  let run ← runOf own.interior
  let bs ← (entries (getD (getD r "frame_v3") "schemas")).mapM fun (p, s) => do
    let .str s := s | none
    pure (p, Json.str (← portBinding own.interior (heldPort own named t p) s run))
  pure (putAt (Json.mkObj bs.toList) (putAt (.str (setName run k)) r ["set"]) ["frame_v3", "bindings"])

/-- Header `h` of a statement for the `k`-th part is template `t`'s reference header placed in the run (`atRun`) but for
the run's fields: the two the same with those blanked (`blankRun`). False for a template the params give no reference,
or one the verifier can't place in the run. -/
def headerOk (own : Own) (named : Bool) (k : Nat) (t : String) (h : Json) : Bool :=
  match refOf own t with
  | some r => match atRun own named k t r with
    | some r' => jeq (blankRun h) (blankRun r')
    | none => false
  | none => false

/-- The header's `units.indices` are part `p`'s slots in order, and its `instances` their count. -/
def populationOk (q : Nat) (p : Part) (h : Json) : Bool :=
  match getD (getD h "units") "indices", nat? (getD h "instances") with
  | .arr is, some m => m == q * p.n && is.toList.map nat? == (List.range (q * p.n)).map fun i => some (p.lo + i)
  | _, _ => false

/-- The header's scheme is the interior record's, its `frame_v3` roots outside the ports `reads` reads are the record's
for template `t`'s ports, each at the record's root, its `domain_ids` entry the record's domain, and each of those ports
is one it roots or reads. -/
def rootsOk (named : Bool) (rec : Json) (t : String) (h : Json) (reads : Array String) : Bool :=
  let fv := getD h "frame_v3"
  let roots := (entries (getD fv "roots")).filter fun (p, _) => !reads.contains p
  let reg := Record.byPort rec
  jeq (getD h "scheme") (getD rec "scheme") &&
    roots.all (fun (p, r) => match reg.find? (·.1 == portOf named t p) with
      | some (_, e) => jeq r (getD e "root") && jeq (getD (getD fv "domain_ids") p) (getD e "domain")
      | none => false) &&
    (templatePorts named t rec).all fun x =>
      roots.any (fun (p, _) => portOf named t p == x) || reads.any fun p => portOf named t p == x

/-- **The verifier's own sample** for part `p`'s session: the second draw's units among its slots, counted from its
first, as `unit_draw` writes a subset of its population. -/
def ownDraw (q : Nat) (p : Part) (drawn : Array Nat) : Json :=
  let units := (drawn.filter (p.holds q)).map (· - p.lo)
  Json.mkObj [("law", Json.str "subset"), ("population", Lean.toJson (q * p.n)), ("k", Lean.toJson units.size),
    ("units", natsJson units)]

/-- Value `v`'s entry among registration `rec`'s registered values. -/
def registeredEntry (rec : Json) (v : String) : Option Json :=
  (match getD rec "registered" with | .arr es => es | _ => #[]).find? fun e => jsonStrIs (getD e "value") v

/-- Entries `e` and `x` name one root, domain and leaf count. -/
def sameAt (e x : Json) : Bool :=
  jeq (getD e "root") (getD x "root") && jeq (getD e "domain") (getD x "domain") && jeq (getD e "leaves") (getD x "leaves")

/-- Every read of registered-reads entry `e` (slot `reads_from + k` reads row `reads[k]`, null for none) is by one of
part `p`'s slots `u`, of a row of the unit `unit u` names, in a tree of `units` units' rows, `leaves / units` each. -/
def rowsOk (q : Nat) (p : Part) (e : Json) (units : Nat) (unit : Nat → Option Nat) : Bool :=
  let first : Option Nat := match getD e "reads_from" with | .null => some 0 | x => nat? x
  match nat? (getD e "leaves"), getD e "reads", first with
  | some leaves, .arr rs, some f =>
    decide (0 < units) && leaves % units == 0 && (List.range rs.size).all fun k =>
      match rs.getD k .null with
      | .null => true
      | x => match nat? x, unit (f + k) with
        | some pos, some b =>
          p.holds q (f + k) && decide (b * (leaves / units) ≤ pos) && decide (pos < (b + 1) * (leaves / units))
        | _, _ => false
  | _, _, _ => false

/-- Every read of registered-reads entry `e` (slot `reads_from + k` reads row `reads[k]`, null for none) is by one of
part `p`'s slots `u`, of row `row u`, in a tree of `leaves` rows. -/
def placedOk (q : Nat) (p : Part) (e : Json) (leaves : Nat) (row : Nat → Option Nat) : Bool :=
  let first : Option Nat := match getD e "reads_from" with | .null => some 0 | x => nat? x
  match nat? (getD e "leaves"), getD e "reads", first with
  | some l, .arr rs, some f =>
    l == leaves && (List.range rs.size).all fun k =>
      match rs.getD k .null with
      | .null => true
      | x => match nat? x, row (f + k) with
        | some pos, some r => p.holds q (f + k) && pos == r
        | _, _ => false
  | _, _, _ => false

/-- The row slot `u` of part `p` reads at a port a link names, of rows `r`: its unit's (`rowAt`), of the `per` its
cell's replay block holds in the boundary port. -/
def linkedRow (own : Own) (p : Part) (r : Rows) (u : Nat) : Option Nat := do
  let b ← own.sample[p.span u]?
  let x ← rowAt r.runs (p.unit u)
  if x < r.per then some (b * r.per + x) else none

/-- The row slot `u` of part `p` reads at a port a wire names, of rows `r`: its unit's writer unit's (`rowAt`), of the
`per` its cell's replay block holds in the writer's output port. -/
def wiredRow (p : Part) (r : Rows) (u : Nat) : Option Nat := do
  let x ← rowAt r.runs (p.unit u)
  if x < r.per then some (p.span u * r.per + x) else none

/-- Template `p.template`'s session port `port` may read registered-reads entry `e`: a port a link names, only the
boundary port it names, of its rows' `per` a replay block, each slot only its unit's row of its cell's replay block's
(`linkedRow`); a port a wire names, only its writer's output port of the interior record, of the writer's units a
block, each slot only its unit's writer unit's row in its cell's replay block (`wiredRow`); any other, its own port of
the interior record, each slot only its own rows, or a value either registration registers, never a boundary port.
Each at its root, domain and leaves. -/
def readOk (own : Own) (named : Bool) (p : Part) (port : String) (e : Json) : Bool :=
  let q := own.sample.size
  match getD e "value", linkOf own p.template port, wireOf own p.template port with
  | .str v, some l, _ =>
    v == l.value && match (Record.byPort own.boundary).find? (·.1 == v), nat? (getD own.boundary "population"),
        rowsOf own p.template port with
      | some (_, b), some m, some r => sameAt e b && placedOk q p e (m * r.per) (linkedRow own p r)
      | _, _, _ => false
  | .str v, none, some w =>
    v == portOf named w.writer w.wport && match (Record.byPort own.interior).find? (·.1 == v), writerOf own w,
        rowsOf own p.template port with
      | some (_, x), some pw, some r => sameAt e x && pw.n == r.per && placedOk q p e (q * r.per) (wiredRow p r)
      | _, _, _ => false
  | .str v, none, none =>
    match (Record.byPort own.interior).find? (·.1 == v), (Record.byPort own.boundary).find? (·.1 == v) with
    | some (_, x), _ => v == portOf named p.template port && sameAt e x && rowsOk q p e (q * p.n) fun u => some (u - p.lo)
    | none, some _ => false
    | none, none => match registeredEntry own.boundary v <|> registeredEntry own.interior v with
      | some x => sameAt e x
      | none => false
  | _, _, _ => false

/-- Every port a link names for template `t` is one of `reads`. -/
def linkedRead (own : Own) (t : String) (reads : Array (String × Json)) : Bool :=
  (own.links.getD #[]).all fun l => l.template != t || reads.any (·.1 == l.port)

/-- Every port a wire names for template `t` is one of `reads`. -/
def wiredRead (own : Own) (t : String) (reads : Array (String × Json)) : Bool :=
  (own.wires.getD #[]).all fun w => w.template != t || reads.any (·.1 == w.port)

/-- Statement `s` states what an accepted verdict of record checked: one names its header's digest, its registered
reads' and its sample (`Stmt.checked`). -/
def verifiedOk (own : Own) (s : Stmt) : Bool :=
  match s.checked with
  | some c => own.verified.any fun v => v.header == c.header && jeq v.registered c.registered && jeq v.draw c.draw
  | none => false

/-- Every accepted verdict of record that names statement `s`'s header, registered reads and sample set its session up
as one table (`ONE_TABLE`): a session split across Flock tables is no session round 2 reads. -/
def tablesOk (own : Own) (s : Stmt) : Bool :=
  match s.checked with
  | some c => own.verified.all fun v =>
    !(v.header == c.header && jeq v.registered c.registered && jeq v.draw c.draw) || jeq v.tables ONE_TABLE
  | none => true

/-- Statement `s` reads registered every port its header shares rows at (`shared_rows`: the tables whose rows its
instances read by the refs in its public file's body). At such a port the session's verifier holds each instance's ref
to the row at the position the read gives its unit (`Flock.Registered.check`); at any other it only bounds a ref by its
table. A header that shares none has no refs. -/
def refsOk (s : Stmt) : Bool :=
  let reads := entries (getD s.registered "ports")
  (entries (getD s.header "shared_rows")).all fun (port, _) => reads.any (·.1 == port)

/-- **The run's fields** of every port of a registered-reads file, which `second-read` holds to the run (`sameAt`): the
value's `root`, `domain` and `leaves`. -/
def READ_RUN_FIELDS : List String := ["root", "domain", "leaves"]

/-- **The rows** a port of a registered-reads file reads, slot `reads_from + k` its row `reads[k]`: run fields only
where the run places them (`runPlacesRows`). -/
def READ_ROW_FIELDS : List String := ["reads", "reads_from"]

/-- Template `t`'s session port `port` reads registered-reads entry `e` at rows the run places, which `second-read`
holds to the run (`placedOk`, `rowsOk`): a port a link names, each slot at its unit's row of its cell's replay block's,
or one reading a port of the interior record, its own port each slot at its own rows or, at a port a wire names, its
writer's at its writer unit's row (`readOk`'s first three cases). Any other port reads a value a
registration registers, which `second-read` holds only to the value's root, domain and leaves (`sameAt`), so
`second-values` holds its rows to the reference statement's, the same rows in every run. A workload whose rows there
depend on the sampled block (a positional table) would need a rule of its own. -/
def runPlacesRows (own : Own) (t port : String) (e : Json) : Bool :=
  (linkOf own t port).isSome ||
    match getD e "value" with
    | .str v => ((Record.byPort own.interior).find? (·.1 == v)).isSome
    | _ => false

/-- Registered-reads file `r` of a session of template `t` with each port's run fields null: `READ_RUN_FIELDS`, and
`READ_ROW_FIELDS` where the run places the port's rows (`runPlacesRows`); anything else unchanged. -/
def blankReads (own : Own) (t : String) (r : Json) : Json :=
  let blank (p : String) (e : Json) : Json :=
    (if runPlacesRows own t p e then READ_RUN_FIELDS ++ READ_ROW_FIELDS else READ_RUN_FIELDS).foldl
      (fun e k => setKey e k .null) e
  match getD r "ports" with
  | .obj _ => setKey r "ports" (Json.mkObj ((entries (getD r "ports")).toList.map fun (p, e) => (p, blank p e)))
  | _ => r

/-- The registered reads of template `t`'s reference statement, as the auditor's params give them (`Own.refReads`). -/
def refReadsOf (own : Own) (t : String) : Option Json := (own.refReads.find? (·.1 == t)).map (·.2)

/-- Statement `s`'s registered reads are its template's reference statement's but for the run's fields: the two the
same with those blanked (`blankReads`), so null only where the reference's is, and each value a registration registers
read at the reference's rows. False for a template the params give no reference reads. -/
def valuesOk (own : Own) (s : Stmt) : Bool :=
  match refReadsOf own s.template with
  | some r => jeq (blankReads own s.template s.registered) (blankReads own s.template r)
  | none => false

/-- Statement `s`'s failing checks against its part `p` (module docstring), in order, `k` the part's position. -/
def stmtCodes (own : Own) (named : Bool) (k : Nat) (p : Part) (s : Stmt) : List String :=
  let q := own.sample.size
  let reads := entries (getD s.registered "ports")
  [("second-verified", verifiedOk own s),
   ("second-tables", tablesOk own s),
   ("second-header", headerOk own named k s.template s.header),
   ("second-population", populationOk q p s.header),
   ("second-roots", rootsOk named own.interior s.template s.header (reads.map (·.1))),
   ("second-sample", jeq s.draw (ownDraw q p own.drawn)),
   ("second-read", reads.all (fun (port, e) => readOk own named p port e) && linkedRead own p.template reads &&
     wiredRead own p.template reads),
   ("second-refs", refsOk s),
   ("second-values", valuesOk own s)].filterMap
    fun (code, ok) => if ok then none else some code

def detailOf (i : Nat) (t : Json) : Json := Json.mkObj [("statement", Lean.toJson i), ("template", t)]

/-- The templates statements `ss` are for, those that are statements. -/
def templates (ss : List Json) : List String := ss.filterMap fun j => (Stmt.ofJson? j).map (·.template)

/-- Template `t`'s position among the parts `ps`, its first part's: the index of its member in the interior window,
whose members are the parts in order (the interior program's strata's); `ps.size`, no member's, for none. -/
def posOf (ps : Array Part) (t : String) : Nat := (ps.findIdx? (·.template == t)).getD ps.size

/-- Statement `i` of `ss`'s failures, `(code, detail)` each: `second-statement` for one that is not a statement,
`second-template` for one for no part's template or an earlier statement's, else its failing checks, its header placed
at its template's position (`posOf`). -/
def stmtFailures (own : Own) (ps : Array Part) (ss : List Json) (i : Nat) : List (String × Json) :=
  match Stmt.ofJson? (ss.getD i .null) with
  | none => [("second-statement", detailOf i .null)]
  | some s =>
    match ps.find? (·.template == s.template) with
    | some p =>
      if (templates (ss.take i)).contains s.template then [("second-template", detailOf i (.str s.template))]
      else (stmtCodes own (byTemplate own.interior) (posOf ps s.template) p s).map
        fun c => (c, detailOf i (.str s.template))
    | none => [("second-template", detailOf i (.str s.template))]

/-- `second-order` for each statement of `ss` for a part, no earlier statement's, other than the part at its position:
round 2 reads the statements by position, the `i`-th the `i`-th part's (the parts' order, the interior program's
strata's). -/
def orderFailures (ps : Array Part) (ss : List Json) : List (String × Json) :=
  (List.range ss.length).filterMap fun i =>
    match Stmt.ofJson? (ss.getD i .null) with
    | some s =>
      if (ps.find? (·.template == s.template)).isSome && !(templates (ss.take i)).contains s.template &&
          ps[i]?.map (·.template) != some s.template then
        some ("second-order", detailOf i (.str s.template))
      else none
    | none => none

/-- `second-cover` for each part no statement is for: every part has one, drawn by the second draw or not. -/
def coverFailures (ps : Array Part) (ss : List Json) : List (String × Json) :=
  ps.toList.filterMap fun p =>
    if !(templates ss).contains p.template then some ("second-cover", Json.mkObj [("template", Json.str p.template)])
    else none

/-- The `{value: root}` statements `ss` read, merged as `Record.sessionRegistered` merges the verdicts': a value read
under two roots maps to null. -/
def statementReads (ss : List Json) : Array (String × Json) :=
  ss.foldl (fun acc j => match Stmt.ofJson? j with
    | some s => (entries (getD s.registered "ports")).foldl (fun acc (_, e) => match getD e "value" with
      | .str v =>
        let root := getD e "root"
        match acc.find? (·.1 == v) with
        | some (_, r) => acc.map fun (k, x) => if k == v then (k, if jeq r root then r else .null) else (k, x)
        | none => acc.push (v, root)
      | _ => acc) acc
    | none => acc) #[]

/-- `second-reported` when the reads the verifiers of record report are not those statements `ss` state. -/
def reportedFailures (own : Own) (ss : List Json) : List (String × Json) :=
  if Record.sameDict (own.reported.getD #[]) (statementReads ss) then [] else [("second-reported", Json.str REPORTED_WHY)]

/-- **The interior record's order**: the replay sample the interior record names (`window.replay_sample`, which the
proof server writes) is the verifier's own, `own.sample`, in its order, each entry a natural number (`nat?`, never a
boolean). -/
def interiorOrderOk (own : Own) : Bool :=
  match getD (getD own.interior "window") "replay_sample" with
  | .arr bs => bs.toList.map nat? == own.sample.toList.map some
  | _ => false

/-- `second-interior-order` unless the interior record names the replay sample in its order (`interiorOrderOk`). -/
def interiorFailures (own : Own) : List (String × Json) :=
  if interiorOrderOk own then [] else [("second-interior-order", Json.str INTERIOR_ORDER_WHY)]

/-- Every failure of statements `ss` over parts `ps`, the interior record's order's first, then, when there is none
and the form isn't allowed, its refusal. -/
def failures (allow : Bool) (own : Own) (ps : Array Part) (ss : List Json) : List (String × Json) :=
  let found := interiorFailures own ++ (List.range ss.length).flatMap (stmtFailures own ps ss) ++
    orderFailures ps ss ++ coverFailures ps ss ++ reportedFailures own ss
  if found.isEmpty && !allow then [("two-stage-form", Json.str FORM_WHY)] else found

/-- The design is derived over parts `ps`: its links (`linksOk`) and its wires (`wiresOk`). -/
def designOk (ps : Array Part) (own : Own) : Bool := linksOk ps own && wiresOk ps own

/-- `second-links` unless the links are derived over parts `ps` (`linksOk`), then `second-wires` unless the wires are
(`wiresOk`). -/
def designFailures (ps : Array Part) (own : Own) : List (String × Json) :=
  (if linksOk ps own then [] else [("second-links", Json.str LINKS_WHY)]) ++
    if wiresOk ps own then [] else [("second-wires", Json.str WIRES_WHY)]

/-- **Round 2's statement check** (module docstring) of statements `stmts`, the two-stage form allowed when `allow`
(`FORM` in the run): each code once, with the detail it first failed with. The interior record's order is checked
before any statement is read (`interiorFailures`). -/
def check (allow : Bool) (own : Own) (stmts : Option (Array Json)) : Check :=
  let found := match own.parts, stmts with
    | none, _ => interiorFailures own ++ [("second-cells", Json.str CELLS_WHY)]
    | some ps, none =>
      interiorFailures own ++ if designOk ps own then [("second-statements", Json.str STATEMENTS_WHY)]
        else designFailures ps own
    | some ps, some ss =>
      if designOk ps own then failures allow own ps ss.toList
      else interiorFailures own ++ designFailures ps own
  found.foldl (fun c x => c.fail x.1 x.2) {}

/-- The interior registration's audit record `rec` with round 2's check `c` as `checks.statements`, failing where `c`
does as a failed check of its own fails it (`Record.record`): not accepted, and no profile, integrity profile or wrong
units. -/
def attach (rec : Json) (c : Check) : Json :=
  let r := rec.setObjVal! "checks" ((getD rec "checks").setObjVal! "statements" c.toJson)
  if c.codes.isEmpty then r
  else (((r.setObjVal! "accepted" (.bool false)).setObjVal! "profile" .null).setObjVal! "integrity_profile" .null).setObjVal!
    "wrong_units" .null

theorem mem_fail (c : Check) (code : String) (d : Json) (x : String) :
    x ∈ (c.fail code d).codes ↔ x ∈ c.codes ∨ x = code := by
  unfold Check.fail
  split
  · rename_i h
    constructor
    · exact Or.inl
    · rintro (hx | rfl)
      · exact hx
      · simpa using h
  · simp

theorem mem_fold (l : List (String × Json)) (c : Check) (x : String) :
    x ∈ (l.foldl (fun c x => c.fail x.1 x.2) c).codes ↔ x ∈ c.codes ∨ ∃ d, (x, d) ∈ l := by
  induction l generalizing c with
  | nil => simp
  | cons a l ih =>
    rw [List.foldl_cons, ih, mem_fail]
    constructor
    · rintro ((h | rfl) | ⟨d, hd⟩)
      · exact Or.inl h
      · exact Or.inr ⟨a.2, by simp⟩
      · exact Or.inr ⟨d, by simp [hd]⟩
    · rintro (h | ⟨d, hd⟩)
      · exact Or.inl (Or.inl h)
      · rcases List.mem_cons.1 hd with rfl | hd
        · exact Or.inl (Or.inr rfl)
        · exact Or.inr ⟨d, hd⟩

/-- A fold of `Check.fail` has no code only over no failure. -/
theorem fold_codes_empty (l : List (String × Json)) :
    (l.foldl (fun c x => c.fail x.1 x.2) ({} : Check)).codes = #[] ↔ l = [] := by
  constructor
  · intro h
    cases l with
    | nil => rfl
    | cons a l =>
      have hm := (mem_fold (a :: l) {} a.1).2 (Or.inr ⟨a.2, by simp⟩)
      rw [h] at hm
      simp at hm
  · rintro rfl
    rfl

theorem failures_ne_nil (own : Own) (ps : Array Part) (ss : List Json) : failures false own ps ss ≠ [] := by
  simp only [failures]
  split
  · simp
  · rename_i h
    intro he
    simp_all

/-- A design not derived fails a code of its own (`second-links` or `second-wires`). -/
theorem designFailures_ne_nil {ps : Array Part} {own : Own} (h : ¬designOk ps own = true) :
    designFailures ps own ≠ [] := by
  revert h
  unfold designOk designFailures
  cases linksOk ps own <;> cases wiresOk ps own <;> simp

/-- **Off, round 2 never passes**: with the two-stage form refused, every run's check fails. -/
theorem check_refused (own : Own) (stmts : Option (Array Json)) : (check false own stmts).codes ≠ #[] := by
  simp only [check, Ne, fold_codes_empty]
  split
  · simp
  · split
    · simp
    · exact fun he => designFailures_ne_nil ‹_› (List.append_eq_nil_iff.1 he).2
  · split
    · simp [failures_ne_nil]
    · exact fun he => designFailures_ne_nil ‹_› (List.append_eq_nil_iff.1 he).2

/-- **By position**: with no `second-order`, statement `i`, one for part `p` and no earlier statement's, is for the
part at its position. -/
theorem orderFailures_nil {ps : Array Part} {ss : List Json} (h : orderFailures ps ss = []) {i : Nat}
    (hi : i < ss.length) {s : Stmt} {p : Part} (hs : Stmt.ofJson? (ss.getD i .null) = some s)
    (hp : ps.find? (·.template == s.template) = some p) (hd : (templates (ss.take i)).contains s.template = false) :
    ps[i]?.map (·.template) = some s.template := by
  have h' := List.filterMap_eq_nil_iff.1 h i (List.mem_range.2 hi)
  simp only [hs] at h'
  by_cases he : ps[i]?.map (·.template) = some s.template
  · exact he
  · have hb : (ps[i]?.map (·.template) != some s.template) = true := bne_iff_ne.2 he
    simp only [hp, hd, hb, Option.isSome_some] at h'
    simp at h'

/-- With no `second-interior-order`, the interior record names the replay sample in its order. -/
theorem interiorFailures_nil {own : Own} (h : interiorFailures own = []) : interiorOrderOk own = true := by
  unfold interiorFailures at h
  split at h
  · assumption
  · simp at h

/-- **A passing `second-interior-order`**: the interior record's `window.replay_sample` is a list whose entries are the
verifier's own replay sample's blocks, in its order. -/
theorem interiorOrderOk_sample {own : Own} (h : interiorOrderOk own = true) :
    ∃ bs, getD (getD own.interior "window") "replay_sample" = .arr bs ∧
      bs.toList.map nat? = own.sample.toList.map some := by
  unfold interiorOrderOk at h
  split at h
  · rename_i bs hb
    exact ⟨bs, hb, by simpa using h⟩
  · simp at h

/-- **What a passing check says**: the form is allowed, the interior record names the replay sample in its order
(`interiorOrderOk_sample`), the interior program's strata make parts, the replay block's Definition makes links, each
for a part's template, and wires, each between parts' templates, each statement is one for a part's template, no
earlier statement's, the part at its position (`orderFailures_nil`), and passes every check at its position
(`stmtCodes_nil`), every part has a statement, and the reads the verifiers of record report are the statements'. -/
theorem check_ok {allow : Bool} {own : Own} {ss : Array Json} (h : (check allow own (some ss)).codes = #[]) :
    allow = true ∧ interiorOrderOk own = true ∧ ∃ ps, own.parts = some ps ∧ linksOk ps own = true ∧
      wiresOk ps own = true ∧
      (∀ i < ss.size, ∃ s p, Stmt.ofJson? (ss.toList.getD i .null) = some s ∧
        ps.find? (·.template == s.template) = some p ∧ (templates (ss.toList.take i)).contains s.template = false ∧
        ps[i]?.map (·.template) = some s.template ∧
        stmtCodes own (byTemplate own.interior) (posOf ps s.template) p s = []) ∧
      (∀ p ∈ ps.toList, (templates ss.toList).contains p.template = true) ∧
      Record.sameDict (own.reported.getD #[]) (statementReads ss.toList) = true := by
  simp only [check, fold_codes_empty] at h
  split at h
  · simp at h
  · split at h
    · simp at h
    · exact absurd (List.append_eq_nil_iff.1 h).2 (designFailures_ne_nil ‹_›)
  · rename_i ps _ hp hs
    cases hs
    by_cases hl : designOk ps own = true
    case neg => rw [if_neg hl] at h; exact absurd (List.append_eq_nil_iff.1 h).2 (designFailures_ne_nil hl)
    rw [if_pos hl] at h
    obtain ⟨hl, hw⟩ : linksOk ps own = true ∧ wiresOk ps own = true := by simpa [designOk] using hl
    simp only [failures] at h
    split at h
    · simp at h
    · rename_i hne
      simp only [h, List.isEmpty_nil, Bool.true_and, Bool.not_eq_true', Bool.not_eq_false] at hne
      simp only [List.append_eq_nil_iff, List.flatMap_eq_nil_iff, List.mem_range, and_assoc] at h
      obtain ⟨hint, hst, hord, hcov, hrep⟩ := h
      refine ⟨by simpa using hne, interiorFailures_nil hint, ps, hp, hl, hw, ?_, ?_, ?_⟩
      · intro i hi
        have hi' : i < ss.toList.length := by simpa using hi
        have := hst i hi'
        simp only [stmtFailures] at this
        split at this
        · simp at this
        · rename_i s hs
          split at this
          · rename_i p hp'
            split at this
            · simp at this
            · rename_i hdup
              have hd : (templates (ss.toList.take i)).contains s.template = false := by simpa using hdup
              exact ⟨s, p, hs, hp', hd, orderFailures_nil hord hi' hs hp' hd, by simpa using this⟩
          · simp at this
      · intro p hp'
        have := List.filterMap_eq_nil_iff.1 hcov p hp'
        split at this
        · simp at this
        · rename_i hc
          simpa using hc
      · unfold reportedFailures at hrep
        split at hrep
        · assumption
        · simp at hrep

/-- A passing statement's checks: an accepted verdict of record checked its header, registered reads and sample, every
one that did checked its session as one table, and its header at its position, population, roots, sample and
registered reads, every port a link or a wire names among them, every port its header shares rows at read registered,
and its registered reads its template's reference statement's but for the run's fields. -/
theorem stmtCodes_nil {own : Own} {named : Bool} {k : Nat} {p : Part} {s : Stmt} (h : stmtCodes own named k p s = []) :
    verifiedOk own s = true ∧ tablesOk own s = true ∧
      headerOk own named k s.template s.header = true ∧ populationOk own.sample.size p s.header = true ∧
      rootsOk named own.interior s.template s.header ((entries (getD s.registered "ports")).map (·.1)) = true ∧
      jeq s.draw (ownDraw own.sample.size p own.drawn) = true ∧
      (entries (getD s.registered "ports")).all (fun (port, e) => readOk own named p port e) = true ∧
      linkedRead own p.template (entries (getD s.registered "ports")) = true ∧
      wiredRead own p.template (entries (getD s.registered "ports")) = true ∧ refsOk s = true ∧
      valuesOk own s = true := by
  simpa [stmtCodes, List.filterMap_eq_nil_iff, and_assoc] using h

/-- The reference header `refOf` gives template `t` is one the auditor's params give it. -/
theorem refOf_mem {own : Own} {t : String} {r : Json} (h : refOf own t = some r) : (t, r) ∈ own.refs := by
  unfold refOf at h
  cases hf : own.refs.find? (·.1 == t) with
  | none => rw [hf] at h; simp at h
  | some e =>
    rw [hf] at h
    obtain ⟨t', r'⟩ := e
    have ht : t' = t := by simpa using Array.find?_some hf
    have hr : r' = r := by simpa using h
    subst ht hr
    exact Array.mem_of_find?_eq_some hf

/-- **A passing `second-header`** names the reference header the auditor's params give template `t`, which header `h`
of a statement for the `k`-th part is, placed in the run (`atRun`), but for the run's fields. -/
theorem headerOk_ref {own : Own} {named : Bool} {k : Nat} {t : String} {h : Json}
    (hh : headerOk own named k t h = true) :
    ∃ r r', (t, r) ∈ own.refs ∧ atRun own named k t r = some r' ∧ jeq (blankRun h) (blankRun r') = true := by
  unfold headerOk at hh
  cases hf : refOf own t with
  | none => simp [hf] at hh
  | some r =>
    cases ha : atRun own named k t r with
    | none => simp [hf, ha] at hh
    | some r' =>
      simp only [hf, ha] at hh
      exact ⟨r, r', refOf_mem hf, ha, hh⟩

/-- **Every header is its template's reference placed in the run** but for the run's fields (`RUN_FIELDS`): in a
passing check, the `i`-th statement is for the template of the part at its position, and its header with those fields
blanked is the reference header the auditor's params give that template placed in the run at the template's position
(`atRun` at `posOf`: its `set` the run's `served/<run>/m<k>`, its `frame_v3.bindings` the verifier's), blanked the same
(`second-header`). -/
theorem check_ok_header {allow : Bool} {own : Own} {ss : Array Json} (h : (check allow own (some ss)).codes = #[]) :
    ∃ ps, own.parts = some ps ∧ ∀ i < ss.size, ∃ s r r', Stmt.ofJson? (ss.toList.getD i .null) = some s ∧
      ps[i]?.map (·.template) = some s.template ∧ (s.template, r) ∈ own.refs ∧
      atRun own (byTemplate own.interior) (posOf ps s.template) s.template r = some r' ∧
      jeq (blankRun s.header) (blankRun r') = true := by
  obtain ⟨-, -, ps, hp, -, -, hst, -, -⟩ := check_ok h
  refine ⟨ps, hp, fun i hi => ?_⟩
  obtain ⟨s, p, hs, -, -, hi', hc⟩ := hst i hi
  obtain ⟨r, r', hr, ha, hj⟩ := headerOk_ref (stmtCodes_nil hc).2.2.1
  exact ⟨s, r, r', hs, hi', hr, ha, hj⟩

/-- **A passing `second-refs`**: every port statement `s`'s header shares rows at is one of its registered reads. -/
theorem refsOk_read {s : Stmt} (h : refsOk s = true) {port : String} {c : Json}
    (hp : (port, c) ∈ entries (getD s.header "shared_rows")) :
    ∃ e, (port, e) ∈ entries (getD s.registered "ports") := by
  dsimp only [refsOk] at h
  have h' : (entries (getD s.registered "ports")).any (·.1 == port) = true :=
    Array.all_eq_true_iff_forall_mem.mp h _ hp
  obtain ⟨⟨q, e⟩, hm, hq⟩ := Array.any_eq_true'.mp h'
  obtain rfl : q = port := by simpa using hq
  exact ⟨e, hm⟩

/-- **Every port a header shares rows at is a read round 2 holds**: in a passing check, every port the `i`-th
statement's header shares rows at (`shared_rows`) is one of its registered reads, each `readOk` at its part: for a port
a link names, the boundary port it names, each slot reading only its cell's replay block's rows; else its own port of
the interior record, each slot reading only its own rows, or a value either registration registers. The session's
verifier holds each instance's ref at such a port to the row at the position the read gives its unit
(`Flock.Registered.check`), so no ref there points a drawn unit at another unit's rows. -/
theorem check_ok_refs {allow : Bool} {own : Own} {ss : Array Json} (h : (check allow own (some ss)).codes = #[]) :
    ∃ ps, own.parts = some ps ∧ ∀ i < ss.size, ∃ s p, Stmt.ofJson? (ss.toList.getD i .null) = some s ∧
      ps.find? (·.template == s.template) = some p ∧ ∀ port c, (port, c) ∈ entries (getD s.header "shared_rows") →
        ∃ e, (port, e) ∈ entries (getD s.registered "ports") ∧ readOk own (byTemplate own.interior) p port e = true := by
  obtain ⟨-, -, ps, hp, -, -, hst, -, -⟩ := check_ok h
  refine ⟨ps, hp, fun i hi => ?_⟩
  obtain ⟨s, p, hs, hf, -, -, hc⟩ := hst i hi
  obtain ⟨-, -, -, -, -, -, hr, -, -, hrefs, -⟩ := stmtCodes_nil hc
  refine ⟨s, p, hs, hf, fun port c hm => ?_⟩
  obtain ⟨e, he⟩ := refsOk_read hrefs hm
  exact ⟨e, he, Array.all_eq_true_iff_forall_mem.mp hr _ he⟩

/-- **A wired read reads its writer unit's value** (one value a gate, one leaf a gate): at a port a wire names and no
link does, a read `readOk` passes is of the wire's writer's output port of the interior record, at that port's root,
domain and leaves, the writer's `Q·n_w` slots' (`pw`, the writer's part, `n_w` its rows' `per`), each slot `u` of part
`p` reading only the row of its unit's writer unit in its cell's replay block (`wiredRow`), as the port's rows give it
(`r`, the verifier's own). -/
theorem readOk_wired {own : Own} {named : Bool} {p : Part} {port : String} {e : Json} {w : Wire}
    (hl : linkOf own p.template port = none) (hw : wireOf own p.template port = some w)
    (h : readOk own named p port e = true) :
    ∃ x pw r, getD e "value" = .str (portOf named w.writer w.wport) ∧
      (portOf named w.writer w.wport, x) ∈ Record.byPort own.interior ∧ writerOf own w = some pw ∧
      rowsOf own p.template port = some r ∧ pw.n = r.per ∧ sameAt e x = true ∧
      placedOk own.sample.size p e (own.sample.size * r.per) (wiredRow p r) = true := by
  unfold readOk at h
  rw [hl, hw] at h
  cases hv : getD e "value" with
  | str v =>
    rw [hv] at h
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨rfl, h⟩ := h
    cases hf : (Record.byPort own.interior).find? (·.1 == portOf named w.writer w.wport) with
    | none => rw [hf] at h; simp at h
    | some kx =>
      obtain ⟨k, x⟩ := kx
      cases hp : writerOf own w with
      | none => rw [hf, hp] at h; simp at h
      | some pw =>
        cases hr : rowsOf own p.template port with
        | none => rw [hf, hp, hr] at h; simp at h
        | some r =>
          rw [hf, hp, hr] at h
          have hk : k = portOf named w.writer w.wport := by simpa using Array.find?_some hf
          subst hk
          simp only [Bool.and_eq_true, beq_iff_eq] at h
          exact ⟨x, pw, r, rfl, Array.mem_of_find?_eq_some hf, rfl, rfl, h.1.2, h.1.1, h.2⟩
  | _ => rw [hv] at h; simp at h

/-- **Every wired port reads its writer unit's value**: in a passing check, the `i`-th statement reads registered every
port a wire names for its template (`wiredRead`), and each of its reads at a port a wire names and no link does is of
that wire's writer's output port of the interior record, each slot at its unit's writer unit's row in its cell's
replay block (`readOk_wired`). -/
theorem check_ok_wired {allow : Bool} {own : Own} {ss : Array Json} (h : (check allow own (some ss)).codes = #[]) :
    ∃ ps, own.parts = some ps ∧ ∀ i < ss.size, ∃ s p, Stmt.ofJson? (ss.toList.getD i .null) = some s ∧
      ps.find? (·.template == s.template) = some p ∧
      wiredRead own s.template (entries (getD s.registered "ports")) = true ∧
      ∀ port e w, (port, e) ∈ entries (getD s.registered "ports") → linkOf own s.template port = none →
        wireOf own s.template port = some w →
        ∃ x pw r, getD e "value" = .str (portOf (byTemplate own.interior) w.writer w.wport) ∧
          (portOf (byTemplate own.interior) w.writer w.wport, x) ∈ Record.byPort own.interior ∧
          writerOf own w = some pw ∧ rowsOf own p.template port = some r ∧ pw.n = r.per ∧ sameAt e x = true ∧
          placedOk own.sample.size p e (own.sample.size * r.per) (wiredRow p r) = true := by
  obtain ⟨-, -, ps, hp, -, -, hst, -, -⟩ := check_ok h
  refine ⟨ps, hp, fun i hi => ?_⟩
  obtain ⟨s, p, hs, hf, -, -, hc⟩ := hst i hi
  obtain ⟨-, -, -, -, -, -, hr, -, hwr, -, -⟩ := stmtCodes_nil hc
  have hpt : p.template = s.template := by simpa using Array.find?_some hf
  rw [hpt] at hwr
  refine ⟨s, p, hs, hf, hwr, fun port e w he hl hw => ?_⟩
  rw [← hpt] at hl hw
  exact readOk_wired hl hw (Array.all_eq_true_iff_forall_mem.mp hr _ he)

/-- The reference reads `refReadsOf` gives template `t` are ones the auditor's params give it. -/
theorem refReadsOf_mem {own : Own} {t : String} {r : Json} (h : refReadsOf own t = some r) :
    (t, r) ∈ own.refReads := by
  unfold refReadsOf at h
  cases hf : own.refReads.find? (·.1 == t) with
  | none => rw [hf] at h; simp at h
  | some e =>
    rw [hf] at h
    obtain ⟨t', r'⟩ := e
    have ht : t' = t := by simpa using Array.find?_some hf
    have hr : r' = r := by simpa using h
    subst ht hr
    exact Array.mem_of_find?_eq_some hf

/-- **A passing `second-values`** names the registered reads the auditor's params give statement `s`'s template's
reference statement, which `s`'s are but for the run's fields. -/
theorem valuesOk_ref {own : Own} {s : Stmt} (h : valuesOk own s = true) :
    ∃ r, (s.template, r) ∈ own.refReads ∧
      jeq (blankReads own s.template s.registered) (blankReads own s.template r) = true := by
  unfold valuesOk at h
  cases hf : refReadsOf own s.template with
  | none => simp [hf] at h
  | some r =>
    simp only [hf] at h
    exact ⟨r, refReadsOf_mem hf, h⟩

/-- **Every statement reads registered what its template's reference statement reads**: in a passing check, the `i`-th
statement's registered reads, each port's run fields blanked (`blankReads`: every port's root, domain and leaves, and
the rows of a port whose rows the run places, `runPlacesRows`), are those the auditor's params give its template's
reference statement, blanked the same (`second-values`). The ports it reads registered, the value each reads, and the
rows at which it reads a value a registration registers are fixed before the run. -/
theorem check_ok_values {allow : Bool} {own : Own} {ss : Array Json} (h : (check allow own (some ss)).codes = #[]) :
    ∀ i < ss.size, ∃ s r, Stmt.ofJson? (ss.toList.getD i .null) = some s ∧ (s.template, r) ∈ own.refReads ∧
      jeq (blankReads own s.template s.registered) (blankReads own s.template r) = true := by
  obtain ⟨-, -, _ps, -, -, -, hst, -, -⟩ := check_ok h
  intro i hi
  obtain ⟨s, p, hs, -, -, -, hc⟩ := hst i hi
  obtain ⟨-, -, -, -, -, -, -, -, -, -, hv⟩ := stmtCodes_nil hc
  obtain ⟨r, hr, hj⟩ := valuesOk_ref hv
  exact ⟨s, r, hs, hr, hj⟩

/-- **A verified entry is a verdict's**: each of `verifiedOf verdicts` is read off an accepted verdict of record among
`verdicts`, its header's digest that verdict's `public_header_sha512`, its registered reads' its `registered_sha512`,
its sample its `unit_draw` and its tables its `session_tables`. -/
theorem mem_verifiedOf {verdicts : Array (String × Json)} {c : Checked} (h : c ∈ verifiedOf verdicts) :
    ∃ x ∈ verdicts, getD x.2 "public_header_sha512" = .str c.header ∧ truthy (getD x.2 "of_record") = true ∧
      jeq (getD x.2 "accepted") (.bool true) = true ∧ c.registered = getD x.2 "registered_sha512" ∧
      c.draw = getD x.2 "unit_draw" ∧ c.tables = getD x.2 "session_tables" := by
  obtain ⟨⟨k, v⟩, hm, hv⟩ := Array.mem_filterMap.mp h
  refine ⟨(k, v), hm, ?_⟩
  dsimp only at hv ⊢
  generalize hg : getD v "public_header_sha512" = g at hv
  cases g with
  | str hdr =>
    dsimp only at hv
    split at hv
    · next ha =>
      cases hv
      simp only [Bool.and_eq_true] at ha
      exact ⟨rfl, ha.1, ha.2, rfl, rfl, rfl⟩
    · cases hv
  | _ => simp at hv

/-- **What `second-verified` says**: a statement passing it is, by what a verdict names of it (`Stmt.checked`), one an
accepted verdict of record checked (`mem_verifiedOf`): its header's digest, its registered reads' and its sample are
that verdict's. -/
theorem verifiedOk_some {own : Own} {s : Stmt} (h : verifiedOk own s = true) :
    ∃ c, s.checked = some c ∧ ∃ v ∈ own.verified, v.header = c.header ∧ jeq v.registered c.registered = true ∧
      jeq v.draw c.draw = true := by
  unfold verifiedOk at h
  cases hc : s.checked with
  | none => simp [hc] at h
  | some c =>
    have h' : ∃ (i : Nat) (_ : i < own.verified.size), (own.verified[i].header = c.header ∧
        jeq own.verified[i].registered c.registered = true) ∧ jeq own.verified[i].draw c.draw = true := by
      simpa [hc, Bool.and_eq_true, beq_iff_eq] using h
    obtain ⟨i, hi, ⟨h1, h2⟩, h3⟩ := h'
    exact ⟨c, rfl, own.verified[i], Array.getElem_mem hi, h1, h2, h3⟩

/-- **What `second-tables` says**: every verified entry that names a statement passing it as `Stmt.checked` does, its
header's digest, its registered reads' and its sample, set its session up as one table (`mem_verifiedOf`: that verdict's
`session_tables` is 1). -/
theorem tablesOk_one {own : Own} {s : Stmt} {c : Checked} (h : tablesOk own s = true) (hc : s.checked = some c)
    {v : Checked} (hv : v ∈ own.verified) (h1 : v.header = c.header) (h2 : jeq v.registered c.registered = true)
    (h3 : jeq v.draw c.draw = true) : jeq v.tables ONE_TABLE = true := by
  unfold tablesOk at h
  simp only [hc] at h
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hv
  have := Array.all_eq_true.1 h i hi
  simpa [h1, h2, h3] using this

private def exRoot (i : Nat) : String := String.ofList (List.replicate 128 (Char.ofNat (48 + i)))

private def exRec (population : Nat) (roots : List (String × Nat × Nat)) (window : Json := Json.mkObj []) : Json :=
  Json.mkObj [("scheme", "frame-v3-sha512/hm96-sha512"), ("population", Lean.toJson population),
    ("program", "program-0"), ("partition", "partition-0"), ("window", window),
    ("roots", Json.arr (roots.toArray.map fun (p, r, d) =>
      Json.mkObj [("port", p), ("root", exRoot r), ("domain", exRoot d), ("leaves", Lean.toJson population)]))]

private def exRun : String := "run-0"

/-- The interior registration's window: the run, and the replay sample `sample` it lays out. -/
private def exWindow (sample : Json) (run : Json := .str exRun) : Json :=
  Json.mkObj [("kind", "served"), ("run", run), ("interior", Json.bool true), ("replay_sample", sample)]

private def exInterior (window : Json := exWindow (natsJson #[1, 3])) : Json :=
  exRec 6 [("x", 1, 2), ("out", 3, 4)] window

private def exSchemas : Json := Json.mkObj [("x", "hm96-sha512/row/v2"), ("out", "hm96-sha512/row/v2")]

/-- Port `p`'s binding in run `exRun`, which the served header names. -/
private def exBinding (p : String) : Json := .str ((portBinding exInterior p "hm96-sha512/row/v2" exRun).getD "")

/-- `exStmt`'s reference header, fixed before the run: its scheme, the set it was staged under, its ports' schemas and
their bindings there, and none of the run's fields. -/
private def exRef : Json :=
  Json.mkObj [("scheme", "frame-v3-sha512/hm96-sha512"), ("set", "unit-Pair_v0"), ("units", Json.mkObj []),
    ("frame_v3", Json.mkObj [("schemas", exSchemas), ("bindings", Json.mkObj [("x", "00"), ("out", "00")])])]

/-- A replay sample of blocks 1 and 3 of 4, three proof blocks of one template in each, units 2 and 4 drawn. -/
private def exOwn : Own :=
  { sample := #[1, 3], parts := parts 2 #[("Pair_v0", #[(0, 6)])], interior := exInterior,
    boundary := exRec 4 [("bx", 5, 6)], drawn := #[2, 4], refs := #[("Pair_v0", exRef)],
    refReads := #[("Pair_v0", .null)] }

private def exStmt (indices units : List Nat) : Json :=
  Json.mkObj [("template", "Pair_v0"), ("registered", .null),
    ("header", Json.mkObj [("scheme", "frame-v3-sha512/hm96-sha512"), ("set", .str (setName exRun 0)),
      ("instances", Lean.toJson indices.length), ("units", Json.mkObj [("indices", natsJson indices.toArray)]),
      ("frame_v3", Json.mkObj [("roots", Json.mkObj [("x", exRoot 1), ("out", exRoot 3)]),
        ("domain_ids", Json.mkObj [("x", exRoot 2), ("out", exRoot 4)]), ("schemas", exSchemas),
        ("bindings", Json.mkObj [("x", exBinding "x"), ("out", exBinding "out")])])]),
    ("draw", Json.mkObj [("law", "subset"), ("population", (6 : Nat)), ("k", Lean.toJson units.length),
      ("units", natsJson units.toArray)])]

/-- `own` with a verifier of record that checked statements `ss`' sessions, naming each one's header, registered reads
and sample (`Stmt.checked`). -/
private def exVerifying (ss : Array Json) (own : Own := exOwn) : Own :=
  { own with verified := ss.filterMap fun j => (Stmt.ofJson? j).bind Stmt.checked }

/-- `own` with the params giving each of statements `ss`' templates the registered reads its statement reads as its
reference statement's, as an honest run's are. -/
private def exReferring (ss : Array Json) (own : Own) : Own :=
  { own with refReads := ss.filterMap fun j => (Stmt.ofJson? j).map fun s => (s.template, s.registered) }

/-- Round 2's check of statements `ss`, each one's session checked by a verifier of record and its template's reference
statement reading registered what it reads. -/
private def exCheck (allow : Bool) (own : Own) (ss : Array Json) : Check :=
  check allow (exVerifying ss (exReferring ss own)) (some ss)

private def exHonest : Json := exStmt [0, 1, 2, 3, 4, 5] [2, 4]

-- Python's `window_binding("program-0", "partition-0", "x", "hm96-sha512/row/v2", 6, "run-0")`
#guard portBinding exInterior "x" "hm96-sha512/row/v2" exRun == some
  "009d0ab56f7953138bd8853c525c88a1ede372e270f1243e8c7b1980de7390130205daa6b324fe330e87a215a663b19a011fd6dbe3391eb47c5e7116ca32192f"
#guard portBinding exInterior "x" "hm96-sha512/row/v2" exRun != portBinding exInterior "out" "hm96-sha512/row/v2" exRun
#guard portBinding exInterior "y" "hm96-sha512/row/v2" exRun == none
#guard setName exRun 2 == "served/run-0/m2"

#guard exOwn.parts == some #[{ template := "Pair_v0", lo := 0, n := 3 }]
#guard (exCheck false exOwn #[exHonest]).codes == #["two-stage-form"]
#guard (exCheck true exOwn #[exHonest]).codes.isEmpty
#guard (exCheck true exOwn #[exStmt [0, 1, 2, 3, 4, 6] [2, 4]]).codes == #["second-population"]
#guard (exCheck true exOwn #[exStmt [0, 1, 2, 3, 4, 5] [1, 4]]).codes == #["second-sample"]
#guard (exCheck true exOwn #[exHonest, exHonest]).codes == #["second-template"]
#guard (check true exOwn (some #[])).codes == #["second-cover"]
#guard (check true exOwn none).codes == #["second-statements"]
#guard (check true { exOwn with parts := parts 4 #[("Pair_v0", #[(0, 6)])] } none).codes == #["second-cells"]

/-- The honest statement, its header naming circuit file `c` by digest. -/
private def exNaming (c : Nat) : Json :=
  exHonest.setObjVal! "header" ((getD exHonest "header").setObjVal! "circuit_sha512" (exRoot c))

/-- `exOwn`, the params giving its template the reference header that names circuit file `c` (`exNaming`). -/
private def exNamed (c : Nat) : Own := { exOwn with refs := #[("Pair_v0", getD (exNaming c) "header")] }

/-- The verdict of record a verifier that checked statement `j`'s session as `tables` tables writes, accepted or not;
naming no table count with none. -/
private def exVerdict (j : Json) (accepted : Bool := true) (tables : Option Nat := some 1) : Json :=
  Json.mkObj ([("of_record", Json.bool true), ("accepted", Json.bool accepted),
    ("public_header_sha512", Json.str ((canonSha512 (getD j "header")).toOption.getD "")),
    ("unit_draw", getD j "draw")] ++ (tables.map fun n => [("session_tables", Lean.toJson n)]).getD [])

#guard (check true exOwn (some #[exHonest])).codes == #["second-verified"]
#guard (check true { exOwn with verified := verifiedOf #[("lean", exVerdict exHonest)] } (some #[exHonest])).codes.isEmpty
#guard (check true { exOwn with verified := verifiedOf #[("lean", exVerdict exHonest false)] } (some #[exHonest])).codes ==
  #["second-verified"]
#guard (verifiedOf #[("lean", (exVerdict exHonest).setObjVal! "of_record" (.bool false))]).isEmpty
#guard (check true (exVerifying #[exNaming 7] (exNamed 7)) (some #[exNaming 7])).codes.isEmpty
#guard (check true (exVerifying #[exNaming 7] (exNamed 8)) (some #[exNaming 8])).codes == #["second-verified"]
#guard (check true (exVerifying #[exStmt [0, 1, 2, 3, 4, 5] [1, 4]]) (some #[exHonest])).codes == #["second-verified"]
#guard (check true { exOwn with verified := verifiedOf #[("lean", exVerdict exHonest (tables := some 2))] }
  (some #[exHonest])).codes == #["second-tables"]
#guard (check true { exOwn with verified := verifiedOf #[("lean", exVerdict exHonest (tables := none))] }
  (some #[exHonest])).codes == #["second-tables"]
#guard (check true { exOwn with verified := verifiedOf #[("a", exVerdict exHonest),
  ("b", exVerdict exHonest (tables := some 2))] } (some #[exHonest])).codes == #["second-tables"]
#guard (check true { exOwn with verified := verifiedOf #[("a", exVerdict exHonest),
  ("b", exVerdict (exNaming 9) (tables := some 2))] } (some #[exHonest])).codes.isEmpty

/-- `exStmt`'s honest statement, its header `f` of the honest one. -/
private def exHeader (f : Json → Json) : Json :=
  let s := exStmt [0, 1, 2, 3, 4, 5] [2, 4]
  s.setObjVal! "header" (f (getD s "header"))

/-- Header `h` with its object `k`'s field `f` set to `v`. -/
private def exIn (k f : String) (v : Json) (h : Json) : Json := h.setObjVal! k ((getD h k).setObjVal! f v)

/-- Round 2's codes, the form allowed, for `exHeader f` alone, its session checked by a verifier of record. -/
private def exCodes (own : Own) (f : Json → Json) : Array String := (exCheck true own #[exHeader f]).codes

#guard (exCodes exOwn id).isEmpty
#guard (exCodes { exOwn with refs := #[("Pair_v0", getD (exHeader id) "header")] } id).isEmpty
#guard (exCodes exOwn (·.setObjVal! "registered" (Json.mkObj [("ports", Json.mkObj [])]))).isEmpty
#guard exCodes exOwn (·.setObjVal! "set" (.str "served/other/m0")) == #["second-header"]
#guard exCodes exOwn (·.setObjVal! "set" (.str (setName exRun 1))) == #["second-header"]
#guard exCodes exOwn (exIn "frame_v3" "bindings" (Json.mkObj [("x", exBinding "x"), ("out", exBinding "x")])) ==
  #["second-header"]
#guard exCodes exOwn (exIn "frame_v3" "bindings" (Json.mkObj [("x", exBinding "x")])) == #["second-header"]
#guard exCodes exOwn (exIn "frame_v3" "schemas" (Json.mkObj [("x", "hm96-sha512/row/v2")])) == #["second-header"]
#guard exCodes { exOwn with interior := exInterior (exWindow (natsJson #[1, 3]) (.str "run-1")) } id ==
  #["second-header"]
#guard exCodes { exOwn with interior := exInterior (exWindow (natsJson #[1, 3]) .null) } id == #["second-header"]
#guard exCodes { exOwn with interior := exInterior (exWindow (natsJson #[3, 1])) } id == #["second-interior-order"]
#guard exCodes { exOwn with interior := exInterior (exWindow (natsJson #[1])) } id == #["second-interior-order"]
#guard exCodes { exOwn with interior := exInterior (exWindow (.arr #[.bool true, Lean.toJson (3 : Nat)])) } id ==
  #["second-interior-order"]
#guard exCodes { exOwn with interior := exInterior (exWindow .null) } id == #["second-interior-order"]
#guard (exCheck false { exOwn with interior := exInterior (exWindow (natsJson #[3, 1])) } #[exHeader id]).codes ==
  #["second-interior-order"]
#guard (check true { exOwn with interior := exInterior (exWindow (natsJson #[3, 1])) } none).codes ==
  #["second-interior-order", "second-statements"]
#guard (check true { exOwn with interior := exInterior (exWindow .null), links := none } none).codes ==
  #["second-interior-order", "second-links"]
#guard (exCodes exOwn (exIn "units" "blocks" (natsJson #[0]))).isEmpty
#guard (exCodes { exOwn with refs := #[("Pair_v0", exIn "units" "blocks" (natsJson #[9]) exRef)] } id).isEmpty
#guard exCodes exOwn (exIn "units" "units_per_vu" (Lean.toJson (2 : Nat))) == #["second-header"]
#guard exCodes exOwn (exIn "frame_v3" "owner" (.str "other")) == #["second-header"]
#guard exCodes exOwn (·.setObjVal! "scheme" (.str "frame-v3-sha512/other")) == #["second-header", "second-roots"]
#guard exCodes exOwn (·.setObjVal! "instances" (Lean.toJson (7 : Nat))) == #["second-population"]
#guard exCodes exOwn (exIn "units" "indices" (natsJson #[1, 2, 3, 4, 5, 6])) == #["second-population"]
#guard exCodes exOwn (exIn "frame_v3" "roots" (Json.mkObj [("x", exRoot 1), ("out", exRoot 5)])) == #["second-roots"]
#guard exCodes exOwn (exIn "frame_v3" "domain_ids" (Json.mkObj [("x", exRoot 2)])) == #["second-roots"]
#guard exCodes { exOwn with refs := #[] } id == #["second-header"]
#guard exCodes { exOwn with refs := #[("Other_v0", exRef)] } id == #["second-header"]
#guard (exCheck false { exOwn with refs := #[] } #[exHeader id]).codes == #["second-header"]

/-- Port `x`'s statement reading `value`'s tree of `leaves` leaves at root `root`, domain `dom`, slot `k` its `reads[k]`. -/
private def exReading (value : String) (root dom leaves : Nat) (reads : List Nat) : Json :=
  (exStmt [0, 1, 2, 3, 4, 5] [2, 4]).setObjVal! "registered" (Json.mkObj [("ports", Json.mkObj [("x", Json.mkObj [
    ("value", value), ("root", exRoot root), ("domain", exRoot dom), ("leaves", Lean.toJson leaves),
    ("reads_from", (0 : Nat)), ("reads", natsJson reads.toArray)])])])

private def exReported (value : String) (root : Nat) (own : Own := exOwn) : Own :=
  { own with reported := some #[(value, .str (exRoot root))] }

/-- `exOwn` with `Pair_v0`'s `x` linked to the boundary's `bx`, one row a block, which its three units a block each
read. -/
private def exLinked : Own :=
  { exOwn with links := some #[⟨"Pair_v0", "x", "bx"⟩], rows := #[⟨"Pair_v0", "x", 1, #[⟨0, 3, 0, 0⟩]⟩] }

/-- The boundary registration's record of population 4, its port `bx` of `leaves` leaves. -/
private def exBoundary (leaves : Nat) : Json :=
  (exRec 4 [("bx", 5, 6)]).setObjVal! "roots" (Json.arr #[Json.mkObj [("port", "bx"), ("root", exRoot 5),
    ("domain", exRoot 6), ("leaves", Lean.toJson leaves)]])

/-- `exLinked` with `bx` of `per` rows a block, its units reading rows `runs`. -/
private def exLinkedAt (per : Nat) (runs : Array Run) : Own :=
  { exLinked with boundary := exBoundary (4 * per), rows := #[⟨"Pair_v0", "x", per, runs⟩] }

/-- Round 2's codes for the statement reading `bx`, of `leaves` leaves, at rows `reads` over `exLinkedAt per runs`, its
`bx` of `held` leaves. -/
private def exAt (per : Nat) (runs : Array Run) (reads : List Nat) (held : Nat := 4 * per) (leaves : Nat := held) :
    Array String :=
  (exCheck true (exReported "bx" 5 { exLinkedAt per runs with boundary := exBoundary held })
    #[exReading "bx" 5 6 leaves reads]).codes

#guard (exCheck true (exReported "x" 1) #[exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]]).codes.isEmpty
#guard (exCheck true exOwn #[exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]]).codes == #["second-reported"]
#guard (exCheck true (exReported "x" 1) #[exReading "x" 1 2 6 [1, 1, 2, 3, 4, 5]]).codes == #["second-read"]
#guard (exCheck true (exReported "out" 3) #[exReading "out" 3 4 6 [0, 1, 2, 3, 4, 5]]).codes == #["second-read"]
#guard (exCheck true (exReported "bx" 5 exLinked) #[exReading "bx" 5 6 4 [1, 1, 1, 3, 3, 3]]).codes.isEmpty
#guard (exCheck true (exReported "bx" 5 exLinked) #[exReading "bx" 5 6 4 [1, 1, 1, 3, 3, 2]]).codes == #["second-read"]
#guard (exCheck true (exReported "bx" 5) #[exReading "bx" 5 6 4 [1, 1, 1, 3, 3, 3]]).codes == #["second-read"]
#guard (exCheck false (exReported "bx" 5 exLinked) #[exReading "bx" 5 6 4 [1, 1, 1, 3, 3, 3]]).codes ==
  #["two-stage-form"]
#guard (exCheck true (exReported "x" 1 exLinked) #[exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]]).codes == #["second-read"]
#guard (exCheck true exLinked #[exHonest]).codes == #["second-read"]
#guard (exCheck true { exOwn with links := none } #[exHonest]).codes == #["second-links"]
#guard (check true { exOwn with links := none } none).codes == #["second-links"]
#guard (check true { exOwn with links := some #[⟨"Other_v0", "x", "bx"⟩] } none).codes == #["second-links"]
-- a link's units each read the row their caller's slice names, of the block's rows: here units 0, 1 and 2 rows 0, 1
-- and 2 of three a block, in blocks 1 and 3; another unit's row, or one past the block's, refused
#guard (exAt 3 #[⟨0, 3, 0, 1⟩] [3, 4, 5, 9, 10, 11]).isEmpty
#guard exAt 3 #[⟨0, 3, 0, 1⟩] [4, 3, 5, 9, 10, 11] == #["second-read"]
#guard exAt 3 #[⟨0, 3, 0, 1⟩] [3, 3, 3, 9, 9, 9] == #["second-read"]
#guard (exAt 3 #[⟨0, 1, 2, 0⟩, ⟨1, 2, 0, 1⟩] [5, 3, 4, 11, 9, 10]).isEmpty
#guard exAt 3 #[⟨0, 3, 1, 1⟩] [4, 5, 6, 10, 11, 12] == #["second-read"]
-- the boundary port holds its population's rows a block, and the read is at its leaves
#guard exAt 3 #[⟨0, 3, 0, 1⟩] [3, 4, 5, 9, 10, 11] (held := 4) == #["second-read"]
#guard exAt 3 #[⟨0, 3, 0, 1⟩] [3, 4, 5, 9, 10, 11] (leaves := 16) == #["second-read"]
-- rows that don't place the part's units, or no rows, make no links
#guard (check true (exLinkedAt 3 #[⟨0, 2, 0, 1⟩]) none).codes == #["second-links"]
#guard (check true (exLinkedAt 3 #[⟨0, 3, 0, 1⟩, ⟨3, 1, 0, 0⟩]) none).codes == #["second-links"]
#guard (check true (exLinkedAt 3 #[⟨1, 3, 0, 1⟩]) none).codes == #["second-links"]
#guard (check true { exLinked with rows := #[] } none).codes == #["second-links"]
#guard (check true { exLinked with rows := #[⟨"Pair_v0", "out", 1, #[⟨0, 3, 0, 0⟩]⟩] } none).codes == #["second-links"]
#guard (check true (exReported "x" 1 (exVerifying #[exReading "x" 1 2 6 [0, 1, 2, 3, 4, 4]]
  (exReferring #[exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]] exOwn)))
  (some #[exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]])).codes == #["second-verified"]

/-- `exOwn`, the interior registration registering values `w1` and `w2`, six leaves each at its own root and domain. -/
private def exRegistering : Own :=
  { exOwn with interior := exInterior.setObjVal! "registered" (Json.arr #[
      Json.mkObj [("value", "w1"), ("root", exRoot 7), ("domain", exRoot 8), ("leaves", Lean.toJson (6 : Nat))],
      Json.mkObj [("value", "w2"), ("root", exRoot 9), ("domain", exRoot 5), ("leaves", Lean.toJson (6 : Nat))]]) }

/-- Round 2's codes for statement `j` alone, its session checked by a verifier of record, the params giving its
template's reference statement the registered reads of statement `r`. -/
private def exValues (own : Own) (r j : Json) : Array String :=
  (check true (exVerifying #[j] (exReferring #[r] own)) (some #[j])).codes

/-- Statement `j`, its port `x`'s registered read's field `k` set to `v`. -/
private def exReadField (k : String) (v : Json) (j : Json) : Json :=
  let r := getD j "registered"
  j.setObjVal! "registered" (r.setObjVal! "ports" (Json.mkObj [("x", (getD (getD r "ports") "x").setObjVal! k v)]))

-- a registered value's rows are the reference's; a linked or own port's the run places
#guard jeq (blankReads exRegistering "Pair_v0" (getD (exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 5]) "registered"))
  (Json.mkObj [("ports", Json.mkObj [("x", Json.mkObj [("value", "w2"), ("root", .null), ("domain", .null),
    ("leaves", .null), ("reads_from", (0 : Nat)), ("reads", natsJson #[0, 1, 2, 3, 4, 5])])])])
#guard jeq (blankReads exOwn "Pair_v0" (getD (exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]) "registered")) (Json.mkObj
  [("ports", Json.mkObj [("x", Json.mkObj [("value", "x"), ("root", .null), ("domain", .null), ("leaves", .null),
    ("reads_from", .null), ("reads", .null)])])])
#guard jeq (blankReads exLinked "Pair_v0" (getD (exReading "bx" 5 6 4 [1, 1, 1, 3, 3, 3]) "registered")) (Json.mkObj
  [("ports", Json.mkObj [("x", Json.mkObj [("value", "bx"), ("root", .null), ("domain", .null), ("leaves", .null),
    ("reads_from", .null), ("reads", .null)])])])
#guard jeq (blankReads exOwn "Pair_v0" .null) .null
-- a port no link names reads a value either registration registers (`second-read`): the one its reference reads, at
-- the run's root, domain and leaves and the reference's rows
#guard (exCheck true (exReported "w2" 9 exRegistering) #[exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 5]]).codes.isEmpty
#guard (exValues (exReported "w2" 9 exRegistering) (exReading "w2" 1 2 3 [0, 1, 2, 3, 4, 5])
  (exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 5])).isEmpty
-- the rows at which it reads a registered value, which `second-read` leaves to the run: refused at any but the
-- reference's, while a linked or own port's rows are the run's to place
#guard exValues (exReported "w2" 9 exRegistering) (exReading "w2" 9 5 6 [5])
  (exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 5]) == #["second-values"]
#guard exValues (exReported "w2" 9 exRegistering) (exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 4])
  (exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 5]) == #["second-values"]
#guard exValues (exReported "w2" 9 exRegistering) (exReadField "reads_from" (3 : Nat)
  (exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 5])) (exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 5]) == #["second-values"]
#guard (exValues (exReported "x" 1) (exReading "x" 1 2 6 [5, 4, 3, 2, 1, 0])
  (exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5])).isEmpty
#guard (exValues (exReported "bx" 5 exLinked) (exReading "bx" 5 6 4 [0, 0, 0, 0, 0, 0])
  (exReading "bx" 5 6 4 [1, 1, 1, 3, 3, 3])).isEmpty
-- another value a registration registers, which `second-read` alone takes, so the run's choice: refused
#guard exValues (exReported "w2" 9 exRegistering) (exReading "w1" 7 8 6 [0, 1, 2, 3, 4, 5])
  (exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 5]) == #["second-values"]
#guard exValues (exReported "w1" 7 exRegistering) (exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 5])
  (exReading "w1" 7 8 6 [0, 1, 2, 3, 4, 5]) == #["second-values"]
#guard (exValues (exReported "x" 1) (exReadField "reads_from" (3 : Nat) (exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]))
  (exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5])).isEmpty
-- reads where the reference reads none, none where it reads one, a field of a port's other than the run's, or no
-- reference reads for the template
#guard exValues (exReported "x" 1) exHonest (exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]) == #["second-values"]
#guard exValues exOwn (exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]) exHonest == #["second-values"]
#guard exValues (exReported "x" 1) (exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5])
  (exReadField "row_schema" "hm96-sha512/row/v1" (exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5])) == #["second-values"]
#guard (check true (exVerifying #[exHonest] { exOwn with refReads := #[] }) (some #[exHonest])).codes ==
  #["second-values"]
#guard (check true (exVerifying #[exHonest] { exOwn with refReads := #[("Other_v0", .null)] })
  (some #[exHonest])).codes == #["second-values"]

/-- Port `p`'s read of the interior record's own port `p` at root `root`, domain `dom`, slot `k` reading row `reads[k]`. -/
private def exOwnRead (p : String) (root dom : Nat) (reads : List Nat := [0, 1, 2, 3, 4, 5]) : String × Json :=
  (p, Json.mkObj [("value", .str p), ("root", exRoot root), ("domain", exRoot dom), ("leaves", Lean.toJson (6 : Nat)),
    ("reads_from", (0 : Nat)), ("reads", natsJson reads.toArray)])

/-- Tables of six rows at `ports`, as a header shares them (`shared_rows`). -/
private def exTables (ports : List String) : Json := Json.mkObj (ports.map fun p => (p, Lean.toJson (6 : Nat)))

/-- The honest statement, its header sharing rows at `shared`, reading `reads` registered (`none`: no registered
reads). -/
private def exSharing (shared : List String) (reads : Option (List (String × Json))) : Json :=
  (exHeader (·.setObjVal! "shared_rows" (exTables shared))).setObjVal! "registered"
    ((reads.map fun rs => Json.mkObj [("ports", Json.mkObj rs)]).getD .null)

/-- `exOwn` with its template's reference sharing rows at `shared`, and the verifiers of record reporting reads of
`read`, each value at its root. -/
private def exShares (shared : List String) (read : List (String × Nat)) : Own :=
  { exOwn with
    refs := #[("Pair_v0", exRef.setObjVal! "shared_rows" (exTables shared))],
    reported := some (read.toArray.map fun (v, r) => (v, .str (exRoot r))) }

-- a header sharing rows at `out` and `x` reads both registered, each slot its own row
#guard (exCheck true (exShares ["out", "x"] [("out", 3), ("x", 1)])
  #[exSharing ["out", "x"] (some [exOwnRead "out" 3 4, exOwnRead "x" 1 2])]).codes.isEmpty
#guard (exCheck true (exShares ["x"] [("x", 1)]) #[exSharing ["x"] (some [exOwnRead "x" 1 2])]).codes.isEmpty
-- `out` shared and not read registered: its refs are bounded only, so one could name another unit's row
#guard (exCheck true (exShares ["out", "x"] [("x", 1)]) #[exSharing ["out", "x"] (some [exOwnRead "x" 1 2])]).codes ==
  #["second-refs"]
#guard (exCheck true (exShares ["out", "x"] []) #[exSharing ["out", "x"] none]).codes == #["second-refs"]
-- a shared port read at another slot's row, which the session's verifier would hold slot 0's ref to
#guard (exCheck true (exShares ["out", "x"] [("out", 3), ("x", 1)])
  #[exSharing ["out", "x"] (some [exOwnRead "out" 3 4, exOwnRead "x" 1 2 [1, 0, 2, 3, 4, 5]])]).codes ==
  #["second-read"]
-- rows shared where the template's reference shares none
#guard (exCheck true (exReported "x" 1) #[exSharing ["out", "x"] (some [exOwnRead "x" 1 2])]).codes ==
  #["second-header", "second-refs"]
-- a session reading registered a port its template's reference statement doesn't, or not one it does
#guard exValues (exReported "x" 1) (exSharing [] (some [exOwnRead "out" 3 4]))
  (exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]) == #["second-values"]
#guard exValues (exReported "x" 1) (exSharing [] (some [exOwnRead "out" 3 4, exOwnRead "x" 1 2]))
  (exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]) == #["second-values"]

/-- A statement for template `t`'s part. -/
private def exFor (t : String) : Json := (exStmt [] []).setObjVal! "template" (.str t)

private def exTwoParts : Array Part := #[⟨"Pair_v0", 0, 3⟩, ⟨"Triple_v0", 6, 1⟩]

#guard [posOf exTwoParts "Pair_v0", posOf exTwoParts "Triple_v0", posOf exTwoParts "Other_v0"] == [0, 1, 2]
#guard (orderFailures exTwoParts [exFor "Pair_v0", exFor "Triple_v0"]).isEmpty
#guard (orderFailures exTwoParts [exFor "Triple_v0", exFor "Pair_v0"]).map (·.1) == ["second-order", "second-order"]
#guard (orderFailures exTwoParts [exFor "Triple_v0"]).map (·.1) == ["second-order"]
#guard (orderFailures exTwoParts [exFor "Pair_v0", exFor "Pair_v0", exFor "Other_v0", .null]).isEmpty
#guard (check true { exOwn with parts := some exTwoParts } (some #[exFor "Triple_v0", exFor "Pair_v0"])).codes.contains
  "second-order"
#guard (exCheck true { exOwn with parts := some exTwoParts, drawn := #[2] } #[exStmt [0, 1, 2, 3, 4, 5] [2]]).codes ==
  #["second-cover"]

private def exArg (p : Bool) (i : Nat) : Flock.Program.Arg :=
  { ty := .value 8, seq := Flock.Program.Seq.ofParts #[.affine p i 0 1] }

private def exNode (form : Flock.Program.Form) (fn : Nat) (args : Array Flock.Program.Arg) : Flock.Program.Node :=
  { form, fn, args, out := .value 8 }

private def exDefn (key : String) (prim : Bool) (params : Array String) (nodes : Array Flock.Program.Node)
    (ret : Array Flock.Program.Part) (rty : Flock.Program.Ty := .value 8) : Flock.Program.Defn :=
  { key, id := key, prim, params := params.map (·, .value 8), ret := rty, gates := 0, paramLeaves := 0, outLeaves := 0,
    body := { params := params.map (·, .value 8), retType := rty, nodes, ret := .ofParts ret, offsets := #[] } }

/-- A template of one parameter `x` (0), calling a primitive (2); a replay block of parameters `x` and `w` (1), of body
`block` returning `ret`. -/
private def exDefs (block : Array Flock.Program.Node) (ret : Array Flock.Program.Part)
    (rty : Flock.Program.Ty := .value 8) : Array Flock.Program.Defn :=
  #[exDefn "Pair_v0" false #["x"] #[exNode .prim 2 #[exArg true 0]] #[.affine false 0 0 1],
    exDefn "Block_v0" false #["x", "w"] block ret rty, exDefn "Xor_v1" true #["a"] #[] #[]]

#guard linksOf (exDefs #[exNode .batch 0 #[exArg true 0]] #[.affine false 0 0 1]) 1 ==
  some #[⟨"Pair_v0", "x", "x"⟩, ⟨"Pair_v0", "o0", "o0"⟩]
/-- An argument reading `parts`. -/
private def exParts (parts : Array Flock.Program.Part) : Flock.Program.Arg :=
  { ty := .value 8, seq := .ofParts parts }

-- a caller reads one aligned row of one parameter alone, never explicit leaves; a block returning none of its leaves
#guard linksOf (exDefs #[exNode .call 0 #[exArg true 1]] #[] (.tuple #[])) 1 == some #[⟨"Pair_v0", "x", "w"⟩]
#guard linksOf (exDefs #[exNode .call 0 #[exParts #[.explicit #[⟨true, 1, 0⟩]]]] #[] (.tuple #[])) 1 == none
#guard linksOf (exDefs #[exNode .call 0 #[exArg true 1]] #[]) 1 == none
#guard linksOf (exDefs #[exNode .call 0 #[exArg false 0]] #[.affine false 0 0 1]) 1 ==
  some #[⟨"Pair_v0", "o0", "o0"⟩]
-- several callers of a template read a linked port alike, of one value
#guard linksOf (exDefs #[exNode .call 0 #[exArg true 0], exNode .call 0 #[exArg true 1]] #[] (.tuple #[])) 1 == none
#guard linkRuns (exDefs #[exNode .call 0 #[exArg true 0], exNode .call 0 #[exArg true 0]] #[] (.tuple #[])) 1 ==
  some #[(⟨"Pair_v0", "x", "x"⟩, 1, #[⟨0, 2, 0, 0⟩])]
#guard linksOf (exDefs #[exNode .call 0 #[exArg true 0], exNode .call 0 #[exArg false 0]] #[] (.tuple #[])) 1 == none
#guard linksOf (exDefs #[exNode .call 0 #[exParts #[.affine true 0 0 1, .affine false 1 0 1]]] #[]) 1 == none
#guard linksOf (exDefs #[exNode .prim 2 #[exArg true 0]] #[]) 1 == none
#guard linksOf (exDefs #[exNode .batch 0 #[exArg true 0]] #[.affine true 1 0 1]) 1 == none
#guard linksOf (exDefs #[exNode .batch 0 #[exArg true 0]] #[.affine false 0 0 1] (.tuple #[.value 8, .value 8])) 1 ==
  none
#guard linksOf (exDefs #[exNode .batch 0 #[exArg true 0]] #[.affine false 0 0 1]) 2 == none

/-- `exDefs` with a second template (3), the replay block calling `Pair_v0` on `x` and it on `w`, returning `ret` of
type `rty`. -/
private def exTwo (ret : Array Flock.Program.Part) (rty : Flock.Program.Ty) : Array Flock.Program.Defn :=
  (exDefs #[exNode .call 0 #[exArg true 0], exNode .call 3 #[exArg true 1]] ret rty).push
    (exDefn "Triple_v0" false #["x"] #[exNode .prim 2 #[exArg true 0]] #[.affine false 0 0 1])

#guard linksOf (exTwo #[.affine false 0 0 1, .affine false 1 0 1] (.tuple #[.value 8, .value 8])) 1 ==
  some #[⟨"Pair_v0", "x", "x"⟩, ⟨"Triple_v0", "x", "w"⟩, ⟨"Pair_v0", "o0", "o0"⟩, ⟨"Triple_v0", "o0", "o1"⟩]
#guard linksOf (exTwo #[.affine false 1 0 1, .affine false 0 0 1] (.record #[("a", .value 8), ("b", .value 8)])) 1 ==
  some #[⟨"Pair_v0", "x", "x"⟩, ⟨"Triple_v0", "x", "w"⟩, ⟨"Triple_v0", "o0", "o0"⟩, ⟨"Pair_v0", "o0", "o1"⟩]
#guard linksOf (exTwo #[.affine false 0 0 1, .affine false 0 0 1] (.tuple #[.value 8, .value 8])) 1 == none
#guard linksOf (exTwo #[.affine false 0 0 1, .affine true 1 0 1] (.tuple #[.value 8, .value 8])) 1 == none
#guard linksOf (exTwo #[.affine false 0 0 1, .affine false 1 0 1] (.tuple #[.value 8, .array 2 (.value 8)])) 1 == none
#guard linksOf (exTwo #[.affine false 0 0 1] (.tuple #[.value 8, .value 8])) 1 == none

/-- `exDefs` with the replay block's parameter `x` an array of `n` leaves, its body `block` returning `ret` of type
`rty`. -/
private def exDefsN (n : Nat) (block : Array Flock.Program.Node) (ret : Array Flock.Program.Part)
    (rty : Flock.Program.Ty) : Array Flock.Program.Defn :=
  (exDefs block ret rty).modify 1 fun d =>
    let ps : Array (String × Flock.Program.Ty) := #[("x", .array n (.value 8)), ("w", .value 8)]
    { d with params := ps, body := { d.body with params := ps } }

/-- `Pair_v0` called on leaf `i` of the replay block's `x`. -/
private def exPairAt (i : Nat) : Flock.Program.Node := exNode .call 0 #[exParts #[.affine true 0 i 1]]

/-- `exDefsN 2` calling `Pair_v0` on `x`'s two leaves in order and returning nodes `ks`' outputs as its two leaves. -/
private def exReturning (ks : List Nat) : Array Flock.Program.Defn :=
  exDefsN 2 #[exPairAt 0, exPairAt 1] (ks.toArray.map fun k => .affine false k 0 1) (.array 2 (.value 8))

-- each caller's units read the row of the parameter its slice is, and a template returned whole links its `o0` to
-- its field's rows, in its units' order; a caller returned twice, or out of order, makes none
#guard linkRuns (exReturning [0, 1]) 1 ==
  some #[(⟨"Pair_v0", "x", "x"⟩, 2, #[⟨0, 2, 0, 1⟩]), (⟨"Pair_v0", "o0", "o0"⟩, 2, #[⟨0, 2, 0, 1⟩])]
#guard linkRuns (exDefsN 3 #[exPairAt 2, exPairAt 0, exPairAt 1] #[] (.tuple #[])) 1 ==
  some #[(⟨"Pair_v0", "x", "x"⟩, 3, #[⟨0, 1, 2, 0⟩, ⟨1, 2, 0, 1⟩])]
#guard linksOf (exReturning [1, 0]) 1 == none
#guard linksOf (exReturning [0, 0]) 1 == none
-- a batch's members read one row each, an outer axis mapped, or the one row a shared argument is
#guard linkRuns (exDefsN 2 #[{ exNode .batch 0 #[{ exParts #[.affine true 0 0 2] with dims := some (2, 1) }] with
    n := 2 }] #[] (.tuple #[])) 1 == some #[(⟨"Pair_v0", "x", "x"⟩, 2, #[⟨0, 2, 0, 1⟩])]
#guard linkRuns (exDefsN 2 #[{ exNode .batch 0 #[exParts #[.affine true 0 1 1]] with n := 2 }] #[] (.tuple #[])) 1 ==
  some #[(⟨"Pair_v0", "x", "x"⟩, 2, #[⟨0, 2, 1, 0⟩])]
#guard linksOf (exDefsN 2 #[{ exNode .batch 0 #[{ exParts #[.affine true 0 0 1] with dims := some (2, 1) }] with
    n := 2 }] #[] (.tuple #[])) 1 == none
#guard rowAt #[⟨0, 1, 2, 0⟩, ⟨1, 2, 0, 1⟩] 0 == some 2 && rowAt #[⟨0, 1, 2, 0⟩, ⟨1, 2, 0, 1⟩] 2 == some 1 &&
  rowAt #[⟨0, 1, 2, 0⟩, ⟨1, 2, 0, 1⟩] 3 == none
#guard fits #[⟨0, 1, 2, 0⟩, ⟨1, 2, 0, 1⟩] 3 && !fits #[⟨0, 1, 2, 0⟩, ⟨1, 2, 0, 1⟩] 2 && !fits #[⟨1, 2, 0, 1⟩] 2 &&
  !fits #[⟨0, 0, 0, 0⟩, ⟨0, 2, 0, 1⟩] 2

private def exProg : Flock.Program.Prog :=
  { defs := exDefs #[exNode .batch 0 #[exArg true 0]] #[.affine false 0 0 1], root := 1, programFn := "Block_v0",
    index := Std.HashMap.ofList [("Pair_v0", 0), ("Block_v0", 1), ("Xor_v1", 2)], sha512 := ByteArray.empty }

#guard replayLinks exProg "Block_v0" (exRec 4 [("Block_v0/x", 5, 6)]) ==
  some #[⟨"Pair_v0", "x", "Block_v0/x"⟩, ⟨"Pair_v0", "o0", "Block_v0/o0"⟩]
#guard replayLinks exProg "Block_v0" (exRec 4 [("x", 5, 6)]) ==
  some #[⟨"Pair_v0", "x", "x"⟩, ⟨"Pair_v0", "o0", "o0"⟩]
#guard replayLinks exProg "Other_v0" (exRec 4 [("x", 5, 6)]) == none

/-- `exDefs` with a second template (3), the replay block calling `Pair_v0` on `x` and it on `arg`, which reads
`Pair_v0`'s node, returning its output. -/
private def exChain (arg : Flock.Program.Arg) : Array Flock.Program.Defn :=
  (exDefs #[exNode .call 0 #[exArg true 0], exNode .call 3 #[arg]] #[.affine false 1 0 1]).push
    (exDefn "Triple_v0" false #["x"] #[exNode .prim 2 #[exArg true 0]] #[.affine false 0 0 1])

-- a template's parameter that is another's whole output is a wire from it; a link reads only a parameter
#guard wiresOf (exChain (exArg false 0)) 1 == some #[⟨"Triple_v0", "x", "Pair_v0", "o0"⟩]
#guard linksOf (exChain (exArg false 0)) 1 == some #[⟨"Pair_v0", "x", "x"⟩, ⟨"Triple_v0", "o0", "o0"⟩]
#guard wiresOf (exChain (exArg true 1)) 1 == some #[]
#guard wiresOf (exDefs #[exNode .batch 0 #[exArg true 0]] #[.affine false 0 0 1]) 1 == some #[]
-- part of a row, a row twice, explicit rows, a node with a parameter, a node no template's, or no replay block's
#guard wiresOf (exChain (exParts #[.affine false 0 1 1])) 1 == none
#guard wiresOf (exChain (exParts #[.affine false 0 0 1, .affine false 0 0 1])) 1 == none
#guard wiresOf (exChain (exParts #[.explicit #[⟨false, 0, 0⟩]])) 1 == none
#guard wiresOf (exChain (exParts #[.affine false 0 0 1, .affine true 1 0 1])) 1 == none
#guard wiresOf (exDefs #[exNode .prim 2 #[exArg true 0], exNode .call 0 #[exArg false 0]] #[]) 1 == none
#guard wiresOf (exChain (exArg false 0)) 2 == none
private def exChainProg : Flock.Program.Prog :=
  { exProg with
    defs := exChain (exArg false 0)
    index := Std.HashMap.ofList [("Pair_v0", 0), ("Block_v0", 1), ("Xor_v1", 2), ("Triple_v0", 3)] }

#guard replayWires exChainProg "Block_v0" == some #[⟨"Triple_v0", "x", "Pair_v0", "o0"⟩]
#guard replayWires exProg "Block_v0" == some #[]
#guard replayWires exProg "Other_v0" == none
#guard replayRows exChainProg "Block_v0" == #[⟨"Pair_v0", "x", 1, #[⟨0, 1, 0, 0⟩]⟩,
  ⟨"Triple_v0", "o0", 1, #[⟨0, 1, 0, 0⟩]⟩, ⟨"Triple_v0", "x", 1, #[⟨0, 1, 0, 0⟩]⟩]
#guard replayRows exProg "Other_v0" == #[]

/-- `exDefs` with `Triple_v0` (3) and a template `Two_v0` (4) returning two fields, the replay block calling `Two_v0` on
`x` and `Triple_v0` on `arg`, returning `Triple_v0`'s output. -/
private def exFields (arg : Flock.Program.Arg) : Array Flock.Program.Defn :=
  ((exDefs #[{ exNode .call 4 #[exArg true 0] with out := .tuple #[.value 8, .value 8] }, exNode .call 3 #[arg]]
      #[.affine false 1 0 1]).push
    (exDefn "Triple_v0" false #["x"] #[exNode .prim 2 #[exArg true 0]] #[.affine false 0 0 1])).push
    (exDefn "Two_v0" false #["x"] #[exNode .prim 2 #[exArg true 0], exNode .prim 2 #[exArg true 0]]
      #[.affine false 0 0 1, .affine false 1 0 1] (.tuple #[.value 8, .value 8]))

/-- `exDefs` with `Triple_v0` (3), the replay block calling `Pair_v0` on `x` twice and `Triple_v0` on `arg`, returning
`Triple_v0`'s output. -/
private def exTwice (arg : Flock.Program.Node) : Array Flock.Program.Defn :=
  (exDefs #[exNode .call 0 #[exArg true 0], exNode .call 0 #[exArg true 0], arg] #[]
      (.tuple #[])).push
    (exDefn "Triple_v0" false #["x"] #[exNode .prim 2 #[exArg true 0]] #[.affine false 0 0 1])

-- a parameter that is one field of a writer's output wires that field, `o<f>`; one of several callers' outputs wires
-- its unit's row, at every member of a batch reading it alike
#guard wiresOf (exFields (exParts #[.affine false 0 1 1])) 1 == some #[⟨"Triple_v0", "x", "Two_v0", "o1"⟩]
#guard wiresOf (exFields (exParts #[.affine false 0 0 1])) 1 == some #[⟨"Triple_v0", "x", "Two_v0", "o0"⟩]
#guard wireRuns (exTwice (exNode .call 3 #[exArg false 1])) 1 ==
  some #[(⟨"Triple_v0", "x", "Pair_v0", "o0"⟩, 2, #[⟨0, 1, 1, 0⟩])]
#guard wireRuns (exTwice { exNode .batch 3 #[exArg false 0] with n := 2 }) 1 ==
  some #[(⟨"Triple_v0", "x", "Pair_v0", "o0"⟩, 2, #[⟨0, 2, 0, 0⟩])]
/-- `exTwice` with `Pair_v0`'s first caller a batch of two members. -/
private def exBatchWriter (arg : Flock.Program.Node) : Array Flock.Program.Defn :=
  (exTwice arg).modify 1 fun d => { d with body := { d.body with nodes := d.body.nodes.modify 0 fun x =>
    { x with form := .batch, n := 2, out := .array 2 (.value 8) } } }

-- a writer's units count its batches' members
#guard wireRuns (exBatchWriter (exNode .call 3 #[exArg false 1])) 1 ==
  some #[(⟨"Triple_v0", "x", "Pair_v0", "o0"⟩, 3, #[⟨0, 1, 2, 0⟩])]
-- both fields, a batch's member, a batched argument, or a gather of two writers' outputs, none
#guard wiresOf (exFields (exParts #[.affine false 0 0 2])) 1 == none
#guard wiresOf (exBatchWriter (exNode .call 3 #[exArg false 0])) 1 == none
#guard wiresOf (exTwice { exNode .batch 3 #[{ exArg false 0 with dims := some (1, 1) }] with n := 1 }) 1 == none
#guard wiresOf (exTwice (exNode .call 3 #[exParts #[.affine false 0 0 1, .affine false 1 0 1]])) 1 == none

private def exWire : Wire := ⟨"Triple_v0", "x", "Pair_v0", "out"⟩

#guard wiresOf? (Json.arr #[exWire.toJson]) == some #[exWire]
#guard wiresOf? (Json.arr #[exWire.toJson.setObjVal! "value" (.str "x")]) == none
#guard wiresOf? (Json.arr #[exWire.toJson.setObjVal! "wport" (Lean.toJson (0 : Nat))]) == none
#guard wiresOf? .null == none

/-- An interior record naming its ports by template, six leaves each: `Pair_v0`'s `x` and `out` and `Triple_v0`'s
`out`, and no tree of `Triple_v0`'s wired `x`. -/
private def exWiredInterior : Json :=
  exRec 6 [("Pair_v0/x", 1, 2), ("Pair_v0/out", 3, 4), ("Triple_v0/out", 5, 6)] (exWindow (natsJson #[1, 3]))

/-- `exTwoParts` over `exWiredInterior`, `Triple_v0`'s `x` wired from `Pair_v0`'s `out`, its unit reading its writer's
first unit of three a block. -/
private def exWired : Own :=
  { exOwn with parts := some exTwoParts, interior := exWiredInterior, wires := some #[exWire],
    rows := #[⟨"Triple_v0", "x", 3, #[⟨0, 1, 0, 0⟩]⟩] }

/-- `exWired`, `Triple_v0`'s unit reading its writer's unit `j`, of `per` a block. -/
private def exWiredAt (per j : Nat) : Own := { exWired with rows := #[⟨"Triple_v0", "x", per, #[⟨0, 1, j, 0⟩]⟩] }

private def exTriple : Part := ⟨"Triple_v0", 6, 1⟩

/-- A read of value `v` at root `root` and domain `dom`, slot `6 + k` reading row `reads[k]`. -/
private def exWiredRead (v : String) (root dom : Nat) (reads : List Nat) : Json :=
  Json.mkObj [("value", .str v), ("root", exRoot root), ("domain", exRoot dom), ("leaves", Lean.toJson (6 : Nat)),
    ("reads_from", (6 : Nat)), ("reads", natsJson reads.toArray)]

-- a wired port reads its writer's output value, each slot the row of its writer's unit its rows name, in its replay
-- block
#guard readOk exWired true exTriple "x" (exWiredRead "Pair_v0/out" 3 4 [0, 3])
#guard readOk exWired true exTriple "x" (exWiredRead "Pair_v0/out" 3 4 [0])
#guard readOk (exWiredAt 3 2) true exTriple "x" (exWiredRead "Pair_v0/out" 3 4 [2, 5])
#guard !readOk (exWiredAt 3 2) true exTriple "x" (exWiredRead "Pair_v0/out" 3 4 [0, 3])
#guard !readOk (exWiredAt 3 3) true exTriple "x" (exWiredRead "Pair_v0/out" 3 4 [3, 6])
#guard !readOk (exWiredAt 2 1) true exTriple "x" (exWiredRead "Pair_v0/out" 3 4 [1, 3])
#guard !readOk { exWired with rows := #[] } true exTriple "x" (exWiredRead "Pair_v0/out" 3 4 [0, 3])
-- its own port, which has no tree, another of its writer's rows, another port, or at another root
#guard !readOk exWired true exTriple "x" (exWiredRead "Triple_v0/x" 3 4 [0, 3])
#guard !readOk exWired true exTriple "x" (exWiredRead "Pair_v0/out" 3 4 [1, 3])
#guard !readOk exWired true exTriple "x" (exWiredRead "Pair_v0/out" 3 4 [0, 4])
#guard !readOk exWired true exTriple "x" (exWiredRead "Pair_v0/x" 1 2 [0, 3])
#guard !readOk exWired true exTriple "x" (exWiredRead "Pair_v0/out" 5 4 [0, 3])
-- no wire: the port reads only its own
#guard !readOk { exWired with wires := some #[] } true exTriple "x" (exWiredRead "Pair_v0/out" 3 4 [0, 3])
#guard !readOk { exWired with parts := some #[exTriple] } true exTriple "x" (exWiredRead "Pair_v0/out" 3 4 [0, 3])
-- a wired port not read
#guard wiredRead exWired "Triple_v0" #[("x", .null)]
#guard !wiredRead exWired "Triple_v0" #[("out", .null)]
#guard wiredRead exWired "Pair_v0" #[]

/-- The binding round 2 gives `Triple_v0`'s `x`, of row schema v2, placed in the run (`atRun`). -/
private def exWiredBinding (own : Own) : Option String := do
  let r ← atRun own true 1 "Triple_v0"
    (Json.mkObj [("frame_v3", Json.mkObj [("schemas", Json.mkObj [("x", "hm96-sha512/row/v2")])])])
  let .str b := getD (getD (getD r "frame_v3") "bindings") "x" | none
  pure b

-- a wired port is bound by its writer's port, which holds it
#guard heldPort exWired true "Triple_v0" "x" == "Pair_v0/out"
#guard heldPort exWired true "Triple_v0" "out" == "Triple_v0/out"
#guard (exWiredBinding exWired).isSome
#guard exWiredBinding exWired == portBinding exWiredInterior "Pair_v0/out" "hm96-sha512/row/v2" exRun
#guard exWiredBinding { exWired with wires := some #[] } == none

#guard (check true exWired none).codes == #["second-statements"]
#guard (check true { exWired with wires := none } none).codes == #["second-wires"]
#guard (check true { exWired with wires := some #[{ exWire with template := "Other_v0" }] } none).codes ==
  #["second-wires"]
#guard (check true { exWired with wires := some #[{ exWire with writer := "Other_v0" }] } none).codes ==
  #["second-wires"]
#guard (check true { exWired with wires := none, links := none } none).codes == #["second-links", "second-wires"]
#guard (check true { exWired with wires := none } (some #[])).codes == #["second-wires"]
-- rows that don't place the reader's units, or its writer's, make no wires
#guard (check true { exWired with rows := #[] } none).codes == #["second-wires"]
#guard (check true (exWiredAt 2 1) none).codes == #["second-wires"]
#guard (check true { exWired with rows := #[⟨"Triple_v0", "x", 3, #[⟨0, 2, 0, 0⟩]⟩] } none).codes == #["second-wires"]
#guard (check true (exWiredAt 3 2) none).codes == #["second-statements"]

end Spot.SecondStage
