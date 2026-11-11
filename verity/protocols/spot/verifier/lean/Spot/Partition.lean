/-
The partition as a named query over the verifier's own program (`one_stage/partition.py`): core's `verity/partition/v1`
object and its digest (Flock's `HmRow.checkPartitionObject` and `HmRow.partitionDigest`), and the queries one-stage
applies to the program: the template queries (Flock's `TemplateQuery`), `Q_word` v1 at `X = 16` (`Flock.Extract`) and
`Q_call` v1 at `X ≤ 16` (`Flock.Qcall`), each on a topological program only, as core's `evaluate` refuses any other.
Every unit's committed bits fit a public size class (`UNIT_CLASSES`, powers of two from 16 to `UNIT_CLASS_MAX` bits,
provisional), its class the least that holds them (`unitClass`), a public property of its Definition: a program with a unit
wider than the widest class is refused (a template instance's bits are its distinct returned gates', `templateBits`; a
`Q_call` unit's count the constants its activation holds, `committedBits`, which `Q_call`'s width rule leaves out; a
`Q_word` unit's are its cut's), and so is a program with a template of no gate. The verifier holds its own copy of the
program and applies the query itself; it never takes a population or a unit index on the prover's word.
-/
module

public import Flock.HmRow
public import Flock.Registered
public import Spot.Json
public import Spot.Program

@[expose] public section

namespace Spot.Partition

open Lean (Json)
open Flock (V)
open Flock.TemplateQuery (TRange)

/-- `partition_object.QUERIES`, the queries a partition may name. Flock's form check leaves the names open. -/
def QUERIES : List (String × Nat) :=
  [("Q_word", 1), ("Q_word", 2), Flock.HmRow.Q_CALL, ("Q_template_instance", 0), ("Q_template_instances", 0)]

/-- `partition_object.validate`: Flock's form check, narrowed to core's queries, with `Q_word`'s `X` and `W` at most
`PARAM_MAX` as core's are. -/
def valid (obj : Json) : Bool :=
  match Flock.HmRow.checkPartitionObject obj with
  | .error _ => false
  | .ok (name, version) =>
    let params := getD (getD obj "query") "params"
    QUERIES.contains (name, version) &&
      (name != "Q_word" || ["X", "W"].all fun k => (nat? (getD params k)).any (· ≤ Flock.Qcall.PARAM_MAX))

/-- `partition_object.query()`, the query `build` falls back to for an empty one (Python's `q or query()`). -/
def defaultQuery : Json :=
  Json.mkObj [("name", "Q_word"), ("version", Lean.toJson (1 : Nat)),
    ("params", Json.mkObj [("X", Lean.toJson (16 : Nat)), ("W", Lean.toJson (32 : Nat))])]

/-- `partition_object.build`: the object naming the partition of the program (its SHA-512, hex) as the query. -/
def build (program : String) (query : Json) : Json :=
  Json.mkObj [("format", Flock.HmRow.PARTITION_V1), ("program", program),
    ("query", if keys? query == some #[] then defaultQuery else query)]

/-- `partition.digest`: core's digest of a valid object, or of a pre-canonical `{program, query}` (the envelope's rule);
`none` for anything else (Python's `ValueError`). -/
def digest (obj : Json) : Option String :=
  if keys? obj == some #["program", "query"] || valid obj then
    (Flock.HmRow.partitionDigest obj).toOption.map Flock.hexOf
  else none

/-- `Q_word` v1's `X`, and the most `Q_call` v1's `X` may be: the width rule of the queries applied here. -/
def QUERY_X : Nat := 16

/-- The widest public size class of a unit's committed bits (the claims' `Claims.Sampling.unitClassMax`,
provisional). 16 sets the classes back to the former literal limit alone. -/
def UNIT_CLASS_MAX : Nat := 262144

/-- The public size classes, ascending: the powers of two from 16 to `UNIT_CLASS_MAX` (the claims' `sizeClasses`). -/
def UNIT_CLASSES : List Nat := (List.range (Nat.log2 (UNIT_CLASS_MAX / 16) + 1)).map (16 * 2 ^ ·)

/-- `partition.unit_class`: the size class of a unit that commits `bits` bits, the least of `UNIT_CLASSES` that holds
them; `none` when none does. -/
def unitClass (bits : Nat) : Option Nat := UNIT_CLASSES.find? (bits ≤ ·)

/-- `partition._call_ranges`: the ranges of the Calls `(root-body index, Definition id, units)`, numbered by Call from
`0`, one per Call with units. -/
def callRanges (calls : Array (Nat × String × Nat)) : Array TRange :=
  (calls.foldl (fun (acc : Array TRange × Nat) (x : Nat × String × Nat) =>
    if x.2.2 == 0 then acc else (acc.1.push ⟨x.1, x.2.1, acc.2, x.2.2⟩, acc.2 + x.2.2)) (#[], 0)).1

/-- `Q_word` v1's Calls with their cuts at `X = 16` (`partition._word_calls`): refused when a Definition's unit commits
bits no public class holds (`unitClass`, `Flock.Partition.boundaryWidths` on its Call's graph). -/
def wordCalls (P : Flock.Program.Prog) : V (Array (Nat × Nat × Flock.Qword.DCut)) := do
  let calls ← Flock.Extract.evaluate P QUERY_X
  let mut seen : Array Nat := #[]
  for (_, d, c) in calls do
    if seen.contains d then continue
    seen := seen.push d
    let gr ← Flock.Extract.graph P d (full := false)
    if c.owner.size != gr.g.n then throw s!"query-refused: {(P.defs[d]!).key}'s cut leaves a computing gate out"
    let (ob, _) := Flock.Partition.boundaryWidths gr.g c.owner
    if let some u := (List.range ob.size).find? (fun u => (unitClass ob[u]!).isNone) then
      throw s!"query-refused: unit {u} of {(P.defs[d]!).key} commits {ob[u]!} bits, more than the widest class, {UNIT_CLASS_MAX}"
  return calls

/-- `partition._committed_bits`: per unit `[lo, hi)` of one activation of a Definition, laid out as `G`
(`Flock.Qcall.gates`), the bits it commits: the summed declared widths of its gates that are Input gates, or that a later
gate of the activation reads or the activation returns (`Gates.readTo`), constants included. Nothing outside an activation
reads its gates but through its returns, so in any program this bounds the claims' `Prog.Boundary` count, as `Q_call`'s
width rule bounds its computing gates'. `committedBitsRef` is the walk of its own it was, which the difftest holds it to. -/
def committedBits (G : Flock.Qcall.Gates) (units : Array (Nat × Nat)) : Array Nat :=
  units.map fun (a, b) => (List.range (b - a)).foldl (fun bits i =>
    if Flock.Program.isInputId G.prim[a + i]! || G.readTo[a + i]! ≥ b then bits + G.width[a + i]! else bits) 0

/-- `partition._template_bits`: the bits one instance of template `d` commits, the summed widths, as its return type gives
them, of the distinct gates of its activation that it returns. Decoding keeps Input gates to the root's body, and nothing
outside an instance reads its gates but through its returns, so this is `committedBits` of the whole activation, read off
the return's references alone rather than the activation's gates. -/
def templateBits (P : Flock.Program.Prog) (d : Nat) : V Nat := do
  let defn := P.defs[d]!
  let ws := defn.ret.widths
  if defn.prim then return ws.foldl (· + ·) 0
  let lo := defn.paramLeaves
  let starts := (defn.params.foldl (fun (acc : Array Nat × Nat) (_, t) => (acc.1.push acc.2, acc.2 + t.leaves)) (#[], 0)).1
  let top : Flock.Program.Scope := .top d lo (some starts)
  let mut own : Array (Nat × Nat) := #[]
  for j in [0:min ws.size defn.body.ret.count] do
    let g ← Flock.Program.resolve P top (defn.body.ret.get j)
    if lo ≤ g && g < lo + defn.gates then own := own.push (g, ws[j]!)
  let sorted := own.qsort (fun a b => a.1 < b.1)
  return (List.range sorted.size).foldl (fun bits i =>
    if i > 0 && sorted[i - 1]!.1 == sorted[i]!.1 then bits else bits + sorted[i]!.2) 0

/-- `partition._check_units` on a template query's ranges: refused for a template of no gate, or one whose instance commits
bits no public class holds (`unitClass`, `templateBits`), each template once. -/
def checkTemplates (P : Flock.Program.Prog) (rs : Array TRange) : V Unit := do
  let root := (P.defs[P.root]!).body
  let mut seen : Array String := #[]
  for r in rs do
    if r.count == 0 || seen.contains r.template then continue
    seen := seen.push r.template
    let d := root.nodes[r.node]!.fn
    if (P.defs[d]!).gates == 0 then throw s!"query-refused: template {r.template} has no gate"
    let bits ← templateBits P d
    if (unitClass bits).isNone then
      throw s!"query-refused: an instance of {r.template} commits {bits} bits, more than the widest class, {UNIT_CLASS_MAX}"

/-- `partition._check_units` on `Q_call`'s Calls: refused when a unit commits bits no public class holds (`unitClass`,
`committedBits`, constants counted, which `Q_call`'s width rule leaves out), each Definition once. -/
def checkCalls (P : Flock.Program.Prog) (cs : Array Flock.Qcall.Call) : V Unit := do
  let mut seen : Array Nat := #[]
  for c in cs do
    if seen.contains c.d then continue
    seen := seen.push c.d
    let bits := committedBits c.c.gates (c.c.units.map fun u => (u.lo, u.hi))
    if let some u := (List.range bits.size).find? (fun u => (unitClass bits[u]!).isNone) then
      throw s!"query-refused: unit {u} of {(P.defs[c.d]!).key} commits {bits[u]!} bits, more than the widest class, {UNIT_CLASS_MAX}"

/-- The `Q_word` or `Q_call` query's Calls with their units, as `(root-body index, Definition id, units)`: `Q_word` v1 at
`X = 16` (`wordCalls`) or `Q_call` v1 at `X ≤ 16` (any `W` core accepts, its units `checkCalls`'); `query-refused` for
any other parameters or query, and where the query is inapplicable. -/
def calls (p : Program) (name : String) (version : Nat) (params : Json) : V (Array (Nat × String × Nat)) := do
  let X := (nat? (getD params "X")).getD 0
  let key := fun d => (p.prog.defs[d]!).key
  if name == "Q_word" && version == 1 && X == QUERY_X then
    return (← wordCalls p.prog).map fun (k, d, c) => (k, key d, c.units)
  if name == "Q_call" && version == 1 && X ≤ QUERY_X then
    let cs ← Flock.Qcall.evaluate p.prog X
    checkCalls p.prog cs
    return cs.map fun c => (c.k, key c.d, c.c.units.size)
  throw s!"query-refused: one-stage applies the template queries, Q_word v1 at X = 16 and Q_call v1 at X <= 16, not {name} v{version} at X = {X}"

/-- `partition.evaluate`'s ranges: the query applied to the program, once core accepts its form, per root node or Call
with units its template or Definition, first unit and count; `query-refused` when it is malformed, the program is not
topological, the query is not one one-stage applies or does not apply, or a unit is wider than the widest class or a
template has no gate (`wordCalls`, `checkCalls`, `checkTemplates`). -/
def evaluate (p : Program) (query : Json) : V (Array TRange) := do
  let obj := build (Flock.hexOf p.sha512) query
  if !valid obj then throw "query-refused: not a verity/partition/v1 query"
  if let some v := p.prog.orderViolation then throw s!"query-refused: the program is not topological: {v.compress}"
  let q := getD obj "query"
  let name := (q.getObjValAs? String "name").toOption.getD ""
  let version := (nat? (getD q "version")).getD 0
  match Flock.TemplateQuery.templatesOf name version (getD q "params") with
  | some ts =>
    let rs ← Flock.TemplateQuery.evaluate p.desc (← ts)
    checkTemplates p.prog rs
    return rs
  | none => return callRanges (← calls p name version (getD q "params"))

/-- `partition.units`: the population. -/
def units (p : Program) (query : Json) : V Nat := return Flock.TemplateQuery.population (← evaluate p query)

/-- `partition.strata`: one stratum per template or Definition, its units as ascending, disjoint ranges, in the order of
their first unit. -/
def strata (p : Program) (query : Json) : V (Array (String × Array (Nat × Nat))) := do
  return Flock.Draw.strataOf ((← evaluate p query).map fun r => (r.template, r.first, r.count))

/-- The gate span `[lo, hi)` of each of `Q_word` Call `(k, d, c)`'s units in root gates: from the least to past the
greatest gate it owns. -/
def wordGates (P : Flock.Program.Prog) (k d : Nat) (c : Flock.Qword.DCut) : V (Array (Nat × Nat)) := do
  let gr ← Flock.Extract.graph P d (full := false)
  let base := ((P.defs[P.root]!).body.offsets[k]!)
  let own := c.owner
  let mut span : Array (Option (Nat × Nat)) := Array.replicate c.units none
  for i in [0:own.size] do
    let g := base + gr.keep[i]!
    span := span.modify own[i]! fun
      | none => some (g, g + 1)
      | some (lo, hi) => some (min lo g, max hi (g + 1))
  span.mapM fun s => match s with
    | some s => pure s
    | none => throw s!"query-refused: a unit of {(P.defs[d]!).key} owns no gate"

/-- `partition.unit_gates` with a template query's units laid out by `instances` (`TemplateQuery.unitGates` or
`TemplateQuery.unitGatesRef`): each unit's gates `[lo, hi)` in root gates, by unit: an instance's
(`TemplateQuery.gates`), a `Q_call` unit's interval of its Call, or the span of the gates a `Q_word` unit owns. -/
def unitGatesBy (instances : Flock.TemplateQuery.Desc → Array TRange → Except Nat (Array (Nat × Nat))) (p : Program)
    (query : Json) : V (Array (Nat × Nat)) := do
  let rs ← evaluate p query
  let q := getD (build (Flock.hexOf p.sha512) query) "query"
  let name := (q.getObjValAs? String "name").toOption.getD ""
  let version := (nat? (getD q "version")).getD 0
  let X := (nat? (getD (getD q "params") "X")).getD 0
  if (Flock.TemplateQuery.templatesOf name version (getD q "params")).isSome then
    return ← match instances p.desc rs with
      | .ok gs => pure gs
      | .error u => throw s!"unit {u} has no instance"
  let root := (p.prog.defs[p.prog.root]!).body
  if name == "Q_word" then
    return ← (← wordCalls p.prog).foldlM (fun acc (k, d, c) => return acc ++ (← wordGates p.prog k d c)) #[]
  return (← Flock.Qcall.evaluate p.prog X).foldl (fun acc c =>
    acc ++ c.c.units.map fun u => (root.offsets[c.k]! + u.lo, root.offsets[c.k]! + u.hi)) #[]

/-- `partition.unit_gates`, a template query's instances a range at a time (`TemplateQuery.unitGates`). -/
def unitGates (p : Program) (query : Json) : V (Array (Nat × Nat)) := unitGatesBy Flock.TemplateQuery.unitGates p query

/-- `partition._once`: a strided run of `count` references, `inner` a row, names each of its leaves at most once, its
rows (or its columns) lying apart. A run whose rows and columns interleave counts as naming a leaf twice. -/
def once (inner : Nat) (os is : Int) (count : Nat) : Bool :=
  let rows := (count + inner - 1) / inner
  let cols := min inner count
  if rows == 1 || cols == 1 then (cols == 1 || is != 0) && (rows == 1 || os != 0)
  else (is != 0 && is.natAbs * (cols - 1) < os.natAbs) || (os != 0 && os.natAbs * (rows - 1) < is.natAbs)

/-- `partition._runs`: `(node, lo, hi, once)` per run of root-node references in `s`: each leaf of the run is one of
leaves `[lo, hi)` of root node `node`'s result, and with `once` no two of its references name one leaf. A strided run is
taken at its span (its corners, as `Program.decode` bounds it), an explicit one a run a reference. -/
def runs (s : Flock.Program.Seq) : Array (Nat × Nat × Nat × Bool) :=
  s.parts.foldl (init := #[]) fun out part => match part with
    | .affine p i b c => if p || c == 0 then out else out.push (i, b, b + c, true)
    | .strided p i b inner os is c =>
      if p || c == 0 then out else
      let q := (c - 1) / inner
      let corners : List (Nat × Nat) :=
        [(0, 0), (0, min inner c - 1), (q, 0), (q, (c - 1) % inner)] ++ (if q > 0 then [(q - 1, inner - 1)] else [])
      let at_ := corners.map fun (x, y) => b + (x : Int) * os + (y : Int) * is
      out.push (i, (at_.foldl min (at_.headD 0)).toNat, (at_.foldl max (at_.headD 0) + 1).toNat, once inner os is c)
    | .explicit items => items.foldl (init := out) fun o r => if r.p then o else o.push (r.i, r.j, r.j + 1, true)

/-- `partition.fanout` over a template query's ranges `rs`: F, the most proof blocks the root nodes of one joined set
hold, two nodes joined when one reads any part of the other's result; at least 1. It reads each instance node's
references once, never its members or its gates. -/
def fanout (P : Flock.Program.Prog) (rs : Array TRange) : Nat := Id.run do
  let nodes := Flock.Registered.rootNodes P
  let mut members : Array (Option Nat) := Array.replicate nodes.size none
  for r in rs do members := members.set! r.node (some r.count)
  let mut label : Array Nat := Array.range nodes.size
  let mut joined : Array Bool := Array.replicate nodes.size false
  for r in rs do
    for a in (nodes.getD r.node default).args do
      for (i, _) in runs a.seq do
        if (members.getD i none).isSome then
          let (li, lk) := (label.getD i i, label.getD r.node r.node)
          if li != lk then label := label.map fun l => if l == li then lk else l
          joined := (joined.set! i true).set! r.node true
  let mut held : Array Nat := Array.replicate nodes.size 0
  for k in [0:nodes.size] do
    if joined.getD k false then
      let g := label.getD k k
      held := held.set! g (held.getD g 0 + (members.getD k none).getD 0)
  return held.foldl max 1

/-- Spans `(node, lo, hi)` in lexicographic order. -/
def spanLt (a b : Nat × Nat × Nat) : Bool :=
  a.1 < b.1 || (a.1 == b.1 && (a.2.1 < b.2.1 || (a.2.1 == b.2.1 && a.2.2 < b.2.2)))

/-- `partition.shared_inputs` over a template query's ranges `rs`: the run Input nodes (an input bound at `run`,
`Program.inputId?`; a registered input, and an input node whose gates lie in one of `held`, the gates `(first, count)`
of the parameters the verifier expects registered, are registered values) whose leaves the proof blocks don't each
read alone, ascending: a batch of more than one member reading one whole (an argument it broadcasts) or by a run that
names a leaf twice, two of a batch's runs that overlap, or two instance nodes' runs that overlap. -/
def sharedInputs (P : Flock.Program.Prog) (rs : Array TRange) (held : Array (Nat × Nat) := #[]) : Array Nat := Id.run do
  let nodes := Flock.Registered.rootNodes P
  let offs := ((P.defs[P.root]?).map (·.body.offsets)).getD #[]
  let run := fun (i : Nat) => match nodes[i]? >>= fun x => P.defs[x.fn]? with
    | some f => f.prim && f.binding == some Flock.Program.RUN &&
        !held.any fun (first, count) => first ≤ offs.getD i 0 && offs.getD (i + 1) 0 ≤ first + count
    | none => false
  let mut shared : Array Nat := #[]
  let mut spans : Array (Nat × Nat × Nat) := #[]
  for r in rs do
    let x := nodes.getD r.node default
    let batch := x.form == .batch
    let mut mine : Array (Nat × Nat × Nat) := #[]
    for ai in [0:if batch then min x.args.size x.axes.size else x.args.size] do
      let mapped := batch && (x.axes.getD ai none).isSome
      for (i, lo, hi, one) in runs (x.args.getD ai default).seq do
        if !run i then continue
        if r.count > 1 && (!mapped || !one) then shared := shared.push i
        mine := mine.push (i, lo, hi)
    let mut last : Option (Nat × Nat) := none
    let mut merged : Array (Nat × Nat × Nat) := #[]
    for (i, lo, hi) in mine.qsort spanLt do
      if let some (j, h) := last then
        if j == i && r.count > 1 && lo < h then shared := shared.push i
      last := some (i, hi)
      merged := match merged.back? with
        | some (j, l, h) => if j == i && lo < h then merged.pop.push (j, l, max h hi) else merged.push (i, lo, hi)
        | none => merged.push (i, lo, hi)
    spans := spans ++ merged
  let mut top : Option (Nat × Nat) := none
  for (i, lo, hi) in spans.qsort spanLt do
    match top with
    | some (j, t) =>
      if j == i then
        if lo < t then shared := shared.push i
        top := some (i, max t hi)
      else top := some (i, hi)
    | none => top := some (i, hi)
  return (shared.qsort (· < ·)).foldl (init := #[]) fun o i => if o.back? == some i then o else o.push i

/-- A template query's ranges (`partition.is_template_query`) applied to `p`; none for another query or one that
doesn't apply. -/
def templateRanges? (p : Program) (query : Json) : Option (Array TRange) :=
  let name := (query.getObjValAs? String "name").toOption.getD ""
  if nat? (getD query "version") == some 0 && (name == "Q_template_instance" || name == "Q_template_instances") then
    (evaluate p query).toOption
  else none

/-- The fan-out a registration verdict publishes (`registration._readers`): a template query's, else none. -/
def fanout? (p : Program) (query : Json) : Option Nat := (templateRanges? p query).map (fanout p.prog)

/-- R8: no run Input of a template query's partition, outside the gates `held`, is one its proof blocks don't each read
alone (`sharedInputs`). -/
def readAlone (p : Program) (query : Json) (held : Array (Nat × Nat) := #[]) : Bool :=
  ((templateRanges? p query).map (sharedInputs p.prog · held)).all (·.isEmpty)

/-- The run Input nodes of a template query's partition, outside the gates `held`, that its proof blocks don't each
read alone (`sharedInputs`); none for another query or one that doesn't apply. -/
def shared? (p : Program) (query : Json) (held : Array (Nat × Nat) := #[]) : Array Nat :=
  ((templateRanges? p query).map (sharedInputs p.prog · held)).getD #[]

/-- Whether root node `i`'s gates lie in `[first, first + count)` (`partition.within`). -/
def within (P : Flock.Program.Prog) (first count i : Nat) : Bool :=
  let offs := ((P.defs[P.root]?).map (·.body.offsets)).getD #[]
  first ≤ offs.getD i 0 && offs.getD (i + 1) 0 ≤ first + count

/-- Whether the root's nodes whose gates lie in `[first, first + count)` are run Input nodes, and one does
(`partition.run_inputs`): a parameter R8 may let through as tied is one the run binds. -/
def runInputs (P : Flock.Program.Prog) (first count : Nat) : Bool :=
  let nodes := Flock.Registered.rootNodes P
  let inside := (List.range nodes.size).filter (within P first count)
  !inside.isEmpty && inside.all fun i => match nodes[i]? >>= fun x => P.defs[x.fn]? with
    | some f => f.prim && f.binding == some Flock.Program.RUN
    | none => false

end Spot.Partition
