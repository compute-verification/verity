/-
The partition as a named query over the verifier's own program (`one_stage/partition.py`): core's `verity/partition/v1`
object and its digest (C-Flock's `HmRow.checkPartitionObject` and `HmRow.partitionDigest`), and the queries one-stage
applies to the program: the template queries (C-Flock's `TemplateQuery`), `Q_word` v1 at `X = 16` (`Flock.Extract`) and
`Q_call` v1 at `X ≤ 16` (`Flock.Qcall`), each on a topological program only, as core's `evaluate` refuses any other.
Every unit's committed bits fit a public size class (`UNIT_CLASSES`, powers of two from 16 to `UNIT_CLASS_MAX` bits; ASK-61,
provisional), its class the least that holds them (`unitClass`), a public property of its Definition: a program with a unit
wider than the widest class is refused (a template instance's bits are its distinct returned gates', `templateBits`; a
`Q_call` unit's count the constants its activation holds, `committedBits`, which `Q_call`'s width rule leaves out; a
`Q_word` unit's are its cut's), and so is a program with a template of no gate. The verifier holds its own copy of the
program and applies the query itself; it never takes a population or a unit index on the prover's word.
-/
import Flock.HmRow
import Flock.TemplateQuery
import Flock.Draw
import Flock.Extract
import Flock.Qcall
import SampledProofs.Json
import SampledProofs.Program

namespace SampledProofs.Partition

open Lean (Json)
open Flock (V)
open Flock.TemplateQuery (TRange)

/-- `partition_object.QUERIES`, the queries a partition may name. C-Flock's form check leaves the names open. -/
def QUERIES : List (String × Nat) :=
  [("Q_word", 1), ("Q_word", 2), Flock.HmRow.Q_CALL, ("Q_template_instance", 0), ("Q_template_instances", 0)]

/-- `partition_object.validate`: C-Flock's form check, narrowed to core's queries, with `Q_word`'s `X` and `W` at most
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

/-- The widest public size class of a unit's committed bits (the claims' `Claims.Sampling.unitClassMax`; ASK-61, Daniel,
19 Oct 2:59 PM PDT, provisional). 16 sets the classes back to the former literal limit alone. -/
def UNIT_CLASS_MAX : Nat := 262144

/-- The public size classes, ascending: the powers of two from 16 to `UNIT_CLASS_MAX` (the claims' `unitClasses`). -/
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

/-- `partition.unit_gates`: each unit's gates `[lo, hi)` in root gates, by unit: an instance's (`TemplateQuery.gates`),
a `Q_call` unit's interval of its Call, or the span of the gates a `Q_word` unit owns. -/
def unitGates (p : Program) (query : Json) : V (Array (Nat × Nat)) := do
  let rs ← evaluate p query
  let q := getD (build (Flock.hexOf p.sha512) query) "query"
  let name := (q.getObjValAs? String "name").toOption.getD ""
  let version := (nat? (getD q "version")).getD 0
  let X := (nat? (getD (getD q "params") "X")).getD 0
  if (Flock.TemplateQuery.templatesOf name version (getD q "params")).isSome then
    return ← (Array.range (Flock.TemplateQuery.population rs)).mapM fun u =>
      match Flock.TemplateQuery.gates p.desc rs u with
      | some g => pure g
      | none => throw s!"unit {u} has no instance"
  let root := (p.prog.defs[p.prog.root]!).body
  if name == "Q_word" then
    return ← (← wordCalls p.prog).foldlM (fun acc (k, d, c) => return acc ++ (← wordGates p.prog k d c)) #[]
  return (← Flock.Qcall.evaluate p.prog X).foldl (fun acc c =>
    acc ++ c.c.units.map fun u => (root.offsets[c.k]! + u.lo, root.offsets[c.k]! + u.hi)) #[]

end SampledProofs.Partition
