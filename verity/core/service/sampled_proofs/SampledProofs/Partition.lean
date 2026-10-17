/-
The partition as a named query over the verifier's own program (`one_stage/partition.py`): core's `verity/partition/v1`
object and its digest (C-Flock's `HmRow.checkPartitionObject` and `HmRow.partitionDigest`), and the queries one-stage
applies to the program: the template queries (C-Flock's `TemplateQuery`), `Q_word` v1 at `X = 16` (`Flock.Extract`) and
`Q_call` v1 at `X ≤ 16` (`Flock.Qcall`), each on a topological program only, as core's `evaluate` refuses any other. A
`Q_word` or `Q_call` unit commits at most 16 bits, the literal limit; `Q_word` refuses a program with a unit that commits
more, which its width rule's `W` allows a unit of one value. The verifier holds its own copy of the program and applies
the query itself; it never takes a population or a unit index on the prover's word.
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

/-- The literal limit: the bits a unit commits. -/
def UNIT_BITS : Nat := 16

/-- `partition._call_ranges`: the ranges of the Calls `(root-body index, Definition id, units)`, numbered by Call from
`0`, one per Call with units. -/
def callRanges (calls : Array (Nat × String × Nat)) : Array TRange :=
  (calls.foldl (fun (acc : Array TRange × Nat) (x : Nat × String × Nat) =>
    if x.2.2 == 0 then acc else (acc.1.push ⟨x.1, x.2.1, acc.2, x.2.2⟩, acc.2 + x.2.2)) (#[], 0)).1

/-- `Q_word` v1's Calls with their cuts at `X = 16` (`partition._word_calls`): refused when a Definition's unit commits
more than `UNIT_BITS` bits (`Flock.Partition.boundaryWidths` on its Call's graph). -/
def wordCalls (P : Flock.Program.Prog) : V (Array (Nat × Nat × Flock.Qword.DCut)) := do
  let calls ← Flock.Extract.evaluate P UNIT_BITS
  let mut seen : Array Nat := #[]
  for (_, d, c) in calls do
    if seen.contains d then continue
    seen := seen.push d
    let gr ← Flock.Extract.graph P d (full := false)
    if c.owner.size != gr.g.n then throw s!"query-refused: {(P.defs[d]!).key}'s cut leaves a computing gate out"
    let (ob, _) := Flock.Partition.boundaryWidths gr.g c.owner
    if let some u := (List.range ob.size).find? (ob[·]! > UNIT_BITS) then
      throw s!"query-refused: unit {u} of {(P.defs[d]!).key} commits {ob[u]!} bits, more than {UNIT_BITS}"
  return calls

/-- The `Q_word` or `Q_call` query's Calls with their units, as `(root-body index, Definition id, units)`: `Q_word` v1 at
`X = 16` (`wordCalls`) or `Q_call` v1 at `X ≤ 16` (any `W` core accepts); `query-refused` for any other parameters or
query, and where the query is inapplicable. -/
def calls (p : Program) (name : String) (version : Nat) (params : Json) : V (Array (Nat × String × Nat)) := do
  let X := (nat? (getD params "X")).getD 0
  let key := fun d => (p.prog.defs[d]!).key
  if name == "Q_word" && version == 1 && X == UNIT_BITS then
    return (← wordCalls p.prog).map fun (k, d, c) => (k, key d, c.units)
  if name == "Q_call" && version == 1 && X ≤ UNIT_BITS then
    return (← Flock.Qcall.evaluate p.prog X).map fun c => (c.k, key c.d, c.c.units.size)
  throw s!"query-refused: one-stage applies the template queries, Q_word v1 at X = 16 and Q_call v1 at X <= 16, not {name} v{version} at X = {X}"

/-- `partition.evaluate`'s ranges: the query applied to the program, once core accepts its form, per root node or Call
with units its template or Definition, first unit and count; `query-refused` when it is malformed, the program is not
topological, or the query is not one one-stage applies or does not apply. -/
def evaluate (p : Program) (query : Json) : V (Array TRange) := do
  let obj := build (Flock.hexOf p.sha512) query
  if !valid obj then throw "query-refused: not a verity/partition/v1 query"
  if let some v := p.prog.orderViolation then throw s!"query-refused: the program is not topological: {v.compress}"
  let q := getD obj "query"
  let name := (q.getObjValAs? String "name").toOption.getD ""
  let version := (nat? (getD q "version")).getD 0
  match Flock.TemplateQuery.templatesOf name version (getD q "params") with
  | some ts => Flock.TemplateQuery.evaluate p.desc (← ts)
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
