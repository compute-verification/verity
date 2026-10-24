/-
The partition as a named query over the verifier's own program (`one_stage/partition.py`): core's `verity/partition/v1`
object and its digest (C-Flock's `HmRow.checkPartitionObject` and `HmRow.partitionDigest`), and the queries one-stage
applies to the program: the template queries (C-Flock's `TemplateQuery`), on a topological program only. `Q_word` and
`Q_call` are retired from the verifier of record: their objects keep core's form and digest, and `evaluate` refuses them.
Every unit's committed bits fit a public size class (`UNIT_CLASSES`, powers of two from 16 to `UNIT_CLASS_MAX` bits; ASK-61,
provisional), its class the least that holds them (`unitClass`), a public property of its Definition: a program with a unit
wider than the widest class is refused (a template instance's bits are its distinct returned gates', `templateBits`), and so
is a program with a template of no gate. The verifier holds its own copy of the
program and applies the query itself; it never takes a population or a unit index on the prover's word.
-/
import Flock.HmRow
import Flock.TemplateQuery
import Flock.Draw
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
      (name != "Q_word" || ["X", "W"].all fun k => (nat? (getD params k)).any (· ≤ Flock.HmRow.PARAM_MAX))

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

/-- `partition._template_bits`: the bits one instance of template `d` commits, the summed widths, as its return type gives
them, of the distinct gates of its activation that it returns. Decoding keeps Input gates to the root's body, and nothing
outside an instance reads its gates but through its returns, so this bounds the claims' `Prog.Boundary` count of the whole
activation, read off the return's references alone rather than the activation's gates. -/
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

/-- The Call queries' ranges `(root-body index, Definition id, units)`: one-stage applies none, as `Q_word` and `Q_call` are
retired from the verifier of record, so every query that is not a template query is refused. -/
def calls (_p : Program) (name : String) (version : Nat) (_params : Json) : V (Array (Nat × String × Nat)) :=
  throw s!"query-refused: one-stage applies the template queries only, not {name} v{version}: Q_word and Q_call are retired from the verifier of record"

/-- `partition.evaluate`'s ranges: the query applied to the program, once core accepts its form, per root node or Call
with units its template or Definition, first unit and count; `query-refused` when it is malformed, the program is not
topological, the query is not a template query (`calls`) or does not apply, or a unit is wider than the widest class or a
template has no gate (`checkTemplates`). -/
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

/-- `partition.unit_gates`: each unit's gates `[lo, hi)` in root gates, by unit: an instance's (`TemplateQuery.gates`). -/
def unitGates (p : Program) (query : Json) : V (Array (Nat × Nat)) := do
  let rs ← evaluate p query
  (Array.range (Flock.TemplateQuery.population rs)).mapM fun u =>
    match Flock.TemplateQuery.gates p.desc rs u with
    | some g => pure g
    | none => throw s!"unit {u} has no instance"

end SampledProofs.Partition
