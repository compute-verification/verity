/-
The partition as a named query over the verifier's own program (`one_stage/partition.py`): core's `verity/partition/v1`
object and its digest (C-Flock's `HmRow.checkPartitionObject` and `HmRow.partitionDigest`), and the queries one-stage
applies to the program: the template queries (C-Flock's `TemplateQuery`), `Q_word` v1 at `X = 16` (`Flock.Extract`) and
`Q_call` v1 at `X ≤ 16` (`Flock.Qcall`), each on a topological program only, as core's `evaluate` refuses any other. A
`Q_word` or `Q_call` unit commits at most 16 bits, the literal limit: a program with a unit that commits more is refused,
which `Q_word`'s width rule's `W` allows a unit of one value, and `Q_call`'s leaves out the constants a unit holds
(`committedBits`); so is a program with a template of no gate. A template query has no width rule. The verifier holds its
own copy of the program and applies the query itself; it never takes a population or a unit index on the prover's word.
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

/-- `partition._committed_bits`: per unit `[lo, hi)` of one activation of definition `d` (its local gates, as
`Flock.Qcall.gates` lays them out), the bits it commits: the summed declared widths of its gates that are Input gates, or
that a later gate of the activation reads or the activation returns, constants included. Nothing outside an activation
reads its gates but through its returns, so in any program this bounds the claims' `Prog.Boundary` count, as `Q_call`'s
width rule bounds its computing gates'. -/
def committedBits (P : Flock.Program.Prog) (d : Nat) (units : Array (Nat × Nat)) : V (Array Nat) := do
  let defn := P.defs[d]!
  let lo := defn.paramLeaves
  let n := defn.gates
  let starts := (defn.params.foldl (fun (acc : Array Nat × Nat) (_, t) => (acc.1.push acc.2, acc.2 + t.leaves)) (#[], 0)).1
  let top : Flock.Program.Scope := .top d lo (some starts)
  let mut prim : Array Nat := if defn.prim then #[d] else Array.mkEmpty n
  let mut last : Array Int := Array.replicate n (-1)
  let mut stack : Array (Flock.Program.Scope × Nat × Nat) := if defn.prim then #[] else #[(top, 0, 0)]
  while stack.size > 0 do
    let (sc, k, m) := stack.back!
    let body := (P.defs[sc.d]!).body
    if k ≥ body.nodes.size then
      stack := stack.pop
      continue
    let node := body.nodes[k]!
    if m ≥ node.members then
      stack := stack.set! (stack.size - 1) (sc, k + 1, 0)
      continue
    stack := stack.set! (stack.size - 1) (sc, k, m + 1)
    let callee := P.defs[node.fn]!
    let o := sc.off + body.offsets[k]!
    if !callee.prim then
      stack := stack.push (.child sc node.fn (o + m * callee.gates) k m, 0, 0)
      continue
    let j := o + m - lo
    for g in ← Flock.Program.operands P sc node m do
      if lo ≤ g && g - lo < j then last := last.set! (g - lo) j
    prim := prim.push node.fn
  if prim.size != n then throw s!"{defn.key}: {prim.size} gates laid out, {n} declared"
  for g in ← if defn.prim then pure #[lo] else defn.body.ret.toArray.mapM (Flock.Program.resolve P top) do
    if lo ≤ g && g < lo + n then last := last.set! (g - lo) n
  return units.map fun (a, b) => (List.range (b - a)).foldl (fun bits i =>
    let f := P.defs[prim[a + i]!]!
    if Flock.Program.isInputId f.key || last[a + i]! ≥ b then
      bits + match f.ret with
        | .value w => w
        | _ => 0
    else bits) 0

/-- `partition._check_units` on a template query's ranges: refused for a template of no gate. -/
def checkTemplates (P : Flock.Program.Prog) (rs : Array TRange) : V Unit := do
  let root := (P.defs[P.root]!).body
  for r in rs do
    if r.count > 0 && (P.defs[root.nodes[r.node]!.fn]!).gates == 0 then
      throw s!"query-refused: template {r.template} has no gate"

/-- `partition._check_units` on `Q_call`'s Calls: refused when a unit commits more than `UNIT_BITS` bits (`committedBits`,
constants counted, which `Q_call`'s width rule leaves out), each Definition once. -/
def checkCalls (P : Flock.Program.Prog) (cs : Array Flock.Qcall.Call) : V Unit := do
  let mut seen : Array Nat := #[]
  for c in cs do
    if seen.contains c.d then continue
    seen := seen.push c.d
    let bits ← committedBits P c.d (c.c.units.map fun u => (u.lo, u.hi))
    if let some u := (List.range bits.size).find? (bits[·]! > UNIT_BITS) then
      throw s!"query-refused: unit {u} of {(P.defs[c.d]!).key} commits {bits[u]!} bits, more than {UNIT_BITS}"

/-- The `Q_word` or `Q_call` query's Calls with their units, as `(root-body index, Definition id, units)`: `Q_word` v1 at
`X = 16` (`wordCalls`) or `Q_call` v1 at `X ≤ 16` (any `W` core accepts, its units `checkCalls`'); `query-refused` for
any other parameters or query, and where the query is inapplicable. -/
def calls (p : Program) (name : String) (version : Nat) (params : Json) : V (Array (Nat × String × Nat)) := do
  let X := (nat? (getD params "X")).getD 0
  let key := fun d => (p.prog.defs[d]!).key
  if name == "Q_word" && version == 1 && X == UNIT_BITS then
    return (← wordCalls p.prog).map fun (k, d, c) => (k, key d, c.units)
  if name == "Q_call" && version == 1 && X ≤ UNIT_BITS then
    let cs ← Flock.Qcall.evaluate p.prog X
    checkCalls p.prog cs
    return cs.map fun c => (c.k, key c.d, c.c.units.size)
  throw s!"query-refused: one-stage applies the template queries, Q_word v1 at X = 16 and Q_call v1 at X <= 16, not {name} v{version} at X = {X}"

/-- `partition.evaluate`'s ranges: the query applied to the program, once core accepts its form, per root node or Call
with units its template or Definition, first unit and count; `query-refused` when it is malformed, the program is not
topological, the query is not one one-stage applies or does not apply, or a unit is wider than the literal limit or a
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
