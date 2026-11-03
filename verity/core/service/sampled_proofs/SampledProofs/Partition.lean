/-
The partition as a named query over the verifier's own program (`one_stage/partition.py`): core's `verity/partition/v1`
object and its digest (C-Flock's `HmRow.checkPartitionObject` and `HmRow.partitionDigest`), and the queries one-stage
applies to the program: the template queries (C-Flock's `TemplateQuery`), `Q_word` v1 at `X = 16` (`Flock.Extract`) and
`Q_call` v1 at `X ≤ 16` (`Flock.Qcall`), each on a topological program only, as core's `evaluate` refuses any other.
Every unit's committed bits fit a public size class (`UNIT_CLASSES`, powers of two from 16 to `UNIT_CLASS_MAX` bits,
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

/-- The widest public size class of a unit's committed bits (the claims' `Claims.Sampling.unitClassMax`,
provisional). 16 sets the classes back to the former literal limit alone. -/
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

/-- Gate `g` added to a unit's gates, as ascending, disjoint intervals with adjacent ones merged. -/
def pushGate (ivs : Array (Nat × Nat)) (g : Nat) : Array (Nat × Nat) :=
  match ivs.back? with
  | some (lo, hi) => if hi == g then ivs.pop.push (lo, g + 1) else ivs.push (g, g + 1)
  | none => #[(g, g + 1)]

/-- The gates each of `Q_word` Call `(k, d, c)`'s units owns, relative to the Call's first gate, as ascending intervals. -/
def wordGates (P : Flock.Program.Prog) (d : Nat) (c : Flock.Qword.DCut) : V (Array (Array (Nat × Nat))) := do
  let gr ← Flock.Extract.graph P d (full := false)
  let own := c.owner
  let mut out : Array (Array (Nat × Nat)) := Array.replicate c.units #[]
  for i in [0:own.size] do
    out := out.modify own[i]! (pushGate · gr.keep[i]!)
  if out.any (·.isEmpty) then throw s!"query-refused: a unit of {(P.defs[d]!).key} owns no gate"
  return out

/-- An activation the query cuts into units: member `m` of root node `k`, of Definition `d`, its units from `first` on,
each unit's gates relative to the activation's first gate, as ascending intervals. -/
structure Act where
  k : Nat
  m : Nat
  d : Nat
  first : Nat
  units : Array (Array (Nat × Nat))
  deriving Inhabited

/-- The activation's first gate in root gates. -/
def Act.base (P : Flock.Program.Prog) (a : Act) : Nat :=
  (P.defs[P.root]!).body.offsets[a.k]! + a.m * (P.defs[a.d]!).gates

/-- The query's units by activation, in unit order: each instance of a template query its member's whole activation, and
each `Q_call` or `Q_word` Call its units, a `Q_call` unit its interval and a `Q_word` unit the gates it owns. -/
def acts (p : Program) (query : Json) : V (Array Act) := do
  let rs ← evaluate p query
  let q := getD (build (Flock.hexOf p.sha512) query) "query"
  let name := (q.getObjValAs? String "name").toOption.getD ""
  let version := (nat? (getD q "version")).getD 0
  let X := (nat? (getD (getD q "params") "X")).getD 0
  let P := p.prog
  let root := (P.defs[P.root]!).body
  if (Flock.TemplateQuery.templatesOf name version (getD q "params")).isSome then
    return rs.flatMap fun r =>
      let d := root.nodes[r.node]!.fn
      (Array.range r.count).map fun m =>
        ({ k := r.node, m, d, first := r.first + m, units := #[#[(0, (P.defs[d]!).gates)]] } : Act)
  if name == "Q_word" then
    let mut out : Array Act := #[]
    let mut first := 0
    for (k, d, c) in ← wordCalls P do
      let units ← wordGates P d c
      out := out.push { k, m := 0, d, first, units }
      first := first + units.size
    return out
  let mut out : Array Act := #[]
  let mut first := 0
  for c in ← Flock.Qcall.evaluate P X do
    out := out.push { k := c.k, m := 0, d := c.d, first, units := c.c.units.map fun u => #[(u.lo, u.hi)] }
    first := first + c.c.units.size
  return out

/-- The activations' units' gates in root gates, by unit. -/
def gatesOf (P : Flock.Program.Prog) (as : Array Act) : Array (Array (Nat × Nat)) :=
  as.flatMap fun a => let b := a.base P; a.units.map (·.map fun (lo, hi) => (b + lo, b + hi))

/-- `partition.unit_gates`: each unit's gates in root gates, as ascending intervals, by unit: an instance's whole activation,
a `Q_call` unit's interval of its Call, or the gates a `Q_word` unit owns. -/
def unitGates (p : Program) (query : Json) : V (Array (Array (Nat × Nat))) := return gatesOf p.prog (← acts p query)

/-- The gates of one activation of Definition `d` that read each of its parameter leaves, relative to its first gate,
ascending: every operand of every gate, a primitive's being its parameters' leaves in order. -/
def leafReaders (P : Flock.Program.Prog) (d : Nat) : V (Array (Array Nat)) := do
  let defn := P.defs[d]!
  let lo := defn.paramLeaves
  if defn.prim then return Array.replicate lo #[0]
  let starts := (defn.params.foldl (fun (acc : Array Nat × Nat) (_, t) => (acc.1.push acc.2, acc.2 + t.leaves)) (#[], 0)).1
  let mut out : Array (Array Nat) := Array.replicate lo #[]
  let mut stack : Array (Flock.Program.Scope × Nat × Nat) := #[(.top d lo (some starts), 0, 0)]
  for _ in [0:Flock.Steps.FUEL] do
    if stack.size == 0 then break
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
    for i in [0:node.args.size] do
      let a := node.args[i]!
      let cnt := if node.form == .batch && a.dims.isSome then (callee.params[i]!.2).leaves else a.seq.count
      for t in [0:cnt] do
        let g ← Flock.Program.resolve P sc (if node.form == .batch then a.memberRef m t else a.seq.get t)
        if g < lo then out := out.modify g fun rs => if rs.back? == some j then rs else rs.push j
  if stack.size > 0 then throw "a walk of more steps than this verifier follows"
  return out

/-- Each unit's inputs, by unit: the gates of the root's input nodes whose owner, the first unit in unit order with a gate
that reads it, is that unit, in gate order. An input no unit reads is no unit's. -/
def unitInputs (p : Program) (as : Array Act) : V (Array (Array Nat)) := do
  let P := p.prog
  let root := (P.defs[P.root]!).body
  let ins := (Array.range root.nodes.size).filterMap fun k =>
    if Flock.Program.isInputId (P.defs[root.nodes[k]!.fn]!).key then some (root.offsets[k]!, root.offsets[k + 1]!) else none
  let mut owner : Std.HashMap Nat Nat := {}
  let mut memo : Std.HashMap Nat (Array (Option Nat)) := {}
  for a in as do
    let defn := P.defs[a.d]!
    let lead ← match memo[a.d]? with
      | some l => pure l
      | none => do
        let rs ← leafReaders P a.d
        if a.units.size == 1 && a.units[0]! == #[(0, defn.gates)] then
          pure (rs.map fun r => if r.isEmpty then none else some 0)
        else
          let mut at_ : Array (Option Nat) := Array.replicate defn.gates none
          for lu in [0:a.units.size] do
            for (x, y) in a.units[lu]! do
              for g in [x:y] do at_ := at_.set! g (some lu)
          pure (rs.map fun r => r.foldl (fun acc j => match at_[j]!, acc with
            | some u, some v => some (min u v)
            | some u, none => some u
            | none, acc => acc) none)
    memo := memo.insert a.d lead
    let sc := P.child P.rootScope a.k a.m
    let mut ℓ := 0
    for i in [0:defn.params.size] do
      for t in [0:(defn.params[i]!.2).leaves] do
        if let some lu := lead[ℓ]! then
          let g ← Flock.Program.resolve P sc ⟨true, i, t⟩
          if ins.any fun (x, y) => x ≤ g && g < y then
            let u := a.first + lu
            owner := owner.insert g (min u (owner.getD g u))
        ℓ := ℓ + 1
  let n := as.foldl (· + ·.units.size) 0
  return (owner.toArray.qsort (·.1 < ·.1)).foldl (fun out (g, u) => out.modify u (·.push g)) (Array.replicate n (#[] : Array Nat))

end SampledProofs.Partition
