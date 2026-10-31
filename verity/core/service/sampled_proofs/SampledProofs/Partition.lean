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
`unitGates` runs as `unitGatesFast` (`@[csimp]`, `unitGates_eq_unitGatesFast`): a template query's units read their
gates off one `table`, each root callee's gate count computed once through a memo (`gatesMemo`) and each range's offset
once, where `TemplateQuery.gates` recomputes both per unit.
-/
import Flock.HmRow
import Flock.TemplateQuery
import Flock.Draw
import Flock.Extract
import Flock.Qcall
import SampledProofs.Json
import SampledProofs.Program
import Std.Data.HashMap

namespace SampledProofs.Partition

open Lean (Json)
open Flock (V)
open Flock.TemplateQuery (TRange Desc DDef DNode)

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

/-- Each Definition of `d` by its id, the first of an id as `Desc.def?` finds it. -/
def defMap (d : Desc) : Std.HashMap String DDef := d.defs.foldr (fun p m => m.insert p.1 p.2) {}

theorem defMap_get (d : Desc) (id : String) : (defMap d)[id]? = d.def? id := by
  unfold defMap Flock.TemplateQuery.Desc.def?
  rw [← Array.foldr_toList, ← Array.find?_toList]
  induction d.defs.toList with
  | nil => simp
  | cons x l ih =>
    simp only [List.foldr_cons, List.find?_cons, Std.HashMap.getElem?_insert, ih]
    by_cases h : x.1 = id
    · simp [h]
    · have h' : (x.1 == id) = false := by simpa using h
      simp [h']

/-- Gate counts by fuel and Definition id. -/
abbrev Memo := Std.HashMap (Nat × String) Nat

/-- `Desc.gatesIn` through the memo `M`: a composite's count is computed once per fuel and recorded, so a Definition
reached again at the same depth is a lookup, and the walk is proportional to the descriptor's nodes times its depth. -/
def gatesMemo (d : Desc) (dm : Std.HashMap String DDef) : Nat → String → Memo → Nat × Memo
  | 0, _, M => (0, M)
  | n + 1, id, M =>
    match M[(n + 1, id)]? with
    | some v => (v, M)
    | none =>
      match dm[id]? with
      | none => (0, M)
      | some df =>
        if df.prim then (1, M) else
        let r := df.nodes.foldl (fun (r : Nat × Memo) x =>
          let s := gatesMemo d dm n (d.callees.getD x.fn "") r.2
          (r.1 + x.members * s.1, s.2)) (0, M)
        (r.1, r.2.insert (n + 1, id) r.1)

/-- Every count `M` holds is `Desc.gatesIn`'s. -/
def MemoExact (d : Desc) (M : Memo) : Prop := ∀ n id v, M[(n, id)]? = some v → v = d.gatesIn n id

theorem MemoExact.insert {d : Desc} {M : Memo} (h : MemoExact d M) (n : Nat) (id : String) :
    MemoExact d (M.insert (n, id) (d.gatesIn n id)) := by
  intro m id' v hv
  rw [Std.HashMap.getElem?_insert] at hv
  split at hv
  · rename_i he
    cases eq_of_beq he
    simp only [Option.some.injEq] at hv
    exact hv.symm
  · exact h _ _ _ hv

theorem gatesMemo_spec (d : Desc) (dm : Std.HashMap String DDef) (hdm : ∀ id, dm[id]? = d.def? id) :
    ∀ n id M, MemoExact d M →
      (gatesMemo d dm n id M).1 = d.gatesIn n id ∧ MemoExact d (gatesMemo d dm n id M).2 := by
  intro n
  induction n with
  | zero => intro id M hM; exact ⟨by simp [gatesMemo, Flock.TemplateQuery.Desc.gatesIn], hM⟩
  | succ n ih =>
    intro id M hM
    simp only [gatesMemo]
    split
    · rename_i v hv
      exact ⟨hM _ _ _ hv, hM⟩
    · rw [hdm]
      simp only [Flock.TemplateQuery.Desc.gatesIn]
      cases hd : d.def? id with
      | none => exact ⟨rfl, hM⟩
      | some df =>
        simp only
        split
        next => exact ⟨rfl, hM⟩
        next hp =>
          have key : ∀ (l : List DNode) (acc : Nat) (M : Memo), MemoExact d M →
              (l.foldl (fun (r : Nat × Memo) x =>
                let s := gatesMemo d dm n (d.callees.getD x.fn "") r.2
                (r.1 + x.members * s.1, s.2)) (acc, M)).1 =
                l.foldl (fun acc x => acc + x.members * d.gatesIn n (d.callees.getD x.fn "")) acc ∧
              MemoExact d (l.foldl (fun (r : Nat × Memo) x =>
                let s := gatesMemo d dm n (d.callees.getD x.fn "") r.2
                (r.1 + x.members * s.1, s.2)) (acc, M)).2 := by
            intro l
            induction l with
            | nil => intro acc M hM; exact ⟨rfl, hM⟩
            | cons x l ihl =>
              intro acc M hM
              obtain ⟨h1, h2⟩ := ih (d.callees.getD x.fn "") M hM
              simp only [List.foldl_cons]
              rw [← h1]
              exact ihl _ _ h2
          obtain ⟨k1, k2⟩ := key df.nodes.toList 0 M hM
          rw [Array.foldl_toList, Array.foldl_toList] at k1
          rw [Array.foldl_toList] at k2
          refine ⟨k1, ?_⟩
          have e : df.nodes.foldl (fun acc x => acc + x.members * d.gatesIn n (d.callees.getD x.fn "")) 0 =
              d.gatesIn (n + 1) id := by
            simp only [Flock.TemplateQuery.Desc.gatesIn, hd, hp]; rfl
          rw [k1, e]
          exact k2.insert (n + 1) id

/-- `Desc.gates` of every callee of `d`'s root nodes, through one memo (`gatesMemo`) shared across them. -/
def gTable (d : Desc) : Std.HashMap String Nat :=
  let dm := defMap d
  match dm[d.root]? with
  | none => {}
  | some root => (root.nodes.foldl (fun (r : Std.HashMap String Nat × Memo) x =>
      let c := d.callees.getD x.fn ""
      let s := gatesMemo d dm Flock.Steps.FUEL c r.2
      (r.1.insert c s.1, s.2)) ({}, {})).1

theorem gTable_spec (d : Desc) (c : String) (v : Nat) (h : (gTable d)[c]? = some v) : v = d.gates c := by
  unfold gTable at h
  dsimp only at h
  split at h
  · simp at h
  · have key : ∀ (l : List DNode) (r : Std.HashMap String Nat × Memo),
        (∀ c v, r.1[c]? = some v → v = d.gates c) → MemoExact d r.2 →
        (∀ c v, (l.foldl (fun (r : Std.HashMap String Nat × Memo) x =>
          let c := d.callees.getD x.fn ""
          let s := gatesMemo d (defMap d) Flock.Steps.FUEL c r.2
          (r.1.insert c s.1, s.2)) r).1[c]? = some v → v = d.gates c) := by
      intro l
      induction l with
      | nil => intro r h1 _; exact h1
      | cons x l ihl =>
        intro r h1 h2
        obtain ⟨s1, s2⟩ :=
          gatesMemo_spec d (defMap d) (defMap_get d) Flock.Steps.FUEL (d.callees.getD x.fn "") r.2 h2
        apply ihl _ _ s2
        intro c' v' hv'
        simp only [Std.HashMap.getElem?_insert] at hv'
        split at hv'
        · rename_i he
          cases eq_of_beq he
          simp only [Option.some.injEq] at hv'
          rw [← hv', s1]; rfl
        · exact h1 _ _ hv'
    rw [← Array.foldl_toList] at h
    exact key _ _ (by simp) (by intro n id v hv; simp at hv) c v h

/-- `Desc.gates` of `c` read off `G`, or computed where `G` has no entry. -/
def look (d : Desc) (G : Std.HashMap String Nat) (c : String) : Nat :=
  match G[c]? with
  | some v => v
  | none => d.gates c

theorem look_gTable (d : Desc) : look d (gTable d) = d.gates := by
  funext c
  unfold look
  split
  · rename_i v hv; exact gTable_spec d c v hv
  · rfl

/-- The gates of `root`'s nodes before node `k`, each callee's count by `g` (`TemplateQuery.gates`' offset at
`g = d.gates`). -/
def offWith (d : Desc) (g : String → Nat) (root : DDef) (k : Nat) : Nat :=
  (root.nodes.extract 0 k).foldl (fun acc x => acc + x.members * g (d.callees.getD x.fn "")) 0

/-- What `TemplateQuery.gates` reads per unit, once per program and ranges: the root, each root callee's gates
(`gTable`) and the offset of each range's node. -/
structure GateTable where
  root : Option DDef
  G : Std.HashMap String Nat
  offs : Std.HashMap Nat Nat

def table (d : Desc) (rs : Array TRange) : GateTable :=
  let G := gTable d
  let root := d.def? d.root
  { root, G, offs := match root with
      | none => {}
      | some r => rs.foldl (fun o x => o.insert x.node (offWith d (look d G) r x.node)) {} }

/-- `TemplateQuery.gates` off a `GateTable`. -/
def gatesBy (d : Desc) (tb : GateTable) (rs : Array TRange) (i : Nat) : Option (Nat × Nat) := do
  let (k, m, t) ← Flock.TemplateQuery.locate rs i
  let root ← tb.root
  let off := match tb.offs[k]? with
    | some o => o
    | none => offWith d (look d tb.G) root k
  let g := look d tb.G t
  return (off + m * g, off + m * g + g)

theorem offs_spec (d : Desc) (g : String → Nat) (r : DDef) (rs : Array TRange) (k o : Nat)
    (h : (rs.foldl (fun (o : Std.HashMap Nat Nat) (x : TRange) => o.insert x.node (offWith d g r x.node)) {})[k]? =
      some o) : o = offWith d g r k := by
  rw [← Array.foldl_toList] at h
  have key : ∀ (l : List TRange) (O : Std.HashMap Nat Nat), (∀ k o, O[k]? = some o → o = offWith d g r k) →
      ∀ k o, (l.foldl (fun (o : Std.HashMap Nat Nat) (x : TRange) => o.insert x.node (offWith d g r x.node)) O)[k]? =
        some o → o = offWith d g r k := by
    intro l
    induction l with
    | nil => intro O h; exact h
    | cons x l ihl =>
      intro O hO
      apply ihl
      intro k' o' h'
      simp only [Std.HashMap.getElem?_insert] at h'
      split at h'
      · rename_i he
        cases eq_of_beq he
        simp only [Option.some.injEq] at h'
        exact h'.symm
      · exact hO _ _ h'
  exact key _ _ (by intro k o h; simp at h) k o h

theorem gatesBy_table (d : Desc) (rs : Array TRange) (i : Nat) :
    gatesBy d (table d rs) rs i = Flock.TemplateQuery.gates d rs i := by
  simp only [gatesBy, table, Flock.TemplateQuery.gates, look_gTable]
  cases Flock.TemplateQuery.locate rs i with
  | none => rfl
  | some kmt =>
    obtain ⟨k, m, t⟩ := kmt
    cases d.def? d.root with
    | none => rfl
    | some r =>
      simp only [Option.bind_eq_bind, Option.bind_some]
      split
      · rename_i o ho
        rw [offs_spec d d.gates r rs k o ho]
        rfl
      · rfl

/-- `partition.unit_gates`: each unit's gates `[lo, hi)` in root gates, by unit: an instance's (`TemplateQuery.gates`),
a `Q_call` unit's interval of its Call, or the span of the gates a `Q_word` unit owns. It runs as `unitGatesFast`. -/
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

/-- `unitGates` with a template query's units read off one `table` of the program and ranges (`gatesBy`), rather than
each unit recomputing its node's offset and its template's gates. -/
def unitGatesFast (p : Program) (query : Json) : V (Array (Nat × Nat)) := do
  let rs ← evaluate p query
  let q := getD (build (Flock.hexOf p.sha512) query) "query"
  let name := (q.getObjValAs? String "name").toOption.getD ""
  let version := (nat? (getD q "version")).getD 0
  let X := (nat? (getD (getD q "params") "X")).getD 0
  if (Flock.TemplateQuery.templatesOf name version (getD q "params")).isSome then
    let tb := table p.desc rs
    return ← (Array.range (Flock.TemplateQuery.population rs)).mapM fun u =>
      match gatesBy p.desc tb rs u with
      | some g => pure g
      | none => throw s!"unit {u} has no instance"
  let root := (p.prog.defs[p.prog.root]!).body
  if name == "Q_word" then
    return ← (← wordCalls p.prog).foldlM (fun acc (k, d, c) => return acc ++ (← wordGates p.prog k d c)) #[]
  return (← Flock.Qcall.evaluate p.prog X).foldl (fun acc c =>
    acc ++ c.c.units.map fun u => (root.offsets[c.k]! + u.lo, root.offsets[c.k]! + u.hi)) #[]

@[csimp] theorem unitGates_eq_unitGatesFast : @unitGates = @unitGatesFast := by
  funext p query
  simp only [unitGates, unitGatesFast, gatesBy_table]

end SampledProofs.Partition
