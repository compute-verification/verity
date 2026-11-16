module

public import Proofs.Properties.Verity.Sampled.HonestSession.ProgramParse
public import Proofs.Properties.Verity.Sampled.HonestSession.Program
public import Proofs.Flock.Bridge.PopMatches
import Proofs.Flock.Verifier.Hash
import Spot.Partition
import Spot.Slots
import Spot.Program
import Spot.Json
import Flock.HmRow
import Flock.TemplateQuery
import Commitments.Draw
import Circuits.Program
import Circuits.Steps
import all Init.Data.Array.QSort.Basic
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic
import all Std.Data.TreeMap.Raw.Basic
import all Std.Data.DTreeMap.Raw.Basic
import all Std.Data.DTreeMap.Internal.Operations
import all Std.Data.DTreeMap.Internal.Queries
import all Std.Data.DTreeMap.Internal.Balancing
import all Std.Data.DTreeMap.Internal.Def
import all Init.Data.Ord.String
import all Init.Data.Ord.Basic
import all Init.Data.String.Basic
import all Init.Data.String.Defs
import all Init.Data.String.Legacy
import all Init.Data.String.Slice
import all Init.Data.String.Search
import all Init.Data.String.TakeDrop
import all Init.Data.String.Pattern.String
import all Init.Data.String.Lemmas.Pattern.String.ForwardPattern
import all Init.Data.ToString.Basic
import all Init.Data.Repr
import all Init.Data.Nat.ToString
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Stream
import all Init.Data.Slice.Basic
import all Init.Data.Slice.Array.Basic
import all Init.Data.Array.Subarray
import all Init.Data.Range.Basic

/-!
# `ProgramFacts.popMatches`, and `ProgramFacts`

`popMatches` is the walk of `Spot.Slots.layout` on the parsed program, over a copy of `decV` whose SHA-512 and name
index are variables (`progB b ix`): the partition object names the hash through `hexOf b`, which the walk reads only
through `isSha512Hex (hexOf b)` (`valid_obj`), so no step evaluates a hash. `decV = progB decV.sha512 decV.index`
(`decV_progB`) carries it to `program`. What the kernel does not evaluate is read by a rule:

* `orderViolation`'s `HashSet Nat` (its bucket index is a `USize`): `orderViolation` is `ovGen`, the same walk over
  abstract set operations, by `rfl`; `ovGen_map` moves it along any state map that commutes with `contains` and `insert`,
  and the kernel decides the instance on characteristic functions (`ov_fn`).
* `Desc.ofProg`'s `Array.map`s (`descB`, `amap`), `Ty.widths` (`widthsK`), the sort of one pair (`qsort_one`) and of
  ascending keys (`keys3`, `qsort_sorted`), `checkPartitionObject`'s let-bound `keys` (`checkPartitionObjectK`), and
  the loops in `Except` (`forIn_range_ex`, `forIn_arr_ex`, `forInEx.eq_def`).

`programFacts` assembles the six fields.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession.ProgramPop

open Flock
open Lean (Json)
open ProgramProof ProgramParse

/-- `checkPartitionObject`'s `keys`, lifted out of its `let` so that a rule can name it (`cpo_K`). -/
def cpoKeys (j : Json) : Array String := match j with
  | .obj kvs => (kvs.toArray.map (·.1)).qsort (· < ·)
  | _ => #[]

def checkPartitionObjectK (obj : Json) : V (String × Nat) := do
  if cpoKeys obj != #["format", "program", "query"] then throw "partition: the object's keys are not format, program, query"
  if (obj.getObjValAs? String "format").toOption != some Flock.HmRow.PARTITION_V1 then throw s!"partition: not a {Flock.HmRow.PARTITION_V1} object"
  if !Flock.HmRow.isSha512Hex ((obj.getObjValAs? String "program").toOption.getD "") then throw "partition: program is not a SHA-512 digest"
  let q := (obj.getObjVal? "query").toOption.getD Json.null
  if cpoKeys q != #["name", "params", "version"] then throw "partition: the query's keys are not name, version, params"
  let name := (q.getObjValAs? String "name").toOption.getD ""
  if name.isEmpty then throw "partition: the query's name is not a non-empty string"
  let version ← match (q.getObjVal? "version").toOption with
    | some (.num n) => if n.exponent == 0 && n.mantissa ≥ 0 then pure n.mantissa.toNat else throw "partition: the query's version is not a natural number"
    | _ => throw "partition: the query's version is not a natural number"
  let ps := (q.getObjVal? "params").toOption.getD Json.null
  match ps with
  | .obj _ => pure ()
  | _ => throw "partition: the query's params are not an object"
  if Flock.HmRow.EVALUABLE.contains (name, version) then
    if cpoKeys ps != #["W", "X"] then throw s!"partition: Q_word v{version}'s parameters are not X, W"
    for k in ["X", "W"] do
      match (ps.getObjVal? k).toOption with
      | some (.num n) => if !(n.exponent == 0 && n.mantissa > 0) then throw s!"partition: {k} is not a positive integer"
      | _ => throw s!"partition: {k} is not a positive integer"
  if (name, version) == Flock.HmRow.Q_CALL then
    if cpoKeys ps != #["X"] then throw "partition: Q_call v1's parameters are not X"
    match (ps.getObjVal? "X").toOption with
    | some (.num n) =>
      if !(n.exponent == 0 && n.mantissa > 0 && n.mantissa ≤ Flock.Qcall.PARAM_MAX) then
        throw s!"partition: X is not an integer in 1..{Flock.Qcall.PARAM_MAX}"
    | _ => throw s!"partition: X is not an integer in 1..{Flock.Qcall.PARAM_MAX}"
  if let some ts := Flock.TemplateQuery.templatesOf name version ps then discard ts
  return (name, version)

theorem cpo_K : @Flock.HmRow.checkPartitionObject = checkPartitionObjectK := rfl

/-- The sorted keys of a three-member object whose keys are already ascending. -/
theorem keys3 (a b c : String) (x y z : Json) (h : [a, b, c].Pairwise (· < ·)) :
    (Array.map (fun p : String × Json => p.1) #[(a, x), (b, y), (c, z)]).qsort (· < ·) = #[a, b, c] := by
  have hm : Array.map (fun p : String × Json => p.1) #[(a, x), (b, y), (c, z)] = #[a, b, c] := by
    simp only [List.map_toArray, List.map_cons, List.map_nil]
  rw [hm]
  exact qsort_sorted _ h

theorem keys_query : Spot.keys? query = some #["name", "params", "version"] := by
  show some (Spot.sortStrings (Array.map (fun p : String × Json => p.1) #[("name", _), ("params", _), ("version", _)])) = _
  unfold Spot.sortStrings
  rw [keys3 _ _ _ _ _ _ (by decide)]

theorem build_query (h : String) : Spot.Partition.build h query =
    Json.mkObj [("format", Flock.HmRow.PARTITION_V1), ("program", h), ("query", query)] := by
  unfold Spot.Partition.build
  rw [keys_query]
  rfl

theorem cpoKeys_build (h : String) :
    cpoKeys (Json.mkObj [("format", Flock.HmRow.PARTITION_V1), ("program", h), ("query", query)]) =
      #["format", "program", "query"] := by
  show (Array.map (fun p : String × Json => p.1) #[("format", _), ("program", _), ("query", _)]).qsort (· < ·) = _
  exact keys3 _ _ _ _ _ _ (by decide)

theorem cpoKeys_query (j : Json) (hj : j = query) : cpoKeys j = #["name", "params", "version"] := by
  subst hj
  show (Array.map (fun p : String × Json => p.1) #[("name", _), ("params", _), ("version", _)]).qsort (· < ·) = _
  exact keys3 _ _ _ _ _ _ (by decide)

theorem templatesOf_q (n : String) (v : Nat) (P : Json) (hn : n = "Q_template_instance") (hv : v = 0)
    (hP : P = Json.mkObj [("template", .str templateId)]) :
    Flock.TemplateQuery.templatesOf n v P = some (pure #[templateId]) := by
  subst hn hv hP
  unfold Flock.TemplateQuery.templatesOf
  dsimp only
  split
  · split
    · rename_i kvs heq
      have hk : ((Array.map (fun x => x.fst) kvs.toArray).qsort fun x1 x2 => decide (x1 < x2)) = #["template"] := by
        unfold Json.mkObj at heq
        injection heq with h
        subst h
        show (Array.map (fun p : String × Json => p.1) #[("template", Json.str templateId)]).qsort (· < ·) = _
        have hm : Array.map (fun p : String × Json => p.1) #[("template", Json.str templateId)] = #["template"] := by
          simp only [List.map_toArray, List.map_cons, List.map_nil]
        rw [hm]
        exact qsort_sorted _ (by decide)
      rw [hk]
      rfl
    · rename_i hx
      exact (hx _ rfl).elim
  · rename_i h; exact absurd rfl h

theorem hex_field (h : String) (hh : Flock.HmRow.isSha512Hex h = true) :
    Flock.HmRow.isSha512Hex (((Json.mkObj [("format", Flock.HmRow.PARTITION_V1), ("program", h), ("query", query)]).getObjValAs?
      String "program").toOption.getD "") = true := hh

fwd_def cpoV cpo_V (h : String) (hh : Flock.HmRow.isSha512Hex h = true) :
    checkPartitionObjectK (Json.mkObj [("format", Flock.HmRow.PARTITION_V1), ("program", h), ("query", query)])
  by [hex_field h hh, templatesOf_q, cpoKeys_build, cpoKeys_query, qsort_sorted] block [String.hash, Sha512.hash] shallow []

theorem valid_build (h : String) (hh : Flock.HmRow.isSha512Hex h = true) :
    Spot.Partition.valid (Spot.Partition.build h query) = true := by
  unfold Spot.Partition.valid
  rw [cpo_K, build_query, cpo_V h hh]
  rfl

/-- The decoded program with its digest a variable: the walk below is generic in it. -/
def ovGen {S : Type} (e : S) (c : S → Nat → Bool) (ins : S → Nat → S) (P : Flock.Program.Prog) : Option Json := Id.run do
  let mut seen : S := e
  let mut todo : Array Nat := #[P.root]
  let mut at_ := 0
  for _ in [0:Steps.FUEL] do
    if at_ ≥ todo.size then break
    let d := todo[at_]!
    at_ := at_ + 1
    if c seen d then continue
    seen := ins seen d
    let defn := P.defs[d]!
    if defn.prim then continue
    let body := defn.body
    for k in [0:body.nodes.size] do
      let node := body.nodes[k]!
      for i in [0:node.args.size] do
        for part in node.args[i]!.seq.parts do
          let names : Array (Bool × Nat) := match part with
            | .affine p idx _ c => if c > 0 then #[(p, idx)] else #[]
            | .strided p idx _ _ _ _ c => if c > 0 then #[(p, idx)] else #[]
            | .explicit it => it.map fun r => (r.p, r.i)
          for (p, idx) in names do
            if !(if p then idx < body.params.size else idx < k) then
              return some (Json.mkObj [("definition", Json.str defn.key), ("node", Lean.toJson k), ("arg", Lean.toJson i),
                ("reference", Json.arr #[Json.str (if p then "p" else "n"), Lean.toJson idx]), ("why", Json.str "not topological")])
      todo := todo.push node.fn
  if at_ < todo.size then return some (Json.mkObj [("why", Json.str "a walk of more steps than this verifier follows")])
  return none

theorem ov_gen (P : Flock.Program.Prog) : P.orderViolation = ovGen (∅ : Std.HashSet Nat) (·.contains ·) (·.insert ·) P := rfl

theorem ov_fn : ovGen (fun _ => false) (fun f d => f d) (fun f d x => d == x || f x) decV = none := by decide +kernel

def stepMap {β γ : Type} (Φ : β → γ) : ForInStep β → ForInStep γ
  | .done b => .done (Φ b)
  | .yield b => .yield (Φ b)

theorem forIn_sim {α β γ δ : Type} (xs : List α) (b : β) (f : α → β → Id (ForInStep β)) (g : α → γ → Id (ForInStep γ))
    (Φ : β → γ) (post1 : β → Id δ) (post2 : γ → Id δ) (hpost : ∀ r, post2 (Φ r) = post1 r)
    (h : ∀ a b, g a (Φ b) = stepMap Φ (f a b)) :
    (forIn xs b f >>= post1 : Id δ) = (forIn xs (Φ b) g >>= post2 : Id δ) := by
  induction xs generalizing b with
  | nil => exact (hpost b).symm
  | cons x xs ih =>
    simp only [List.forIn_cons, h]
    cases f x b with
    | done b' => exact (hpost b').symm
    | yield b' => exact ih b'

theorem ovGen_map {S T : Type} (e : S) (c : S → Nat → Bool) (i : S → Nat → S) (c' : T → Nat → Bool)
    (i' : T → Nat → T) (φ : S → T) (hc : ∀ s d, c s d = c' (φ s) d)
    (hi : ∀ s d, φ (i s d) = i' (φ s) d) (P : Flock.Program.Prog) : ovGen e c i P = ovGen (φ e) c' i' P := by
  unfold ovGen
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range']
  refine forIn_sim _ ((none : Option (Option Json)), e, #[P.root], 0) _ _
    (fun st => (st.1, φ st.2.1, st.2.2)) _ _ (fun _ => rfl) ?_
  rintro a ⟨r, s, t, n⟩
  dsimp only
  rw [← hc]
  split
  · rfl
  split
  · rfl
  split
  · show _ = ForInStep.yield (_, φ _, _, _); rw [hi]; rfl
  simp only [bind]
  generalize (forIn (m := Id) (List.range' 0 _) ((none : Option (Option Json)), t) _) = res
  rcases res with ⟨_ | x, t'⟩ <;> (rw [← hi]; rfl)

theorem ov_hs : decV.orderViolation = none := by
  refine (ov_gen decV).trans ((ovGen_map (∅ : Std.HashSet Nat) (fun s x => s.contains x) (fun s x => s.insert x)
    (fun f d => f d) (fun f d x => d == x || f x) (fun s x => s.contains x)
    (fun _ _ => rfl) (fun s d => by funext x; simp [Std.HashSet.contains_insert]) decV).trans ?_)
  have : (fun x => (∅ : Std.HashSet Nat).contains x) = fun _ => false := by funext x; simp
  rw [this]; exact ov_fn

theorem valid_obj (h : String) (hh : Flock.HmRow.isSha512Hex h = true) :
    Spot.Partition.valid (Json.mkObj [("format", Flock.HmRow.PARTITION_V1), ("program", h), ("query", query)]) = true := by
  rw [← build_query]; exact valid_build h hh

def progB (b : ByteArray) (ix : Std.HashMap String Nat) : Flock.Program.Prog :=
  { defs := #[defIn16, defNot16, defUnit, defRoot], index := ix, root := 3, programFn := "HsProg_v0", sha512 := b }

theorem progB_defs (b : ByteArray) (ix : Std.HashMap String Nat) :
    (progB b ix).defs = #[defIn16, defNot16, defUnit, defRoot] := rfl
theorem progB_root (b : ByteArray) (ix : Std.HashMap String Nat) : (progB b ix).root = 3 := rfl
theorem ovGen_congr {S : Type} (e : S) (c : S → Nat → Bool) (i : S → Nat → S) (P Q : Flock.Program.Prog)
    (h1 : P.defs = Q.defs) (h2 : P.root = Q.root) : ovGen e c i P = ovGen e c i Q := by
  unfold ovGen; rw [h1, h2]

theorem progB_order (b : ByteArray) (ix : Std.HashMap String Nat) : (progB b ix).orderViolation = none := by
  rw [ov_gen]
  exact (ovGen_congr _ _ _ (progB b ix) decV (by rw [decV_defs]; rfl) (by rw [decV_root]; rfl)).trans
    ((ov_gen decV).symm.trans ov_hs)

def amap {α β : Type} (f : α → β) (a : Array α) : Array β := (a.toList.map f).toArray

theorem map_amap {α β : Type} (f : α → β) (a : Array α) : a.map f = amap f a := by
  unfold amap; rw [← Array.toList_map, Array.toArray_toList]

def dnodeOf (x : Flock.Program.Node) : Flock.TemplateQuery.DNode := { form := x.form.name, fn := x.fn, n := x.n }

def ddefOf (d : Flock.Program.Defn) : String × Flock.TemplateQuery.DDef :=
  (d.key, { prim := d.prim, nodes := amap dnodeOf d.body.nodes })

def descB (b : ByteArray) : Flock.TemplateQuery.Desc where
  callees := amap (·.key) #[defIn16, defNot16, defUnit, defRoot]
  defs := amap ddefOf #[defIn16, defNot16, defUnit, defRoot]
  root := (#[defIn16, defNot16, defUnit, defRoot].getD 3 default).key
  sha512 := b

theorem ofProg_progB (b : ByteArray) (ix : Std.HashMap String Nat) :
    Flock.TemplateQuery.Desc.ofProg (progB b ix) = descB b := by
  unfold Flock.TemplateQuery.Desc.ofProg descB
  rw [progB_defs, progB_root]
  simp only [map_amap]
  rfl

theorem sha_ofProg (b : ByteArray) (ix : Std.HashMap String Nat) : Spot.Program.sha512 ⟨descB b, progB b ix⟩ = b := rfl


theorem isInputNode_eq (d : Flock.TemplateQuery.Desc) (x : Flock.TemplateQuery.DNode) :
    d.isInputNode x = (Flock.TemplateQuery.isInput (d.callees.getD x.fn "") &&
      (d.def? (d.callees.getD x.fn "")).any (·.prim)) := rfl

theorem isInput_in16 (s : String) (h : s = "Input16_v1") : Flock.TemplateQuery.isInput s = true := by
  subst h; show (Program.inputId? _).isSome = _; rw [inputId_in16 _ rfl]; rfl

theorem inputId_unit : Flock.Program.inputId? "Unit_v1" = none := by
  simp +decide [ProgramProof.inputId_K, ProgramProof.widthAfterK]

theorem isInput_unit (s : String) (h : s = "Unit_v1") : Flock.TemplateQuery.isInput s = false := by
  subst h; show (Program.inputId? _).isSome = _; rw [inputId_unit]; rfl

/-- `Ty.widths`, by structural recursion where the kernel needs it: a value. -/
def widthsK : Flock.Program.Ty → Array Nat
  | .value w => #[w]
  | t => t.widths

theorem widths_eq : @Flock.Program.Ty.widths = widthsK := by
  funext t
  cases t with
  | value w => simp [widthsK, Flock.Program.Ty.widths]
  | array n e => simp [widthsK]
  | tuple fs => simp [widthsK]
  | record fs => simp [widthsK]

theorem qsort_one {α : Type} (a : Array α) (lt : α → α → Bool) (h : a.size = 1) : a.qsort lt = a := by
  unfold Array.qsort
  simp only [h]
  unfold Array.qsort.sort
  simp

fwd_def popL layout_L (b : ByteArray) (ix : Std.HashMap String Nat) (hb : Flock.HmRow.isSha512Hex (Flock.hexOf b) = true) :
    Spot.Slots.layout ⟨.ofProg (progB b ix), progB b ix⟩ query
  by [valid_obj (Flock.hexOf b) hb, build_query, ofProg_progB, sha_ofProg b ix, forIn_range_ex, forIn_arr_ex, forIn_list_ex, forInEx.eq_def, isInputNode_eq, isInput_in16, isInput_unit, progB_order, templatesOf_q, qsort_sorted,
    leaves_eq, widths_eq, qsort_one, inputId_in16, inputId_not16', isInputId_not16, registered_not16, tyBeq_v16, tyBeq_a16]
  block [Sha512.hash] shallow []

theorem pop_L (b : ByteArray) (ix : Std.HashMap String Nat) :
    (Spot.Slots.popOf (popL b ix)).map (·.unit) = (List.range 1).map some := by kernel_rfl'

theorem decV_progB : decV = progB decV.sha512 decV.index := by
  cases h : decV with
  | mk d ix r f sh =>
    have hd : d = #[defIn16, defNot16, defUnit, defRoot] := by rw [← decV_defs, h]
    have hr : r = 3 := by rw [← decV_root, h]
    have hf : f = "HsProg_v0" := by rw [← decV_programFn, h]
    subst hd hr hf; rfl

theorem decV_hex : Flock.HmRow.isSha512Hex (Flock.hexOf decV.sha512) = true := by
  rw [decV_sha512]; exact hexOf_sha512Hex _ (Flock.Sha512.hash_size _)

/-- **`ProgramFacts.popMatches`**. -/
theorem popMatches : Flock.Bridge.PopMatches desc query nUnits := by
  refine Flock.Bridge.popMatches_iff.2 ⟨program, popL decV.sha512 decV.index, program_parse, ?_, pop_L _ _⟩
  have h := layout_L decV.sha512 decV.index decV_hex
  rw [← decV_progB] at h
  rw [program_eq]; exact h

/-- **`ProgramFacts`**, proved. -/
theorem programFacts : ProgramFacts where
  program_parse := program_parse
  program_sha := program_sha
  popMatches := popMatches
  design_eq := design_eq
  lawAt_of := lawAt_of
  partitionDigest_ok := partitionDigest_ok

end Verity.Sampled.Instance.HonestSession.ProgramPop
