module

public import Proofs.Flock.SpotFlock.Instance
public import Flock.Program
public import Flock.HmRow
public import Flock.TemplateQuery
public import Spot.Partition
public import Spot.Slots
public import Spot.Select
public import Spot.Draw
public import Spot.Registration
import Proofs.Flock.Soundness.Refine.Walk
import Std.Data.HashSet.Lemmas
import Init.Data.Range.Lemmas
import all Init.Data.String.Legacy
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
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Init.Data.String.Extra
import all Init.Data.Array.QSort.Basic
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Range.Basic
import all Init.Data.Int.Repr
import all Init.Data.Repr
import all Init.Data.Nat.Fold
import all Init.Data.Array.Attach
import all Init.WF
import all Init.Data.Nat.Log2
import all Init.Data.String.Pattern.String
import all Init.Data.String.Search
import all Init.Data.String.Slice
import all Init.Data.String.TakeDrop
import all Init.Data.String.Decode
import all Init.Data.Nat.Bitwise.Basic
import all Init.Data.Vector.Basic

/-!
# A boundary registration `SpotFlock.Instance.Bnd` holds at

`bnd0 : Bnd`: the verifier's program `program0` is one primitive template `T` of one 16-bit gate, called once by the
root, its query `Q_template_instance` of `T` (`query0`), and its rule `subset:1` with no closure map and no work table.
Its layout is one proof block, unit `0` at the gates `[0, 1)` (`layout0`), its population is that one unit in slot order
(`popOk0`), and its first stage's design is `subset:1` (`design_reg0`, `firstK_reg0`).

Each fact is the verifier's own function at these values, which the kernel decides once the functions it can't unfold
are rewritten to ones it can: `Array.qsort` to `QK.qsortK` (`QK.qsort_eq`) and `Ty.widths` to `widthsK` (`widths_eq`),
each by fuel; `String.all` to a list's (`String.all_bool_eq`); the program's order check by its three steps
(`orderViolation_prog0`); and the law's text by `simp` (`split0`, `toNat_one`).
-/

namespace SpotFlock.Instance

namespace QK

variable {α : Type} (lt : α → α → Bool)

/-- `Array.qpartition.loop` by fuel. -/
public def loopK {n : ℕ} (lo hi : ℕ) (hhi : hi < n) (pivot : α) :
    (fuel : ℕ) → (as : Vector α n) → (i k : ℕ) → lo ≤ i → i ≤ k → k ≤ hi → { m // lo ≤ m ∧ m ≤ hi } × Vector α n
  | 0, as, i, _, ilo, ik, w => (⟨i, ilo, by omega⟩, as.swap i hi (by omega) hhi)
  | fuel + 1, as, i, k, ilo, ik, w =>
    if h : k < hi then
      if lt as[k] pivot then
        loopK lo hi hhi pivot fuel (as.swap i k (by omega) (by omega)) (i + 1) (k + 1) (by omega) (by omega) (by omega)
      else
        loopK lo hi hhi pivot fuel as i (k + 1) ilo (by omega) (by omega)
    else
      (⟨i, ilo, by omega⟩, as.swap i hi (by omega) hhi)

theorem loop_eq {n : ℕ} (lo hi : ℕ) (hhi : hi < n) (pivot : α) :
    ∀ (fuel : ℕ) (as : Vector α n) (i k : ℕ) (ilo : lo ≤ i) (ik : i ≤ k) (w : k ≤ hi), hi - k < fuel →
      Array.qpartition.loop lt lo hi hhi pivot as i k ilo ik w = loopK lt lo hi hhi pivot fuel as i k ilo ik w := by
  intro fuel
  induction fuel with
  | zero => intro _ _ _ _ _ _ h; omega
  | succ fuel ih =>
    intro as i k ilo ik w hf
    rw [Array.qpartition.loop, loopK]
    by_cases h : k < hi
    · simp only [h, dite_true]
      split
      · exact ih _ _ _ _ _ _ (by omega)
      · exact ih _ _ _ _ _ _ (by omega)
    · simp only [h, dite_false]

/-- `Array.qpartition` over `loopK`. -/
public def qpartitionK {n : ℕ} (as : Vector α n) (lo hi : ℕ) (w : lo ≤ hi) (hlo : lo < n) (hhi : hi < n) :
    { m // lo ≤ m ∧ m ≤ hi } × Vector α n :=
  let mid := (lo + hi) / 2
  let as := if lt as[mid] as[lo] then as.swap lo mid hlo (by omega) else as
  let as := if lt as[hi] as[lo] then as.swap lo hi hlo hhi else as
  let as := if lt as[mid] as[hi] then as.swap mid hi (by omega) hhi else as
  let pivot := as[hi]
  loopK lt lo hi hhi pivot (hi + 1) as lo lo (Nat.le_refl _) (Nat.le_refl _) w

theorem qpartition_eq {n : ℕ} (as : Vector α n) (lo hi : ℕ) (w : lo ≤ hi) (hlo : lo < n) (hhi : hi < n) :
    Array.qpartition as lt lo hi w hlo hhi = qpartitionK lt as lo hi w hlo hhi := by
  unfold Array.qpartition qpartitionK
  apply loop_eq
  omega

/-- `Array.qsort.sort` by fuel. -/
public def sortK : (fuel : ℕ) → {n : ℕ} → (as : Vector α n) → (lo hi : ℕ) → lo ≤ hi → lo < n → hi < n → Vector α n
  | 0, _, as, _, _, _, _, _ => as
  | fuel + 1, _, as, lo, hi, w, hlo, hhi =>
    if h₁ : lo < hi then
      match qpartitionK lt as lo hi w hlo hhi with
      | ⟨⟨mid, hmid⟩, as⟩ =>
        if h₂ : mid ≥ hi then as
        else sortK fuel (sortK fuel as lo mid hmid.1 hlo (by omega)) (mid + 1) hi (by omega) (by omega) hhi
    else as

theorem sort_eq : ∀ (fuel : ℕ) {n : ℕ} (as : Vector α n) (lo hi : ℕ) (w : lo ≤ hi) (hlo : lo < n) (hhi : hi < n),
    hi - lo < fuel → Array.qsort.sort lt as lo hi w hlo hhi = sortK lt fuel as lo hi w hlo hhi := by
  intro fuel
  induction fuel with
  | zero => intro _ _ _ _ _ _ _ h; omega
  | succ fuel ih =>
    intro n as lo hi w hlo hhi hf
    rw [Array.qsort.sort, sortK]
    by_cases h₁ : lo < hi
    · simp only [h₁, dite_true, qpartition_eq]
      have hm := (qpartitionK lt as lo hi w hlo hhi).1.property
      by_cases h₂ : (qpartitionK lt as lo hi w hlo hhi).1.val ≥ hi
      · simp only [h₂, dite_true]
      · simp only [h₂, dite_false]
        have key : ∀ m, lo ≤ m → m ≤ hi → ¬m ≥ hi → hi - (m + 1) < fuel ∧ m - lo < fuel := by intro m; omega
        rw [ih, ih]
        · exact (key _ hm.1 hm.2 h₂).2
        · exact (key _ hm.1 hm.2 h₂).1
    · simp only [h₁, dite_false]

/-- `Array.qsort` by fuel, which the kernel evaluates. -/
public def qsortK (as : Array α) (lo hi : ℕ) : Array α :=
  if h : as.size = 0 then as
  else
    let lo := min lo (as.size - 1)
    let hi := max lo (min hi (as.size - 1))
    (sortK lt as.size as.toVector lo hi (Nat.le_max_left _ _) (by omega) (by omega)).toArray

public theorem qsort_eq (as : Array α) (lo hi : ℕ) : as.qsort lt lo hi = qsortK lt as lo hi := by
  unfold Array.qsort qsortK
  split
  · rfl
  · dsimp only
    congr 1
    apply sort_eq
    omega

end QK

open Flock.Program

/-- `Ty.widths` by fuel, which the kernel evaluates. -/
public def widthsK : ℕ → Ty → Array ℕ
  | 0, _ => #[]
  | _ + 1, .value w => #[w]
  | f + 1, .array n e => let w := widthsK f e; (Array.range n).foldl (fun a _ => a ++ w) #[]
  | f + 1, .tuple fs => fs.foldl (fun a t => a ++ widthsK f t) #[]
  | f + 1, .record fs => fs.foldl (fun a t => a ++ widthsK f t.2) #[]

theorem widths_fuel : ∀ (f : ℕ) (t : Ty), sizeOf t < f → t.widths = widthsK f t := by
  intro f
  induction f with
  | zero => intro _ h; omega
  | succ f ih =>
    intro t ht
    cases t with
    | value w => rw [Ty.widths]; rfl
    | array n e =>
      have he : sizeOf e < f := by simp only [Ty.array.sizeOf_spec] at ht; omega
      rw [Ty.widths, ih e he]
      rfl
    | tuple fs =>
      have hx : ∀ x : {t // t ∈ fs}, Ty.widths x.val = widthsK f x.val := fun x => ih _ (by
        have := Array.sizeOf_lt_of_mem x.2
        simp only [Ty.tuple.sizeOf_spec] at ht; omega)
      rw [Ty.widths, ← Array.foldl_attach (f := fun a t => a ++ Ty.widths t)]
      simp only [hx]
      rw [Array.foldl_attach (f := fun a t => a ++ widthsK f t)]
      rfl
    | record fs =>
      have hx : ∀ x : {t // t ∈ fs}, Ty.widths x.val.2 = widthsK f x.val.2 := fun x => ih _ (by
        have := Array.sizeOf_lt_of_mem x.2
        have h2 : sizeOf x.val.2 < sizeOf x.val := by
          obtain ⟨⟨a, b⟩, _⟩ := x
          simp only [Prod.mk.sizeOf_spec]; omega
        simp only [Ty.record.sizeOf_spec] at ht; omega)
      rw [Ty.widths, ← Array.foldl_attach (f := fun a (t : String × Ty) => a ++ Ty.widths t.2)]
      simp only [hx]
      rw [Array.foldl_attach (f := fun a (t : String × Ty) => a ++ widthsK f t.2)]
      rfl

public theorem widths_eq (t : Ty) : t.widths = widthsK (sizeOf t + 1) t := widths_fuel _ t (by omega)

theorem splitOn0 : "subset:1".splitOn ":" = ["subset", "1"] := by
  simp (config := {decide := true}) [String.splitOn, String.splitOnAux, String.Pos.Raw.get, String.Pos.Raw.next,
    String.Pos.Raw.extract]

theorem toNat_one : "1".toNat! = 1 := by
  simp [String.toNat!, String.Slice.toNat!, String.Slice.isNat, String.Slice.forIn_eq_forIn_toList,
    String.Slice.foldl_eq_foldl_toList]

open private split1 from Spot.Select in
theorem split0 : split1 "subset:1" ":" = ("subset", "1") := by
  unfold split1
  rw [splitOn0]
  kernel_rfl

@[expose] public section

open Lean (Json)

set_option maxHeartbeats 0 -- the closed checks below leave the verifier's evaluation to the kernel
set_option maxRecDepth 100000

/-- **The template** `T`: a primitive of one 16-bit gate. -/
def tDef : Defn :=
  { key := "T", id := "T_v1", prim := true, params := #[], ret := .value 16, gates := 1, paramLeaves := 0,
    outLeaves := 1 }

/-- **The root** `main`: one call of `T`, which it returns. -/
def mainDef : Defn :=
  { key := "main", id := "main_v1", prim := false, params := #[], ret := .value 16,
    body :=
      { params := #[], retType := .value 16, nodes := #[{ form := .call, fn := 0, args := #[], out := .value 16 }],
        ret := Seq.ofParts #[.affine false 0 0 1], offsets := #[0] },
    gates := 1, paramLeaves := 0, outLeaves := 1 }

def prog0 : Prog :=
  { defs := #[tDef, mainDef], index := Std.HashMap.ofList [("T", 0), ("main", 1)], root := 1, programFn := "main",
    sha512 := ⟨Array.replicate 64 0⟩ }

theorem orderViolation_prog0 : prog0.orderViolation = none := by
  unfold Prog.orderViolation
  dsimp only
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  rw [show (Std.Legacy.Range.mk 0 Flock.Steps.FUEL 1 (by decide)).size = (2 ^ 62 - 3) + 1 + 1 + 1 by
    simp only [Std.Legacy.Range.size]; unfold Flock.Steps.FUEL; decide]
  rw [List.range'_succ, List.range'_succ, List.range'_succ]
  simp only [List.forIn_cons]
  simp only [prog0, mainDef, tDef]
  simp

/-- **The verifier's program**: `prog0` with its descriptor. -/
def program0 : Spot.Program := { desc := .ofProg prog0, prog := prog0 }

theorem orderViolation_program0 : program0.prog.orderViolation = none := orderViolation_prog0

/-- **The query**: the instances of `T`. -/
def query0 : Json :=
  Json.mkObj [("name", "Q_template_instance"), ("version", Lean.toJson (0 : ℕ)),
    ("params", Json.mkObj [("template", "T")])]

theorem checkPartitionObject0 :
    Flock.HmRow.checkPartitionObject (Spot.Partition.build (Flock.hexOf program0.sha512) query0) =
      .ok ("Q_template_instance", 0) := by
  simp only [Flock.HmRow.checkPartitionObject, Spot.Partition.build, Spot.keys?, Spot.sortStrings, QK.qsort_eq,
    Flock.HmRow.isSha512Hex, String.all_bool_eq]
  unfold Flock.TemplateQuery.templatesOf
  simp (config := {iota := false}) only [QK.qsort_eq]
  kernel_rfl

theorem valid0 : Spot.Partition.valid (Spot.Partition.build (Flock.hexOf program0.sha512) query0) = true := by
  simp only [Spot.Partition.valid, checkPartitionObject0]
  kernel_rfl

/-- **The query's ranges**: root node `0`, `T`'s one instance, unit `0`. -/
def rs0 : Array Flock.TemplateQuery.TRange := #[⟨0, "T", 0, 1⟩]

theorem evaluate0 : Spot.Partition.evaluate program0 query0 = .ok rs0 := by
  unfold Spot.Partition.evaluate
  simp (config := {iota := false}) only [valid0, orderViolation_program0]
  unfold Flock.TemplateQuery.templatesOf Spot.Partition.checkTemplates Spot.Partition.templateBits
  simp (config := {iota := false}) only [QK.qsort_eq, widths_eq, Spot.Partition.build, Spot.keys?, Spot.sortStrings]
  kernel_rfl

theorem unitGates0 : Spot.Partition.unitGates program0 query0 = .ok #[(0, 1)] := by
  unfold Spot.Partition.unitGates Spot.Partition.unitGatesBy
  simp (config := {iota := false}) only [evaluate0]
  unfold Flock.TemplateQuery.templatesOf
  simp (config := {iota := false}) only [QK.qsort_eq, Spot.Partition.build, Spot.keys?, Spot.sortStrings]
  kernel_rfl

theorem layout0 : Spot.Slots.layout program0 query0 = .ok (oneLayout (0, 1)) := by
  unfold Spot.Slots.layout
  simp (config := {iota := false}) only [evaluate0, unitGates0]
  kernel_rfl

theorem units0 : Spot.Partition.units program0 query0 = .ok 1 := by
  unfold Spot.Partition.units
  simp (config := {iota := false}) only [evaluate0]
  kernel_rfl

theorem popOk0 : Spot.Slots.popOk program0 query0 1 = true := by
  unfold Spot.Slots.popOk
  simp (config := {iota := false}) only [layout0]
  kernel_rfl

/-- **The law** `subset:1` as Spot's design object. -/
def subsetJ : Json := Json.mkObj [("law", "subset"), ("k", Lean.toJson (1 : ℕ))]

open private digits from Spot.Select in
theorem lawOver0 : (Spot.Select.Rule.law "subset:1").lawOver 1 = .ok subsetJ := by
  unfold Spot.Select.Rule.lawOver Spot.Select.parseDesign digits
  simp only [split0, toNat_one, String.all_bool_eq]
  kernel_rfl

theorem deriveLaw0 : Spot.Draw.deriveLaw program0 query0 subsetJ none = .ok none := by kernel_rfl

theorem flockLaw0 : Spot.Draw.flockLaw subsetJ 1 = .ok (.subset 1) := by
  unfold Spot.Draw.flockLaw
  simp (config := {iota := false}) only [Spot.keys?, Spot.sortStrings, QK.qsort_eq]
  kernel_rfl

open Claims Claims.Sampling SamplingDeploy Properties.Spot

/-- **The verifier's copies**: `program0`, `query0` and the law `subset:1`, with no work table, ports, anchors or
earlier records. -/
def verifier0 : Spot.Registration.Verifier :=
  { program := program0, query := query0, law := subsetJ, workTable := none, scheme := "", ports := #[],
    leafLayer := none, anchors := #[], log := #[], partition := none }

/-- **The boundary registration**: `verifier0` under the rule `subset:1`, with no closure map. -/
def reg0 : Reg := { verifier := verifier0, rule := .law "subset:1", closure := none }

theorem design_reg0 : slotDesign reg0.rule reg0.program reg0.query reg0.table = .ok design1 := by
  show slotDesign (.law "subset:1") program0 query0 none = .ok design1
  simp only [slotDesign, units0, bind, Except.bind, lawOver0, deriveLaw0, Option.map_none, Option.getD_none, flockLaw0]
  rfl

theorem firstK_reg0 : firstK reg0 = some 1 := by
  unfold firstK firstKOf
  simp only [reg0, Reg.program, Reg.query, verifier0, units0, lawOver0, flockLaw0]

/-- **A boundary registration `Bnd` holds at**: `reg0`, its layout `oneLayout (0, 1)`. -/
def bnd0 : Bnd where
  g := reg0
  gs := (0, 1)
  layout := layout0
  design := design_reg0
  firstK := firstK_reg0
  pop := fun n h => by
    have h' : Spot.Partition.units program0 query0 = .ok n := h
    rw [units0] at h'
    cases h'
    exact popOk0

end

end SpotFlock.Instance
