module

public import Proofs.Flock.Claims.Sampling.Select.SecondWitness
public import Properties.Spot.Draw
public import Batteries.Tactic.OpenPrivate
import all Init.Data.Array.QSort.Basic
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.Nat.Bitwise.Basic
import all Init.Data.String.Defs
import all Init.Data.Ord.String
import all Init.Data.Range.Basic
import all Init.Data.Repr
import all Init.Data.Int.Repr
import all Lean.Data.Json.Basic

/-!
# `DrawMisses` and `DesignWellFormed` say something: the sample of every unit accepts, and misses nothing

The registration is `Batch`'s: a root that batches four instances of a one-gate primitive template, under the template
query naming it, sampled by the `All` rule, whose design samples the one class with probability 1.

* `drawMisses_honest`: the deployed `Slots.select` accepts it, on no bytes. The kernel evaluates it (`decide +kernel`)
  once the definitions it can't unfold are their twins: `Array.qsort` and `Flock.canon` with fuel (`qsort_fuel`,
  `canon_fuel`), `Flock.Program.Ty.widths` with fuel (`widths_fuel`), `String.startsWith` and `String.all` on the
  strings' characters (`startsWith_toList`, `String.all_bool_eq`), `[a:b]` loops over `List.range'`, and
  `Prog.orderViolation`, whose loop runs to `Flock.Steps.FUEL` over a hash set, evaluated once (`orderViolation_batch`).
* `drawMisses_nontrivial`: there `P_a({0}) = 0`, since the design samples slot 0 surely, and `acceptAllSelect`, which
  accepts everything and checks nothing, misses slot 0 with probability 1.
* `designWellFormed_honest`: `Slots.fileLayout` lays the registration out, and `slotDesign` gives its design.
* `designWellFormed_nontrivial`: `subset 0`, a design an accept-all `slotDesign` could give, is not well formed on a
  layout of one proof block.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

open Spot
open private Array.qpartition.loop Array.qsort.sort from Init.Data.Array.QSort.Basic
open private Flock.Program.widthAfter? from Flock.Program
open private Spot.Draw.lawKeys from Spot.Draw

namespace Properties.Spot.SecurityProofs

open Lean (Json)
open Claims.Sampling

/-! ## The kernel's twins -/

/-- `Ty.widths` with fuel: its recursion is well founded, which the kernel does not unfold. -/
def widthsFuel : ℕ → Flock.Program.Ty → Array ℕ
  | 0, t => t.widths
  | _ + 1, .value w => #[w]
  | f + 1, .array n e => let w := widthsFuel f e; (Array.range n).foldl (fun a _ => a ++ w) #[]
  | f + 1, .tuple fs => fs.foldl (fun a t => a ++ widthsFuel f t) #[]
  | f + 1, .record fs => fs.foldl (fun a t => a ++ widthsFuel f t.2) #[]

theorem widths_fuel : ∀ (f : ℕ) (t : Flock.Program.Ty), t.widths = widthsFuel f t
  | 0, _ => rfl
  | f + 1, t => by
    cases t with
    | value w => rw [Flock.Program.Ty.widths]; rfl
    | array n e => rw [Flock.Program.Ty.widths]; simp only [widthsFuel, widths_fuel f]
    | tuple fs => rw [Flock.Program.Ty.widths]; simp only [widthsFuel, widths_fuel f]
    | record fs => rw [Flock.Program.Ty.widths]; simp only [widthsFuel, widths_fuel f]

/-- `String.startsWith` on a string pattern, on the strings' characters. -/
theorem startsWith_toList (s pat : String) : s.startsWith pat = pat.toList.isPrefixOf s.toList := by
  rw [Bool.eq_iff_iff, String.startsWith_string_iff, List.isPrefixOf_iff_prefix]

theorem eq_ok_of_toOption {ε α : Type} {e : Except ε α} {a : α} (h : e.toOption = some a) : e = .ok a := by
  cases e <;> simp_all [Except.toOption]

/-! ## The registration evaluated -/

deriving instance DecidableEq for Flock.TemplateQuery.TRange
deriving instance DecidableEq for Claims.Sampling.OneStage
deriving instance DecidableEq for Claims.Sampling.Design

/-- The registration's one range: root node 0, template `t`, units `0, …, 3`. -/
def batchRanges : Array Flock.TemplateQuery.TRange := #[⟨0, "t", 0, 4⟩]

theorem orderViolation_batch : Batch.program.prog.orderViolation = none := by
  obtain ⟨k, hk⟩ : ∃ k, Flock.Steps.FUEL = k + 3 := ⟨2 ^ 62 - 3, by unfold Flock.Steps.FUEL; decide⟩
  unfold Flock.Program.Prog.orderViolation
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, hk, Nat.sub_zero, Nat.add_sub_cancel,
    Nat.div_one, List.range'_succ]
  simp [Batch.program, Batch.prog, Batch.root, Batch.template, Batch.body, Batch.node, Std.HashSet.contains_insert]

theorem valid_batch :
    Spot.Partition.valid (Spot.Partition.build (Flock.hexOf Batch.program.sha512) Batch.query) = true := by
  unfold Spot.Partition.valid Spot.Partition.build Flock.HmRow.checkPartitionObject Flock.HmRow.isSha512Hex
    Flock.TemplateQuery.templatesOf Spot.keys? Spot.sortStrings
  simp only [qsort_fuel, String.all_bool_eq]
  decide +kernel

theorem templateQuery_batch :
    (Flock.TemplateQuery.evaluate Batch.program.desc #["t"]).toOption = some batchRanges := by
  unfold Flock.TemplateQuery.evaluate Flock.TemplateQuery.isInput Flock.Program.isInputId Flock.Program.inputId?
    Flock.Program.widthAfter?
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.forIn'_eq_forIn'_range', startsWith_toList]
  decide +kernel

theorem evaluate_batch : Spot.Partition.evaluate Batch.program Batch.query = .ok batchRanges := by
  apply eq_ok_of_toOption
  unfold Spot.Partition.evaluate
  simp only [valid_batch, orderViolation_batch]
  unfold Spot.Partition.build Spot.keys? Spot.sortStrings Flock.TemplateQuery.templatesOf
    Flock.TemplateQuery.evaluate Flock.TemplateQuery.isInput Flock.Program.isInputId Flock.Program.inputId?
    Flock.Program.widthAfter? Spot.Partition.checkTemplates Spot.Partition.templateBits
  simp only [qsort_fuel, widths_fuel 4, startsWith_toList, Std.Legacy.Range.forIn_eq_forIn_range',
    Std.Legacy.Range.forIn'_eq_forIn'_range']
  decide +kernel

theorem layout_batch : ∃ l, Slots.fileLayout default Batch.program Batch.query = .ok l ∧
    (slotPartition l).classes.map (fun c => c.flatten.length) = [4] := by
  have hb : (match Slots.fileLayout default Batch.program Batch.query with
      | .ok l => decide ((slotPartition l).classes.map (fun c => c.flatten.length) = [4])
      | .error _ => false) = true := by
    unfold Slots.fileLayout Spot.Partition.strata
    simp only [evaluate_batch]
    decide +kernel
  revert hb
  cases Slots.fileLayout default Batch.program Batch.query with
  | error => simp
  | ok l => exact fun hb => ⟨l, rfl, of_decide_eq_true hb⟩

theorem slotDesign_batch : slotDesign .all Batch.program Batch.query none = .ok ⟨[.all]⟩ := by
  apply eq_ok_of_toOption
  unfold slotDesign Spot.Partition.units
  simp only [evaluate_batch]
  unfold Select.Rule.lawOver Draw.deriveLaw
  decide +kernel

/-- `P_a({0})` of the `All` design on the registration's layout: slot 0 is sampled surely. -/
theorem missBound_batch : missBound .all default Batch.program Batch.query none {0} = 0 := by
  obtain ⟨l, hl, hc⟩ := layout_batch
  obtain ⟨c, hcl, hlen⟩ := List.map_eq_singleton_iff.1 hc
  have hw : window 0 4 ({0} : Finset ℕ) ≠ ∅ := Finset.ne_empty_of_mem (a := 0) (by simp [window])
  simp only [missBound, hl, slotDesign_batch, Design.Pa, hcl, classesPa, hlen, OneStage.Pa, hw, ↓reduceIte, zero_mul]

/-! ## The witnesses -/

theorem drawMisses_honest :
    (Slots.select .all default Batch.program Batch.query none none ByteArray.empty).isOk = true := by
  unfold Slots.select Slots.fileLayout Spot.Partition.strata Select.select Spot.Partition.units
  simp only [evaluate_batch]
  unfold Select.Rule.lawOver Draw.deriveLaw Draw.checkDraw Draw.checkDrawFull Draw.sameLaw Draw.lawOf
    Spot.Draw.lawKeys keys? sortStrings jeq
  simp only [canon_fuel 16, qsort_fuel]
  decide +kernel

theorem drawMisses_nontrivial : ¬ Claims.Pr (fun src : Fin 0 → Fin 256 =>
      (acceptAllSelect .all default Batch.program Batch.query none none (bytesOf src)).isOk = true ∧
        Misses {0} (checkedOf (acceptAllSelect .all default Batch.program Batch.query none none (bytesOf src)))) ≤
      missBound .all default Batch.program Batch.query none {0} := by
  have h1 : Claims.Pr (fun src : Fin 0 → Fin 256 =>
      (acceptAllSelect .all default Batch.program Batch.query none none (bytesOf src)).isOk = true ∧
        Misses {0} (checkedOf (acceptAllSelect .all default Batch.program Batch.query none none (bytesOf src)))) = 1 :=
    (Claims.Pr_congr fun _ => ⟨fun _ => trivial, fun _ => ⟨rfl, fun _ _ => by
      simp [checkedOf, acceptAllSelect]⟩⟩).trans Pr_true
  rw [h1, missBound_batch]
  norm_num

theorem designWellFormed_honest : ∃ l D, Slots.fileLayout default Batch.program Batch.query = .ok l ∧
    slotDesign .all Batch.program Batch.query none = .ok D :=
  let ⟨l, hl, _⟩ := layout_batch
  ⟨l, _, hl, slotDesign_batch⟩

theorem designWellFormed_nontrivial : ¬ (⟨[.subset 0]⟩ : Design).WellFormed (slotPartition Batch.layout) := by
  rintro ⟨-, h⟩
  have := h _ (List.mem_singleton.2 rfl)
  simp [OneStage.WellFormed] at this

end Properties.Spot.SecurityProofs
