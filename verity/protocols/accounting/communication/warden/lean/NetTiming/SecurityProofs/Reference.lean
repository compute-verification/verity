import NetTiming.Protocol
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Finset.Dedup

/-!
# The Program's `List` definitions and their `Finset` reference forms

The Program (`NetTiming.Protocol`) is Lean core only, so it counts and sums over `List.finRange` and keeps `Σ_sync` as
two duplicate-free lists, as Python does. Each definition here is the plain `Finset` form of one of them, the form the
proofs reason with, and each theorem says the two agree: `rowCount` and `offset` are equal to their references, and a
sync set and its reference have the same values and the same size, every reference set being the image of one.
-/

namespace NetTiming

open Finset

namespace Reference

/-- `NetTiming.rowCount`'s reference: the occupied slots of a row, counted as a `Finset`. -/
def rowCount {S : ℕ} {F : Type} (r : Row S F) : Fin (S + 1) :=
  ⟨(univ.filter fun s => (r s).isSome).card, Nat.lt_succ_of_le ((card_filter_le _ _).trans (by simp))⟩

/-- `NetTiming.offset`'s reference: the frames released before bucket `t`, summed over a `Finset`. -/
def offset {T S : ℕ} (n : Fin T → Fin (S + 1)) (t : Fin T) : ℕ :=
  ∑ u ∈ univ.filter (fun u => u < t), (n u : ℕ)

/-- `NetTiming.SyncSet`'s reference: `Σ_sync` as two `Finset`s. -/
structure SyncSet where
  deltas : Finset ℕ
  rhos : Finset Rat
  deltas_nonempty : deltas.Nonempty
  rhos_nonempty : rhos.Nonempty
  rhos_pos : ∀ ρ ∈ rhos, 0 < ρ

/-- `NetTiming.SyncSet.Elem`'s reference: a δ of the set and a ρ of the set. -/
abbrev SyncSet.Elem (ss : SyncSet) : Type := {d // d ∈ ss.deltas} × {ρ // ρ ∈ ss.rhos}

/-- `NetTiming.SyncSet.size`'s reference: `#Σ_sync` as a product of `Finset` cards. -/
def SyncSet.size (ss : SyncSet) : ℕ := ss.deltas.card * ss.rhos.card

end Reference

variable {T S : ℕ} {F : Type}

theorem val_rowCount (r : Row S F) : (rowCount r : ℕ) = (univ.filter fun s => (r s).isSome).card := by
  simp only [rowCount, List.countP_eq_length_filter]
  rw [← List.toFinset_card_of_nodup ((List.nodup_finRange S).filter _), List.toFinset_filter, List.toFinset_finRange]

theorem rowCount_eq_reference (r : Row S F) : rowCount r = Reference.rowCount r :=
  Fin.ext (val_rowCount r)

theorem offset_eq_reference (n : Fin T → Fin (S + 1)) (t : Fin T) : offset n t = Reference.offset n t := by
  rw [offset, Reference.offset, ← List.sum_toFinset _ ((List.nodup_finRange T).filter _), List.toFinset_filter,
    List.toFinset_finRange]
  simp only [decide_eq_true_eq]

namespace SyncSet

/-- The reference form of a sync set: its lists as `Finset`s. -/
def toReference (ss : SyncSet) : Reference.SyncSet where
  deltas := ss.deltas.toFinset
  rhos := ss.rhos.toFinset
  deltas_nonempty := List.toFinset_nonempty_iff _ |>.mpr ss.deltas_nonempty
  rhos_nonempty := List.toFinset_nonempty_iff _ |>.mpr ss.rhos_nonempty
  rhos_pos ρ h := ss.rhos_pos ρ (List.mem_toFinset.mp h)

theorem mem_toReference_deltas (ss : SyncSet) (d : ℕ) : d ∈ ss.toReference.deltas ↔ d ∈ ss.deltas :=
  List.mem_toFinset

theorem mem_toReference_rhos (ss : SyncSet) (ρ : Rat) : ρ ∈ ss.toReference.rhos ↔ ρ ∈ ss.rhos :=
  List.mem_toFinset

theorem size_toReference (ss : SyncSet) : ss.toReference.size = ss.size := by
  simp only [Reference.SyncSet.size, toReference, SyncSet.size, List.toFinset_card_of_nodup ss.deltas_nodup,
    List.toFinset_card_of_nodup ss.rhos_nodup]

theorem card_subtype_mem {α : Type} [DecidableEq α] {l : List α} (h : l.Nodup) :
    Fintype.card {x // x ∈ l} = l.length := by
  rw [Fintype.card_of_subtype l.toFinset (fun _ => List.mem_toFinset), List.toFinset_card_of_nodup h]

/-- `#Σ_sync` is the number of values of the set. -/
theorem card_elem (ss : SyncSet) : Fintype.card ss.Elem = ss.size := by
  rw [Fintype.card_prod, card_subtype_mem ss.deltas_nodup, card_subtype_mem ss.rhos_nodup, SyncSet.size]

end SyncSet

/-- A sync set with a reference's values, its `Finset`s listed in some order. -/
noncomputable def Reference.SyncSet.toCode (ss : Reference.SyncSet) : NetTiming.SyncSet where
  deltas := ss.deltas.toList
  rhos := ss.rhos.toList
  deltas_nonempty := by simpa using ss.deltas_nonempty.ne_empty
  rhos_nonempty := by simpa using ss.rhos_nonempty.ne_empty
  rhos_pos ρ h := ss.rhos_pos ρ (Finset.mem_toList.mp h)
  deltas_nodup := ss.deltas.nodup_toList
  rhos_nodup := ss.rhos.nodup_toList

/-- Every reference sync set is the reference form of a sync set. -/
theorem Reference.SyncSet.toReference_toCode (ss : Reference.SyncSet) : ss.toCode.toReference = ss := by
  obtain ⟨d, r, _, _, _⟩ := ss
  simp only [SyncSet.toReference, toCode, Finset.toList_toFinset]

end NetTiming
