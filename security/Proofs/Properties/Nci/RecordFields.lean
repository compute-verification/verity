module

public import NciCheck.RecordText
public import Proofs.Properties.Nci.LeafWidthsType

/-!
# A field's leaf is a leaf of the return

An output row is one field of a Definition's return type (`RecordText.fieldsOf`), and its leaves are the return's
leaves from the field's first, the leaves of the fields before it.

* `fields_leaf`: of a list of types, a leaf of one of them is a leaf of the list's, after the leaves of the types
  before it, and the fold `Widths.wAt` runs over a Tuple's fields reads it at the field's width.
* `field_leaf`: leaf `l` of field `f` of a type is the type's leaf `o + l`, `o` the leaves of the fields before
  `f`, and the type's leaf there is as wide as the field's leaf `l`.
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Flock.Program
open NciCheck.RecordText (fieldsOf)
open NciCheck.Properties.SecurityProofs.LeafWidths (wStep foldl_leaves field_le fold_wAt leaves_tuple leaves_record
  wAt_tuple wAt_record)

/-- A list is its elements before one, the one and those after it. -/
theorem split_at {α : Type} : ∀ (L : List α) (f : ℕ) (hf : f < L.length), L = L.take f ++ L[f] :: L.drop (f + 1)
  | [], _, hf => absurd hf (Nat.not_lt_zero _)
  | _ :: _, 0, _ => rfl
  | x :: xs, f + 1, hf => congrArg (x :: ·) (split_at xs f (Nat.lt_of_succ_lt_succ hf))

/-- The fold `Widths.wAt` runs over a Record's fields is the fold over their types. -/
theorem foldl_map_wStep {α : Type} (g : α → Ty) (j : ℕ) : ∀ (L : List α) (acc : Option ℕ × ℕ),
    (L.map g).foldl (wStep (fun t : Ty => t) j) acc = L.foldl (wStep g j) acc
  | [], _ => rfl
  | x :: xs, acc => foldl_map_wStep g j xs (wStep g j acc x)

/-- **A leaf of one of a list of types is a leaf of the list's**: leaf `l` of the type at `f` is past the leaves of
the types before it and within the list's, and the fold `Widths.wAt` runs over the list reads it there at its
width. -/
theorem fields_leaf (L : List Ty) (f l : ℕ) (hf : f < L.length) (hl : l < L[f]!.leaves) :
    (L.take f).foldl (fun a t => a + t.leaves) 0 + l < (L.map fun t => t.leaves).sum ∧
      (L.foldl (wStep (fun t : Ty => t) ((L.take f).foldl (fun a t => a + t.leaves) 0 + l)) (none, 0)).1 =
        some (Widths.wAt L[f]! l) := by
  rw [getElem!_pos L f hf] at hl ⊢
  have hsplit := split_at L f hf
  have hp : (L.take f).foldl (fun a t => a + t.leaves) 0 = 0 + ((L.take f).map fun t => t.leaves).sum :=
    foldl_leaves (fun t : Ty => t) (L.take f) 0
  have hle : ((L.take f).map fun t => t.leaves).sum + L[f].leaves ≤ (L.map fun t => t.leaves).sum := by
    have h := field_le (fun t : Ty => t) (L.take f) L[f] (L.drop (f + 1))
    rw [← hsplit] at h
    exact h
  have hw := fold_wAt (fun t : Ty => t) ((L.take f).foldl (fun a t => a + t.leaves) 0 + l) (L.take f) L[f]
    (L.drop (f + 1)) 0 ((L.take f).foldl (fun a t => a + t.leaves) 0) hp (Nat.le_add_right _ _)
    (Nat.add_lt_add_left hl _)
  rw [← hsplit, Nat.add_sub_cancel_left] at hw
  exact ⟨by omega, hw⟩

/-- `field_leaf`, of a type that is its own one field. -/
theorem field_leaf_one (t : Ty) (h : fieldsOf t = [t]) (f l : ℕ) (hf : f < (fieldsOf t).length)
    (hl : l < (fieldsOf t)[f]!.leaves) :
    ((fieldsOf t).take f).foldl (fun a t => a + t.leaves) 0 + l < t.leaves ∧
      Widths.wAt t (((fieldsOf t).take f).foldl (fun a t => a + t.leaves) 0 + l) =
        Widths.wAt (fieldsOf t)[f]! l := by
  rw [h] at hf hl ⊢
  have hf0 : f = 0 := by
    have hf1 : f < 1 := hf
    omega
  subst hf0
  have h0 : ([t] : List Ty)[0]! = t := rfl
  have hz : (([t] : List Ty).take 0).foldl (fun a t => a + t.leaves) 0 = 0 := rfl
  rw [h0] at hl ⊢
  rw [hz, Nat.zero_add]
  exact ⟨hl, rfl⟩

/-- **A field's leaf is a leaf of the return**: leaf `l` of field `f` of the type `t` is `t`'s leaf `o + l`, `o` the
leaves of the fields before `f` (an output row's first leaf, `outRow`), and `t`'s leaf there is as wide as the
field's leaf `l`. -/
theorem field_leaf (t : Ty) (f l : ℕ) (hf : f < (fieldsOf t).length) (hl : l < (fieldsOf t)[f]!.leaves) :
    ((fieldsOf t).take f).foldl (fun a t => a + t.leaves) 0 + l < t.leaves ∧
      Widths.wAt t (((fieldsOf t).take f).foldl (fun a t => a + t.leaves) 0 + l) =
        Widths.wAt (fieldsOf t)[f]! l := by
  cases t with
  | value w => exact field_leaf_one (.value w) rfl f l hf hl
  | array n e => exact field_leaf_one (.array n e) rfl f l hf hl
  | tuple fs =>
    by_cases he : fs.isEmpty = true
    · exact field_leaf_one (.tuple fs) (if_pos he) f l hf hl
    · have hfo : fieldsOf (.tuple fs) = fs.toList := if_neg he
      rw [hfo] at hf hl ⊢
      obtain ⟨h1, h2⟩ := fields_leaf _ f l hf hl
      rw [leaves_tuple, wAt_tuple, h2, Option.getD_some]
      exact ⟨h1, rfl⟩
  | record fs =>
    by_cases he : fs.isEmpty = true
    · exact field_leaf_one (.record fs) (if_pos he) f l hf hl
    · have hfo : fieldsOf (.record fs) = fs.toList.map (·.2) := if_neg he
      rw [hfo] at hf hl ⊢
      obtain ⟨h1, h2⟩ := fields_leaf _ f l hf hl
      rw [foldl_map_wStep] at h2
      rw [List.map_map] at h1
      rw [leaves_record, wAt_record, h2, Option.getD_some]
      exact ⟨h1, rfl⟩

end NciCheck.Properties.Record.SecurityProofs
