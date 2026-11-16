module

public import Batteries.Tactic.OpenPrivate
import all Init.Data.Array.QSort.Basic

/-!
# `Array.qsort` permutes its input

The samples' sampler sorts its values and a plan's check sorts its runs, both with `Array.qsort`. Every step of `qsort`
is a swap of two elements of a vector (`qpartition`'s three pivot swaps and its loop's, `sort`'s recursion on two
parts), and a swap permutes, so the result is a permutation of the input (`qsort_perm`): it holds the same elements,
each as often.
-/

@[expose] public section

open private Array.qpartition.loop Array.qsort.sort from Init.Data.Array.QSort.Basic

namespace Properties.Poub

variable {α : Type}

theorem swap_perm' {n : Nat} (as : Vector α n) (i j : Nat) (hi : i < n) (hj : j < n) :
    (as.swap i j hi hj).toArray.Perm as.toArray := by
  rw [Vector.toArray_swap]
  exact Array.swap_perm _ _

private theorem qpartition_loop_perm {n : Nat} (lt : α → α → Bool) {lo hi : Nat} (hhi : hi < n) (pivot : α) :
    ∀ (d : Nat) (as : Vector α n) (i k : Nat) (ilo : lo ≤ i) (ik : i ≤ k) (w : k ≤ hi), hi - k = d →
      (Array.qpartition.loop lt lo hi hhi pivot as i k ilo ik w).2.toArray.Perm as.toArray := by
  intro d
  induction d using Nat.strongRecOn with
  | _ d ih =>
    intro as i k ilo ik w hd
    unfold Array.qpartition.loop
    split
    · split
      · exact (ih (hi - (k + 1)) (by omega) _ _ _ _ _ _ rfl).trans (swap_perm' _ _ _ (by omega) (by omega))
      · exact ih (hi - (k + 1)) (by omega) _ _ _ _ _ _ rfl
    · exact swap_perm' _ _ _ (by omega) hhi

theorem ite_swap_perm {n : Nat} (as : Vector α n) (b : Bool) (i j : Nat) (hi : i < n) (hj : j < n) :
    (if b then as.swap i j hi hj else as).toArray.Perm as.toArray := by
  split
  · exact swap_perm' _ _ _ _ _
  · exact Array.Perm.refl _


/-- **`qsort` permutes its input.** -/
theorem qsort_perm (as : Array α) (lt : α → α → Bool) : (as.qsort lt).Perm as := by
  unfold Array.qsort
  split
  · exact Array.Perm.refl _
  · exact sort_perm lt _ as.toVector _ _ _ _ _ rfl

theorem qsort_toList_perm (as : Array α) (lt : α → α → Bool) : (as.qsort lt).toList.Perm as.toList :=
  Array.perm_iff_toList_perm.1 (qsort_perm as lt)

end Properties.Poub
