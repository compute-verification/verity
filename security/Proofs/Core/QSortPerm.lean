module

public import Batteries.Tactic.OpenPrivate
import all Init.Data.Array.QSort.Basic

/-!
# `Array.qsort` permutes its array

`qsort_perm`: `Array.qsort` returns a permutation of the array it was given, under any comparison. Every step of it
swaps two elements: the three swaps that pick `qpartition`'s pivot, its loop's (`qpartition_loop_perm`,
`qpartition_perm`), and `sort`'s two halves (`sort_perm`).
-/

@[expose] public section

set_option Elab.async false

open private Array.qsort.sort Array.qpartition.loop from Init.Data.Array.QSort.Basic

namespace Verity.QSort

variable {α : Type} {n : Nat}

theorem perm_ite_swap (as : Vector α n) (b : Bool) {i j : Nat} (hi : i < n) (hj : j < n) :
    (if b then as.swap i j hi hj else as).Perm as := by
  split
  · exact Vector.swap_perm hi hj
  · exact Vector.Perm.refl as

/-- `qpartition`'s loop permutes the array. -/
private theorem qpartition_loop_perm (lt : α → α → Bool) {lo hi : Nat} (hhi : hi < n) (pivot : α) :
    ∀ (d : Nat) (as : Vector α n) (i k : Nat) (ilo : lo ≤ i) (ik : i ≤ k) (w : k ≤ hi), hi - k = d →
      (Array.qpartition.loop lt lo hi hhi pivot as i k ilo ik w).2.Perm as := by
  intro d
  induction d using Nat.strongRecOn with
  | _ d ih =>
    intro as i k ilo ik w hd
    unfold Array.qpartition.loop
    split
    · rename_i hk
      split
      · exact (ih (hi - (k + 1)) (by omega) (as.swap i k (by omega) (by omega)) (i + 1) (k + 1) (by omega)
          (by omega) (by omega) rfl).trans (Vector.swap_perm _ _)
      · exact ih (hi - (k + 1)) (by omega) as i (k + 1) ilo (by omega) (by omega) rfl
    · exact Vector.swap_perm (by omega) hhi

/-- `qpartition` permutes the array. -/
theorem qpartition_perm (lt : α → α → Bool) (as : Vector α n) (lo hi : Nat) (w : lo ≤ hi) (hlo : lo < n)
    (hhi : hi < n) : (Array.qpartition as lt lo hi w hlo hhi).2.Perm as := by
  unfold Array.qpartition
  dsimp only
  exact (qpartition_loop_perm lt hhi _ (hi - lo) _ lo lo (Nat.le_refl _) (Nat.le_refl _) w rfl).trans
    (((perm_ite_swap _ _ _ _).trans (perm_ite_swap _ _ _ _)).trans (perm_ite_swap _ _ _ _))

/-- `qsort`'s `sort` permutes the array. -/
private theorem sort_perm (lt : α → α → Bool) :
    ∀ (d : Nat) (as : Vector α n) (lo hi : Nat) (w : lo ≤ hi) (hlo : lo < n) (hhi : hi < n), hi - lo = d →
      (Array.qsort.sort lt as lo hi w hlo hhi).Perm as := by
  intro d
  induction d using Nat.strongRecOn with
  | _ d ih =>
    intro as lo hi w hlo hhi hd
    rw [Array.qsort.sort]
    split
    · rename_i h₁
      have hp := qpartition_perm lt as lo hi w hlo hhi
      revert hp
      match Array.qpartition as lt lo hi w hlo hhi with
      | (⟨mid, hmid⟩, as') =>
        intro hp
        dsimp only at hp ⊢
        split
        · exact hp
        · exact ((ih (hi - (mid + 1)) (by omega) _ (mid + 1) hi (by omega) (by omega) hhi rfl).trans
            (ih (mid - lo) (by omega) as' lo mid hmid.1 hlo (by omega) rfl)).trans hp
    · exact Vector.Perm.refl as

/-- **`Array.qsort` returns a permutation of its array.** -/
theorem qsort_perm (as : Array α) (lt : α → α → Bool) : (as.qsort lt).Perm as := by
  unfold Array.qsort
  split
  · exact Array.Perm.refl as
  · dsimp only
    exact (sort_perm lt _ as.toVector _ _ (by omega) (by omega) (by omega) rfl).toArray

end Verity.QSort
