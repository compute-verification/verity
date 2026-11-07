module

public import Batteries.Tactic.OpenPrivate
import all Init.Data.Array.QSort.Basic

/-!
# `Array.qsort` leaves a sorted array as it is

`qsort_sorted`: an array strictly sorted by `lt` is `qsort lt`'s own output. Each partition's pivot is the middle element,
which splits a sorted range in place (`qpartition_sorted`), and both halves stay sorted (`sort_sorted`).
-/

@[expose] public section

set_option Elab.async false

open private Array.qsort.sort Array.qpartition.loop from Init.Data.Array.QSort.Basic

namespace Verity.QSort

variable {α : Type} {n : Nat}

theorem swap_self (xs : Vector α n) (i : Nat) (h : i < n) : xs.swap i i h h = xs := by
  ext k hk
  rw [Vector.getElem_swap]
  split
  · subst_vars; rfl
  · rfl

section Loop

variable (lt : α → α → Bool) {lo hi : Nat} (hhi : hi < n) (pivot : α) (as : Vector α n)

private theorem loop_skip (i : Nat) (ilo : lo ≤ i) :
    ∀ (d k : Nat) (ik : i ≤ k) (w : k ≤ hi), hi - k = d →
      (∀ j (hj : j < n), k ≤ j → j < hi → lt as[j] pivot = false) →
      Array.qpartition.loop lt lo hi hhi pivot as i k ilo ik w =
        (⟨i, ilo, by omega⟩, as.swap i hi (by omega) hhi) := by
  intro d
  induction d with
  | zero =>
    intro k ik w hd _
    rw [Array.qpartition.loop.eq_def]
    have : ¬ k < hi := by omega
    simp only [this, dite_false]
  | succ d ih =>
    intro k ik w hd hb
    rw [Array.qpartition.loop.eq_def]
    have hk : k < hi := by omega
    simp only [hk, dite_true, hb k (by omega) (Nat.le_refl _) hk, Bool.false_eq_true, ite_false]
    exact ih (k + 1) (by omega) (by omega) (by omega) (fun j hj h1 h2 => hb j hj (by omega) h2)

private theorem loop_take (m : Nat) (hm : m ≤ hi) :
    ∀ (d k : Nat) (ilo : lo ≤ k) (w : k ≤ m), m - k = d →
      (∀ j (hj : j < n), k ≤ j → j < m → lt as[j] pivot = true) →
      (∀ j (hj : j < n), m ≤ j → j < hi → lt as[j] pivot = false) →
      Array.qpartition.loop lt lo hi hhi pivot as k k ilo (Nat.le_refl _) (by omega) =
        (⟨m, by omega, hm⟩, as.swap m hi (by omega) hhi) := by
  intro d
  induction d with
  | zero =>
    intro k ilo w hd _ hb
    obtain rfl : k = m := by omega
    exact loop_skip lt hhi pivot as k ilo (hi - k) k (Nat.le_refl _) _ rfl hb
  | succ d ih =>
    intro k ilo w hd ht hb
    rw [Array.qpartition.loop.eq_def]
    have hk : k < hi := by omega
    simp only [hk, dite_true, ht k (by omega) (Nat.le_refl _) (by omega), ite_true, swap_self]
    exact ih (k + 1) (by omega) (by omega) (by omega) (fun j hj h1 h2 => ht j hj (by omega) h2) hb

end Loop

/-- `as` is strictly sorted by `lt` on `[lo, hi]`: `lt` holds of two of its elements there exactly in their order. -/
def SortedOn (lt : α → α → Bool) (lo hi : Nat) (as : Vector α n) : Prop :=
  ∀ i j (hi' : i < n) (hj : j < n), lo ≤ i → i ≤ hi → lo ≤ j → j ≤ hi → lt as[i] as[j] = decide (i < j)

theorem qpartition_sorted (lt : α → α → Bool) (as : Vector α n) {lo hi : Nat} (w : lo ≤ hi)
    (hlo : lo < n) (hhi : hi < n) (hlt : lo < hi) (hs : SortedOn lt lo hi as) :
    Array.qpartition as lt lo hi w hlo hhi = (⟨(lo + hi) / 2, by omega, by omega⟩, as) := by
  have hm1 : lo ≤ (lo + hi) / 2 := by omega
  have hm2 : (lo + hi) / 2 < hi := by omega
  unfold Array.qpartition
  have h1 : lt (as[(lo + hi) / 2]'(by omega)) as[lo] = false := by
    rw [hs ((lo + hi) / 2) lo _ _ hm1 (by omega) (Nat.le_refl _) w]; simp; omega
  have h2 : lt as[hi] as[lo] = false := by
    rw [hs hi lo _ _ w (Nat.le_refl _) (Nat.le_refl _) w]; simp; omega
  have h3 : lt (as[(lo + hi) / 2]'(by omega)) as[hi] = true := by
    rw [hs ((lo + hi) / 2) hi _ _ hm1 (by omega) w (Nat.le_refl _)]; simp; omega
  simp only [h1, h2, h3, Bool.false_eq_true, ite_false, ite_true, Vector.getElem_swap_right]
  rw [loop_take lt hhi _ _ ((lo + hi) / 2) (Nat.le_of_lt hm2) _ lo (Nat.le_refl _) hm1 rfl]
  · simp only [Vector.swap_swap]
  · intro j hj h4 h5
    rw [Vector.getElem_swap_of_ne (by omega) (by omega), hs j ((lo + hi) / 2) _ _ h4 (by omega) hm1 (by omega)]
    simp; omega
  · intro j hj h4 h5
    rw [Vector.getElem_swap]
    split
    · rename_i e; subst e
      rw [hs hi ((lo + hi) / 2) _ _ w (Nat.le_refl _) hm1 (by omega)]; simp; omega
    · split
      · omega
      · rw [hs j ((lo + hi) / 2) _ _ (by omega) (by omega) hm1 (by omega)]; simp; omega

private theorem sort_sorted (lt : α → α → Bool) :
    ∀ (d : Nat) (as : Vector α n) (lo hi : Nat) (w : lo ≤ hi) (hlo : lo < n) (hhi : hi < n), hi - lo < d →
      SortedOn lt lo hi as → Array.qsort.sort lt as lo hi w hlo hhi = as := by
  intro d
  induction d with
  | zero => intro _ _ _ _ _ _ hd; omega
  | succ d ih =>
    intro as lo hi w hlo hhi hd hs
    rw [Array.qsort.sort.eq_def]
    split
    · rename_i hlt
      rw [qpartition_sorted lt as w hlo hhi hlt hs]
      simp only
      split
      · omega
      · rw [ih as lo ((lo + hi) / 2) _ hlo _ (by omega)
          (fun i j _ _ h1 h2 h3 h4 => hs i j _ _ h1 (by omega) h3 (by omega))]
        exact ih as ((lo + hi) / 2 + 1) hi _ _ hhi (by omega)
          (fun i j _ _ h1 h2 h3 h4 => hs i j _ _ (by omega) h2 (by omega) h4)
    · rfl

/-- `Array.qsort` leaves a strictly sorted array as it is: its pivot, the middle element, splits it in place. -/
theorem qsort_sorted (as : Array α) (lt : α → α → Bool)
    (hs : ∀ i (hi : i < as.size) j (hj : j < as.size), lt as[i] as[j] = decide (i < j)) : as.qsort lt = as := by
  unfold Array.qsort
  split
  · rfl
  · simp only
    rw [sort_sorted lt _ as.toVector _ _ _ _ _ (Nat.lt_succ_self _)
      (fun i j hi hj _ _ _ _ => by simpa using hs i hi j hj)]

/-- `qsort` keeps its array's size. -/
theorem size_qsort {α : Type} (xs : Array α) (lt : α → α → Bool) : (xs.qsort lt).size = xs.size := by
  unfold Array.qsort
  split
  · rfl
  · simp

end Verity.QSort
