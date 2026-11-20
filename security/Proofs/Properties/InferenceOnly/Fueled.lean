module

public import Base.Canon
public import Circuits.Descriptor
public import Batteries.Tactic.OpenPrivate
import all Init.Data.Array.QSort.Basic
import all Circuits.Descriptor

/-!
# Each fuelled twin is the function it unfolds

`qsortFueled_eq`: `qsortFueled` is `Array.qsort` of the whole array: the fuel of a range `[lo, hi]` it passes,
`hi - lo + 1`, is more than the recursion on it takes, which each twin carries as a hypothesis, so fuel `0` is refuted. `canonFueled_eq`, `fnRefsFueled_eq` and `subFnsFueled_eq`: at
every fuel, a twin is its original, each falling back to it at fuel `0`.
-/

@[expose] public section

set_option Elab.async false

open private Array.qsort.sort Array.qpartition.loop from Init.Data.Array.QSort.Basic
open private fnRefs fnRefsFueled from Circuits.Descriptor

namespace Flock.Fueled

open Lean (Json)
open Flock.Program

variable {α : Type} {n : Nat}

private theorem qpartLoopFueled_eq (lt : α → α → Bool) {lo hi : Nat} (hhi : hi < n) (pivot : α) :
    ∀ (f : Nat) (as : Vector α n) (i k : Nat) (ilo : lo ≤ i) (ik : i ≤ k) (w : k ≤ hi) (hf : hi + 1 ≤ k + f),
      qpartLoopFueled lt lo hi hhi pivot f as i k ilo ik w hf = Array.qpartition.loop lt lo hi hhi pivot as i k ilo ik w := by
  intro f
  induction f with
  | zero => intro _ _ _ _ _ w h; omega
  | succ f ih =>
    intro as i k ilo ik w hf
    unfold qpartLoopFueled
    rw [Array.qpartition.loop.eq_def]
    by_cases hk : k < hi
    · simp only [hk, dite_true]
      cases hc : lt as[k] pivot
      · simp only [Bool.false_eq_true, ite_false]
        exact ih _ _ _ _ _ _ _
      · simp only [ite_true]
        exact ih _ _ _ _ _ _ _
    · simp only [hk, dite_false]

theorem qpartitionFueled_eq (as : Vector α n) (lt : α → α → Bool) (lo hi : Nat) (w : lo ≤ hi) (hlo : lo < n)
    (hhi : hi < n) : qpartitionFueled as lt lo hi w hlo hhi = Array.qpartition as lt lo hi w hlo hhi := by
  unfold qpartitionFueled Array.qpartition
  exact qpartLoopFueled_eq lt hhi _ _ _ _ _ _ _ _ _

private theorem qsortSortFueled_eq (lt : α → α → Bool) :
    ∀ (f : Nat) (as : Vector α n) (lo hi : Nat) (w : lo ≤ hi) (hlo : lo < n) (hhi : hi < n) (hf : hi + 1 ≤ lo + f),
      qsortSortFueled lt f as lo hi w hlo hhi hf = Array.qsort.sort lt as lo hi w hlo hhi := by
  intro f
  induction f with
  | zero => intro _ _ _ w _ _ h; omega
  | succ f ih =>
    intro as lo hi w hlo hhi hf
    unfold qsortSortFueled
    rw [Array.qsort.sort.eq_def]
    by_cases h₁ : lo < hi
    · simp only [h₁, dite_true, qpartitionFueled_eq]
      obtain ⟨mid, hmid, as', hq⟩ : ∃ mid hmid as',
          Array.qpartition as lt lo hi w hlo hhi = (⟨mid, hmid⟩, as') := ⟨_, _, _, rfl⟩
      first | (rw [hq]; simp only) | (simp only [hq])
      by_cases h₂ : mid ≥ hi
      · simp only [h₂, dite_true]
      · simp only [h₂, dite_false]
        have hm : mid < n := by omega
        rw [ih as' lo mid hmid.1 hlo hm (by omega)]
        exact ih _ (mid + 1) hi (by omega) (by omega) hhi _
    · simp only [h₁, dite_false]

/-- The sort's twin sorts: at the fuel it passes, the array's size, it is `Array.qsort` of the whole array. -/
theorem qsortFueled_eq (as : Array α) (lt : α → α → Bool) : qsortFueled as lt = as.qsort lt := by
  unfold qsortFueled Array.qsort
  split
  · rfl
  · simp only [Nat.zero_min, Nat.min_self, Nat.zero_max]
    rw [qsortSortFueled_eq lt _ _ _ _ _ _ _ _]

theorem canonFueled_eq : ∀ (f : Nat) (j : Json), canonFueled f j = canon j
  | 0, _ => by simp only [canonFueled]
  | f + 1, j => by
    cases j <;> rw [canon] <;> simp only [canonFueled, canonFueled_eq f, qsortFueled_eq]

theorem fnRefsFueled_eq : ∀ (f : Nat) (acc : Array String) (j : Json), fnRefsFueled f acc j = fnRefs acc j
  | 0, _, _ => by simp only [fnRefsFueled]
  | f + 1, acc, j => by
    cases j <;> unfold fnRefs <;> simp only [fnRefsFueled, fnRefsFueled_eq f, qsortFueled_eq]

theorem subFnsFueled_eq (d : String → Option String) : ∀ (f : Nat) (j : Json), subFnsFueled d f j = subFns d j
  | 0, _ => by simp only [subFnsFueled]
  | f + 1, j => by
    cases j <;> rw [subFns.eq_def] <;> simp only [subFnsFueled, subFnsFueled_eq d f]

end Flock.Fueled
