import Proofs.Flock.Soundness.Discharge.Layout.Drawn
import Batteries.Tactic.OpenPrivate

set_option Elab.async false

/-!
# The drawn units are in the population, with a closure

`Draw.UnitDraw.proved` sorts the drawn units with their closure (`Array.qsort`) when the closure isn't empty. Every swap
of `qsort` keeps a property every element has (`all_qsort`), so every proved unit is below the population
(`proved_lt_population`), as each drawn unit and each closure unit is (`ofJson_lt`).
-/

open private Array.qpartition.loop Array.qsort.sort from Init.Data.Array.QSort.Basic

namespace FlockSoundness.Discharge.Layout

open Flock

variable {α : Type} {P : α → Prop}

theorem all_swap {n : ℕ} {as : Vector α n} (h : ∀ k (hk : k < n), P as[k]) (i j : ℕ) (hi : i < n) (hj : j < n) :
    ∀ k (hk : k < n), P (as.swap i j hi hj)[k] := by
  intro k hk
  rw [Vector.getElem_swap]
  split_ifs <;> apply h

theorem all_qpartition_loop {n : ℕ} (lt : α → α → Bool) {lo hi : ℕ} (hhi : hi < n) (pivot : α) :
    ∀ (d : ℕ) (as : Vector α n) (i k : ℕ) (ilo : lo ≤ i) (ik : i ≤ k) (w : k ≤ hi), hi - k = d →
      (∀ x (hx : x < n), P as[x]) →
      ∀ x (hx : x < n), P (Array.qpartition.loop lt lo hi hhi pivot as i k ilo ik w).2[x] := by
  intro d
  induction d using Nat.strong_induction_on with
  | _ d ih =>
    intro as i k ilo ik w hd h
    unfold Array.qpartition.loop
    split
    · rename_i hk
      split
      · exact ih (hi - (k + 1)) (by omega) _ _ _ _ _ _ rfl (all_swap h _ _ _ _)
      · exact ih (hi - (k + 1)) (by omega) _ _ _ _ _ _ rfl h
    · exact all_swap h _ _ (by omega) hhi

theorem all_ite_swap {n : ℕ} {as : Vector α n} (h : ∀ k (hk : k < n), P as[k]) (b : Bool) (i j : ℕ) (hi : i < n)
    (hj : j < n) : ∀ k (hk : k < n), P (if b then as.swap i j hi hj else as)[k] := by
  split
  · exact all_swap h i j hi hj
  · exact h

theorem all_qpartition {n : ℕ} (as : Vector α n) (lt : α → α → Bool) (lo hi : ℕ) (w : lo ≤ hi) (hlo : lo < n)
    (hhi : hi < n) (h : ∀ k (hk : k < n), P as[k]) :
    ∀ k (hk : k < n), P (Array.qpartition as lt lo hi w hlo hhi).2[k] := by
  unfold Array.qpartition
  exact all_qpartition_loop lt hhi _ _ _ _ _ _ _ _ rfl
    (all_ite_swap (all_ite_swap (all_ite_swap h _ _ _ _ _) _ _ _ _ _) _ _ _ _ _)

variable (P) in
/-- Every element of a vector has `P`. -/
def AllP {n : ℕ} (as : Vector α n) : Prop := ∀ k (hk : k < n), P as[k]

theorem all_sort {n : ℕ} (lt : α → α → Bool) :
    ∀ (d : ℕ) (as : Vector α n) (lo hi : ℕ) (w : lo ≤ hi) (hlo : lo < n) (hhi : hi < n), hi - lo = d →
      AllP P as → AllP P (Array.qsort.sort lt as lo hi w hlo hhi) := by
  intro d
  induction d using Nat.strong_induction_on with
  | _ d ih =>
    intro as lo hi w hlo hhi hd h
    rw [Array.qsort.sort]
    split
    · rename_i h₁
      have hp : AllP P (Array.qpartition as lt lo hi w hlo hhi).2 := all_qpartition as lt lo hi w hlo hhi h
      revert hp
      match Array.qpartition as lt lo hi w hlo hhi with
      | (⟨mid, hmid⟩, as') =>
        intro hp
        dsimp only at hp ⊢
        split
        · exact hp
        · rename_i h₂
          exact ih (hi - (mid + 1)) (by omega) _ _ _ _ _ _ rfl
            (ih (mid - lo) (by omega) _ _ _ _ _ _ rfl hp)
    · exact h

/-- **Every element of `qsort`'s result has a property every element of its input has.** -/
theorem all_qsort (as : Array α) (lt : α → α → Bool) (h : ∀ x ∈ as, P x) : ∀ x ∈ as.qsort lt, P x := by
  unfold Array.qsort
  split
  · exact h
  · intro x hx
    obtain ⟨k, hk, rfl⟩ := Array.mem_iff_getElem.1 hx
    rw [Vector.getElem_toArray]
    refine all_sort lt _ as.toVector _ _ _ _ _ rfl (fun k hk => ?_) k (by simpa using hk)
    exact h _ (by simp)

/-- **Every proved unit of a draw is below the population**, with or without a closure. -/
theorem proved_lt_population {d : Lean.Json} {D : Draw.UnitDraw} (h : Draw.UnitDraw.ofJson d = .ok D) :
    ∀ u ∈ D.proved, u < D.population := by
  obtain ⟨hu, hc⟩ := ofJson_lt h
  unfold Draw.UnitDraw.proved
  split
  · exact hu
  · refine all_qsort _ _ fun x hx => ?_
    rcases Array.mem_append.1 hx with hx | hx
    · exact hu x hx
    · exact hc x hx

end FlockSoundness.Discharge.Layout
