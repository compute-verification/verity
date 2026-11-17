module

public import InferenceOnlyCheck.Record
public import Mathlib.Tactic

/-! `slotOf`, a unit's slot by its template's grids, which is the record check's `RecordText.slotIn`. -/

@[expose] public section

namespace InferenceOnlyCheck.Record.SecurityProofs

open Spot.RecordText

def slotOf (S : ℕ) (gs : List Grid) (u : ℕ) : Option ℕ :=
  (gs.find? fun g => g.start ≤ u && u < g.start + g.units).map fun g => (u - g.start) / g.q % S

theorem slotIn_eq (S : ℕ) (grids : List (String × List Grid)) (tid : String) (u : ℕ) :
    slotIn S grids tid u = slotOf S ((grids.lookup tid).getD []) u := rfl

def OneGrid (gs : List Grid) : Prop :=
  gs.Pairwise fun a b => a.start + a.units ≤ b.start ∨ b.start + b.units ≤ a.start

theorem slotOf_cons (S : ℕ) (g : Grid) (gs : List Grid) (u : ℕ) :
    slotOf S (g :: gs) u =
      if g.start ≤ u ∧ u < g.start + g.units then some ((u - g.start) / g.q % S) else slotOf S gs u := by
  unfold slotOf
  rw [List.find?_cons]
  by_cases h : g.start ≤ u ∧ u < g.start + g.units
  · have hb : (decide (g.start ≤ u) && decide (u < g.start + g.units)) = true := by
      simp only [Bool.and_eq_true, decide_eq_true_eq]; exact h
    rw [hb, ite_eq_left h]; rfl
  · have hb : (decide (g.start ≤ u) && decide (u < g.start + g.units)) = false := by
      rw [Bool.eq_false_iff]; simp only [ne_eq, Bool.and_eq_true, decide_eq_true_eq]; exact h
    rw [hb, ite_eq_right h]

theorem slotOf_none (S : ℕ) (gs : List Grid) (u : ℕ)
    (h : ∀ g ∈ gs, ¬ (g.start ≤ u ∧ u < g.start + g.units)) : slotOf S gs u = none := by
  induction gs with
  | nil => rfl
  | cons a gs ih =>
    rw [slotOf_cons, ite_eq_right (h a (List.mem_cons_self ..))]
    exact ih fun g hg => h g (List.mem_cons_of_mem _ hg)

theorem slotOf_of_mem (S : ℕ) (gs : List Grid) (u : ℕ) (hg : OneGrid gs) (g : Grid) (hmem : g ∈ gs)
    (h1 : g.start ≤ u) (h2 : u < g.start + g.units) : slotOf S gs u = some ((u - g.start) / g.q % S) := by
  induction gs with
  | nil => cases hmem
  | cons a gs ih =>
    have hp := List.pairwise_cons.1 hg
    rw [slotOf_cons]
    rcases List.mem_cons.1 hmem with rfl | hmem
    · rw [ite_eq_left ⟨h1, h2⟩]
    · have hd := hp.1 g hmem
      rw [ite_eq_right (by omega)]
      exact ih hp.2 hmem

end InferenceOnlyCheck.Record.SecurityProofs
