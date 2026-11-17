module

public import Proofs.Slowdown.Fit

/-! A window across optimizer steps, where `FitBoundSlots` applies and `FitBound` can't. -/

@[expose] public section

open InferenceOnly

namespace Slowdown.SecurityProofs.FitSlotsWitness

open Verity Finset

abbrev circuit : Circuit where
  N := 10
  w g := if g.val = 0 ∨ g.val = 1 ∨ g.val = 6 ∨ g.val = 7 then 4 else 1
  isInput g := decide (g.val < 4)
  reads g :=
    if g.val = 4 then {0, 2} else if g.val = 5 then {1, 2} else if g.val = 6 then {0, 4}
    else if g.val = 7 then {1, 5} else if g.val = 8 then {6, 3} else if g.val = 9 then {7, 3} else ∅
  reads_lt := by decide +kernel
  input_reads := by decide +kernel
  op _ _ := 0
  op_congr _ _ _ _ := rfl
  outputs := {8, 9}

abbrev decl : RegisteredCircuit := .ofCircuit circuit ∅ (by decide +kernel)

abbrev threading : Threading decl where
  m := 2
  W := ({0, 1, 6, 7} : Finset (Fin 10))
  thread g := if g.val = 2 then some 0 else if g.val = 3 then some 1 else none
  thread_W := by decide +kernel
  M := ({4, 5, 8, 9} : Finset (Fin 10))
  M_computed := by decide +kernel
  weight g := (if g.val = 4 then 0 else if g.val = 5 then 1 else if g.val = 8 then 6 else 7 : Fin 10)
  weight_W := by decide +kernel
  weight_reads := by decide +kernel
  act g := (if g.val = 4 ∨ g.val = 5 then 2 else 3 : Fin 10)
  act_reads := by decide +kernel
  act_thread := by decide +kernel
  reuse := 1
  reuse_le := by decide +kernel

abbrev slots : threading.Slots where
  p := 2
  slot g := (if g.val = 0 ∨ g.val = 6 then 0 else 1 : Fin 2)
  reuse_le := by decide +kernel

abbrev executionPhases : Partition decl.D 2 where
  unit g :=
    if g.val = 4 ∨ g.val = 6 ∨ g.val = 8 then some 0 else if g.val = 5 ∨ g.val = 7 ∨ g.val = 9 then some 1 else none
  input_iff := by decide +kernel

abbrev start : Fin threading.m → Fin 2 := fun _ => 0

theorem M_nonempty : threading.M.Nonempty := by decide +kernel

theorem gap : (6 : ℝ) < ((2 : ℕ) : ℝ) * 4 := by norm_num

theorem reads_le : ∀ j, (freshInput decl (executionPhases.gates j) : ℝ) ≤ 6 := by
  have h : ∀ j, freshInput decl (executionPhases.gates j) ≤ 6 := by decide +kernel
  exact fun j => by exact_mod_cast h j

theorem reachSlots : Assumptions.ReachSlots decl threading slots 2 := by
  unfold Assumptions.ReachSlots
  decide +kernel

theorem fitsSlots : Assumptions.FitsSlots decl threading slots executionPhases start 4 1 := by
  have h : ∀ j, 4 * (slots.slotsUsed (executionPhases.gates j)).card + 1 * (threading.entered executionPhases start j).card ≤
      freshInput decl (executionPhases.gates j) := by decide +kernel
  exact fun j => by exact_mod_cast h j

theorem executionPhase0 :
    (threading.weightsUsed (executionPhases.gates 0)).card = 2 ∧ (slots.slotsUsed (executionPhases.gates 0)).card = 1 := by
  decide +kernel

theorem not_fits (Nstar : ℕ) (S b_w w_e : ℝ) (hgap : S < Nstar * b_w)
    (hS : ∀ j, (freshInput decl (executionPhases.gates j) : ℝ) ≤ S) (hR : Assumptions.Reach decl threading Nstar) :
    ¬ Assumptions.Fits decl threading executionPhases start b_w w_e := by
  intro hF
  have hw : (threading.weightsUsed (executionPhases.gates 0)).card = 2 := by decide +kernel
  have he : (threading.entered executionPhases start 0).card = 0 := by decide +kernel
  have hr : freshInput decl (executionPhases.gates 0) = 6 := by decide +kernel
  have hN : Nstar ≤ 2 := by
    have h2 : (threading.threadWeights univ ⟨0, by decide⟩).card = 2 := by decide +kernel
    exact (hR ⟨0, by decide⟩ (by decide +kernel)).trans h2.le
  have h0 := hF 0
  have hS0 := hS 0
  rw [hw, he, hr] at h0
  rw [hr] at hS0
  push_cast at h0 hS0
  have hN' : (Nstar : ℝ) ≤ 2 := by exact_mod_cast hN
  have hN0 : (0 : ℝ) ≤ Nstar := Nat.cast_nonneg _
  rcases le_or_gt 0 b_w with hb | hb
  · linarith [mul_le_mul_of_nonneg_right hN' hb]
  · nlinarith [mul_nonneg hN0 (neg_nonneg.2 hb.le)]

theorem strict : threading.M.Nonempty ∧ (6 : ℝ) < ((2 : ℕ) : ℝ) * 4 ∧
    (∀ j, (freshInput decl (executionPhases.gates j) : ℝ) ≤ 6) ∧ Assumptions.ReachSlots decl threading slots 2 ∧
    Assumptions.FitsSlots decl threading slots executionPhases start 4 1 ∧
    ∀ (Nstar : ℕ) (S b_w w_e : ℝ), S < Nstar * b_w → (∀ j, (freshInput decl (executionPhases.gates j) : ℝ) ≤ S) →
      Assumptions.Reach decl threading Nstar → ¬ Assumptions.Fits decl threading executionPhases start b_w w_e :=
  ⟨M_nonempty, gap, reads_le, reachSlots, fitsSlots, not_fits⟩

theorem bound :
    4 * (1 : ℝ) * ((2 : ℕ) * 4 - 6) * threading.M.card ≤ (2 : ℕ) * threading.reuse * (2 : ℕ) * (6 : ℝ) ^ 2 :=
  FitBoundSlots decl threading slots 2 executionPhases start 2 6 4 1 gap reads_le reachSlots fitsSlots

end Slowdown.SecurityProofs.FitSlotsWitness
