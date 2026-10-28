import Proofs.Nci.Fit

/-!
# A window across optimizer steps: `FitBoundSlots` applies where `FitBound` can't

A ten-gate reference circuit of two optimizer steps on two parameters. Gates 0 and 1 are the parameters' step-1
versions (four-bit inputs), gates 2 and 3 one-bit activations, of thread 0 (a token at step 1) and thread 1 (a token at
step 2). Gates 4 and 5 are thread 0's counted multiplications, of parameter 0 and parameter 1 by activation 2. Gates 6
and 7 are the step-2 versions, each computed from its step-1 version and the product on it (the toy's gradient), and
gates 8 and 9 are thread 1's counted multiplications, of the step-2 versions by activation 3. The slot map sends 0 and 6
to slot 0 and 1 and 7 to slot 1. The baseline is empty.

Window 0 holds parameter 0 across both steps (gates 4, 6 and 8) and starts both threads; window 1 holds parameter 1
(gates 5, 7 and 9) and enters both. Each window reads its parameter's step-1 version and the two activations, 6 bits,
and computes the step-2 version itself.

Counted in slots, each window uses one parameter: `FitsSlots` holds at `b_w = 4`, `w_e = 1` (`4 ≤ 6`, and `4 + 2 ≤ 6`,
tight), `ReachSlots` at `N* = 2`, and `S = 6 < N* b_w = 8`, so `FitBoundSlots` applies. Counted in weight gates, each
window uses two, so `Fits` charges window 0 `2 b_w ≤ 6 ≤ S`, while `Reach` allows only `N* ≤ 2` and `FitBound` needs
`S < N* b_w ≤ 2 b_w`: no choice of `N*`, `S`, `b_w`, `w_e` meets `FitBound`'s hypotheses here (`not_fits`).
-/

namespace Nci.SecurityProofs.FitSlotsWitness

open Verity Finset

/-- The circuit: parameter versions 0, 1 (step 1) and 6, 7 (step 2), four bits each; activations 2 and 3; the
multiplications 4, 5, 8 and 9. -/
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

/-- Two threads, of activations 2 and 3; each counted multiplication uses a weight once. -/
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

/-- The slot map: 0 and 6 are versions of parameter 0, 1 and 7 of parameter 1. -/
abbrev slots : threading.Slots where
  p := 2
  slot g := (if g.val = 0 ∨ g.val = 6 then 0 else 1 : Fin 2)
  reuse_le := by decide +kernel

/-- Two windows, one per parameter across both steps. -/
abbrev windows : Partition decl.D 2 where
  unit g :=
    if g.val = 4 ∨ g.val = 6 ∨ g.val = 8 then some 0 else if g.val = 5 ∨ g.val = 7 ∨ g.val = 9 then some 1 else none
  input_iff := by decide +kernel

/-- Window 0 starts both threads. -/
abbrev start : Fin threading.m → Fin 2 := fun _ => 0

theorem M_nonempty : threading.M.Nonempty := by decide +kernel

theorem gap : (6 : ℝ) < ((2 : ℕ) : ℝ) * 4 := by norm_num

theorem reads_le : ∀ j, (novelReads decl (windows.gates j) : ℝ) ≤ 6 := by
  have h : ∀ j, novelReads decl (windows.gates j) ≤ 6 := by decide +kernel
  exact fun j => by exact_mod_cast h j

theorem reachSlots : Assumptions.ReachSlots decl threading slots 2 := by
  unfold Assumptions.ReachSlots
  decide +kernel

theorem fitsSlots : Assumptions.FitsSlots decl threading slots windows start 4 1 := by
  have h : ∀ j, 4 * (slots.slotsUsed (windows.gates j)).card + 1 * (threading.entered windows start j).card ≤
      novelReads decl (windows.gates j) := by decide +kernel
  exact fun j => by exact_mod_cast h j

/-- Window 0 uses two weight gates, both versions of one slot. -/
theorem window0 :
    (threading.weightsUsed (windows.gates 0)).card = 2 ∧ (slots.slotsUsed (windows.gates 0)).card = 1 := by
  decide +kernel

/-- **`Fits` fails wherever the rest of `FitBound`'s hypotheses hold**: `Reach` allows `N* ≤ 2`, so `S < N* b_w` needs
`b_w > 0` and `S < 2 b_w`, while `Fits` charges window 0 `2 b_w ≤ 6 ≤ S`. -/
theorem not_fits (Nstar : ℕ) (S b_w w_e : ℝ) (hgap : S < Nstar * b_w)
    (hS : ∀ j, (novelReads decl (windows.gates j) : ℝ) ≤ S) (hR : Assumptions.Reach decl threading Nstar) :
    ¬ Assumptions.Fits decl threading windows start b_w w_e := by
  intro hF
  have hw : (threading.weightsUsed (windows.gates 0)).card = 2 := by decide +kernel
  have he : (threading.entered windows start 0).card = 0 := by decide +kernel
  have hr : novelReads decl (windows.gates 0) = 6 := by decide +kernel
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

/-- **`FitBoundSlots` strictly generalizes `FitBound`**: here, with counted multiplications to count, all of
`FitBoundSlots`'s hypotheses hold, and no `N*`, `S`, `b_w`, `w_e` meets all of `FitBound`'s. -/
theorem strict : threading.M.Nonempty ∧ (6 : ℝ) < ((2 : ℕ) : ℝ) * 4 ∧
    (∀ j, (novelReads decl (windows.gates j) : ℝ) ≤ 6) ∧ Assumptions.ReachSlots decl threading slots 2 ∧
    Assumptions.FitsSlots decl threading slots windows start 4 1 ∧
    ∀ (Nstar : ℕ) (S b_w w_e : ℝ), S < Nstar * b_w → (∀ j, (novelReads decl (windows.gates j) : ℝ) ≤ S) →
      Assumptions.Reach decl threading Nstar → ¬ Assumptions.Fits decl threading windows start b_w w_e :=
  ⟨M_nonempty, gap, reads_le, reachSlots, fitsSlots, not_fits⟩

/-- `FitBoundSlots` at the witness. -/
theorem bound :
    4 * (1 : ℝ) * ((2 : ℕ) * 4 - 6) * threading.M.card ≤ (2 : ℕ) * threading.reuse * (2 : ℕ) * (6 : ℝ) ^ 2 :=
  FitBoundSlots decl threading slots 2 windows start 2 6 4 1 gap reads_le reachSlots fitsSlots

end Nci.SecurityProofs.FitSlotsWitness
