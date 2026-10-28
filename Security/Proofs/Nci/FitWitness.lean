import Proofs.Nci.Fit

/-!
# The fit bound's hypotheses hold together

A five-gate reference circuit where `FitBound`'s hypotheses all hold with counted multiplications to count, so its
premises are not contradictory. Gates 0 and 1 are two-bit weights, gate 2 a one-bit activation of the one thread, and
gates 3 and 4 the thread's counted multiplications, of weight 0 and weight 1 by the activation. The baseline is empty.
Window 0 holds gate 3 and starts the thread; window 1 holds gate 4 and enters it.

Each window reads 3 bits (a weight and the activation), so `S = 3 < N* b_w = 2 · 2`. The thread uses both weights
(`Reach` at `N* = 2`). Window 0 uses one weight (`2 ≤ 3`) and window 1 one weight and one entered thread
(`2 + 1 ≤ 3`, `Fits` at `b_w = 2`, `w_e = 1`, tight). `FitBound` then gives `4 · 1 · (4 − 3) · 2 ≤ 2 · 1 · 2 · 9`.
-/

namespace Nci.SecurityProofs.FitWitness

open Verity Finset

/-- The circuit: weights 0 and 1 (two bits), the activation 2 (one bit), and the multiplications 3 and 4. -/
abbrev circuit : Circuit where
  N := 5
  w g := if g.val < 2 then 2 else 1
  isInput g := decide (g.val < 3)
  reads g := if g.val = 3 then {0, 2} else if g.val = 4 then {1, 2} else ∅
  reads_lt := by decide +kernel
  input_reads := by decide +kernel
  op _ _ := 0
  op_congr _ _ _ _ := rfl
  outputs := {3, 4}

abbrev decl : Registered := .ofCircuit circuit ∅ (by decide +kernel)

/-- One thread, of the activation 2; the counted multiplications 3 and 4 each use a weight once. -/
abbrev threading : Threading decl where
  m := 1
  W := ({0, 1} : Finset (Fin 5))
  thread g := if g.val = 2 then some 0 else none
  thread_W := by decide +kernel
  M := ({3, 4} : Finset (Fin 5))
  M_computed := by decide +kernel
  weight g := (if g.val = 3 then 0 else 1 : Fin 5)
  weight_W := by decide +kernel
  weight_reads := by decide +kernel
  act _ := (2 : Fin 5)
  act_reads := by decide +kernel
  act_thread := by decide +kernel
  reuse := 1
  reuse_le := by decide +kernel

/-- Two windows: gate 3 in window 0, gate 4 in window 1. -/
abbrev windows : Partition decl.D 2 where
  unit g := if g.val = 3 then some 0 else if g.val = 4 then some 1 else none
  input_iff := by decide +kernel

/-- Window 0 starts the thread. -/
abbrev start : Fin threading.m → Fin 2 := fun _ => 0

theorem M_nonempty : threading.M.Nonempty := by decide +kernel

theorem gap : (3 : ℝ) < ((2 : ℕ) : ℝ) * 2 := by norm_num

theorem reads_le : ∀ j, (novelReads decl (windows.gates j) : ℝ) ≤ 3 := by
  have h : ∀ j, novelReads decl (windows.gates j) ≤ 3 := by decide +kernel
  exact fun j => by exact_mod_cast h j

theorem reach : Assumptions.Reach decl threading 2 := by
  unfold Assumptions.Reach
  decide +kernel

theorem fits : Assumptions.Fits decl threading windows start 2 1 := by
  have h : ∀ j, 2 * (threading.weightsUsed (windows.gates j)).card + 1 * (threading.entered windows start j).card ≤
      novelReads decl (windows.gates j) := by decide +kernel
  exact fun j => by exact_mod_cast h j

/-- **The fit bound's hypotheses are consistent**: with two counted multiplications, every window reads at most
`S = 3 < N* b_w`, and `Reach` and `Fits` hold. -/
theorem nonvacuous : threading.M.Nonempty ∧ (3 : ℝ) < ((2 : ℕ) : ℝ) * 2 ∧
    (∀ j, (novelReads decl (windows.gates j) : ℝ) ≤ 3) ∧ Assumptions.Reach decl threading 2 ∧
    Assumptions.Fits decl threading windows start 2 1 :=
  ⟨M_nonempty, gap, reads_le, reach, fits⟩

/-- `FitBound` at the witness. -/
theorem bound : 4 * (1 : ℝ) * ((2 : ℕ) * 2 - 3) * threading.M.card ≤ (2 : ℕ) * threading.reuse * (2 : ℕ) * (3 : ℝ) ^ 2 :=
  FitBound decl threading 2 windows start 2 3 2 1 gap reads_le reach fits

end Nci.SecurityProofs.FitWitness
