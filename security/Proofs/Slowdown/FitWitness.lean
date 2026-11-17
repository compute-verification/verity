module

public import Proofs.Slowdown.Fit

/-! The fit bound's hypotheses hold together, at a five-gate reference training algorithm. -/

@[expose] public section

open InferenceOnly

namespace Slowdown.SecurityProofs.FitWitness

open Verity Finset

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

abbrev decl : RegisteredCircuit := .ofCircuit circuit ∅ (by decide +kernel)

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

abbrev serverBlocks : Partition decl.D 2 where
  unit g := if g.val = 3 then some 0 else if g.val = 4 then some 1 else none
  input_iff := by decide +kernel

abbrev start : Fin threading.m → Fin 2 := fun _ => 0

theorem M_nonempty : threading.M.Nonempty := by decide +kernel

theorem gap : (3 : ℝ) < ((2 : ℕ) : ℝ) * 2 := by norm_num

theorem reads_le : ∀ j, (freshInput decl (serverBlocks.gates j) : ℝ) ≤ 3 := by
  have h : ∀ j, freshInput decl (serverBlocks.gates j) ≤ 3 := by decide +kernel
  exact fun j => by exact_mod_cast h j

theorem reach : Assumptions.Reach decl threading 2 := by
  unfold Assumptions.Reach
  decide +kernel

theorem fits : Assumptions.Fits decl threading serverBlocks start 2 1 := by
  have h : ∀ j, 2 * (threading.weightsUsed (serverBlocks.gates j)).card + 1 * (threading.entered serverBlocks start j).card ≤
      freshInput decl (serverBlocks.gates j) := by decide +kernel
  exact fun j => by exact_mod_cast h j

theorem bound : 4 * (1 : ℝ) * ((2 : ℕ) * 2 - 3) * threading.M.card ≤ (2 : ℕ) * threading.reuse * (2 : ℕ) * (3 : ℝ) ^ 2 :=
  FitBound decl threading 2 serverBlocks start 2 3 2 1 gap reads_le reach fits

end Slowdown.SecurityProofs.FitWitness
