import Definitions.Nci.Guarantees.Ldl.Witnesses
import Proofs.Nci.Ldl
import Mathlib.Tactic.NormNum

/-!
# Local description length: the honest and nontrivial lemmas' proofs

The one-product instance's facts (`Nci.Ldl.Honest`): the block is `{1}` and reads one unregistered input bit, gate 0's;
the decoder that prints its program describes a string by itself, so `LDL = length`; the block's activation tuple is
gate 0's bit in any run, and its one activation occurrence costs one bit, so `ActivationRate` holds at `β = 1` and fails
at `β = 2`. Each nontrivial lemma's changed instance fails its named premise (`not_readCap_zero`, `not_weightCap_two`,
`not_activationRate_two`) and meets the others, which are the honest lemma's conjuncts at parameters they don't read.
-/

namespace Nci.Ldl.Honest

open Verity
open scoped ENNReal

theorem gates_eq (u : Fin 1) : P.gates u = {1} := by
  fin_cases u
  decide

theorem computed_eq : C.computed = {1} := by
  decide

theorem size_computed : C.size C.computed = 1 := by
  decide

theorem novelReads_block (u : Fin 1) : novelReads d (R.P.gates u) = 1 := by
  fin_cases u
  decide

/-- The decoder that prints its program describes a string by itself. -/
theorem LDL_D (t : ℕ) (y : List Bool) : LDL D t y = y.length := by
  refine le_antisymm (LDL_le (D := D) rfl) (le_iInf₂ fun p hp => ?_)
  rw [Option.some.inj hp]

theorem localEval_zero (X : d.D.Transcript) (u : Fin 1) : R.P.localEval X u 0 = X 0 :=
  Partition.localEval_of_not_mem _ _ (by simp)

theorem consistent (X : d.D.Transcript) : R.Consistent X := fun _ h => absurd h (Finset.notMem_empty _)

theorem readCap : ReadCap d R.P 1 := fun u => (novelReads_block u).le

theorem ckPartition : CKPartition d R.P 1 1 :=
  ⟨fun u => by fin_cases u; decide, readCap⟩

theorem evaluatesBlocks : Assumptions.EvaluatesBlocks D 0 R a 0 := by
  refine ⟨fun _ X => [(X 0).getLsbD 0], fun u X _ => ⟨fun X' _ hX' => ?_, ?_, ?_⟩⟩
  · have h0 : (0 : Fin d.D.N) ∈ d.D.into (R.P.gates u) \ d.θ := by
      fin_cases u
      decide
    show [(X' 0).getLsbD 0] = [(X 0).getLsbD 0]
    rw [hX' 0 h0]
  · show 1 ≤ Nat.clog 2 1 + novelReads d (R.P.gates u) + 0
    rw [novelReads_block]
    simp
  · show some [(X 0).getLsbD 0] = some [(R.P.localEval X u 0).getLsbD 0]
    rw [localEval_zero]

theorem fanOut : M.FanOut 1 := fun j => by
  cases j
  decide

theorem acts_card (u : Fin 1) : (M.acts R.P u).card = 1 := by
  fin_cases u
  decide

theorem activationRate (X : d.D.Transcript) : Assumptions.ActivationRate 1 0 0 D R M a X := fun u => by
  rw [acts_card, LDL_D]
  simp

theorem not_activationRate_two (X : d.D.Transcript) : ¬ Assumptions.ActivationRate 2 0 0 D R M a X := fun h => by
  have h0 := h 0
  rw [acts_card, LDL_D] at h0
  norm_num at h0

theorem not_readCap_zero : ¬ ReadCap d R.P 0 := fun h => by
  have h0 := h 0
  rw [novelReads_block] at h0
  omega

theorem not_weightCap_two : ¬ ∀ u, ∑ _v ∈ ({1} : Finset (Fin C.N)) ∩ P.gates u, (2 : ℝ) ≤ 1 := fun h => by
  have h0 := h 0
  rw [gates_eq, Finset.inter_self, Finset.sum_singleton] at h0
  norm_num at h0

end Nci.Ldl.Honest

namespace Nci.Ldl.SecurityProofs

open Verity Honest

theorem BlockDescription_honest : Guarantees.BlockDescription_honest :=
  ⟨readCap, consistent _, evaluatesBlocks⟩

theorem BlockDescription_nontrivial : Guarantees.BlockDescription_nontrivial := by
  unfold Guarantees.BlockDescription_nontrivial
  rw [LDL_D]
  simp

theorem Counting_honest : Guarantees.Counting_honest := by
  refine ⟨fun u => by fin_cases u; decide, by decide, fun u => ?_⟩
  rw [gates_eq, Finset.inter_self, Finset.sum_singleton]

theorem Counting_nontrivial : Guarantees.Counting_nontrivial := by
  unfold Guarantees.Counting_nontrivial
  rw [size_computed, Finset.sum_singleton]
  norm_num

theorem FanOutBound_honest : Guarantees.FanOutBound_honest :=
  ⟨ckPartition, consistent _, fanOut, evaluatesBlocks, activationRate _⟩

theorem FanOutBound_nontrivial : Guarantees.FanOutBound_nontrivial := by
  unfold Guarantees.FanOutBound_nontrivial
  rw [show d.D.size d.D.computed = 1 from size_computed, show M.J.card = 1 from rfl, Nat.clog_one_right]
  norm_num

end Nci.Ldl.SecurityProofs
