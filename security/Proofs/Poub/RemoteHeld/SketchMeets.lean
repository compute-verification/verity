module

public import Proofs.Poub.RemoteHeld.Sketch
public import Definitions.Poub.Remote.SketchToy
import Proofs.Pouw.Prob

/-! The deployed sketch bound's premises meet at an honest session, and its conclusion fails on a trace that accepts
everything. -/

@[expose] public section

namespace ResourceAccounting.SecurityProofs.Remote

open Finset ResourceAccounting.RateSketch ResourceAccounting.Remote.Held ResourceAccounting.Remote.Held.Assumptions ResourceAccounting.Remote.Sketch
  ResourceAccounting.Remote.Sketch.Assumptions ResourceAccounting.Remote.Sketch.Toy

theorem SketchMeets.honest_accepts (ω : Outcome Unit Hs₁ 1 128) : honestTrace.accepts ω := by
  show (0 : Fin 1) ∈ honest.correctAt ω 0 0
  simp only [Run.correctAt, Run.devAt, Run.coins, Prover.correct, Prover.tags, Prover.prog, honest, device, Chip.run,
    mem_filter, mem_univ, true_and]
  exact congrArg _ (Prod.ext (Subsingleton.elim _ _) rfl)

/-- **`SketchBoundDeployed`'s premises meet** at the honest session: one pass of one candidate at the deployed receipts,
its device holding the block and tagging it with one query, SHA-512's keys the outcome's own. -/
theorem SketchBoundDeployed_honest :
    KeyBudget honest (fun _ => 1) ∧ Receipted honest honestTrace ∧ Present honest honestTrace ∧
      ResourceAccounting.Assumptions.SketchHashRandom honest ((((1 : ℕ) : ℝ) + ((1 : ℕ) : ℝ)) ^ 2 / 2 ^ 513) := by
  refine ⟨fun b j η κ D => ?_, fun ω hω j b _ => ⟨0, ?_, ?_⟩, fun ω _ j _ _ b _ => ⟨j, le_rfl, Nat.lt_succ_self _, ?_⟩,
    fun b D E => le_self_add⟩
  · simp [honest, device, Prover.prog, Chip.queries]
  · rw [Subsingleton.elim j 0]
    exact hω
  · exact (Nat.lt_one_iff.1 b.isLt).symm
  · unfold Trace.present
    exact mem_union_left _ (mem_univ _)

/-- **`SketchBoundDeployed`'s conclusion bites**: on an audit that accepts everything, a pass's correct receipt of a
block never present breaks the sketch bound at the deployed δs (`Q = 0`, `P = 1`, `ε = 0`, `N = 1`). -/
theorem SketchBoundDeployed_nontrivial :
    ¬ SketchBound acceptAll (((0 : ℕ) : ℝ) / 2 ^ 216 + (((0 : ℕ) : ℝ) + ((1 : ℕ) : ℝ)) * (0 + ((1 : ℕ) : ℝ) / 2 ^ 512) +
      (((0 : ℕ) : ℝ) + ((1 : ℕ) : ℝ)) ^ 2 / 2 ^ 513) := by
  intro h
  have hev : ∀ ω : Unit, acceptAll.accepts ω ∧ ¬ (∀ p < acceptAll.passes, ∀ j ∈ acceptAll.correct ω p,
      ∃ t, acceptAll.tick ω p ≤ t ∧ t < acceptAll.tick ω (p + 1) ∧ j ∈ acceptAll.present ω t) := by
    intro ω
    refine ⟨trivial, fun hp => ?_⟩
    obtain ⟨t, -, -, ht⟩ := hp 0 Nat.zero_lt_one ⟨0, Nat.zero_lt_one⟩ (mem_univ _)
    unfold Trace.present at ht
    exact notMem_empty _ ((mem_union.1 ht).elim id id)
  have h1 : (1 : ℝ) ≤ _ := le_trans (Pouw.pr_eq_one hev).symm.le h
  have a : ((2 : ℝ) ^ 512)⁻¹ ≤ 1 / 4 :=
    le_trans (inv_anti₀ (by positivity) (pow_le_pow_right₀ (by norm_num) (by norm_num : 2 ≤ 512))) (by norm_num)
  have b : ((2 : ℝ) ^ 513)⁻¹ ≤ 1 / 4 :=
    le_trans (inv_anti₀ (by positivity) (pow_le_pow_right₀ (by norm_num) (by norm_num : 2 ≤ 513))) (by norm_num)
  norm_num at h1
  linarith

end ResourceAccounting.SecurityProofs.Remote
