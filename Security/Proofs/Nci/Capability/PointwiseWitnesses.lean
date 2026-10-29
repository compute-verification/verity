import Definitions.Nci.Guarantees.Capability.PointwiseWitnesses
import Proofs.Nci.Capability.Pointwise
import Proofs.Nci.Capability.Witnesses
import Mathlib.Tactic.Linarith

/-!
# The pointwise slowdown: proofs of its witnesses

Each witness of `Definitions.Nci.Guarantees.Capability.PointwiseWitnesses`, at the toy instance
(`Nci.Capability.PointwiseToy`). The facts about the toy they share come first: its operation gates, what `D`'s region
computes whole and splits, what its remainder represents, the toy's `Γ`, the executions' probabilities, and that the
heavy execution has no decomposition meeting `D`'s caps (Theorem A would carry its weight `10` on two gates).
-/

namespace Nci.Capability.PointwiseToy

open Verity Finset Nci.Capability

theorem N_X : X.N = 2 := by decide

theorem N_heavy : heavy.N = 2 := by decide

theorem U_D : D.U = 1 := by decide

theorem sum_X (s : Finset X.κ) : ∑ o ∈ s, X.ω o = s.card := by
  show ∑ _o ∈ s, (1 : ℝ) = s.card
  simp

theorem Ω_X : X.Ω = 3 := by
  rw [Counted.Ω, sum_X, card_univ, Fintype.card_fin]
  norm_num

theorem Ω_heavy : heavy.Ω = 10 := by
  show ∑ _o : Fin 1, (10 : ℝ) = 10
  simp

theorem whole_D : ∀ b, X.whole (D.region b) = 1 := by
  have h : ∀ b, (univ.filter fun o => X.G o ⊆ D.region b).card = 1 := by decide
  intro b
  rw [Counted.whole, sum_X, h b]
  norm_num

theorem split_D : ∀ b, D.split (D.region b) = 1 := by
  have h : ∀ b, (univ.filter fun o => D.rep o ∈ D.region b ∧ ¬ X.G o ⊆ D.region b).card = 1 := by decide
  intro b
  rw [Counted.Decomposition.split, sum_X, h b]
  norm_num

theorem remainder_D : D.represented D.remainder = 1 := by
  have h : (univ.filter fun o => D.rep o ∈ D.remainder).card = 1 := by decide
  rw [Counted.Decomposition.represented, sum_X, h]
  norm_num

theorem wholeCap (q : ℝ) : D.WholeCap q ↔ 1 ≤ q :=
  ⟨fun h => (whole_D ⟨0, Nat.one_pos⟩).symm.le.trans (h _), fun h b => (whole_D b).le.trans h⟩

theorem splitCap (q : ℝ) : D.SplitCap q ↔ 1 ≤ q :=
  ⟨fun h => (split_D ⟨0, Nat.one_pos⟩).symm.le.trans (h _), fun h b => (split_D b).le.trans h⟩

theorem remainderCeiling (ν : ℝ) : D.RemainderCeiling ν ↔ 1 ≤ ν := by
  rw [Counted.Decomposition.RemainderCeiling, remainder_D, U_D]
  norm_num

theorem utilization (f : ℝ) : D.Utilization f ↔ 1 ≤ f * 2 := by
  rw [Counted.Decomposition.Utilization, U_D, N_X]
  norm_num

theorem Gamma_two : Gamma (1 + 1) 1 1 (1 / 2) = 2 := by
  unfold Gamma
  rw [max_eq_right (by norm_num)]
  norm_num

theorem Gamma_one : Gamma (1 + 0) 1 1 (1 / 2) = 1 := by
  unfold Gamma
  norm_num

theorem decomposes_X : X.Decomposes 1 1 1 1 (1 / 2) :=
  ⟨D, (wholeCap 1).2 le_rfl, (splitCap 1).2 le_rfl, (remainderCeiling 1).2 le_rfl, (utilization _).2 (by norm_num)⟩

/-- Theorem A at `D`'s caps puts at most `Γ·N = 4` on the toy's two operation gates, below the heavy weight `10`. -/
theorem not_decomposes_heavy : ¬ heavy.Decomposes 1 1 1 1 (1 / 2) := by
  rintro ⟨D', hW, hS, hR, hU⟩
  have := SecurityProofs.PointwiseCount heavy 1 D' 1 1 1 (1 / 2) hW hS hR hU
  rw [Ω_heavy, N_heavy, Gamma_two] at this
  norm_num at this

open Classical in
theorem prob_game (p : Bool → Prop) :
    Game.prob p game strat = (if p true then 2⁻¹ else 0) + if p false then 2⁻¹ else 0 := by
  simp [game, Game.draw, Game.prob]

theorem prob_true {p : Bool → Prop} (ht : p true) (hf : ¬ p false) : Game.prob p game strat = 2⁻¹ := by
  simp [prob_game, ht, hf]

theorem prob_false {p : Bool → Prop} (ht : ¬ p true) (hf : p false) : Game.prob p game strat = 2⁻¹ := by
  simp [prob_game, ht, hf]

theorem prob_both {p : Bool → Prop} (ht : p true) (hf : p false) : Game.prob p game strat = 1 := by
  simp [prob_game, ht, hf, ENNReal.inv_two_add_inv_two]

/-- `Xp`'s one occurrence, the product at gate `2`, lies whole in `Dp`'s region, weighing its slot's one bit. -/
theorem whole_Dp : ∀ b, Xp.whole (Dp.region b) = 1 := by
  have h : ∀ b, (univ.filter fun o => Xp.G o ⊆ Dp.region b) = univ := by decide
  intro b
  rw [Counted.whole, h b]
  show ∑ _o : Fin 1, Toy.W.price 0 * (1 : ℝ) = 1
  simp [Toy.W]

end Nci.Capability.PointwiseToy

namespace Nci.SecurityProofs

open Verity Finset Nci.Capability Nci.Capability.PointwiseToy
open scoped ENNReal

theorem DecompositionCountHonest : Guarantees.DecompositionCountHonest := ⟨D⟩

theorem DecompositionCountNontrivial : Guarantees.DecompositionCountNontrivial := by
  unfold Guarantees.DecompositionCountNontrivial
  rw [N_X, U_D]
  decide

theorem PointwiseCountHonest : Guarantees.PointwiseCountHonest := by
  unfold Guarantees.PointwiseCountHonest
  rw [wholeCap, wholeCap, splitCap, splitCap, remainderCeiling, remainderCeiling, utilization, utilization]
  norm_num

theorem PointwiseCountNontrivial : Guarantees.PointwiseCountNontrivial := by
  unfold Guarantees.PointwiseCountNontrivial
  rw [Ω_X, N_X, Gamma_one]
  norm_num

theorem GammaBelowHonest : Guarantees.GammaBelowHonest := by
  unfold Guarantees.GammaBelowHonest
  rw [remainderCeiling]
  norm_num

theorem GammaBelowNontrivial : Guarantees.GammaBelowNontrivial := by
  unfold Guarantees.GammaBelowNontrivial
  rw [Gamma_two]
  norm_num

theorem PointwiseThresholdHonest : Guarantees.PointwiseThresholdHonest := by
  unfold Guarantees.PointwiseThresholdHonest
  rw [wholeCap, splitCap, remainderCeiling, utilization, Ω_X, Gamma_two]
  norm_num

theorem PointwiseThresholdNontrivial : Guarantees.PointwiseThresholdNontrivial := by
  unfold Guarantees.PointwiseThresholdNontrivial
  rw [Gamma_two, N_X]
  norm_num

theorem PointwiseEventHonest : Guarantees.PointwiseEventHonest := by
  have e1 : Game.prob (fun a => True ∧ ¬ (exec a).Decomposes 1 1 1 1 (1 / 2)) game strat = 2⁻¹ :=
    prob_false (fun h => h.2 decomposes_X) ⟨trivial, not_decomposes_heavy⟩
  have e2 : Game.prob (fun a => True ∧ ((exec a).N : ℝ) < 10 ∧ ¬ (6 : ℝ) ≤ (exec a).Ω) game strat = 2⁻¹ := by
    refine prob_true ⟨trivial, ?_, ?_⟩ fun h => h.2.2 ?_
    · show (X.N : ℝ) < 10
      rw [N_X]
      norm_num
    · show ¬ (6 : ℝ) ≤ X.Ω
      rw [Ω_X]
      norm_num
    · show (6 : ℝ) ≤ heavy.Ω
      rw [Ω_heavy]
      norm_num
  refine ⟨by rw [Gamma_two]; norm_num, e1.le, e2.le, ?_, ?_⟩
  · rw [e1]
    simp
  · rw [e2]
    simp

theorem PointwiseEventNontrivial : Guarantees.PointwiseEventNontrivial := by
  unfold Guarantees.PointwiseEventNontrivial
  rw [Gamma_two]
  have hX : (X.N : ℝ) < min (6 / 2) 10 := by
    rw [N_X]
    norm_num
  have hh : (heavy.N : ℝ) < min (6 / 2) 10 := by
    rw [N_heavy]
    norm_num
  rw [prob_both ⟨trivial, trivial, hX⟩ ⟨trivial, trivial, hh⟩, zero_add]
  exact not_le.2 (ENNReal.inv_lt_one.2 ENNReal.one_lt_two)

theorem WholeCapLocalHonest : Guarantees.WholeCapLocalHonest := by
  refine ⟨Toy.isCK, Toy.slotsRead, Toy.held_W, Toy.upstream, Toy.overlapWeights, one_pos,
    fun _ => by simp [Toy.K, Consumers.ofUses], mem_singleton_self _, subset_refl _, by decide,
    fun _ => ⟨0, show ({2, 3} : Finset (Fin 4)) = Toy.R1.P.gates 0 by decide⟩, fun h => ?_, by decide⟩
  have := h 0 {0} (fun _ => {0}) (fun _ _ => mem_singleton_self _)
    (fun _ _ => show ({0} : Finset (Fin 4)) ⊆ holds Toy.C (Toy.P1.gates 0) by decide)
  norm_num [Toy.W] at this

theorem WholeCapLocalNontrivial : Guarantees.WholeCapLocalNontrivial := by
  intro h
  have := (whole_Dp ⟨0, Nat.one_pos⟩).symm.le.trans (h ⟨0, Nat.one_pos⟩)
  norm_num at this

end Nci.SecurityProofs
