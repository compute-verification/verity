import Proofs.Accounting.ServingBlocks.Straddle
import Proofs.Accounting.ServingBlocks.Witnesses
import Definitions.Accounting.Guarantees.ServingBlocks.Straddle.Witnesses

/-!
# The uncertified cap across the certified boundary: honest and nontrivial lemmas

The proofs of `Security/Definitions/Accounting/Guarantees/ServingBlocks/Straddle/Witnesses.lean`. Every instance weighs
each counted operation 1, so each weight is a number of operations, which `decide` computes. The `example`s at the end
apply each guarantee to its lemmas, so the lemmas are the premises as the guarantee states them.
-/

namespace Accounting.SecurityProofs.ServingBlocks

open Verity Finset Accounting.ServingBlocks

namespace Counted

variable {C : Circuit} (J : Counted C)

/-- With every weight 1, the straddling weight is the number of straddling operations. -/
theorem straddle_eq (h1 : ∀ v ∈ J.V, J.ω v = 1) (R : Finset (Fin C.N)) {ι : Type*} [Fintype ι]
    (B : ι → Finset (Fin C.N)) :
    J.straddle R B =
      ((J.V.filter fun v => v ∈ R ∧ (∀ i, ¬ J.body v ⊆ B i) ∧ ¬ J.body v ⊆ uncertified R B).card : ℝ) := by
  unfold Counted.straddle
  rw [Finset.sum_congr rfl fun v hv => h1 v (Finset.mem_of_mem_filter v hv), Finset.sum_const, nsmul_one]

end Counted

theorem UncertifiedStructural_honest : Guarantees.ServingBlocks.UncertifiedStructural_honest := by
  refine ⟨by decide, zero_le_one, Honest.ω_le, ?_, ?_, ?_⟩
  · unfold Overlap; decide
  · unfold BodySize; decide
  · unfold NoStraddle; decide

theorem UncertifiedStructural_nontrivial : Guarantees.ServingBlocks.UncertifiedStructural_nontrivial := by
  refine ⟨by decide, zero_le_one, fun _ _ => le_rfl, ?_, ?_, ?_, ?_, fun h => ?_⟩
  · unfold Overlap; decide
  · unfold BodySize; decide
  · unfold OperationBodies; decide
  · unfold NoStraddle; decide
  · rw [UncertifiedCap, Counted.outside_eq _ Straddle.ω_one,
      show (uncertified Straddle.server Straddle.certified).card = 1 by decide] at h
    exact absurd (Nat.cast_lt_one.1 (h.trans_lt (show _ < (1 : ℝ) by norm_num))) (by decide)

theorem StraddleBound_honest : Guarantees.ServingBlocks.StraddleBound_honest := by
  refine ⟨?_, ?_, zero_le_one, Honest.ω_le⟩
  · unfold OperationBodies; decide
  · unfold Overlap; decide

theorem StraddleBound_nontrivial : Guarantees.ServingBlocks.StraddleBound_nontrivial := by
  refine ⟨?_, ?_, zero_le_one, fun _ _ => le_rfl, ?_⟩
  · unfold OperationBodies; decide
  · unfold Overlap; decide
  · rw [Counted.straddle_eq _ Detached.ω_one,
      show crossing Detached.d Detached.server Detached.certified = 0 by decide, Nat.cast_zero, mul_zero]
    exact Nat.cast_pos.2 (by decide)

theorem StraddleBound_nontrivial_overlap : Guarantees.ServingBlocks.StraddleBound_nontrivial_overlap := by
  refine ⟨?_, ?_, ?_, zero_le_one, fun _ _ => le_rfl, ?_, ?_⟩
  · unfold OperationBodies; decide
  · unfold Overlap; decide
  · unfold Overlap; decide
  · rw [Counted.straddle_eq _ Straddle.ω_one,
      show crossing Straddle.d Straddle.server Straddle.certified = 1 by decide, Nat.cast_one, mul_one, mul_one]
    exact Nat.cast_eq_one.2 (by decide)
  · rw [Counted.straddle_eq _ Straddle.ω_one, Nat.cast_zero, mul_zero, zero_mul]
    exact Nat.cast_pos.2 (by decide)

theorem UncertifiedCrossing_honest : Guarantees.ServingBlocks.UncertifiedCrossing_honest := by
  refine ⟨by decide, ?_, ?_, ?_, zero_le_one, Honest.ω_le⟩
  · unfold OperationBodies; decide
  · unfold Overlap; decide
  · unfold BodySize; decide

theorem UncertifiedCrossing_nontrivial : Guarantees.ServingBlocks.UncertifiedCrossing_nontrivial := by
  refine ⟨by decide, ?_, ?_, ?_, zero_le_one, fun _ _ => le_rfl, fun h => ?_⟩
  · unfold OperationBodies; decide
  · unfold Overlap; decide
  · unfold BodySize; decide
  · rw [UncertifiedCap, Counted.outside_eq _ Counter.ω_one,
      show (uncertified (Counter.P.gates 0) (Fin.elim0 : Fin 0 → Finset (Fin Counter.d.D.N))).card = 1 by decide,
      show crossing Counter.d (Counter.P.gates 0) (Fin.elim0 : Fin 0 → Finset (Fin Counter.d.D.N)) = 0 by decide]
      at h
    exact absurd (Nat.cast_lt_one.1 (h.trans_lt (show _ < (1 : ℝ) by norm_num))) (by decide)

theorem UncertifiedCrossing_nontrivial_bodies : Guarantees.ServingBlocks.UncertifiedCrossing_nontrivial_bodies := by
  refine ⟨by decide, ?_, ?_, ?_, zero_le_one, fun _ _ => le_rfl, fun h => ?_⟩
  · unfold OperationBodies; decide
  · unfold Overlap; decide
  · unfold BodySize; decide
  · rw [UncertifiedCap, Counted.outside_eq _ Detached.ω_one,
      show (uncertified Detached.server Detached.certified).card = 1 by decide,
      show crossing Detached.d Detached.server Detached.certified = 0 by decide] at h
    exact absurd (Nat.cast_lt_one.1 (h.trans_lt (show _ < (1 : ℝ) by norm_num))) (by decide)

theorem UncertifiedLocal_honest : Guarantees.ServingBlocks.UncertifiedLocal_honest := by
  refine ⟨by decide, ?_, ?_, ?_, zero_le_one, Honest.ω_le⟩
  · unfold OperationBodies; decide
  · unfold Overlap; decide
  · unfold UncertifiedBodySize; decide

theorem UncertifiedLocal_nontrivial : Guarantees.ServingBlocks.UncertifiedLocal_nontrivial := by
  refine ⟨by decide, ?_, ?_, ?_, zero_le_one, fun _ _ => le_rfl, fun h => ?_⟩
  · unfold OperationBodies; decide
  · unfold Overlap; decide
  · unfold UncertifiedBodySize; decide
  · rw [UncertifiedCap, Counted.outside_eq _ Straddle.ω_one,
      show (uncertified Straddle.server Straddle.certified).card = 1 by decide,
      show (∑ i, (Straddle.d.unregistered (Straddle.certified i)).card) = 0 by decide] at h
    exact absurd (Nat.cast_lt_one.1 (h.trans_lt (show _ < (1 : ℝ) by norm_num))) (by decide)

theorem UncertifiedLocal_nontrivial_bodies : Guarantees.ServingBlocks.UncertifiedLocal_nontrivial_bodies := by
  refine ⟨by decide, ?_, ?_, ?_, zero_le_one, fun _ _ => le_rfl, fun h => ?_⟩
  · unfold OperationBodies; decide
  · unfold Overlap; decide
  · unfold UncertifiedBodySize; decide
  · rw [UncertifiedCap, Counted.outside_eq _ Detached.ω_one,
      show (uncertified Detached.server Detached.represented).card = 1 by decide,
      show (∑ i, (Detached.d.unregistered (Detached.represented i)).card) = 0 by decide] at h
    exact absurd (Nat.cast_lt_one.1 (h.trans_lt (show _ < (1 : ℝ) by norm_num))) (by decide)

/-! Each lemma is its guarantee's premises as the guarantee states them: the guarantee applies at the honest instance,
and at a counter-example the dropped premise would give the bound the lemma's last conjunct refutes. -/

example : UncertifiedCap Honest.J ((univ \ Honest.S).biUnion Honest.P.gates)
    (Fin.elim0 : Fin 0 → Finset (Fin Honest.d.D.N))
    (1 * (1 : ℕ) * (uncertified ((univ \ Honest.S).biUnion Honest.P.gates)
      (Fin.elim0 : Fin 0 → Finset (Fin Honest.d.D.N))).card / (1 : ℕ)) := by
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := UncertifiedStructural_honest
  exact UncertifiedStructural _ Honest.J _ _ 1 1 1 h1 h2 h3 h4 h5 h6

example (h : NoStraddle Straddle.J Straddle.server Straddle.certified) : False := by
  obtain ⟨h1, h2, h3, h4, h5, -, -, hcap⟩ := UncertifiedStructural_nontrivial
  exact hcap (UncertifiedStructural _ Straddle.J _ _ 1 1 2 h1 h2 h3 h4 h5 h)

example : Honest.J.straddle ((univ \ Honest.S).biUnion Honest.P.gates) (Fin.elim0 : Fin 0 → Finset (Fin Honest.d.D.N)) ≤
    1 * (1 : ℕ) * crossing Honest.d ((univ \ Honest.S).biUnion Honest.P.gates)
      (Fin.elim0 : Fin 0 → Finset (Fin Honest.d.D.N)) := by
  obtain ⟨h1, h2, h3, h4⟩ := StraddleBound_honest
  exact StraddleBound Honest.d Honest.J _ _ 1 1 h1 h2 h3 h4

example (h : OperationBodies Detached.d Detached.J) : False := by
  obtain ⟨-, h2, h3, h4, hlt⟩ := StraddleBound_nontrivial
  exact not_le.2 hlt (StraddleBound Detached.d Detached.J _ _ 1 1 h h2 h3 h4)

example (h : Overlap Straddle.J 0) : False := by
  obtain ⟨h1, -, -, h3, h4, -, hlt⟩ := StraddleBound_nontrivial_overlap
  exact not_le.2 hlt (StraddleBound Straddle.d Straddle.J _ _ 1 0 h1 h h3 h4)

example : UncertifiedCap Honest.J ((univ \ Honest.S).biUnion Honest.P.gates)
    (Fin.elim0 : Fin 0 → Finset (Fin Honest.d.D.N))
    (1 * (1 : ℕ) * ((uncertified ((univ \ Honest.S).biUnion Honest.P.gates)
      (Fin.elim0 : Fin 0 → Finset (Fin Honest.d.D.N))).card / (1 : ℕ) +
      crossing Honest.d ((univ \ Honest.S).biUnion Honest.P.gates)
        (Fin.elim0 : Fin 0 → Finset (Fin Honest.d.D.N)))) := by
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := UncertifiedCrossing_honest
  exact UncertifiedCrossing Honest.d Honest.J _ _ 1 1 1 h1 h2 h3 h4 h5 h6

example (h : BodySize Counter.J 2) : False := by
  obtain ⟨h1, h2, h3, -, h5, h6, hcap⟩ := UncertifiedCrossing_nontrivial
  exact hcap (UncertifiedCrossing Counter.d Counter.J _ _ 1 1 2 h1 h2 h3 h h5 h6)

example (h : OperationBodies Detached.d Detached.J) : False := by
  obtain ⟨h1, -, h3, h4, h5, h6, hcap⟩ := UncertifiedCrossing_nontrivial_bodies
  exact hcap (UncertifiedCrossing Detached.d Detached.J _ _ 1 1 2 h1 h h3 h4 h5 h6)

/-- Where no straddling fails, the crossing cap still applies, with its crossing term. -/
example : UncertifiedCap Straddle.J Straddle.server Straddle.certified
    (1 * (1 : ℕ) * ((uncertified Straddle.server Straddle.certified).card / (2 : ℕ) +
      crossing Straddle.d Straddle.server Straddle.certified)) := by
  obtain ⟨h1, h2, h3, h4, h5, h6, -, -⟩ := UncertifiedStructural_nontrivial
  exact UncertifiedCrossing Straddle.d Straddle.J _ _ 1 1 2 h1 h6 h4 h5 h2 h3

example : UncertifiedCap Honest.J ((univ \ Honest.S).biUnion Honest.P.gates)
    (Fin.elim0 : Fin 0 → Finset (Fin Honest.d.D.N))
    (1 * (1 : ℕ) * ((uncertified ((univ \ Honest.S).biUnion Honest.P.gates)
      (Fin.elim0 : Fin 0 → Finset (Fin Honest.d.D.N))).card / (1 : ℕ) +
      ∑ i, (Honest.d.unregistered ((Fin.elim0 : Fin 0 → Finset (Fin Honest.d.D.N)) i)).card)) := by
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := UncertifiedLocal_honest
  exact UncertifiedLocal Honest.d Honest.J _ _ 1 1 1 h1 h2 h3 h4 h5 h6

example (h : UncertifiedBodySize Straddle.J Straddle.server Straddle.certified 2) : False := by
  obtain ⟨h1, h2, h3, -, h5, h6, hcap⟩ := UncertifiedLocal_nontrivial
  exact hcap (UncertifiedLocal Straddle.d Straddle.J _ _ 1 1 2 h1 h2 h3 h h5 h6)

example (h : OperationBodies Detached.d Detached.J) : False := by
  obtain ⟨h1, -, h3, h4, h5, h6, hcap⟩ := UncertifiedLocal_nontrivial_bodies
  exact hcap (UncertifiedLocal Detached.d Detached.J _ _ 1 1 2 h1 h h3 h4 h5 h6)

/-- Strictly weaker: at `Straddle` uncertified body size `1` holds where no straddling fails. -/
example : UncertifiedBodySize Straddle.J Straddle.server Straddle.certified 1 ∧
    ¬ NoStraddle Straddle.J Straddle.server Straddle.certified := by
  unfold UncertifiedBodySize NoStraddle; decide

end Accounting.SecurityProofs.ServingBlocks
