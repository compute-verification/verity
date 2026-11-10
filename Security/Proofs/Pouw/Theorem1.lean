module

public import Proofs.Pouw.Reduction
public import Proofs.Pouw.Sampling
public import Definitions.Pouw.Guarantees.Game
public import Proofs.Pouw.Lemmas.Game

/-!
# (b) Theorem 1, (d) end to end, (d″) at the milestone
-/

@[expose] public section

namespace Pouw.Proofs

open Finset

variable {Q R S : Type}

/-- Pointwise step of Theorem 1: an accepted run with little uncorrect work and `T < (1 − g)(1 − ε_s)·W` wins. -/
theorem wins_of_accept (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (τ : Transcript) {g T εs : ℝ}
    (hg : g < 1) (hT : T < (1 - g) * (1 - εs) * totalWref P U.layout)
    (hbad : ∑ u ∈ range U.layout.N \ correctSet P U H s τ, (P.Wref U.layout u : ℝ) ≤ εs * totalWref P U.layout) :
    Wins P U g T τ H s := by
  unfold Wins
  have hsplit := Finset.sum_sdiff (f := fun u => (P.Wref U.layout u : ℝ)) (correctSet_subset_range P U H s τ)
  have hK : (1 - εs) * totalWref P U.layout ≤ ∑ u ∈ correctSet P U H s τ, (P.Wref U.layout u : ℝ) := by
    unfold totalWref at hbad ⊢; linarith
  have h1g : 0 < 1 - g := by linarith
  rw [div_lt_iff₀ h1g]
  calc T < (1 - g) * (1 - εs) * totalWref P U.layout := hT
    _ = (1 - g) * ((1 - εs) * totalWref P U.layout) := by ring
    _ ≤ (1 - g) * ∑ u ∈ correctSet P U H s τ, (P.Wref U.layout u : ℝ) := mul_le_mul_of_nonneg_left hK h1g.le
    _ = _ := by ring

/-- **(b) Theorem 1**: `Pr[accept ∧ T < (1 − g)(1 − ε_s)·W] ≤ δ_s + δ`. -/
theorem theorem1 : Pouw.Guarantees.Theorem1 := by
  intro Q R S _ _ _ _ CM P D g δ hg hG U hU T hT q hq A hA C _ accept εs δs hδ
  let E1 := fun x : (Q → R) × S × C => accept x.1 x.2.1 x.2.2 ∧
    εs * totalWref P U.layout <
      ∑ u ∈ range U.layout.N \ correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1), (P.Wref U.layout u : ℝ)
  let E2 := fun y : (Q → R) × S => Wins P U g T (A.transcript U y.1 y.2) y.1 y.2 ∧ ¬ A.Breaks U y.1 y.2
  calc _ ≤ pr (fun x : (Q → R) × S × C => E1 x ∨ E2 (x.1, x.2.1)) := pr_mono fun x hx => by
        by_cases h1 : E1 x
        · exact Or.inl h1
        · refine Or.inr ⟨wins_of_accept P U x.1 x.2.1 _ hg hx.2.1 ?_, hx.2.2⟩
          by_contra h2
          exact h1 ⟨hx.1, lt_of_not_ge h2⟩
    _ ≤ pr E1 + pr (fun x : (Q → R) × S × C => E2 (x.1, x.2.1)) := pr_or_le _ _
    _ ≤ δs + δ q U.layout.N := add_le_add hδ ((pr_fst2_le E2).trans (hG U hU T hT q hq A hA))

/-- **(d) End to end**: TT and the `Ω_max` bound give resource exhaustion with `δ_s = (1 − ε_s)^t` and the game's
failure probability `δ(q)`. -/
theorem endToEnd : Pouw.Guarantees.EndToEnd := by
  intro Q R S _ _ _ _ CM P D g₀ Ωmax δ hg₀ hΩ hW hTT U hU T hT q hq A hA t εs hεs
  have hG := gapFromTT Q R S CM P D g₀ Ωmax δ hg₀ hΩ hW hTT
  refine theorem1 Q R S CM P D (reductionGap g₀ Ωmax) δ (reductionGap_lt_one hg₀ hΩ) hG
    U hU T hT q hq A hA (Fin t → Ticket (P.Wref U.layout) U.layout.N)
    (fun H s c => AuditAccepts (correctSet P U H s (A.transcript U H s)) c) εs ((1 - εs) ^ t) ?_
  let G := fun y : ((Q → R) × S) × (Fin t → Ticket (P.Wref U.layout) U.layout.N) =>
    AuditAccepts (correctSet P U y.1.1 y.1.2 (A.transcript U y.1.1 y.1.2)) y.2 ∧
      εs * ∑ u ∈ range U.layout.N, (P.Wref U.layout u : ℝ) <
        ∑ u ∈ range U.layout.N \ correctSet P U y.1.1 y.1.2 (A.transcript U y.1.1 y.1.2), (P.Wref U.layout u : ℝ)
  calc _ = pr G := pr_equiv (Equiv.prodAssoc _ _ _).symm G
    _ ≤ _ := workWeightedSampling ((Q → R) × S) (P.Wref U.layout) U.layout.N t
        (fun y => correctSet P U y.1 y.2 (A.transcript U y.1 y.2)) εs hεs

theorem gap0_lt_one : Params.gap0 < 1 := by unfold Params.gap0; norm_num

/-- **(d″)** Milestone resource exhaustion for `D1.m1` at `g = 1 − (1 − g₀)/Ω_max`. -/
theorem milestoneExhaustion : GameStatements.MilestoneExhaustion := by
  intro CM Wref Ωmax ε hΩ hW hTT U hU T hT q hq A hA t εs hεs
  exact endToEnd D1.M1Q D1.M1R Salt CM (D1.m1 Wref).toProtocol D1.m1Domain Params.gap0 Ωmax ε gap0_lt_one hΩ hW hTT
    U hU T hT q hq A hA t εs hεs

end Pouw.Proofs
