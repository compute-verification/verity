import Definitions.Pouw.Assumptions.Pinned
import Proofs.Pouw.Prob
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum

/-!
# (a) `G_γ` from TT, and the overhead budget
-/

namespace Pouw.Proofs

open Finset

variable {Q R S : Type}

theorem correctSet_subset_checkedSet (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (τ : Transcript) :
    correctSet P U H s τ ⊆ checkedSet P U H s τ := by
  intro u hu
  simp only [correctSet, checkedSet, Finset.mem_filter] at hu ⊢
  obtain ⟨hu, c, hc, hcor⟩ := hu
  exact ⟨hu, c, hc, hcor.1⟩

theorem correctSet_subset_range (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (τ : Transcript) :
    correctSet P U H s τ ⊆ range U.layout.N := fun u hu => by
  simp only [correctSet, Finset.mem_filter] at hu; exact hu.1

theorem one_sub_reductionGamma (γ₀ Ωmax : ℝ) : 1 - reductionGamma γ₀ Ωmax = (1 - γ₀) / Ωmax := by
  unfold reductionGamma; ring

theorem reductionGamma_lt_one {γ₀ Ωmax : ℝ} (hγ₀ : γ₀ < 1) (hΩ : 0 < Ωmax) : reductionGamma γ₀ Ωmax < 1 := by
  have : 0 < (1 - γ₀) / Ωmax := div_pos (by linarith) hΩ
  unfold reductionGamma; linarith

/-- The pointwise reduction: a win at `γ = 1 − (1 − γ₀)/Ω_max` is a TT event. -/
theorem wins_imp_tt (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (τ : Transcript) {γ₀ Ωmax T : ℝ}
    (hΩ : 0 < Ωmax)
    (hW : ∀ u < U.layout.N, (P.Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ))
    (hwin : Wins P U (reductionGamma γ₀ Ωmax) T τ H s) :
    T / (1 - γ₀) < ∑ u ∈ checkedSet P U H s τ, ((U.layout.shape u).Wmm : ℝ) := by
  unfold Wins at hwin
  rw [one_sub_reductionGamma, div_div_eq_mul_div] at hwin
  have h1 : ∑ u ∈ correctSet P U H s τ, (P.Wref U.layout u : ℝ)
      ≤ Ωmax * ∑ u ∈ correctSet P U H s τ, ((U.layout.shape u).Wmm : ℝ) := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun u hu => hW u (Finset.mem_range.1 (correctSet_subset_range P U H s τ hu))
  have h2 : ∑ u ∈ correctSet P U H s τ, ((U.layout.shape u).Wmm : ℝ)
      ≤ ∑ u ∈ checkedSet P U H s τ, ((U.layout.shape u).Wmm : ℝ) :=
    Finset.sum_le_sum_of_subset_of_nonneg (correctSet_subset_checkedSet P U H s τ) fun _ _ _ => by positivity
  have h3 : T * Ωmax / (1 - γ₀) = Ωmax * (T / (1 - γ₀)) := by ring
  rw [h3] at hwin
  have h4 : Ωmax * (T / (1 - γ₀)) < Ωmax * ∑ u ∈ checkedSet P U H s τ, ((U.layout.shape u).Wmm : ℝ) :=
    lt_of_lt_of_le hwin (h1.trans (mul_le_mul_of_nonneg_left h2 hΩ.le))
  exact lt_of_mul_lt_mul_left h4 hΩ.le

/-- **(a)** `G_γ` from TT, with `γ = 1 − (1 − γ₀)/Ω_max` and `η = ε_TT(q)`. -/
theorem gammaFromTT : Pinned.GammaFromTT := by
  intro Q R S _ _ _ _ CM P D γ₀ Ωmax ε hγ₀ hΩ hW hTT U hU T hT q hq A hA
  refine le_trans (pr_mono ?_) (hTT.2 U hU T hT q hq A hA)
  rintro ⟨H, s⟩ hwin
  exact wins_imp_tt P U H s _ hΩ (hW U hU) hwin

/-- **The overhead budget**: `γ ≤ 1%` under TT(0.5%) iff `Ω_max ≤ 199/198`. -/
theorem gammaBudget : Pinned.GammaBudget := by
  intro Ωmax hΩ
  unfold reductionGamma Params.γ0
  rw [show (1 : ℝ) - 1 / 200 = 199 / 200 by norm_num]
  constructor
  · intro h
    have h' : 99 / 100 ≤ 199 / 200 / Ωmax := by linarith
    rw [le_div_iff₀ hΩ] at h'
    linarith
  · intro h
    have h' : 99 / 100 * Ωmax ≤ 199 / 200 := by linarith
    rw [← le_div_iff₀ hΩ] at h'
    linarith

end Pouw.Proofs
