import Definitions.Nci.Guarantees.Formalism
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.LinearCombination

/-!
# The training formalism v1: Slowdown and Security

`SlowdownEven` and `SlowdownFrozenTail` are `Slowdown` at the `M` and `x₀` their profiles give, and `Security` is
`Slowdown` with A2: `ρ̲ ≤ ρ ≤ B c / |C|`, `p H(ℓ*) ≤ |C|` and `B ≤ N n`.

`Slowdown` itself is not proved yet. For a `(c, k)`-partition into blocks `I`, with `X_I = max_T x_T(I)`,
`W(I) = Σ_{T entered mid-path} |I ∩ T ∩ Dep|` and `w` the work per token, its proof is:

1. `computedReads_subset_unregistered` (about 10 lines, from `mem_inputs_of_mem_θ`): `|in(I) \ in(C)| ≤ |in(I) \ θ| ≤ k`,
   so A1 gives `u(X_I) + a m(I) ≤ k` for every block.
2. `share_le_one`, `maxShare_le_one`, `card_inter_tokenDep_le` (about 25 lines): `x_T(I) ≤ 1`, and
   `|I ∩ T ∩ Dep| ≤ X_I |T ∩ Dep| ≤ X_I w`.
3. `block_mid_work_le` (about 25 lines): `a W(I) ≤ M w`, from 1, 2 and `PeakBound` (`0 ≤ M`, its case `x = 0`).
4. `mem_of_feedsWithin_of_not_enters` (about 15 lines, induction on `FeedsWithin`): a block that holds a gate of `T` and
   does not enter `T` mid-path holds every ancestor of that gate inside `T`.
5. `quiet_block_unique` (about 20 lines): under `TokensLinked`, at most one block that does not enter `T` mid-path meets
   `(T ∩ Dep) \ R`, since two such blocks hold the common ancestor and a gate is in one block.
6. `token_mid_work_ge` (about 40 lines): `Σ_{I entering T} |I ∩ T ∩ Dep| ≥ (1 - x₀ - ε) |T ∩ Dep|`, from 5, `BottomShare`
   (the one quiet block has `u(X_I) ≤ k`, and `0 ≤ x₀` because A1 at `I = ∅` gives `u(0) ≤ 0`) and `|R| ≤ ε |T ∩ Dep|`.
7. `sum_mid_work` (about 30 lines): swapping the sums over blocks and tokens, `a (1 - x₀ - ε) φ |C| ≤ a Σ_I W(I) ≤ B M w`,
   the tokens disjoint so that `Σ_T |T ∩ Dep| = |Dep ∩ ⋃ 𝒯| = φ |C|`.
8. `Slowdown` (about 30 lines): divide by `|C| w M` (the bound is `0` when `w` or `M` is) and take the infimum over the
   partitions in `ℝ≥0∞`.
-/

namespace Nci.Formalism

open Finset
open Verity (Declaration Partition)
open Nci.Formalism.Assumptions
open scoped NNReal ENNReal

namespace SecurityProofs

theorem Slowdown : Guarantees.Slowdown := by
  sorry

theorem SlowdownEven : Guarantees.SlowdownEven := by
  intro d L _ A c k u₁ a ε hA hcut hlink hk hku ha
  have hk0 : (0 : ℝ) < k := Nat.cast_pos.2 hk
  have hu : 0 < u₁ := hk0.trans_le hku
  have h := Slowdown d L A c k (fun x => (x : ℝ) * u₁) a ((k : ℝ) / u₁) ε ((k : ℝ) ^ 2 / (4 * u₁)) hA hcut hlink
    (fun x _ => by
      show (x : ℝ) * (k - (x : ℝ) * u₁) ≤ (k : ℝ) ^ 2 / (4 * u₁)
      rw [le_div_iff₀ (by linarith)]
      nlinarith [sq_nonneg ((k : ℝ) - 2 * u₁ * x)])
    (fun x _ hx => by
      rw [le_div_iff₀ hu]
      exact hx)
    ha
  have heq : A.depShare * (1 - (k : ℝ) / u₁ - ε) * A.tokensPerBlock c * a / ((k : ℝ) ^ 2 / (4 * u₁)) =
      4 * A.depShare * A.tokensPerBlock c * a * (u₁ - k - ε * u₁) / (k : ℝ) ^ 2 := by
    rw [div_div_eq_mul_div]
    congr 1
    have hcancel : (k : ℝ) / u₁ * u₁ = k := div_mul_cancel₀ _ hu.ne'
    linear_combination (-(4 * A.depShare * A.tokensPerBlock c * a)) * hcancel
  rw [← heq]
  exact h

theorem SlowdownFrozenTail : Guarantees.SlowdownFrozenTail := by
  rintro d L _ A c k u a t ε hA hcut hlink ⟨hu0, hut⟩ ht ha
  refine Slowdown d L A c k u a t ε (t * k) hA hcut hlink (fun x hx1 => ?_) (fun x hx1 hxk => ?_) ha
  · have hx0 : (0 : ℝ) ≤ x := x.2
    have hk0 : (0 : ℝ) ≤ k := Nat.cast_nonneg k
    by_cases hxt : (x : ℝ) ≤ t
    · nlinarith [hu0 x]
    · have hku := hut x (not_le.1 hxt) hx1
      nlinarith
  · by_contra hxt
    exact absurd hxk (not_le.2 (hut x (not_le.1 hxt) hx1))

theorem Security : Guarantees.Security := by
  intro methods dec ℓ ℓstar D p r c k n N B P u a x₀ ε M hfloor hamong hA hwithin hreach hcut hlink hpeak hbottom ha
    hp hP hB
  have hslow := Slowdown r.d (Fin r.numTokens) r.A c k u a x₀ ε M hA hcut hlink hpeak hbottom ha
  set R := r.A.depShare * (1 - x₀ - ε) * r.A.tokensPerBlock c * a / M
  have hρ : slowdown r.d c k ≤ ((B * c : ℕ) : ℝ≥0∞) / (r.d.D.computed.card : ℝ≥0∞) := by
    unfold slowdown
    exact iInf_le_of_le B (iInf_le_of_le P (iInf_le_of_le hP le_rfl))
  have hRC : ENNReal.ofReal R * (r.d.D.computed.card : ℝ≥0∞) ≤ ((B * c : ℕ) : ℝ≥0∞) := by
    by_cases hC : (r.d.D.computed.card : ℝ≥0∞) = 0
    · rw [hC, mul_zero]
      exact zero_le
    · exact (ENNReal.le_div_iff_mul_le (Or.inl hC) (Or.inl (ENNReal.natCast_ne_top _))).1 (hslow.trans hρ)
  have hBc : ((B * c : ℕ) : ℝ≥0∞) ≤ ((N * n * c : ℕ) : ℝ≥0∞) := by
    exact_mod_cast Nat.mul_le_mul_right c hB
  have hfl := hfloor r hamong hA hwithin hreach
  by_cases hR0 : R ≤ 0
  · have h0 : R * p ≤ 0 := by nlinarith
    rw [ENNReal.ofReal_eq_zero.2 h0, zero_mul]
    exact zero_le
  · rw [ENNReal.ofReal_mul (not_le.1 hR0).le, mul_assoc]
    calc ENNReal.ofReal R * (ENNReal.ofReal p * honestCost methods ℓ ℓstar)
        ≤ ENNReal.ofReal R * (r.d.D.computed.card : ℝ≥0∞) := by gcongr
      _ ≤ ((B * c : ℕ) : ℝ≥0∞) := hRC
      _ ≤ ((N * n * c : ℕ) : ℝ≥0∞) := hBc

end SecurityProofs

end Nci.Formalism
