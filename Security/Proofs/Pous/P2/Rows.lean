import Proofs.Pous.P2.Count
import Proofs.Pous.TrackA.A3Prime

/-!
# The certificate in model M1 (`Pous.m1`), from `p2/P2Meets.lean`

`m1`'s correctness and sizes; `gammaA2`, A2's bound at a domain of `2^ℓ` points; the certificate `P2Row`, `A7Row`'s log
form, which closes the history union at `λ' = 128` (`hist_gamma_le`); and `meets_m1_of_row`, `Meets` for `m1` given the
open `Pous.Guarantees.A2LemmaAM1` and a row. A3′ is `PousTrackA.a3_prime_lam`, the proof of `Guarantees.A3PrimeLam`.
`Security/Proofs/Pous/P2/Domain.lean` applies the same certificate on P2's exact domain, where Lemma A is proved,
not assumed.

The certificate bounds `log₂ |Hist| ≤ k·log₂ B`, `log₂ C(B, g) ≤ B`, and
`log₂ Σ_{h ≤ g} C((g+1)T, h) ≤ g·log₂ T + (g + 1)/ln 2` (from `C(N, h) ≤ N^h/h!` and `Σ g^h/h! ≤ e^g`), so the
final check is linear arithmetic with `ln 2 > 0.6931471803`.
-/

namespace PousP2

open Pous Pous.Assumptions
open scoped ENNReal

/-! ### Model M1: correctness and sizes -/

theorem m1_correct (B ℓ : ℕ) [NeZero B] : (m1 B ℓ).Correct := by
  intro W ω r
  change codeEquiv B ℓ (fun j => ω j ((ω j).symm ((codeEquiv B ℓ).symm W j))) = W
  simp only [Equiv.apply_symm_apply]
  exact (codeEquiv B ℓ).apply_symm_apply W

theorem m1_codeHoldsW (B ℓ : ℕ) [NeZero B] : (m1 B ℓ).CodeHoldsW := le_refl (B * ℓ)

theorem m1_spaceBound (B ℓ : ℕ) [NeZero B] : (m1 B ℓ).SpaceBound Params.expansion := by
  change (((B * ℓ + 0 : ℕ)) : ℚ) ≤ 21 / 20 * ((B * ℓ : ℕ) : ℚ)
  have : (0 : ℚ) ≤ ((B * ℓ : ℕ) : ℚ) := Nat.cast_nonneg _
  rw [Nat.add_zero]
  linarith

/-! ### Lemma A in M1 from A2 -/

/-- A2's bound, `2^S · C(B, g) · Σ_{h ≤ g} C((g+1)·Q, h) / 2^(ℓ·g)`: `gammaM1` at a domain of `2^ℓ` points. -/
noncomputable def gammaA2 (B ℓ S Q g : ℕ) : ℝ≥0∞ :=
  2 ^ S * (B.choose g) * (∑ h ∈ Finset.range (g + 1), (((g + 1) * Q).choose h : ℝ≥0∞)) / 2 ^ (ℓ * g)

/-- A2 is Lemma A for `m1` with the good event `True` (M1 has no coins, so the setup is the permutations alone). -/
theorem lemmaA_of_A2 (hA2 : Guarantees.A2LemmaAM1) (B ℓ S D Q g : ℕ) [NeZero B] :
    LemmaA (m1 B ℓ) S D Q g (gammaA2 B ℓ S Q g) (fun _ _ => True) := by
  intro W A₁ A₂
  have h := hA2 B ℓ S D Q g W A₁ A₂
  rw [← PousThm1.pr_equiv (Equiv.prodPUnit (perms B ℓ).Ω)] at h
  simp only [true_and]
  exact h

/-! ### The certificate: `A7Row`'s log form at a domain of `2^ℓ` points -/

/-- **P2's certificate row.** `A7Row` with the domain fixed at `M = 2^ℓ`, M1's `Bits ℓ`, instead of A7′'s
`M ≥ 2^2048 − 2^2029` (a 2048-bit `φ(N)`), which does not apply at `ℓ ≠ 2048`: some `g ∈ [1, B]` has
`((g − 1)/B)^k ≤ 1/100` and `log₂(|Hist|·Γ(g)) ≤ −128`. -/
def P2Row (B ℓ S T k : ℕ) : Prop :=
  ∃ g, 1 ≤ g ∧ g ≤ B ∧ (((g - 1 : ℕ) : ℚ) / B) ^ k ≤ 1 / 100 ∧
    logHistGammaM1 B S T g k (2 ^ ℓ) ≤ -128

theorem histCount_le (B : ℕ) (hB : 2 ≤ B) : ∀ k, histCount B k ≤ B ^ k
  | 0 => by simp [histCount]
  | k + 1 => by
      have ih := histCount_le B hB k
      rw [histCount, Finset.sum_range_succ, ← histCount, pow_succ]
      have := Nat.mul_le_mul_left (B ^ k) hB
      linarith

theorem histCount_pos (B k : ℕ) (hk : 1 ≤ k) : 0 < histCount B k := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hk
  rw [histCount, add_comm, Finset.sum_range_succ']
  simp

open Real in
/-- `Σ_{h ≤ g} C(N, h) ≤ (N/g)^g·e^g` for `N = (g + 1)·T ≥ g`. -/
theorem sum_choose_le (g T : ℕ) (hg : 1 ≤ g) (hT : 1 ≤ T) :
    (∑ h ∈ Finset.range (g + 1), (((g + 1) * T).choose h : ℝ)) ≤
      ((((g + 1) * T : ℕ) : ℝ) / g) ^ g * exp g := by
  set N : ℝ := (((g + 1) * T : ℕ) : ℝ) with hN
  have hg0 : (0 : ℝ) < g := by exact_mod_cast hg
  have hr : 1 ≤ N / g := by
    rw [le_div_iff₀ hg0, one_mul, hN]
    exact_mod_cast (by nlinarith : g ≤ (g + 1) * T)
  calc (∑ h ∈ Finset.range (g + 1), (((g + 1) * T).choose h : ℝ))
      ≤ ∑ h ∈ Finset.range (g + 1), N ^ h / (h.factorial : ℝ) :=
        Finset.sum_le_sum fun h _ => Nat.choose_le_pow_div h _
    _ = ∑ h ∈ Finset.range (g + 1), (N / g) ^ h * ((g : ℝ) ^ h / h.factorial) := by
        refine Finset.sum_congr rfl fun h _ => ?_
        rw [div_pow]
        field_simp
    _ ≤ ∑ h ∈ Finset.range (g + 1), (N / g) ^ g * ((g : ℝ) ^ h / h.factorial) := by
        refine Finset.sum_le_sum fun h hh => ?_
        have : h ≤ g := Nat.lt_succ_iff.1 (Finset.mem_range.1 hh)
        exact mul_le_mul_of_nonneg_right (pow_le_pow_right₀ hr this) (by positivity)
    _ = (N / g) ^ g * ∑ h ∈ Finset.range (g + 1), (g : ℝ) ^ h / h.factorial := by
        rw [Finset.mul_sum]
    _ ≤ (N / g) ^ g * exp g :=
        mul_le_mul_of_nonneg_left (sum_le_exp_of_nonneg hg0.le _) (by positivity)

open Real in
/-- The certificate's bound on `log₂(|Hist|·Γ(g))` at `M = 2^ℓ`, linear in `g` once `B` and `T` are fixed. -/
theorem logHistGammaM1_le (B S T g k ℓ : ℕ) (hB : 2 ≤ B) (hk : 1 ≤ k) (hg : 1 ≤ g) (hgB : g ≤ B) (hT : 1 ≤ T) :
    logHistGammaM1 B S T g k (2 ^ ℓ) ≤
      k * logb 2 B + S + B + (g * logb 2 T + (g + 1) / log 2) - g * ℓ := by
  have hl2 : 0 < log 2 := log_pos one_lt_two
  have hg0 : (0 : ℝ) < g := by exact_mod_cast hg
  -- `|Hist| ≤ B^k`
  have hH : logb 2 (histCount B k) ≤ k * logb 2 B := by
    rw [← logb_pow]
    refine logb_le_logb_of_le one_lt_two (by exact_mod_cast histCount_pos B k hk) ?_
    exact_mod_cast histCount_le B hB k
  -- `C(B, g) ≤ 2^B`
  have hC : logb 2 (B.choose g) ≤ B := by
    have hpos : (0 : ℝ) < B.choose g := by exact_mod_cast Nat.choose_pos hgB
    calc logb 2 (B.choose g) ≤ logb 2 ((2 : ℝ) ^ B) :=
          logb_le_logb_of_le one_lt_two hpos (by exact_mod_cast Nat.choose_le_two_pow B g)
      _ = B := by rw [logb_pow, logb_self_eq_one one_lt_two, mul_one]
  -- `Σ_{h ≤ g} C((g+1)T, h) ≤ (N/g)^g e^g`
  have hsum : logb 2 (∑ h ∈ Finset.range (g + 1), (((g + 1) * T).choose h : ℝ)) ≤
      g * logb 2 T + (g + 1) / log 2 := by
    have hpos : (0 : ℝ) < ∑ h ∈ Finset.range (g + 1), (((g + 1) * T).choose h : ℝ) := by
      rw [Finset.sum_range_succ']
      have : (0 : ℝ) ≤ ∑ h ∈ Finset.range g, (((g + 1) * T).choose (h + 1) : ℝ) := by positivity
      simp only [Nat.choose_zero_right, Nat.cast_one]
      linarith
    have hT0 : (0 : ℝ) < T := by exact_mod_cast hT
    have hq : (((g + 1) * T : ℕ) : ℝ) / g = ((g + 1) / g) * T := by push_cast; ring
    have hq0 : (0 : ℝ) < (g + 1) / g := by positivity
    have hlog1 : log ((g + 1) / g) ≤ 1 / g := by
      have := log_le_sub_one_of_pos hq0
      have h1 : ((g : ℝ) + 1) / g - 1 = 1 / g := by field_simp; ring
      linarith
    calc logb 2 (∑ h ∈ Finset.range (g + 1), (((g + 1) * T).choose h : ℝ))
        ≤ logb 2 (((((g + 1) * T : ℕ) : ℝ) / g) ^ g * exp g) :=
          logb_le_logb_of_le one_lt_two hpos (sum_choose_le g T hg hT)
      _ = g * (log ((g + 1) / g) / log 2 + logb 2 T) + g / log 2 := by
          rw [logb_mul (by positivity) (by positivity), logb_pow, hq, logb_mul hq0.ne' hT0.ne']
          simp only [logb, log_exp]
      _ ≤ g * ((1 / g) / log 2 + logb 2 T) + g / log 2 := by
          gcongr
      _ = g * logb 2 T + (g + 1) / log 2 := by
          field_simp
          ring
  have hM : logb 2 (((2 ^ ℓ : ℕ) : ℝ)) = ℓ := by
    rw [Nat.cast_pow, Nat.cast_ofNat, logb_pow, logb_self_eq_one one_lt_two, mul_one]
  unfold logHistGammaM1
  rw [hM]
  linarith

/-- The row's log bound gives A3′'s hypothesis `|Hist|·Γ ≤ 2^-128`. -/
theorem hist_gamma_le (B ℓ S T g k : ℕ) (hk : 1 ≤ k) (hgB : g ≤ B)
    (h : logHistGammaM1 B S T g k (2 ^ ℓ) ≤ -128) :
    (∑ j ∈ Finset.range k, (B : ℝ≥0∞) ^ j) * gammaA2 B ℓ S T g ≤ (2⁻¹ : ℝ≥0∞) ^ 128 := by
  set hist : ℝ := ((histCount B k : ℕ) : ℝ)
  set c : ℝ := ((B.choose g : ℕ) : ℝ)
  set s : ℝ := ∑ h ∈ Finset.range (g + 1), (((g + 1) * T).choose h : ℝ)
  have hhist : 0 < hist := Nat.cast_pos.2 (histCount_pos B k hk)
  have hc : 0 < c := Nat.cast_pos.2 (Nat.choose_pos hgB)
  have hs : 0 < s := by
    simp only [s]
    rw [Finset.sum_range_succ']
    have : (0 : ℝ) ≤ ∑ h ∈ Finset.range g, (((g + 1) * T).choose (h + 1) : ℝ) := by positivity
    simp only [Nat.choose_zero_right, Nat.cast_one]
    linarith
  set X : ℝ := hist * ((2 : ℝ) ^ S * c * s / (2 : ℝ) ^ (ℓ * g))
  have hX0 : 0 < X := by positivity
  have hlogX : Real.logb 2 X = logHistGammaM1 B S T g k (2 ^ ℓ) := by
    unfold logHistGammaM1
    simp only [X]
    rw [Real.logb_mul hhist.ne' (by positivity), Real.logb_div (by positivity) (by positivity),
      Real.logb_mul (by positivity) hs.ne', Real.logb_mul (by positivity) hc.ne', Real.logb_pow, Real.logb_pow,
      Real.logb_self_eq_one one_lt_two, Nat.cast_pow, Nat.cast_ofNat, Real.logb_pow,
      Real.logb_self_eq_one one_lt_two]
    push_cast
    ring
  have hXle : X ≤ (2⁻¹ : ℝ) ^ 128 := by
    rw [← Real.rpow_logb (by norm_num : (0 : ℝ) < 2) (by norm_num) hX0, hlogX]
    calc (2 : ℝ) ^ logHistGammaM1 B S T g k (2 ^ ℓ) ≤ (2 : ℝ) ^ (-128 : ℝ) :=
          Real.rpow_le_rpow_of_exponent_le (by norm_num) h
      _ = (2⁻¹ : ℝ) ^ 128 := by
          rw [Real.rpow_neg (by norm_num), inv_pow]
          norm_num
  have hL : (∑ j ∈ Finset.range k, (B : ℝ≥0∞) ^ j) * gammaA2 B ℓ S T g = ENNReal.ofReal X := by
    have hsum : (∑ j ∈ Finset.range k, (B : ℝ≥0∞) ^ j) = ENNReal.ofReal hist := by
      simp only [hist, histCount, Nat.cast_sum, Nat.cast_pow]
      rw [ENNReal.ofReal_sum_of_nonneg (fun _ _ => by positivity)]
      refine Finset.sum_congr rfl fun j _ => ?_
      rw [ENNReal.ofReal_pow (Nat.cast_nonneg _), ENNReal.ofReal_natCast]
    have hs' : (∑ h ∈ Finset.range (g + 1), (((g + 1) * T).choose h : ℝ≥0∞)) = ENNReal.ofReal s := by
      simp only [s]
      rw [ENNReal.ofReal_sum_of_nonneg (fun _ _ => by positivity)]
      exact Finset.sum_congr rfl fun h _ => (ENNReal.ofReal_natCast _).symm
    simp only [X, gammaA2, hsum, hs']
    rw [ENNReal.ofReal_mul hhist.le, ENNReal.ofReal_div_of_pos (by positivity), ENNReal.ofReal_mul (by positivity),
      ENNReal.ofReal_mul (by positivity), ENNReal.ofReal_pow (by norm_num), ENNReal.ofReal_pow (by norm_num),
      ENNReal.ofReal_ofNat, ENNReal.ofReal_natCast]
    simp only [c, ENNReal.ofReal_natCast]
  rw [hL, ← ENNReal.ofReal_ofNat 2, ← ENNReal.ofReal_inv_of_pos (by norm_num), ← ENNReal.ofReal_pow (by norm_num)]
  exact ENNReal.ofReal_le_ofReal hXle

/-! ### `Meets` for `m1` from A2 and a row -/

theorem pr_false {α : Type} [Fintype α] : pr (fun _ : α => False) = 0 := by
  rw [pr_def, Finset.filter_false, Finset.card_empty, Nat.cast_zero, ENNReal.zero_div]

/-- **`m1` meets the requirement, given A2 and a certificate row at its state bound.** -/
theorem meets_m1_of_row (hA2 : Guarantees.A2LemmaAM1) (B ℓ Q k : ℕ) [NeZero B] (hk : 1 ≤ k)
    (hrow : P2Row B ℓ ((m1 B ℓ).stateBound Params.ρ) Q k) (D : ℕ) :
    Meets (m1 B ℓ) .sequential D Q k Params.εMax := by
  obtain ⟨g, -, hgB, hpow, hlog⟩ := hrow
  refine ⟨le_rfl, m1_correct B ℓ, m1_codeHoldsW B ℓ, m1_spaceBound B ℓ, ?_⟩
  intro W A₁ A₂
  have h := PousTrackA.a3_prime_lam (m1 B ℓ) _ D Q k g 128 _ (fun _ _ => True)
    (lemmaA_of_A2 hA2 B ℓ ((m1 B ℓ).stateBound Params.ρ) D Q g)
    (hist_gamma_le B ℓ _ Q g k hk hgB hlog) W A₁ A₂
  refine h.trans ?_
  simp only [not_true_eq_false, pr_false, add_zero]
  have hB0 : (0 : ℝ) < B := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne B)
  have hδ : ((((g - 1 : ℕ) : ℝ≥0∞) / (m1 B ℓ).B) ^ k) ≤ ENNReal.ofReal Params.δ := by
    change ((((g - 1 : ℕ) : ℝ≥0∞) / (B : ℝ≥0∞)) ^ k) ≤ ENNReal.ofReal ((1 / 100 : ℚ) : ℝ)
    rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_natCast B, ← ENNReal.ofReal_div_of_pos hB0,
      ← ENNReal.ofReal_pow (by positivity)]
    refine ENNReal.ofReal_le_ofReal ?_
    have := (Rat.cast_le (K := ℝ)).2 hpow
    push_cast at this ⊢
    exact this
  exact add_le_add hδ le_rfl

end PousP2
