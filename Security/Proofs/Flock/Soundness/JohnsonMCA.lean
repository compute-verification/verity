import ArkLib.Data.CodingTheory.ReedSolomon.MutualCorrelatedAgreement.Johnson.Probability

/-!
# Mutual correlated agreement up to the Johnson bound, proved (DKT26)

BCHKS25 Theorem 4.6 in its printed form, for Reed–Solomon codes on `Fin n` domains, proved from
Dao–Kominers–Thaler's Johnson-range MCA bound (`ReedSolomon.mcaError_affineLine_johnson_le`,
ePrint 2026/2056, in ArkLib since #907). The two agree on everything but the count: at the radius
`δ = 1 − √ρ − η`, DKT26's exception count is below `(8/3)·n·t'³/ρ` with
`t' = max(⌈√ρ/(2η)⌉, 3) + ½`, and that is at most BCHKS25's printed numerator
`a = (2(m + ½)^5 + 3(m + ½)·δ·ρ)/(3ρ^{3/2})·n + (m + ½)/√ρ`, `m = max(⌈√ρ/η⌉, 3)`, because
`t' ≤ m + ½`, `(m + ½)² ≥ 49/4` and `ρ ≤ 1`. For `k ≥ n` the code is every word, so there is no
MCA event (`mcaError_top_eq_zero`).

So the soundness proof keeps BCHKS25's numerator in its accounting, and names no assumption for it.
-/

namespace FlockSoundness

open scoped NNReal
open CoreDefinitions ReedSolomon.HiddenDerivative

/-- The printed bound for `k < n`, from DKT26 at the radius `1 − (√ρ + η) = δ`. -/
theorem mcaError_le_bchks25_of_lt {n : ℕ} {F : Type} [Field F] [Fintype F] [SampleableType F]
    (domain : Fin n ↪ F) (k : ℕ) (δ : ℝ≥0) (hk : 1 < k) (hkn : k < n)
    (hJ : (δ : ℝ) < 1 - ((((k - 1 : ℕ) : ℝ) / Fintype.card (Fin n)) ^ ((1 : ℝ) / 2))) :
    mcaError (AffineLineGenerator F) (ReedSolomon.code domain k) (δ : ℝ) ≤
      ENNReal.ofReal
        (let n : ℝ := Fintype.card (Fin n)
         let ρ : ℝ := (k - 1 : ℕ) / n
         let m : ℝ := max ⌈(ρ ^ ((1 : ℝ) / 2)) / (1 - ρ ^ ((1 : ℝ) / 2) - δ)⌉ 3
         ((2 * (m + 1/2) ^ 5 + 3 * (m + 1/2) * δ * ρ)
            / (3 * ρ ^ ((3 : ℝ) / 2)) * n
          + (m + 1/2) / ρ ^ ((1 : ℝ) / 2))
           / (Fintype.card F : ℝ)) := by
  classical
  rw [Fintype.card_fin] at hJ ⊢
  set D := k - 1 with hD_def
  have hD1 : 1 ≤ D := by omega
  have hDn : D ≤ n - 2 := by omega
  have hDlt : D < n := by omega
  have hkD : k = D + 1 := by omega
  set ρ : ℝ := (D : ℝ) / n with hρ_def
  have hρ_eq : ρ = johnsonRhoMinus n D := rfl
  have hρpos : 0 < ρ := johnsonRhoMinus_pos hD1 hDlt
  have hρle1 : ρ ≤ 1 := by
    rw [hρ_def, div_le_one (by exact_mod_cast (show 0 < n by omega))]
    exact_mod_cast hDlt.le
  have hsq : ρ ^ ((1 : ℝ) / 2) = Real.sqrt ρ := (Real.sqrt_eq_rpow ρ).symm
  set η : ℝ := 1 - Real.sqrt ρ - δ with hη_def
  have hη : 0 < η := by rw [hη_def]; rw [hsq] at hJ; linarith
  have hagree : johnsonAgreement n D η = 1 - δ := by
    unfold johnsonAgreement; rw [← hρ_eq, hη_def]; ring
  have hagree1 : johnsonAgreement n D η ≤ 1 := by
    rw [hagree]; have := δ.coe_nonneg; linarith
  have hdkt := ReedSolomon.mcaError_affineLine_johnson_le (F := F) domain hD1 hDn hη hagree1
  rw [hagree, sub_sub_cancel] at hdkt
  rw [hkD]
  refine hdkt.trans (min_le_right _ _ |>.trans (ENNReal.ofReal_le_ofReal ?_))
  -- DKT26's count is at most BCHKS25's numerator
  have hq : (0 : ℝ) < Fintype.card F := by exact_mod_cast Fintype.card_pos
  have hE := johnsonExceptionCount_lt_closed (A := ⌈johnsonAgreement n D η * n⌉₊) hD1 hDn hη
    (Nat.le_ceil _)
  rw [← hρ_eq, hagree] at hE
  simp only
  rw [← hρ_def, hsq]
  have hηeq : 1 - Real.sqrt ρ - (δ : ℝ) = η := rfl
  rw [hηeq]
  apply div_le_div_of_nonneg_right _ hq.le
  set m : ℝ := max (⌈Real.sqrt ρ / η⌉ : ℝ) 3 with hm_def
  set t : ℝ := johnsonT n D η with ht_def
  have hsρ : 0 < Real.sqrt ρ := Real.sqrt_pos.2 hρpos
  have hsρ1 : Real.sqrt ρ ≤ 1 := Real.sqrt_le_one.2 hρle1
  -- `t' = max(⌈√ρ/(2η)⌉, 3) + ½ ≤ m + ½`
  have ht_le : t ≤ m + 1 / 2 := by
    have hM : (johnsonM n D η : ℝ) ≤ m := by
      unfold johnsonM
      rw [← hρ_eq]
      push_cast
      apply max_le_max _ le_rfl
      have hx : 0 ≤ Real.sqrt ρ / η := div_nonneg hsρ.le hη.le
      have h1 : (⌈Real.sqrt ρ / (2 * η)⌉₊ : ℝ) ≤ ⌈Real.sqrt ρ / η⌉₊ := by
        exact_mod_cast Nat.ceil_mono (by
          rw [div_le_div_iff₀ (by positivity) hη]; nlinarith)
      have hc : ((⌈Real.sqrt ρ / η⌉₊ : ℕ) : ℝ) = ((⌈Real.sqrt ρ / η⌉ : ℤ) : ℝ) := by
        rw [← Int.natCast_ceil_eq_ceil hx]; norm_cast
      exact h1.trans (le_of_eq hc)
    unfold t johnsonT; linarith
  have hm_ge : 7 / 2 ≤ m + 1 / 2 := by
    have : (3 : ℝ) ≤ m := le_max_right _ _
    linarith
  -- `ρ^{3/2} = ρ·√ρ`
  have h32 : ρ ^ ((3 : ℝ) / 2) = ρ * Real.sqrt ρ := by
    rw [show (3 : ℝ) / 2 = 1 + 1 / 2 by norm_num, Real.rpow_add hρpos, Real.rpow_one, hsq]
  rw [h32]
  have hlast : 0 ≤ (m + 1 / 2) / Real.sqrt ρ := by positivity
  -- `(8/3)·n·t'³/ρ ≤ (2/3)(m + ½)^5·n/(ρ√ρ)`
  have hmain : (8 / 3 : ℝ) * n * t ^ 3 / ρ ≤ 2 * (m + 1 / 2) ^ 5 / (3 * (ρ * Real.sqrt ρ)) * n := by
    have ht3 : t ^ 3 ≤ (m + 1 / 2) ^ 3 :=
      pow_le_pow_left₀ (by linarith [johnsonT_ge_seven_halves n D η]) ht_le 3
    have hsq2 : (49 / 4 : ℝ) ≤ (m + 1 / 2) ^ 2 := by nlinarith
    have key : 4 * t ^ 3 * Real.sqrt ρ ≤ (m + 1 / 2) ^ 5 := by
      have hm3 : 0 ≤ (m + 1 / 2) ^ 3 := by positivity
      calc 4 * t ^ 3 * Real.sqrt ρ ≤ 4 * t ^ 3 * 1 := by
            have : 0 ≤ 4 * t ^ 3 := by
              have := johnsonT_ge_seven_halves n D η; positivity
            exact mul_le_mul_of_nonneg_left hsρ1 this
        _ ≤ 4 * (m + 1 / 2) ^ 3 := by linarith
        _ ≤ (49 / 4) * (m + 1 / 2) ^ 3 := by nlinarith
        _ ≤ (m + 1 / 2) ^ 2 * (m + 1 / 2) ^ 3 := mul_le_mul_of_nonneg_right hsq2 hm3
        _ = (m + 1 / 2) ^ 5 := by ring
    have hρs : 0 < ρ * Real.sqrt ρ := mul_pos hρpos hsρ
    have e1 : 2 * (m + 1 / 2) ^ 5 / (3 * (ρ * Real.sqrt ρ)) * n =
        (2 / 3 * n) * ((m + 1 / 2) ^ 5 / (ρ * Real.sqrt ρ)) := by
      field_simp
    have e2 : (8 / 3 : ℝ) * n * t ^ 3 / ρ =
        (2 / 3 * n) * (4 * t ^ 3 * Real.sqrt ρ / (ρ * Real.sqrt ρ)) := by
      field_simp; ring
    rw [e1, e2]
    exact mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right key hρs.le) (by positivity)
  have hsplit : 2 * (m + 1 / 2) ^ 5 / (3 * (ρ * Real.sqrt ρ)) * n ≤
      (2 * (m + 1 / 2) ^ 5 + 3 * (m + 1 / 2) * (δ : ℝ) * ρ) / (3 * (ρ * Real.sqrt ρ)) * n := by
    have hδρ : 0 ≤ 3 * (m + 1 / 2) * (δ : ℝ) * ρ := by
      have := δ.coe_nonneg; positivity
    gcongr; linarith
  linarith [hE.le, hmain, hsplit, hlast]

/-- **BCHKS25 Theorem 4.6's printed bound, proved**, for a Reed–Solomon code of `k` coefficients on
a `Fin n` domain in the Johnson range `0 < δ < 1 − √((k − 1)/n)`: the MCA error of the affine line
is at most `a/|F|`. -/
theorem mcaError_le_bchks25 {n : ℕ} {F : Type} [Field F] [Fintype F] [SampleableType F]
    (domain : Fin n ↪ F) (k : ℕ) (δ : ℝ≥0) (hk : 1 < k)
    (hJ : (δ : ℝ) < 1 - ((((k - 1 : ℕ) : ℝ) / Fintype.card (Fin n)) ^ ((1 : ℝ) / 2))) :
    mcaError (AffineLineGenerator F) (ReedSolomon.code domain k) (δ : ℝ) ≤
      ENNReal.ofReal
        (let n : ℝ := Fintype.card (Fin n)
         let ρ : ℝ := (k - 1 : ℕ) / n
         let m : ℝ := max ⌈(ρ ^ ((1 : ℝ) / 2)) / (1 - ρ ^ ((1 : ℝ) / 2) - δ)⌉ 3
         ((2 * (m + 1/2) ^ 5 + 3 * (m + 1/2) * δ * ρ)
            / (3 * ρ ^ ((3 : ℝ) / 2)) * n
          + (m + 1/2) / ρ ^ ((1 : ℝ) / 2))
           / (Fintype.card F : ℝ)) := by
  rcases lt_or_ge k n with hkn | hkn
  · exact mcaError_le_bchks25_of_lt domain k δ hk hkn hJ
  · have htop : ReedSolomon.code domain k = ⊤ := by
      apply Submodule.eq_top_of_finrank_eq
      have h := ReedSolomon.dim_eq_min_deg_card (α := domain) (n := k)
      rw [Fintype.card_fin, min_eq_right hkn] at h
      unfold LinearCode.dim at h
      rw [h, Module.finrank_fintype_fun_eq_card, Fintype.card_fin]
    rw [htop, mcaError_top_eq_zero]
    exact zero_le

end FlockSoundness
