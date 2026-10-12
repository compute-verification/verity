import Proofs.Flock.Soundness.Audit.Partitioning.Guarantees

/-!
# Exfiltration: the witnesses' facts

The statements of `Partitioning.Witness.exfiltration` and `Partitioning.Witness.refinement`, proved here so that the
witness file can cite them. `Law.subset n n` draws every unit, so only the empty set escapes, and it certifies any
`ℓ ≥ 0` with the harm bound `0` and the count `0` (`certifies_full`). The rest is arithmetic: `locBits 4 1 = log₂ 5`,
`locBits 4 2 = log₂ 11`, `(1/2)^{1/2} > 7/10`, and `1/2 ≤ ln 2 ≤ 1`.
-/

namespace FlockSoundness.Audit.Partitioning.SecurityProofs.Exfil

open Finset
open scoped ENNReal

/-- **Drawing every unit certifies any `ℓ ≥ 0`**, with the harm bound `0` and the count `0`. -/
theorem certifies_full (n : ℕ) (b : Fin n → ℕ) {δ ℓ : ℝ} (hδ : 0 < δ) (hℓ : 0 ≤ ℓ) :
    CertifiesLeak (Law.subset n n le_rfl) b δ ℓ := by
  have hzero : ∀ B : Finset (Fin n), B.Nonempty → (Law.subset n n le_rfl).escape B = 0 := by
    intro B hB
    have h1 := hB.card_pos
    have h2 : B.card ≤ n := by simpa using Finset.card_le_univ B
    rw [Law.subset_escape, Nat.choose_eq_zero_of_lt (n := n - B.card) (k := n) (by omega), Nat.cast_zero,
      ENNReal.zero_div]
  have hδ' : (0 : ℝ≥0∞) < ENNReal.ofReal δ := ENNReal.ofReal_pos.2 hδ
  unfold CertifiesLeak
  refine ⟨0, 0, fun B hB => ?_, lt_of_le_of_lt (iSup₂_le fun B hB => ?_) hδ', ?_⟩
  · rcases B.eq_empty_or_nonempty with rfl | hne
    · simp
    · rw [hzero B hne] at hB
      exact absurd hB (not_le.2 hδ')
  · exact (hzero B (Finset.card_pos.1 (by omega))).le
  · simpa [locBits] using hℓ

theorem logb_two_pow (k : ℕ) : Real.logb 2 ((2 : ℝ) ^ k) = k := by
  rw [Real.logb_pow, Real.logb_self_eq_one (by norm_num), mul_one]

theorem locBits_four_one : locBits 4 1 = Real.logb 2 5 := by
  have h : ∑ j ∈ range (1 + 1), Nat.choose 4 j = 5 := by decide
  rw [locBits, h, Nat.cast_ofNat]

theorem locBits_four_two : locBits 4 (1 + 1) = Real.logb 2 11 := by
  have h : ∑ j ∈ range (1 + 1 + 1), Nat.choose 4 j = 11 := by decide
  rw [locBits, h, Nat.cast_ofNat]

theorem logb_five_le : Real.logb 2 5 ≤ 3 := by
  have h := Real.logb_le_logb_of_le (b := 2) (by norm_num) (by norm_num : (0 : ℝ) < 5)
    (by norm_num : (5 : ℝ) ≤ 2 ^ 3)
  rwa [logb_two_pow] at h

theorem two_lt_logb_eleven : 2 < Real.logb 2 11 := by
  have h := Real.logb_lt_logb (b := 2) (by norm_num) (by norm_num : (0 : ℝ) < 2 ^ 2)
    (by norm_num : (2 : ℝ) ^ 2 < 11)
  rwa [logb_two_pow] at h

theorem half_le_log_two : 1 / 2 ≤ Real.log 2 := by
  have h := Real.log_le_sub_one_of_pos (by norm_num : (0 : ℝ) < 1 / 2)
  rw [one_div, Real.log_inv] at h
  linarith

theorem log_two_le_one : Real.log 2 ≤ 1 := by
  have h := Real.log_le_sub_one_of_pos (by norm_num : (0 : ℝ) < 2)
  linarith

/-- `Partitioning.Witness.exfiltration`'s statement. -/
theorem witness_exfiltration :
    (1 : ℝ) * 1 + locBits 4 1 ≤ 4 ∧ (4 : ℝ) < 1 * (1 + 1) + locBits 4 (1 + 1) ∧
    CertifiesLeak (Law.subset 4 4 le_rfl) (fun _ => 1) (1 / 2) 4 ∧
    0 < ((4 : ℝ) - 1) * (1 - (1 / 2 : ℝ) ^ ((1 + 1 : ℝ)⁻¹)) ∧
    ((4 : ℝ) - 1) * (1 - (1 / 2 : ℝ) ^ ((1 + 1 : ℝ)⁻¹)) < 4 * Real.log (1 / 2 : ℝ)⁻¹ / (1 + 1) := by
  have he : ((1 + 1 : ℝ))⁻¹ = ((2 : ℕ) : ℝ)⁻¹ := by norm_num
  have hlt : (1 / 2 : ℝ) ^ ((1 + 1 : ℝ)⁻¹) < 1 := Real.rpow_lt_one (by norm_num) (by norm_num) (by norm_num)
  have hgt : 7 / 10 < (1 / 2 : ℝ) ^ ((1 + 1 : ℝ)⁻¹) := by
    have h := Real.rpow_lt_rpow (by norm_num : (0 : ℝ) ≤ (7 / 10) ^ 2) (by norm_num : ((7 / 10 : ℝ)) ^ 2 < 1 / 2)
      (by norm_num : (0 : ℝ) < ((2 : ℕ) : ℝ)⁻¹)
    rwa [Real.pow_rpow_inv_natCast (by norm_num) (by norm_num), ← he] at h
  have hlog : Real.log (1 / 2 : ℝ)⁻¹ = Real.log 2 := by norm_num
  refine ⟨?_, ?_, certifies_full 4 _ (by norm_num) (by norm_num), ?_, ?_⟩
  · rw [locBits_four_one]; linarith [logb_five_le]
  · rw [locBits_four_two]; linarith [two_lt_logb_eleven]
  · linarith
  · rw [hlog]; linarith [half_le_log_two]

/-- `Partitioning.Witness.refinement`'s statement. -/
theorem witness_refinement :
    2 * (1 + 1) ≤ 4 ∧ (16 : ℝ) * 1 + locBits 4 1 ≤ 20 ∧ (20 : ℝ) < 16 * (1 + 1) + locBits 4 (1 + 1) ∧
    (16 : ℝ) + Real.logb 2 (1 + 2) / (1 + 1) ≤ 16 + Real.logb 2 2 ∧
    CertifiesLeak (Law.subset (2 * 4) (2 * 4) le_rfl) (fun _ => 16) (1 / 2) 20 ∧
    0 < (1 - 1 / 4) * (1 - Real.log (1 / 2 : ℝ)⁻¹ / (2 * (1 + 1))) *
      (unitCost 0 1 1 * (4 * Real.log (1 / 2 : ℝ)⁻¹ / (1 + 1))) := by
  have hlog : Real.log (1 / 2 : ℝ)⁻¹ = Real.log 2 := by norm_num
  have hthree : Real.logb 2 (1 + 2) ≤ 2 := by
    have h := Real.logb_le_logb_of_le (b := 2) (by norm_num) (by norm_num : (0 : ℝ) < 1 + 2)
      (by norm_num : (1 + 2 : ℝ) ≤ 2 ^ 2)
    rwa [logb_two_pow] at h
  have htwo : Real.logb 2 2 = 1 := Real.logb_self_eq_one (by norm_num)
  have hpos : 0 < Real.log 2 := Real.log_pos (by norm_num)
  refine ⟨by norm_num, ?_, ?_, ?_, certifies_full (2 * 4) _ (by norm_num) (by norm_num), ?_⟩
  · rw [locBits_four_one]; linarith [logb_five_le]
  · rw [locBits_four_two]; linarith [two_lt_logb_eleven]
  · rw [htwo, add_le_add_iff_left, div_le_one (by norm_num)]; linarith
  · rw [hlog, unitCost]
    have h1 : 0 < 1 - Real.log 2 / (2 * (1 + 1)) := by linarith [log_two_le_one]
    exact mul_pos (mul_pos (by norm_num) h1)
      (mul_pos (by norm_num) (div_pos (mul_pos (by norm_num) hpos) (by norm_num)))

end FlockSoundness.Audit.Partitioning.SecurityProofs.Exfil
