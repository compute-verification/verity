import Proofs.Pous.FreeBlocks.Audit
import Proofs.Pous.P2.Margin

/-!
# `P2SlackFamilyFreeBlocks`: P2's family with `⌊B/100⌋` free blocks, `k = 140`

`a3_prime_free` on `m1p` with Lemma A on `[0, p)` (`lemmaA_m1p`) and the uniform certificate `g = ⌊c·B⌋ + 1`. The free
blocks enter the 1% term only: `((g − 1 + ⌊B/100⌋)/B)^k ≤ (c + 1/100)^k ≤ 1/100`, so `c = 0.9576` at `k = 140`. The
state, `⌊ρ·(99/100)·|C|⌋ + 128·B`, is below `(18/19)·16448·B`, so the row is the `λ = 0` one, with a margin of 146
bits per block.
-/

namespace Pous.FreeBlocks

open Pous Pous.Assumptions PousP2 PousP2p PousMargin
open scoped ENNReal

/-- **`m1p` with `F` free blocks meets the audit at `2^-e`, given a row whose 1% term counts the free blocks.**
`PousMargin.slack_of_row`'s route through `a3_prime_free`. -/
theorem audit_free_of_row (B w p b c F S Q k e : ℕ) [NeZero B] (hb : 2 ^ b ≤ p) (hp2 : p ≤ 2 ^ w)
    (hc : 2 ^ w ≤ p + c) (hBc : 2 * B * c ≤ 2 ^ w) (hk : 1 ≤ k)
    (hrow : ∃ g, 1 ≤ g ∧ g ≤ B ∧ (((g - 1 + F : ℕ) : ℚ) / B) ^ k ≤ 1 / 100 ∧
      logHistGammaM1 B (S + 1) Q g k (2 ^ w) ≤ -(e : ℝ)) (D : ℕ) :
    AuditSecureFree (m1p B w p b) F S D Q k (ENNReal.ofReal Params.δ) ((2⁻¹ : ℝ≥0∞) ^ e) := by
  obtain ⟨g, -, hgB, hpow, hlog⟩ := hrow
  have hp0 : 0 < p := lt_of_lt_of_le (Nat.two_pow_pos _) hb
  intro W A₁ Fr A₂
  have hgc : 2 * g * c ≤ 2 ^ w := le_trans (by gcongr) hBc
  have hΓ : (∑ j ∈ Finset.range k, (B : ℝ≥0∞) ^ j) * gammaP B p S Q g ≤ (2⁻¹ : ℝ≥0∞) ^ e :=
    (mul_le_mul_right (gammaP_le B w p c S Q g hp0 hc hgc) _).trans (hist_gamma_le_e B w _ Q g k e hk hgB hlog)
  have h := a3_prime_free_lam (m1p B w p b) S D Q k g F e _ (fun _ _ => True)
    (lemmaA_m1p B w p b S D Q g hb hp2) hΓ W A₁ Fr A₂
  refine h.trans ?_
  simp only [not_true_eq_false, pr_false, add_zero]
  have hB0 : (0 : ℝ) < B := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne B)
  have hδ : ((((g - 1 + F : ℕ) : ℝ≥0∞) / (m1p B w p b).B) ^ k) ≤ ENNReal.ofReal Params.δ := by
    change ((((g - 1 + F : ℕ) : ℝ≥0∞) / (B : ℝ≥0∞)) ^ k) ≤ ENNReal.ofReal ((1 / 100 : ℚ) : ℝ)
    rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_natCast B, ← ENNReal.ofReal_div_of_pos hB0,
      ← ENNReal.ofReal_pow (by positivity)]
    refine ENNReal.ofReal_le_ofReal ?_
    have := (Rat.cast_le (K := ℝ)).2 hpow
    push_cast at this ⊢
    exact this
  exact add_le_add hδ le_rfl

open Real in
/-- **The uniform row with `F` free blocks** at `w = 16,448`, `T = 2^20`: `PousMargin.row_uniform_gen` with the 1%
term `(c + 1/100)^k`, for `F ≤ B/100`. -/
theorem row_uniform_free (B n S k e lam F : ℕ) (c m : ℚ) (hB2 : 2 ≤ B) (hBn : B ≤ 2 ^ n) (hk : 1 ≤ k)
    (hc0 : 0 ≤ c) (hc1 : c < 1) (hF : (F : ℚ) ≤ B / 100) (hck : (c + 1 / 100) ^ k ≤ 1 / 100)
    (hm : m ≤ c * (16448 - 20 - 14427 / 10000) - 18 / 19 * 16448 - lam - 1)
    (hslack : (k * n + e + 3 : ℚ) ≤ m * B) (hS : (S : ℝ) ≤ (18 / 19 * 16448 + lam) * (B : ℝ)) :
    ∃ g, 1 ≤ g ∧ g ≤ B ∧ (((g - 1 + F : ℕ) : ℚ) / B) ^ k ≤ 1 / 100 ∧
      logHistGammaM1 B (S + 1) (2 ^ 20) g k (2 ^ 16448) ≤ -(e : ℝ) := by
  obtain ⟨g, hg⟩ : ∃ g, g = ⌊c * B⌋₊ + 1 := ⟨_, rfl⟩
  have hcB0 : 0 ≤ c * B := by positivity
  have hB0 : (0 : ℚ) < B := by exact_mod_cast (by omega : 0 < B)
  have hgB : g ≤ B := by
    have : ⌊c * B⌋₊ < B := (Nat.floor_lt hcB0).2 (by nlinarith)
    omega
  refine ⟨g, by omega, hgB, ?_, ?_⟩
  · have hle : ((g - 1 + F : ℕ) : ℚ) / B ≤ c + 1 / 100 := by
      rw [div_le_iff₀ hB0, show g - 1 = ⌊c * B⌋₊ by omega]
      have := Nat.floor_le hcB0
      push_cast
      linarith
    calc (((g - 1 + F : ℕ) : ℚ) / B) ^ k ≤ (c + 1 / 100) ^ k := pow_le_pow_left₀ (by positivity) hle k
      _ ≤ 1 / 100 := hck
  · refine (logHistGammaM1_le _ _ _ _ _ _ hB2 hk (by omega) hgB (by norm_num)).trans ?_
    rw [logb_two_pow_nat]
    have hlogB : logb 2 (B : ℝ) ≤ n := by
      have := logb_le_logb_of_le one_lt_two (by exact_mod_cast (by omega : 0 < B))
        (show (B : ℝ) ≤ ((2 ^ n : ℕ) : ℝ) by exact_mod_cast hBn)
      rwa [logb_two_pow_nat] at this
    have hkn : (k : ℝ) * logb 2 B ≤ k * n := mul_le_mul_of_nonneg_left hlogB (Nat.cast_nonneg _)
    have hgc : c * B < ((⌊c * B⌋₊ : ℕ) : ℚ) + 1 := Nat.lt_floor_add_one _
    have hgR : (c : ℝ) * B < g := by
      have := (Rat.cast_lt (K := ℝ)).2 hgc
      push_cast [hg] at this ⊢
      linarith
    have hl := log_two_gt_d9
    have hL : ((g : ℝ) + 1) / log 2 ≤ ((g : ℝ) + 1) * (14427 / 10000) := by
      rw [div_le_iff₀ (by positivity)]
      have h1 : (1 : ℝ) ≤ 14427 / 10000 * log 2 := by norm_num at hl ⊢; linarith
      calc ((g : ℝ) + 1) = ((g : ℝ) + 1) * 1 := (mul_one _).symm
        _ ≤ ((g : ℝ) + 1) * (14427 / 10000 * log 2) := mul_le_mul_of_nonneg_left h1 (by positivity)
        _ = _ := by ring
    have hmR : (m : ℝ) ≤ c * (16448 - 20 - 14427 / 10000) - 18 / 19 * 16448 - lam - 1 := by
      have := (Rat.cast_le (K := ℝ)).2 hm
      push_cast at this
      exact this
    have hslR : (k : ℝ) * n + e + 3 ≤ m * B := by
      have := (Rat.cast_le (K := ℝ)).2 hslack
      push_cast at this
      exact this
    have hmB : (m : ℝ) * B ≤ (c * (16448 - 20 - 14427 / 10000) - 18 / 19 * 16448 - lam - 1) * B :=
      mul_le_mul_of_nonneg_right hmR (Nat.cast_nonneg _)
    have hgR' : (c : ℝ) * B * (16448 - 20 - 14427 / 10000) ≤ g * (16448 - 20 - 14427 / 10000) :=
      mul_le_mul_of_nonneg_right hgR.le (by norm_num)
    push_cast at hL ⊢
    nlinarith

/-- The free-blocks state `⌊ρ·(99/100)·B·16448⌋ + 128·B` is at most `(18/19)·16448·B`, the `λ = 0` state. -/
theorem stateBound_free_le (B p b : ℕ) [NeZero B] :
    ((((m1p B 16448 p b).stateBound (Params.ρ * (99 / 100)) + 128 * B : ℕ)) : ℝ) ≤
      (18 / 19 * 16448 + (0 : ℕ)) * (B : ℝ) := by
  have h := Nat.floor_le (show (0 : ℚ) ≤ Params.ρ * (99 / 100) * ((B * 16448 : ℕ) : ℚ) by
    simp only [Params.ρ]; positivity)
  have h' : (((m1p B 16448 p b).stateBound (Params.ρ * (99 / 100)) : ℕ) : ℚ) ≤
      18 / 19 * (99 / 100) * ((B : ℚ) * 16448) := by
    simpa [Scheme.stateBound, Params.ρ, m1p] using h
  have h'' := (Rat.cast_le (K := ℝ)).2 h'
  have hB : (0 : ℝ) ≤ B := Nat.cast_nonneg _
  push_cast at h'' ⊢
  nlinarith

/-- `(0.9576 + 0.01)^140 ≤ 1/100`. -/
theorem free_one_percent : ((9576 / 10000 : ℚ) + 1 / 100) ^ 140 ≤ 1 / 100 := by norm_num

/-- **`P2SlackFamilyFreeBlocks`.** For `2^8 ≤ B ≤ 2^23`, every payload width in range and every `D`, the `k = 140`
audit with `⌊B/100⌋` free blocks against a state of `⌊ρ·(99/100)·|C|⌋ + 128·B` bits passes with probability at most
`1/100 + 2^-129`. -/
theorem p2_slack_family_free_blocks : Pous.Guarantees.P2SlackFamilyFreeBlocks := by
  intro B _ h0 h1 b hsp hb D
  obtain ⟨hlo', hhi, hc, hBc⟩ := hyps_16448
  have hF : ((B / 100 : ℕ) : ℚ) ≤ B / 100 := Nat.cast_div_le
  have h256 : (256 : ℚ) ≤ B := by
    have h := h0
    norm_num at h
    exact_mod_cast h
  have hrow := row_uniform_free B 23
    ((m1p B 16448 (2 ^ 16448 - 21065) b).stateBound (Params.ρ * (99 / 100)) + 128 * B) 140 129 0 (B / 100)
    (9576 / 10000) 146 (by omega) h1 (by norm_num) (by norm_num) (by norm_num) hF free_one_percent (by norm_num)
    (le_trans (by norm_num) (mul_le_mul_of_nonneg_left h256 (by norm_num))) (stateBound_free_le B _ b)
  have h := audit_free_of_row B 16448 (2 ^ 16448 - 21065) b 21065 (B / 100)
    ((m1p B 16448 (2 ^ 16448 - 21065) b).stateBound (Params.ρ * (99 / 100)) + 128 * B) (2 ^ 20) 140 129
    (two_pow_le_of_le hb hlo') hhi hc (le_trans (by gcongr) hBc) (by norm_num) hrow D
  intro _ W A₁ Fr A₂
  exact (h W A₁ Fr A₂).trans (add_le_add le_rfl le_add_self)

end Pous.FreeBlocks
