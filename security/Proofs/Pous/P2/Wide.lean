module

public import Definitions.Pous.Guarantees.P2
import Mathlib.Analysis.SpecialFunctions.Pow.NNReal
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.Measurability.Init
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Pous.FreeBlocks.P2
import Proofs.Pous.P2.Domain
import Proofs.Pous.P2.Margin

/-! P2's `m1p` families extended up to `2^64` blocks, the cap a 64-bit block index gives. -/

@[expose] public section

namespace PousP2Wide

open Pous Pous.Assumptions PousP2 PousP2p PousMargin Pous.FreeBlocks
open scoped ENNReal

theorem hBc_64 : 2 * 2 ^ 64 * 21065 ≤ 2 ^ 16448 := le_two_pow_of _ 80 16448 (by norm_num) (by norm_num)

theorem slack_uniform_64 (B n lo lam k e : ℕ) [NeZero B] (c m : ℚ) (hlo : lo ≤ B) (hB2 : 2 ≤ lo) (hBn : B ≤ 2 ^ n)
    (hB64 : B ≤ 2 ^ 64) (hk : 1 ≤ k) (he : 128 ≤ e) (hc0 : 0 ≤ c) (hc1 : c < 1) (hck : c ^ k ≤ 1 / 100)
    (hm : m ≤ c * (16448 - 20 - 14427 / 10000) - 18 / 19 * 16448 - lam - 1) (hm0 : 0 ≤ m)
    (hslack : (k * n + e + 3 : ℚ) ≤ m * lo) (b : ℕ) (hsp : 20 * 16448 ≤ 21 * b) (hb : b ≤ 16447) (D : ℕ) :
    MeetsWithSlack (m1p B 16448 (2 ^ 16448 - 21065) b) .sequential lam D (2 ^ 20) k ((2⁻¹ : ℝ≥0∞) ^ e) := by
  obtain ⟨hlo', hhi, hc, -⟩ := hyps_16448
  refine slack_of_row B 16448 _ b 21065 (2 ^ 20) k lam e (two_pow_le_of_le hb hlo') hhi hc
    (le_trans (by gcongr) hBc_64) hsp hk he ?_ D
  have hloB : (lo : ℚ) ≤ B := by exact_mod_cast hlo
  exact row_uniform_gen B n _ k e lam c m (by omega) hBn hk hc0 hc1 hck hm
    (hslack.trans (mul_le_mul_of_nonneg_left hloB hm0)) (stateBound_slack_le B _ b lam)

theorem famw_0_129 : ∀ (B : ℕ) [NeZero B], 2 ^ 8 ≤ B → B ≤ 2 ^ 64 → ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    MeetsWithSlack (m1p B 16448 (2 ^ 16448 - 21065) b) .sequential 0 D (2 ^ 20) 88 ((2⁻¹ : ℝ≥0∞) ^ 129) := by
  intro B _ h0 h1 b hsp hb D
  by_cases h : B ≤ 2 ^ 23
  · exact fam_0_129 B h0 h b hsp hb D
  · exact slack_uniform_64 B 64 (2 ^ 23 + 1) 0 88 129 (94901 / 100000) 5 (by omega) (by norm_num) h1 h1
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      b hsp hb D

theorem famw_64_129 : ∀ (B : ℕ) [NeZero B], 2 ^ 10 ≤ B → B ≤ 2 ^ 64 → ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    MeetsWithSlack (m1p B 16448 (2 ^ 16448 - 21065) b) .sequential 64 D (2 ^ 20) 95 ((2⁻¹ : ℝ≥0∞) ^ 129) := by
  intro B _ h0 h1 b hsp hb D
  by_cases h : B ≤ 2 ^ 23
  · exact fam_64_129 B h0 h b hsp hb D
  · exact slack_uniform_64 B 64 (2 ^ 23 + 1) 64 95 129 (95268 / 100000) (19 / 10) (by omega) (by norm_num) h1 h1
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      b hsp hb D

theorem famw_64_96_129 : ∀ (B : ℕ) [NeZero B], 2 ^ 8 ≤ B → B ≤ 2 ^ 64 → ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    MeetsWithSlack (m1p B 16448 (2 ^ 16448 - 21065) b) .sequential 64 D (2 ^ 20) 96 ((2⁻¹ : ℝ≥0∞) ^ 129) := by
  intro B _ h0 h1 b hsp hb D
  by_cases h : B ≤ 2 ^ 23
  · exact fam_64_96_129 B h0 h b hsp hb D
  · exact slack_uniform_64 B 64 (2 ^ 23 + 1) 64 96 129 (953161 / 1000000) 9 (by omega) (by norm_num) h1 h1
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      b hsp hb D

theorem famw_128_129 : ∀ (B : ℕ) [NeZero B], 2 ^ 9 ≤ B → B ≤ 2 ^ 64 → ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    MeetsWithSlack (m1p B 16448 (2 ^ 16448 - 21065) b) .sequential 128 D (2 ^ 20) 104 ((2⁻¹ : ℝ≥0∞) ^ 129) := by
  intro B _ h0 h1 b hsp hb D
  by_cases h : B ≤ 2 ^ 23
  · exact fam_128_129 B h0 h b hsp hb D
  · exact slack_uniform_64 B 64 (2 ^ 23 + 1) 128 104 129 (956685 / 1000000) (37 / 10) (by omega) (by norm_num) h1 h1
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      b hsp hb D

theorem famw_128_105_129 : ∀ (B : ℕ) [NeZero B], 2 ^ 8 ≤ B → B ≤ 2 ^ 64 → ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 →
    ∀ D, MeetsWithSlack (m1p B 16448 (2 ^ 16448 - 21065) b) .sequential 128 D (2 ^ 20) 105 ((2⁻¹ : ℝ≥0∞) ^ 129) := by
  intro B _ h0 h1 b hsp hb D
  by_cases h : B ≤ 2 ^ 23
  · exact fam_128_105_129 B h0 h b hsp hb D
  · exact slack_uniform_64 B 64 (2 ^ 23 + 1) 128 105 129 (957089 / 1000000) 10 (by omega) (by norm_num) h1 h1
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      b hsp hb D

theorem famw_256_129 : ∀ (B : ℕ) [NeZero B], 2 ^ 10 ≤ B → B ≤ 2 ^ 64 → ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    MeetsWithSlack (m1p B 16448 (2 ^ 16448 - 21065) b) .sequential 256 D (2 ^ 20) 127 ((2⁻¹ : ℝ≥0∞) ^ 129) := by
  intro B _ h0 h1 b hsp hb D
  by_cases h : B ≤ 2 ^ 23
  · exact fam_256_129 B h0 h b hsp hb D
  · exact slack_uniform_64 B 64 (2 ^ 23 + 1) 256 127 129 (964388 / 1000000) (22 / 10) (by omega) (by norm_num) h1 h1
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      b hsp hb D

theorem famw_256_128_129 : ∀ (B : ℕ) [NeZero B], 2 ^ 8 ≤ B → B ≤ 2 ^ 64 → ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 →
    ∀ D, MeetsWithSlack (m1p B 16448 (2 ^ 16448 - 21065) b) .sequential 256 D (2 ^ 20) 128 ((2⁻¹ : ℝ≥0∞) ^ 129) := by
  intro B _ h0 h1 b hsp hb D
  by_cases h : B ≤ 2 ^ 23
  · exact fam_256_128_129 B h0 h b hsp hb D
  · exact slack_uniform_64 B 64 (2 ^ 23 + 1) 256 128 129 (964661 / 1000000) (65 / 10) (by omega) (by norm_num) h1 h1
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      b hsp hb D

theorem p2_slack_family_wide : Pous.Guarantees.P2SlackFamilyWide := by
  intro B _ h0 h1 b hsp hb D
  obtain ⟨-, c1, c2, c3, a0⟩ := (meetsWithSlack_zero _ _ _ _ _ _).1 (famw_0_129 B h0 h1 b hsp hb D)
  exact ⟨c1, c2, c3, a0, (famw_64_96_129 B h0 h1 b hsp hb D).2.2.2.2, (famw_128_105_129 B h0 h1 b hsp hb D).2.2.2.2,
    (famw_256_128_129 B h0 h1 b hsp hb D).2.2.2.2, fun h9 => (famw_128_129 B h9 h1 b hsp hb D).2.2.2.2,
    fun h10 => ⟨(famw_64_129 B h10 h1 b hsp hb D).2.2.2.2, (famw_256_129 B h10 h1 b hsp hb D).2.2.2.2⟩⟩

theorem p2_slack_family_free_blocks_wide : Pous.Guarantees.P2SlackFamilyFreeBlocksWide := by
  intro B _ h0 h1 b hsp hb D
  obtain ⟨hlo', hhi, hc, -⟩ := hyps_16448
  have hF : ((B / 100 : ℕ) : ℚ) ≤ B / 100 := Nat.cast_div_le
  have h256 : (256 : ℚ) ≤ B := by
    have h := h0
    norm_num at h
    exact_mod_cast h
  have hrow := row_uniform_free B 64
    ((m1p B 16448 (2 ^ 16448 - 21065) b).stateBound (Params.ρ * (99 / 100)) + 128 * B) 140 129 0 (B / 100)
    (9576 / 10000) 146 (by omega) h1 (by norm_num) (by norm_num) (by norm_num) hF free_one_percent (by norm_num)
    (le_trans (by norm_num) (mul_le_mul_of_nonneg_left h256 (by norm_num))) (stateBound_free_le B _ b)
  have h := audit_free_of_row B 16448 (2 ^ 16448 - 21065) b 21065 (B / 100)
    ((m1p B 16448 (2 ^ 16448 - 21065) b).stateBound (Params.ρ * (99 / 100)) + 128 * B) (2 ^ 20) 140 129
    (two_pow_le_of_le hb hlo') hhi hc (le_trans (by gcongr) hBc_64) (by norm_num) hrow D
  intro _ W A₁ Fr A₂
  exact (h W A₁ Fr A₂).trans (add_le_add le_rfl le_add_self)

end PousP2Wide
