module

public import Definitions.Pous.Guarantees.StreamedWipeResidual
import Mathlib.Analysis.Complex.ExponentialBounds
import Proofs.Pous.StreamedWipe.Bound
import Proofs.Pous.StreamedWipe.Deployed

/-!
# The streamed wipe's residual at any fill and challenge count

`streamed_wipe` at `g = N − D`, `D = wipeMissing N k s`, with `M = N · b`, the error split `2^-(s+1) + 2^-(s+1)` as in
`streamed_wipe_meets_2600GB`:

- **sampling** (`sampling_le`): `k · D / N ≥ (s + 1) · 0.6931471808`, since `D` is that ceiling;
- **compression** (`compression_le_log`): `C(N, D − 1) ≤ 2^((⌊log₂ N⌋ + 1)·(D − 1))` and `k ≤ 2^(⌊log₂ k⌋ + 1)`, so
  the term is at most `2^-(s+1)` once `q ≥ E + wipeResidual N k b s`.

At the deployed point the function gives 8,549,712,478 bits, within `StreamedWipeMeets2600GB`'s 8,600,000,000.
-/

@[expose] public section

namespace PousStreamedWipe

open Pous Pous.StreamedWipe
open scoped ENNReal

/-- **Compression.** With `M = N · b`, `N ≤ 2^LN` and `k ≤ 2^Lk`, the compression term at `g = N − D` is at most
`2^-t` once `q ≥ E + Lk + (LN + b) · (D − 1) + t`. -/
theorem compression_le_log {N D k b q M E LN Lk t : ℕ} (hD : 1 ≤ D) (hDN : D ≤ N) (hN : N ≤ 2 ^ LN)
    (hk : k ≤ 2 ^ Lk) (hM : M = N * b) (hq : Lk + LN * (D - 1) + (D - 1) * b + E + t ≤ q) :
    (k : ℝ≥0∞) * (N.choose (N - D + 1) : ℝ≥0∞) * 2 ^ (M + E) / 2 ^ (q + (N - D + 1) * b) ≤ (2⁻¹ : ℝ≥0∞) ^ t := by
  have hC : N.choose (N - D + 1) ≤ 2 ^ (LN * (D - 1)) := by
    rw [show N - D + 1 = N - (D - 1) by omega, Nat.choose_symm (by omega), pow_mul]
    exact (Nat.choose_le_pow N (D - 1)).trans (Nat.pow_le_pow_left hN _)
  have hsplit : M = (N - D + 1) * b + (D - 1) * b := by
    rw [hM, ← add_mul]
    congr 1
    omega
  have hnat : k * N.choose (N - D + 1) * 2 ^ (M + E) * 2 ^ t ≤ 2 ^ (q + (N - D + 1) * b) := by
    calc k * N.choose (N - D + 1) * 2 ^ (M + E) * 2 ^ t
        ≤ 2 ^ Lk * 2 ^ (LN * (D - 1)) * 2 ^ (M + E) * 2 ^ t := by gcongr
      _ = 2 ^ (Lk + LN * (D - 1) + (M + E) + t) := by simp only [← pow_add]
      _ ≤ 2 ^ (q + (N - D + 1) * b) := Nat.pow_le_pow_right (by norm_num) (by rw [hsplit]; omega)
  rw [← ENNReal.inv_pow]
  apply ENNReal.div_le_of_le_mul
  rw [mul_comm ((2 : ℝ≥0∞) ^ t)⁻¹, ← div_eq_mul_inv,
    ENNReal.le_div_iff_mul_le (Or.inl (pow_ne_zero _ two_ne_zero)) (Or.inl (ENNReal.pow_ne_top ENNReal.ofNat_ne_top))]
  exact_mod_cast hnat

/-- `wipeMissing N k s` is the ceiling it names: `(s + 1) · 6931471808 · N ≤ D · (k · 10^10)`. -/
theorem wipeMissing_spec {N k s : ℕ} (hk : 0 < k) :
    (s + 1) * 6931471808 * N ≤ wipeMissing N k s * (k * 10 ^ 10) := by
  unfold wipeMissing
  rw [if_neg (Nat.pos_iff_ne_zero.mp hk)]
  generalize (s + 1) * 6931471808 * N = A
  have hB : 0 < k * 10 ^ 10 := by positivity
  generalize k * 10 ^ 10 = B at hB ⊢
  have h1 := Nat.div_add_mod (A + B - 1) B
  have h2 := Nat.mod_lt (A + B - 1) hB
  rw [mul_comm]
  generalize (A + B - 1) / B = Q at h1 ⊢
  generalize (A + B - 1) % B = R at h1 h2
  omega

/-- **`StreamedWipeResidual`**: `streamed_wipe` at `g = N − wipeMissing N k s`, with the error split
`2^-(s+1) + 2^-(s+1)`. -/
theorem streamed_wipe_residual : Pous.Guarantees.StreamedWipeResidual := by
  intro N k b s q E κ prf P hN hk ⟨hlt, hq⟩
  obtain ⟨D, hD⟩ : ∃ D, wipeMissing N k s = D := ⟨_, rfl⟩
  have hceil := wipeMissing_spec (N := N) (s := s) hk
  unfold wipeResidual at hlt hq
  rw [hD] at hlt hq hceil
  have hD1 : 1 ≤ D := by
    rcases Nat.eq_zero_or_pos D with h | h
    · rw [h, zero_mul] at hceil
      have : 0 < (s + 1) * 6931471808 * N := by positivity
      omega
    · exact h
  have hDN : D ≤ N := by
    by_contra h
    have : N * b ≤ (N.log2 + 1 + b) * (D - 1) :=
      calc N * b ≤ (D - 1) * b := Nat.mul_le_mul_right _ (by omega)
        _ = b * (D - 1) := mul_comm _ _
        _ ≤ (N.log2 + 1 + b) * (D - 1) := Nat.mul_le_mul_right _ (by omega)
    omega
  have hsamp : ((s + 1 : ℕ) : ℝ) * 0.6931471808 ≤ k * D / N := by
    rw [le_div_iff₀ (by exact_mod_cast hN)]
    have hc : ((s + 1) * 6931471808 * N : ℝ) ≤ D * (k * 10 ^ 10) := by exact_mod_cast hceil
    push_cast
    nlinarith [hc]
  have hexp : (N.log2 + 1 + b) * (D - 1) = (N.log2 + 1) * (D - 1) + (D - 1) * b := by ring
  refine (streamed_wipe N k b q (N * b) E κ (N - D) prf P).trans (add_le_add ?_ le_rfl)
  unfold bound
  calc _ ≤ (2⁻¹ : ℝ≥0∞) ^ (s + 1) + (2⁻¹ : ℝ≥0∞) ^ (s + 1) :=
        add_le_add (sampling_le hDN hN hsamp)
          (compression_le_log hD1 hDN (Nat.le_of_lt Nat.lt_log2_self) (Nat.le_of_lt Nat.lt_log2_self) rfl (by omega))
    _ = (2⁻¹ : ℝ≥0∞) ^ s := half_pow_succ_add s

/-- At the deployed point (`StreamedWipeMeets2600GB`'s `N`, `k`, `b` and `2^-40`) the wipe residual is 8,549,712,478
bits, within its 8,600,000,000. -/
example : wipeResidual 40625000000 74000 512 40 = 8549712478 ∧ wipeResidual 40625000000 74000 512 40 ≤ 8600000000 := by
  decide

/-- At the served H200 (`StreamedWipe.h200`'s fill of 2,319,974,400 labels, `wipe_party.READS = 2152` reads, `2^-40`)
the wipe residual is 16,666,681,461 bits, below the fill's 1,187,826,892,800; 3,020 reads is where it falls to 1% of
the fill and 3,019 is not. -/
example : wipeResidual 2319974400 2152 512 40 = 16666681461 ∧ wipeResidual 2319974400 2152 512 40 < 2319974400 * 512 ∧
    100 * wipeResidual 2319974400 3020 512 40 ≤ 2319974400 * 512 ∧
    ¬ 100 * wipeResidual 2319974400 3019 512 40 ≤ 2319974400 * 512 := by
  decide

/-- At `2^-128`: 52,439,071,629 bits at the served H200 (between 4.41% and 4.42% of its fill) and 26,900,316,210 bits
at the deployed point. -/
example : wipeResidual 2319974400 2152 512 128 = 52439071629 ∧
    10000 * wipeResidual 2319974400 2152 512 128 ≤ 442 * (2319974400 * 512) ∧
    441 * (2319974400 * 512) ≤ 10000 * wipeResidual 2319974400 2152 512 128 ∧
    wipeResidual 40625000000 74000 512 128 = 26900316210 := by
  decide

/-- At no challenge the wipe residual is at least the fill: 40,625,000,000 labels of 512 bits and `2^-40`. -/
example : 40625000000 * 512 ≤ wipeResidual 40625000000 0 512 40 := by
  decide

end PousStreamedWipe
