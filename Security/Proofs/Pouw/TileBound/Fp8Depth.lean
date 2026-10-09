import Definitions.Pouw.TileBound.Defs
import Mathlib.Tactic.IntervalCases
import Definitions.Pouw.TileBound.Fp8Defs

/-!
# When a ⟨2,2,2⟩ recursion could pay at E4M3, and the output region it needs (the TT_OUT restatements lane)

A recursion of `L` levels of ⟨2,2,2⟩ at rank 7 (Strassen–Winograd) runs `(7/8)^L` of the MACs in its sub-products. In a
format priced `p` against E4M3's 1 it can undercut the honest E4M3 chain only if `p·(7/8)^L < 1`, and each level halves k,
m and n alike. So a depth-`L` ⟨2,2,2⟩ recursion needs an exact k-range at least `16·2^L` deep for the k16 BF16/FP16 atom,
and an all-exact output region of `16·2^L × 8·2^L` words, since the leaves are m16n8 fragments.

* `bf16_pays_iff`: BF16 or FP16 ⟨2,2,2⟩ sub-products undercut the chain exactly from depth 6, a region of at least
  1,024 × 512 words over a range at least 1,024 deep (`bf16_region_of_pays`).
* `tf32_pays_iff`, `fp32_pays_iff`: TF32 from depth 11, FFMA from depth 16. Their regions exceed an 8,192-row unit.
  FFMA's threshold depends on its price: depth 16 is the first paying depth exactly for prices p with
  (8/7)^15 ≤ p < (8/7)^16, about 7.41 ≤ p < 8.47. The model's 8.46 and GPU 4's measured 8.0–8.15 (16.0–16.3 FP4 slots)
  both lie there. At 8.47 or more it would be depth 17, and below 7.41 depth 15.

**Scope: ⟨2,2,2⟩ only.** Rank 7 per ⟨2,2,2⟩ is not the best known ratio. Smirnov's ⟨3,3,6;40⟩ (coefficients {±1, ±1/8}),
DPS's ⟨4,4,4;48⟩ (coefficients {±1, ±1/2, ±1/4, ±1/8}) and the larger catalogued schemes have better ratios and pay at
smaller depths, over other regions and shorter windows. The region family that `no-aligned-exact-region` covers is
computed over all of them (the restatements report §7). These theorems are arithmetic on the model's prices for
⟨2,2,2⟩ recursion, not lower bounds for general bilinear algorithms.
-/

namespace Pouw.TileBound

/-- **BF16 or FP16 sub-products undercut the honest E4M3 chain exactly from depth 6.** -/
theorem bf16_pays_iff (L : ℕ) : price .bf16 * (7 / 8 : ℚ) ^ L < price .fp8 ↔ 6 ≤ L := by
  simp only [price]
  constructor
  · intro h
    by_contra hL
    replace hL := Nat.lt_of_not_le hL
    interval_cases L <;> norm_num at h
  · intro hL
    have hmono : (7 / 8 : ℚ) ^ L ≤ (7 / 8) ^ 6 := pow_le_pow_of_le_one (by norm_num) (by norm_num) hL
    have : (2 : ℚ) * (7 / 8) ^ 6 < 1 := by norm_num
    linarith

/-- **TF32 sub-products undercut the chain exactly from depth 11.** -/
theorem tf32_pays_iff (L : ℕ) : price .tf32 * (7 / 8 : ℚ) ^ L < price .fp8 ↔ 11 ≤ L := by
  simp only [price]
  constructor
  · intro h
    by_contra hL
    replace hL := Nat.lt_of_not_le hL
    interval_cases L <;> norm_num at h
  · intro hL
    have hmono : (7 / 8 : ℚ) ^ L ≤ (7 / 8) ^ 11 := pow_le_pow_of_le_one (by norm_num) (by norm_num) hL
    have : (4 : ℚ) * (7 / 8) ^ 11 < 1 := by norm_num
    linarith

/-- **FFMA sub-products undercut the chain exactly from depth 16.** -/
theorem fp32_pays_iff (L : ℕ) : price .fp32 * (7 / 8 : ℚ) ^ L < price .fp8 ↔ 16 ≤ L := by
  simp only [price]
  constructor
  · intro h
    by_contra hL
    replace hL := Nat.lt_of_not_le hL
    interval_cases L <;> norm_num at h
  · intro hL
    have hmono : (7 / 8 : ℚ) ^ L ≤ (7 / 8) ^ 16 := pow_le_pow_of_le_one (by norm_num) (by norm_num) hL
    have : (846 / 100 : ℚ) * (7 / 8) ^ 16 < 1 := by norm_num
    linarith

/-- A paying BF16 or FP16 recursion needs an all-exact region of at least 1,024 × 512 words over at least 1,024 of k. -/
theorem bf16_region_of_pays {L : ℕ} (h : price .bf16 * (7 / 8 : ℚ) ^ L < price .fp8) :
    1024 ≤ 16 * 2 ^ L ∧ 512 ≤ 8 * 2 ^ L := by
  have hL := (bf16_pays_iff L).1 h
  have : 2 ^ 6 ≤ 2 ^ L := Nat.pow_le_pow_right (by norm_num) hL
  constructor <;> omega

/-- A paying TF32 recursion needs more than 8,192 rows of output. -/
theorem tf32_region_of_pays {L : ℕ} (h : price .tf32 * (7 / 8 : ℚ) ^ L < price .fp8) : 8192 < 16 * 2 ^ L := by
  have hL := (tf32_pays_iff L).1 h
  have : 2 ^ 11 ≤ 2 ^ L := Nat.pow_le_pow_right (by norm_num) hL
  omega

/-- A paying FFMA recursion needs more than 8,192 rows of output. -/
theorem fp32_region_of_pays {L : ℕ} (h : price .fp32 * (7 / 8 : ℚ) ^ L < price .fp8) : 8192 < 16 * 2 ^ L := by
  have hL := (fp32_pays_iff L).1 h
  have : 2 ^ 16 ≤ 2 ^ L := Nat.pow_le_pow_right (by norm_num) hL
  omega

end Pouw.TileBound
