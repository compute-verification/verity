import Proofs.Flock.Soundness.CROnly.Hankel

/-!
# `hm96-sha512`'s hiding gap

`hm96-sha512` (`verity/commitments/hm96/PROTOCOL.md` §1) hashes a salt of `l = 1536` bits with the Hankel matrix of a key
of `n + l − 1 = 2047` bits into `n = 512` bits, beside the salt's own SHA-512 hash `c`, which takes at most `2^512`
values. The leftover hash lemma (`sum_hidingGap_le`, `hankel_universal`) gives the gap's mean over a uniform key,
`½·√(2^512·2^512/2^1536) = 2^-257` (`hm96_sha512_sum`), and Markov's inequality the keys whose gap exceeds `2^-193`:
at most a `2^-64` fraction (`hm96_sha512_tail`). That the pinned key is not among them is `hash-derived-key`, the
common-reference-string step of §2a. Bits are a field `F` of two elements.
-/

namespace FlockSoundness.CROnly

open Finset

variable {F : Type} [Field F] [Fintype F] [DecidableEq F] (hF : Fintype.card F = 2)
include hF

theorem hm96_root {C : Type} [DecidableEq C] {n l m a : ℕ} (c : (Fin l → F) → C) (hc : #(univ.image c) ≤ 2 ^ m)
    (h : n + m + 2 * a ≤ l) :
    1 / 2 * Real.sqrt (Fintype.card (Fin n → F) * #(univ.image c) / Fintype.card (Fin l → F)) ≤
      1 / 2 ^ (a + 1) := by
  have hB : (Fintype.card (Fin n → F) : ℝ) = 2 ^ n := by simp [hF]
  have hY : (Fintype.card (Fin l → F) : ℝ) = 2 ^ l := by simp [hF]
  rw [hB, hY]
  have hc' : (#(univ.image c) : ℝ) ≤ 2 ^ m := by exact_mod_cast hc
  have hl : (2 : ℝ) ^ (n + m) * 2 ^ (2 * a) ≤ 2 ^ l := by
    rw [← pow_add]
    exact pow_le_pow_right₀ (by norm_num) h
  have h1 : (2 : ℝ) ^ n * #(univ.image c) / 2 ^ l ≤ (1 / 2 ^ a) ^ 2 := by
    rw [div_le_iff₀ (by positivity), div_pow, one_pow, ← pow_mul, mul_comm a 2, div_mul_eq_mul_div, one_mul,
      le_div_iff₀ (by positivity)]
    calc (2 : ℝ) ^ n * #(univ.image c) * 2 ^ (2 * a) ≤ 2 ^ n * 2 ^ m * 2 ^ (2 * a) := by gcongr
      _ = 2 ^ (n + m) * 2 ^ (2 * a) := by rw [pow_add]
      _ ≤ 2 ^ l := hl
  have h2 : Real.sqrt ((2 : ℝ) ^ n * #(univ.image c) / 2 ^ l) ≤ 1 / 2 ^ a :=
    (Real.sqrt_le_sqrt h1).trans_eq (Real.sqrt_sq (by positivity))
  calc 1 / 2 * Real.sqrt ((2 : ℝ) ^ n * #(univ.image c) / 2 ^ l) ≤ 1 / 2 * (1 / 2 ^ a) :=
        mul_le_mul_of_nonneg_left h2 (by norm_num)
    _ = 1 / 2 ^ (a + 1) := by rw [pow_succ]; field_simp

/-- **hm96's hiding gap on average over a uniform key**, for `n` output bits, an `l`-bit salt and a salt hash with at
most `2^m` values: at most `2^-(a+1)` whenever `n + m + 2a ≤ l`. -/
theorem hm96_sum {C : Type} [DecidableEq C] {n l m a : ℕ} (c : (Fin l → F) → C) (hc : #(univ.image c) ≤ 2 ^ m)
    (h : n + m + 2 * a ≤ l) :
    ∑ κ : Fin (n + l - 1) → F, hidingGap (hankel κ) c ≤
      Fintype.card (Fin (n + l - 1) → F) * (1 / 2 ^ (a + 1)) :=
  (sum_hidingGap_le (hankel : (Fin (n + l - 1) → F) → (Fin l → F) → Fin n → F)
    hankel_universal c).trans (mul_le_mul_of_nonneg_left (hm96_root hF c hc h) (Nat.cast_nonneg _))

/-- **hm96's hiding gap at all but a `2^-e` fraction of the keys**: the keys whose gap exceeds `2^-b` are at most a
`2^-e` fraction, whenever `n + m + 2a ≤ l` and `b + e = a + 1`. -/
theorem hm96_tail {C : Type} [DecidableEq C] {n l m a b e : ℕ} (c : (Fin l → F) → C)
    (hc : #(univ.image c) ≤ 2 ^ m) (h : n + m + 2 * a ≤ l) (hbe : b + e = a + 1) :
    (#{κ : Fin (n + l - 1) → F | 1 / 2 ^ b < hidingGap (hankel κ) c} : ℝ) ≤
      Fintype.card (Fin (n + l - 1) → F) / 2 ^ e := by
  have hk := (card_hidingGap_gt (hankel : (Fin (n + l - 1) → F) → (Fin l → F) → Fin n → F)
    hankel_universal c (ε := 1 / 2 ^ b)).trans
    (mul_le_mul_of_nonneg_left (hm96_root hF c hc h) (Nat.cast_nonneg _))
  have eq : (Fintype.card (Fin (n + l - 1) → F) : ℝ) * (1 / 2 ^ (a + 1)) =
      Fintype.card (Fin (n + l - 1) → F) / 2 ^ e * (1 / 2 ^ b) := by
    rw [← hbe, pow_add]
    field_simp
  rw [eq] at hk
  exact le_of_mul_le_mul_right hk (by positivity)

/-- **`hm96-sha512`'s hiding gap on average over a uniform key**: at most `2^-257`, for any salt hash `c` with at most
`2^512` values. -/
theorem hm96_sha512_sum {C : Type} [DecidableEq C] (c : (Fin 1536 → F) → C) (hc : #(univ.image c) ≤ 2 ^ 512) :
    ∑ κ : Fin (512 + 1536 - 1) → F, hidingGap (hankel κ) c ≤
      Fintype.card (Fin (512 + 1536 - 1) → F) * (1 / 2 ^ 257) :=
  hm96_sum hF (n := 512) (a := 256) c hc (by norm_num)

/-- **`hm96-sha512`'s hiding gap at all but `2^-64` of the keys**: the keys whose gap exceeds `2^-193` are at most a
`2^-64` fraction, for any salt hash `c` with at most `2^512` values. -/
theorem hm96_sha512_tail {C : Type} [DecidableEq C] (c : (Fin 1536 → F) → C) (hc : #(univ.image c) ≤ 2 ^ 512) :
    (#{κ : Fin (512 + 1536 - 1) → F | 1 / 2 ^ 193 < hidingGap (hankel κ) c} : ℝ) ≤
      Fintype.card (Fin (512 + 1536 - 1) → F) / 2 ^ 64 :=
  hm96_tail hF (n := 512) (a := 256) c hc (by norm_num) (by norm_num)

end FlockSoundness.CROnly
