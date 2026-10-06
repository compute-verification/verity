import Proofs.Flock.Soundness.Assumptions.Zk

/-!
# `PadNonvanishing40`, proved

`Assumptions.Zk.PadNonvanishing40` says `Ŵ_cols(ω_q) + PAD_C ≠ 0` for `cols ≤ 40` and `q < 2^(cols+1)`. It holds outright:
`Ŵ_cols(ω_q)` is bit `cols` of `q`, so 0 or 1, and `PAD_C = x` is neither.

* `What_omega`: `Ŵ_i(ω_q) ∈ {0, 1}` for `i < 64`, `q < 2^(i+1)`.
* `padNonvanishing40`: the instance Lemma B's row (ii) reads (`Composed.zkViewAt_of_ok`), so no `--zk` headline takes it.
-/

namespace FlockSoundness.ZK

open FlockLevel3 Polynomial

/-- `s_i` is additive: GF(2^128) has characteristic 2, so each step `s ↦ s² + c·s` is. -/
theorem sPoly_eval_add : ∀ (i : ℕ) (a b : GF128), (sPoly i).eval (a + b) = (sPoly i).eval a + (sPoly i).eval b
  | 0, a, b => by simp [sPoly]
  | i + 1, a, b => by
    simp only [sPoly, eval_add, eval_pow, eval_mul, eval_C, sPoly_eval_add i a b, CharTwo.add_sq]
    ring

theorem sPoly_eval_zero : ∀ i, (sPoly i).eval (0 : GF128) = 0
  | 0 => by simp [sPoly]
  | i + 1 => by simp [sPoly, sPoly_eval_zero i]

/-- `s_i` vanishes at `β_j = x^j` for `j < i`. -/
theorem sPoly_eval_beta : ∀ i j, j < i → (sPoly i).eval (beta j) = 0
  | 0, _, h => absurd h (Nat.not_lt_zero _)
  | i + 1, j, h => by
    simp only [sPoly, eval_add, eval_pow, eval_mul, eval_C]
    rcases Nat.lt_succ_iff_lt_or_eq.mp h with h | rfl
    · rw [sPoly_eval_beta i j h]; ring
    · rw [sq]; exact CharTwo.add_self_eq_zero _

theorem omega_zero : omega 0 = 0 := by
  rw [GF128.zero_eq]; rfl

theorem two_pow_xor_of_lt {k r : ℕ} (hr : r < 2 ^ k) : 2 ^ k ^^^ r = 2 ^ k + r := by
  have h := Nat.two_pow_add_eq_or_of_lt hr 1
  rw [Nat.mul_one] at h
  rw [h]
  apply Nat.eq_of_testBit_eq; intro j
  simp only [Nat.testBit_xor, Nat.testBit_or, Nat.testBit_two_pow]
  by_cases hj : k = j
  · subst hj; simp [Nat.testBit_lt_two_pow hr]
  · simp [hj]

/-- `ω_{2^k + r} = x^k + ω_r` for `r < 2^k`: bit `k` of the low word is the coefficient of `x^k`. -/
theorem omega_two_pow_add (k r : ℕ) (hk : k < 64) (hr : r < 2 ^ k) :
    omega (2 ^ k + r) = beta k + omega r := by
  simp only [omega, beta, Flock.F128.monomial, hk, ↓reduceIte]
  rw [← of_add]
  show GF128.of ⟨(2 ^ k + r).toUInt64, 0⟩ = GF128.of ⟨(1 : UInt64) <<< k.toUInt64 ^^^ r.toUInt64, 0 ^^^ 0⟩
  congr 2
  apply UInt64.toNat.inj
  have h64 : 2 ^ k < 2 ^ 64 := Nat.pow_lt_pow_right (by norm_num) hk
  have hk' : k % 2 ^ 64 = k := Nat.mod_eq_of_lt (by omega)
  have hsum : 2 ^ k + r < 2 ^ 64 :=
    lt_of_lt_of_le (by rw [pow_succ]; omega : 2 ^ k + r < 2 ^ (k + 1)) (Nat.pow_le_pow_right (by norm_num) hk)
  simp only [UInt64.toNat_xor, UInt64.toNat_shiftLeft, Nat.toUInt64_eq, UInt64.toNat_ofNat', UInt64.toNat_one]
  rw [hk', Nat.mod_eq_of_lt hk, Nat.one_shiftLeft, Nat.mod_eq_of_lt h64, Nat.mod_eq_of_lt (by omega : r < 2 ^ 64),
    Nat.mod_eq_of_lt hsum, two_pow_xor_of_lt hr]

/-- `s_i(ω_q) = 0` for `q < 2^k`, `k ≤ i`. -/
theorem sPoly_eval_omega_lt : ∀ k ≤ 64, ∀ q < 2 ^ k, ∀ i, k ≤ i → (sPoly i).eval (omega q) = 0
  | 0, _, q, hq, i, _ => by
    obtain rfl : q = 0 := by simpa using hq
    rw [omega_zero, sPoly_eval_zero]
  | k + 1, hk, q, hq, i, hi => by
    by_cases h : q < 2 ^ k
    · exact sPoly_eval_omega_lt k (by omega) q h i (by omega)
    · obtain ⟨r, rfl⟩ : ∃ r, q = 2 ^ k + r := ⟨q - 2 ^ k, by omega⟩
      have hr : r < 2 ^ k := by rw [pow_succ] at hq; omega
      rw [omega_two_pow_add k r (by omega) hr, sPoly_eval_add, sPoly_eval_beta i k (by omega),
        sPoly_eval_omega_lt k (by omega) r hr i (by omega), add_zero]

/-- `Ŵ_i(ω_q)` is 0 or 1 on the first `2^(i+1)` domain points. -/
theorem What_omega (i : ℕ) (hi : i < 64) (q : ℕ) (hq : q < 2 ^ (i + 1)) :
    What i (omega q) = 0 ∨ What i (omega q) = 1 := by
  have hs : (sPoly i).eval (omega q) = 0 ∨ (sPoly i).eval (omega q) = (sPoly i).eval (beta i) := by
    by_cases h : q < 2 ^ i
    · exact Or.inl (sPoly_eval_omega_lt i (by omega) q h i le_rfl)
    · obtain ⟨r, rfl⟩ : ∃ r, q = 2 ^ i + r := ⟨q - 2 ^ i, by omega⟩
      have hr : r < 2 ^ i := by rw [pow_succ] at hq; omega
      right
      rw [omega_two_pow_add i r hi hr, sPoly_eval_add, sPoly_eval_omega_lt i (by omega) r hr i le_rfl, add_zero]
  unfold What
  rcases hs with h | h
  · left; rw [h, zero_mul]
  · by_cases h0 : (sPoly i).eval (beta i) = 0
    · left; rw [h, h0, zero_mul]
    · right; rw [h, mul_inv_cancel₀ h0]

theorem padC_ne_zero : Refine.fOf Flock.Zk.PAD_C ≠ 0 := by
  rw [GF128.zero_eq]; intro h; exact absurd (h : Flock.Zk.PAD_C = 0) (by decide)

theorem padC_ne_one : Refine.fOf Flock.Zk.PAD_C ≠ 1 := by
  rw [GF128.one_eq]; intro h; exact absurd (h : Flock.Zk.PAD_C = 1) (by decide)

/-- **`PadNonvanishing40` holds.** -/
theorem padNonvanishing40 : Assumptions.Zk.PadNonvanishing40 := by
  unfold Assumptions.Zk.PadNonvanishing40 Assumptions.PadNonvanishing
  intro cols hcols q hq
  show What cols (omega q) + Refine.fOf Flock.Zk.PAD_C ≠ 0
  rcases What_omega cols (by omega) q hq with h | h <;> rw [h]
  · rw [zero_add]; exact padC_ne_zero
  · intro h1
    exact padC_ne_one (by rw [eq_neg_of_add_eq_zero_right h1, CharTwo.neg_eq])

end FlockSoundness.ZK
