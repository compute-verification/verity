module

public import Definitions.Claims
public import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# `k = ⌈4·N₀/e₋⌉`: `E^rep`'s accepted reruns per wait

`Claims.Flock.kOf` divides by `e`'s lower bound `e₋ = 2.7182818283` (`eLo`, below `e` by `Real.exp_one_gt_d9`), so `k`
is arithmetic on `ℚ`; `kOfReal` is the same at a real `N₀`, and `kOf` at a rational is `kOfReal` at its cast
(`kOf_eq`). What the proofs read of it:

* `k ≥ 4·N₀/e` at every `N₀` (`le_kOfReal`), which Flock's rate `N₀/(e·k) ≤ 1/4` needs;
* `k < 4·N₀/e₋ + 1` at `N₀ ≥ 0` (`kOfReal_lt`), and `k` against an integer at `e₋` alone (`kOfReal_le`, `lt_kOfReal`,
  `kOfReal_eq`), which the deployed numbers' bounds need;
* `k` is monotone in `N₀` (`kOfReal_mono`) and positive at `N₀ > 0` (`kOfReal_pos`).
-/

@[expose] public section

namespace Claims.Flock

theorem eLo_real : ((eLo : ℚ) : ℝ) = 2.7182818283 := by norm_num [eLo]

theorem eLo_pos : (0 : ℝ) < eLo := by rw [eLo_real]; norm_num

theorem eLo_lt_exp_one : (eLo : ℝ) < Real.exp 1 := by rw [eLo_real]; exact Real.exp_one_gt_d9

theorem kOfReal_def (N₀ : ℝ) : kOfReal N₀ = ⌈4 * N₀ / 2.7182818283⌉₊ := by rw [kOfReal, eLo_real]

/-- **`kOf` is `kOfReal` at a rational.** -/
theorem kOf_eq (q : ℚ) : kOf q = kOfReal q := by
  have h : 4 * (q : ℝ) / (eLo : ℝ) = ((4 * q / eLo : ℚ) : ℝ) := by push_cast; ring
  rw [kOf, kOfReal, h, ← Int.ceil_toNat, ← Int.ceil_toNat, Rat.ceil_cast]

theorem kOf_natCast (n : ℕ) : kOf n = kOfReal n := by rw [kOf_eq]; norm_cast

/-- **`k ≥ 4·N₀/e`** at every `N₀`. -/
theorem le_kOfReal (N₀ : ℝ) : 4 * N₀ / Real.exp 1 ≤ kOfReal N₀ := by
  rcases le_or_gt N₀ 0 with h | h
  · exact (div_nonpos_of_nonpos_of_nonneg (by linarith) (Real.exp_pos 1).le).trans (Nat.cast_nonneg _)
  · exact (div_le_div_of_nonneg_left (by linarith) eLo_pos eLo_lt_exp_one.le).trans (Nat.le_ceil _)

/-- **`k < 4·N₀/e₋ + 1`** at `N₀ ≥ 0`. -/
theorem kOfReal_lt {N₀ : ℝ} (h : 0 ≤ N₀) : (kOfReal N₀ : ℝ) < 4 * N₀ / 2.7182818283 + 1 := by
  rw [kOfReal_def]; exact Nat.ceil_lt_add_one (by positivity)

theorem kOfReal_le {N₀ : ℝ} {n : ℕ} (h : 4 * N₀ ≤ n * 2.7182818283) : kOfReal N₀ ≤ n := by
  rw [kOfReal_def, Nat.ceil_le, div_le_iff₀ (by norm_num)]; exact h

theorem lt_kOfReal {N₀ : ℝ} {n : ℕ} (h : n * 2.7182818283 < 4 * N₀) : n < kOfReal N₀ := by
  rw [kOfReal_def, Nat.lt_ceil, lt_div_iff₀ (by norm_num)]; exact h

/-- `k = n` from `(n − 1)·e₋ < 4·N₀ ≤ n·e₋`. -/
theorem kOfReal_eq {N₀ : ℝ} {n : ℕ} (hn : n ≠ 0) (h1 : ((n : ℝ) - 1) * 2.7182818283 < 4 * N₀)
    (h2 : 4 * N₀ ≤ n * 2.7182818283) : kOfReal N₀ = n := by
  rw [kOfReal_def, Nat.ceil_eq_iff hn, Nat.cast_sub (Nat.one_le_iff_ne_zero.2 hn), Nat.cast_one,
    lt_div_iff₀ (by norm_num), div_le_iff₀ (by norm_num)]
  exact ⟨h1, h2⟩

@[simp] theorem kOfReal_zero : kOfReal 0 = 0 := by simp [kOfReal]

theorem kOfReal_mono : Monotone kOfReal := fun _ _ h =>
  Nat.ceil_mono (div_le_div_of_nonneg_right (by linarith) eLo_pos.le)

theorem kOfReal_pos {N₀ : ℝ} (h : 0 < N₀) : 0 < kOfReal N₀ :=
  Nat.ceil_pos.2 (div_pos (by linarith) eLo_pos)

/-- `k = n` at `N₀ = n·e₋/4`. -/
theorem kOfReal_mul_eLo (n : ℕ) : kOfReal (n * eLo / 4) = n := by
  rw [kOfReal, show 4 * ((n : ℝ) * eLo / 4) / eLo = n by field_simp [eLo_pos.ne']]; exact Nat.ceil_natCast n

section Deploy

variable {Coin Col T : Type}

theorem Deploy.k_def (D : Deploy Coin Col T) (x : Msg) : D.k x = ⌈4 * D.N₀ x / 2.7182818283⌉₊ :=
  kOfReal_def _

end Deploy

end Claims.Flock
