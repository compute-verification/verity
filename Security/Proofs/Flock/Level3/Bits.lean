/-
Level 3, L3-F (PROTOCOL.md §2.1, §17.4 stage 2): the bits of a natural number as a polynomial over GF(2). XOR is addition,
a left shift is multiplication by a power of `X`, and a number split at bit `w` is `X^w · high + low`.
-/
import Mathlib.Algebra.Polynomial.Div
import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Nat.Bitwise

open Polynomial

namespace FlockLevel3

abbrev P := Polynomial (ZMod 2)

/-- `1` for a set bit, `0` otherwise. -/
def b2 (b : Bool) : ZMod 2 := if b then 1 else 0

@[simp] theorem b2_true : b2 true = 1 := rfl
@[simp] theorem b2_false : b2 false = 0 := rfl

theorem b2_xor (x y : Bool) : b2 (x ^^ y) = b2 x + b2 y := by
  cases x <;> cases y <;> decide

theorem b2_and (x y : Bool) : b2 (x && y) = b2 x * b2 y := by
  cases x <;> cases y <;> decide

/-- The polynomial whose coefficient of `X^i` is bit `i` of `n`. -/
noncomputable def natPoly (n : ℕ) : P := ∑ i ∈ Finset.range n, if n.testBit i then X ^ i else 0

theorem testBit_ge (n k : ℕ) (hk : n ≤ k) : n.testBit k = false :=
  Nat.testBit_lt_two_pow (lt_of_lt_of_le Nat.lt_two_pow_self (Nat.pow_le_pow_right (by norm_num) hk))

theorem coeff_natPoly (n k : ℕ) : (natPoly n).coeff k = b2 (n.testBit k) := by
  unfold natPoly
  rw [finsetSum_coeff]
  have h : ∀ i ∈ Finset.range n, (if n.testBit i then (X : P) ^ i else 0).coeff k =
      if i = k then b2 (n.testBit k) else 0 := by
    intro i _
    by_cases hb : n.testBit i
    · simp only [hb, ite_true, coeff_X_pow]
      by_cases hik : i = k
      · subst hik; simp [hb]
      · simp [hik, Ne.symm hik]
    · simp only [hb]
      by_cases hik : i = k
      · subst hik; simp [hb]
      · simp [hik]
  rw [Finset.sum_congr rfl h, Finset.sum_ite_eq']
  by_cases hk : k < n
  · simp [Finset.mem_range.mpr hk]
  · simp [Finset.mem_range, hk, testBit_ge n k (not_lt.mp hk)]

theorem natPoly_ext {p : P} {n : ℕ} (h : ∀ k, p.coeff k = b2 (n.testBit k)) : natPoly n = p := by
  ext k; rw [coeff_natPoly, h]

@[simp] theorem natPoly_zero : natPoly 0 = 0 := by
  ext k; simp [coeff_natPoly]

theorem natPoly_one : natPoly 1 = 1 := by
  ext k; rw [coeff_natPoly, coeff_one]
  rcases k with _ | k
  · decide
  · simp [Nat.testBit_succ]

theorem natPoly_xor (a b : ℕ) : natPoly (a ^^^ b) = natPoly a + natPoly b := by
  ext k; simp only [coeff_natPoly, coeff_add, Nat.testBit_xor, b2_xor]

theorem natPoly_shiftLeft (a t : ℕ) : natPoly (a <<< t) = X ^ t * natPoly a := by
  ext k
  rw [coeff_natPoly, coeff_X_pow_mul', Nat.testBit_shiftLeft]
  by_cases h : t ≤ k
  · simp [h, coeff_natPoly]
  · simp [h]

theorem natPoly_split (a b w : ℕ) (hb : b < 2 ^ w) : natPoly (2 ^ w * a + b) = X ^ w * natPoly a + natPoly b := by
  ext k
  rw [coeff_natPoly, Nat.testBit_two_pow_mul_add a hb, coeff_add, coeff_X_pow_mul', coeff_natPoly, coeff_natPoly]
  by_cases h : k < w
  · simp [h, show ¬ w ≤ k by omega]
  · have hbk : b.testBit k = false := Nat.testBit_lt_two_pow (lt_of_lt_of_le hb (Nat.pow_le_pow_right (by norm_num) (by omega)))
    simp [h, show w ≤ k by omega, hbk]

/-- A number below `2^w` has degree below `w`. -/
theorem natPoly_degree_lt (n w : ℕ) (hn : n < 2 ^ w) : (natPoly n).degree < w := by
  rw [Polynomial.degree_lt_iff_coeff_zero]
  intro k hk
  rw [coeff_natPoly, Nat.testBit_lt_two_pow (lt_of_lt_of_le hn (Nat.pow_le_pow_right (by norm_num) hk))]
  rfl

theorem natPoly_injective (m n : ℕ) (h : natPoly m = natPoly n) : m = n := by
  apply Nat.eq_of_testBit_eq
  intro k
  have := congrArg (fun p => p.coeff k) h
  simp only [coeff_natPoly] at this
  cases hmk : m.testBit k <;> cases hnk : n.testBit k <;> simp_all [b2]

end FlockLevel3
