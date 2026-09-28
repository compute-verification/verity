import Mathlib.Data.Rat.Floor
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
# Rounding to nearest, ties to even

`RoundsEven x q`: the natural number `q` is the rational `x` rounded to an integer, to nearest with ties to even. A
quotient of naturals rounded up by its remainder, as `Verity.Fp32.roundRn` and `Verity.Prims.roundEvenDyadic` round, is
that rounding (`roundsEven_div`), and it is unique (`RoundsEven.unique`).

On a floating-point grid, the points `M · 2^k` with significands `M < 2^(p + 1)` and exponents `k ≥ k₀`, the rounding
`q` of `x / 2^j` gives the nearest point `q · 2^j` to `x ≥ 0` when `j` is `x`'s binade exponent clamped below at `k₀`:
`j = k₀`, or `2^(j + p) ≤ x` (`nearest_grid`). Unless `x / 2^j` is a tie, every other point is strictly farther.

A format's value is a sign and a magnitude, `signed s m`. A magnitude nearest to `x ≥ 0` is, under `x`'s sign, nearest
among the signed values of every sign (`signed_near`), and when every other magnitude is strictly farther, no other
signed value is as near (`signed_tie`): the two zeros are one value.
-/

namespace Verity.Round

/-- `q` is `x` rounded to an integer, to nearest with ties to even. -/
structure RoundsEven (x : ℚ) (q : ℕ) : Prop where
  near : 2 * |(q : ℚ) - x| ≤ 1
  tie : 2 * |(q : ℚ) - x| = 1 → q % 2 = 0

theorem qdiv_lt {a b c : ℚ} (hb : 0 < b) (h : a < c * b) : a / b < c := by
  rwa [div_lt_iff₀ hb]

theorem qle_div {a b c : ℚ} (hb : 0 < b) (h : c * b ≤ a) : c ≤ a / b := by
  rwa [le_div_iff₀ hb]

theorem RoundsEven.of_mul {x D : ℚ} (hD : 0 < D) {q : ℕ} (h1 : 2 * |(q : ℚ) * D - x * D| ≤ D)
    (h2 : 2 * |(q : ℚ) * D - x * D| = D → q % 2 = 0) : RoundsEven x q := by
  have he : |(q : ℚ) * D - x * D| = |(q : ℚ) - x| * D := by rw [← sub_mul, abs_mul, abs_of_pos hD]
  rw [he, ← mul_assoc] at h1 h2
  refine ⟨le_of_mul_le_mul_right (by rw [one_mul]; exact h1) hD, fun h => h2 ?_⟩
  rw [h, one_mul]

/-- The quotient `n / D` rounded up when its remainder is past half of `D`, or half of it and the quotient odd, is
`n / D` rounded to nearest even. -/
theorem roundsEven_div (n D : ℕ) (hD : 0 < D) :
    RoundsEven ((n : ℚ) / D)
      (if 2 * (n % D) > D ∨ (2 * (n % D) = D ∧ n / D % 2 = 1) then n / D + 1 else n / D) := by
  obtain ⟨a, r, hr, rfl⟩ : ∃ a r, r < D ∧ n = D * a + r :=
    ⟨n / D, n % D, Nat.mod_lt n hD, (Nat.div_add_mod n D).symm⟩
  have ha : (D * a + r) / D = a := by
    rw [Nat.add_comm, Nat.add_mul_div_left r a hD, Nat.div_eq_of_lt hr, Nat.zero_add]
  have hr' : (D * a + r) % D = r := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hr]
  rw [ha, hr']
  have hD' : (0 : ℚ) < D := by exact_mod_cast hD
  have hx : ((D * a + r : ℕ) : ℚ) / D * D = ((D * a + r : ℕ) : ℚ) := ((div_eq_iff hD'.ne').1 rfl).symm
  have hrD : (r : ℚ) < D := by exact_mod_cast hr
  split_ifs with hc
  · have e : ((a + 1 : ℕ) : ℚ) * D - ((D * a + r : ℕ) : ℚ) / D * D = (D : ℚ) - r := by rw [hx]; push_cast; ring
    have h2r : (D : ℚ) ≤ 2 * r := by exact_mod_cast (show D ≤ 2 * r by omega)
    refine RoundsEven.of_mul hD' ?_ ?_
    · rw [e, abs_of_nonneg (by linarith)]; linarith
    · rw [e, abs_of_nonneg (by linarith)]; intro h
      have : 2 * r = D := by exact_mod_cast (show (2 * r : ℚ) = D by linarith)
      omega
  · have e : (a : ℚ) * D - ((D * a + r : ℕ) : ℚ) / D * D = -(r : ℚ) := by rw [hx]; push_cast; ring
    have h2r : 2 * (r : ℚ) ≤ D := by exact_mod_cast (show 2 * r ≤ D by omega)
    refine RoundsEven.of_mul hD' ?_ ?_
    · rw [e, abs_neg, abs_of_nonneg (by positivity)]; exact h2r
    · rw [e, abs_neg, abs_of_nonneg (by positivity)]; intro h
      have : 2 * r = D := by exact_mod_cast h
      omega

theorem RoundsEven.half {x : ℚ} {q : ℕ} (h : RoundsEven x q) : -(1 / 2) ≤ (q : ℚ) - x ∧ (q : ℚ) - x ≤ 1 / 2 :=
  abs_le.1 (by linarith [h.near])

theorem RoundsEven.le {x : ℚ} {q c : ℕ} (h : RoundsEven x q) (hx : x < c + 1 / 2) : q ≤ c := by
  have : (q : ℚ) < c + 1 := by linarith [h.half.2]
  have : q < c + 1 := by exact_mod_cast this
  omega

theorem RoundsEven.lt {x : ℚ} {q c : ℕ} (h : RoundsEven x q) (hx : x < c - 1 / 2) : q < c := by
  have : (q : ℚ) < c := by linarith [h.half.2]
  exact_mod_cast this

theorem RoundsEven.ge {x : ℚ} {q c : ℕ} (h : RoundsEven x q) (hx : (c : ℚ) - 1 / 2 < x) : c ≤ q := by
  have : (c : ℚ) < q + 1 := by linarith [h.half.1]
  have : c < q + 1 := by exact_mod_cast this
  omega

/-- At the tie `x = c - 1/2` below an even `c`, the rounding is `c`. -/
theorem RoundsEven.ge_even {x : ℚ} {q c : ℕ} (h : RoundsEven x q) (hx : (c : ℚ) - 1 / 2 ≤ x) (hc : c % 2 = 0) :
    c ≤ q := by
  rcases hx.lt_or_eq with hlt | heq
  · exact h.ge hlt
  · by_contra hq
    have hqc : (q : ℚ) + 1 ≤ c := by exact_mod_cast (show q + 1 ≤ c by omega)
    have e : (q : ℚ) - x = -(1 / 2) := by linarith [h.half.1]
    have t := h.tie (by rw [e, abs_neg, abs_of_pos (by norm_num : (0 : ℚ) < 1 / 2)]; norm_num)
    have : q + 1 = c := by exact_mod_cast (show (q : ℚ) + 1 = c by linarith)
    omega

theorem RoundsEven.not_succ {x : ℚ} {q q' : ℕ} (h : RoundsEven x q) (h' : RoundsEven x q') (hq : q + 1 ≤ q') :
    False := by
  have hqq : (q : ℚ) + 1 ≤ q' := by exact_mod_cast hq
  have e1 : (q : ℚ) - x = -(1 / 2) := by linarith [h.half.1, h'.half.2]
  have e2 : (q' : ℚ) - x = 1 / 2 := by linarith [h.half.1, h'.half.2]
  have t1 := h.tie (by rw [e1, abs_neg, abs_of_pos (by norm_num : (0 : ℚ) < 1 / 2)]; norm_num)
  have t2 := h'.tie (by rw [e2, abs_of_pos (by norm_num : (0 : ℚ) < 1 / 2)]; norm_num)
  have : q' = q + 1 := by exact_mod_cast (show (q' : ℚ) = q + 1 by linarith)
  omega

theorem RoundsEven.unique {x : ℚ} {q q' : ℕ} (h : RoundsEven x q) (h' : RoundsEven x q') : q = q' := by
  by_contra hne
  rcases Nat.lt_or_gt_of_ne hne with hl | hl
  · exact h.not_succ h' hl
  · exact h'.not_succ h hl

/-- An integer within half of `y` is at least as near to `y` as every other integer, and strictly nearer unless it is
exactly half away. -/
theorem near_int {y : ℚ} {q : ℕ} (h : 2 * |(q : ℚ) - y| ≤ 1) (P : ℕ) :
    |(q : ℚ) - y| ≤ |(P : ℚ) - y| ∧ (2 * |(q : ℚ) - y| < 1 → P ≠ q → |(q : ℚ) - y| < |(P : ℚ) - y|) := by
  rcases eq_or_ne P q with rfl | hP
  · exact ⟨le_rfl, fun _ h => absurd rfl h⟩
  · have h1 : (1 : ℚ) ≤ |(P : ℚ) - q| := by
      rcases Nat.lt_or_gt_of_ne hP with hl | hl
      · have : (P : ℚ) + 1 ≤ q := by exact_mod_cast hl
        rw [abs_sub_comm, abs_of_nonneg (by linarith)]; linarith
      · have : (q : ℚ) + 1 ≤ P := by exact_mod_cast hl
        rw [abs_of_nonneg (by linarith)]; linarith
    have h3 : |(P : ℚ) - q| ≤ |(P : ℚ) - y| + |(q : ℚ) - y| := by
      calc |(P : ℚ) - q| ≤ |(P : ℚ) - y| + |y - q| := abs_sub_le _ _ _
        _ = |(P : ℚ) - y| + |(q : ℚ) - y| := by rw [abs_sub_comm y]
    exact ⟨by linarith, fun _ _ => by linarith⟩

/-- **The nearest grid point.** With `j` the exponent of `x`'s binade clamped at `k₀` (`j = k₀`, or `2^(j + p) ≤ x`)
and `q` within half of `x / 2^j`, the point `q · 2^j` is at least as near to `x` as every `M · 2^k` with `M < 2^(p + 1)`
and `k ≥ k₀`, and strictly nearer than every other such point when `q` is strictly within half. -/
theorem nearest_grid {p k₀ j q : ℕ} {x : ℚ} (hlow : j = k₀ ∨ (2 : ℚ) ^ (j + p) ≤ x)
    (hq : 2 * |(q : ℚ) - x / 2 ^ j| ≤ 1) {M k : ℕ} (hM : M < 2 ^ (p + 1)) (hk : k₀ ≤ k) :
    |(q : ℚ) * 2 ^ j - x| ≤ |(M : ℚ) * 2 ^ k - x| ∧
      (2 * |(q : ℚ) - x / 2 ^ j| < 1 → (M : ℚ) * 2 ^ k ≠ q * 2 ^ j →
        |(q : ℚ) * 2 ^ j - x| < |(M : ℚ) * 2 ^ k - x|) := by
  have hpj : (0 : ℚ) < 2 ^ j := by positivity
  have hs : ∀ a : ℚ, |a * 2 ^ j - x| = |a - x / 2 ^ j| * 2 ^ j := fun a => by
    have : a * 2 ^ j - x = (a - x / 2 ^ j) * 2 ^ j := by
      rw [sub_mul, div_mul_cancel₀ x hpj.ne']
    rw [this, abs_mul, abs_of_pos hpj]
  rcases le_or_gt j k with hjk | hjk
  · have hM' : (M : ℚ) * 2 ^ k = ((M * 2 ^ (k - j) : ℕ) : ℚ) * 2 ^ j := by
      push_cast; rw [mul_assoc, ← pow_add, Nat.sub_add_cancel hjk]
    rw [hs, hM', hs]
    obtain ⟨n1, n2⟩ := near_int hq (M * 2 ^ (k - j))
    exact ⟨mul_le_mul_of_nonneg_right n1 hpj.le,
      fun h1 h2 => mul_lt_mul_of_pos_right (n2 h1 fun he => h2 (by rw [he])) hpj⟩
  · have hx0 : (2 : ℚ) ^ (j + p) ≤ x := hlow.resolve_left (by omega)
    have hMk : (M : ℚ) * 2 ^ k < 2 ^ (j + p) := by
      have h1 : (M : ℚ) < 2 ^ (p + 1) := by exact_mod_cast hM
      have h2 : (2 : ℚ) ^ (p + 1) * 2 ^ k ≤ 2 ^ (j + p) := by
        rw [← pow_add]; exact_mod_cast Nat.pow_le_pow_right (by norm_num) (by omega)
      calc (M : ℚ) * 2 ^ k < 2 ^ (p + 1) * 2 ^ k := mul_lt_mul_of_pos_right h1 (by positivity)
        _ ≤ 2 ^ (j + p) := h2
    have hP : |((2 ^ p : ℕ) : ℚ) * 2 ^ j - x| = x - 2 ^ (j + p) := by
      push_cast; rw [← pow_add, Nat.add_comm p j, abs_of_nonpos (by linarith)]; ring
    have hMx : |(M : ℚ) * 2 ^ k - x| = x - M * 2 ^ k := by
      rw [abs_of_nonpos (by linarith)]; ring
    have hle : |(q : ℚ) * 2 ^ j - x| ≤ x - 2 ^ (j + p) := by
      rw [← hP, hs, hs]; exact mul_le_mul_of_nonneg_right (near_int hq (2 ^ p)).1 hpj.le
    rw [hMx]
    exact ⟨by linarith, fun _ _ => by linarith⟩

/-! ## Signed values -/

/-- A sign and a magnitude as a signed value. -/
def signed (negative : Bool) (m : ℚ) : ℚ := if negative then -m else m

theorem signed_sub (s : Bool) (a b : ℚ) : |signed s a - signed s b| = |a - b| := by
  cases s
  · rfl
  · show |-a - -b| = |a - b|
    rw [show -a - -b = -(a - b) by ring, abs_neg]

theorem signed_cross {s s' : Bool} (hs : s' ≠ s) {a b : ℚ} (ha : 0 ≤ a) (hb : 0 ≤ b) :
    |signed s' a - signed s b| = a + b := by
  cases s <;> cases s'
  · exact absurd rfl hs
  · show |-a - b| = a + b
    rw [show -a - b = -(a + b) by ring, abs_neg, abs_of_nonneg (by linarith)]
  · show |a - -b| = a + b
    rw [show a - -b = a + b by ring, abs_of_nonneg (by linarith)]
  · exact absurd rfl hs

theorem signed_le (s s' : Bool) {a b : ℚ} (ha : 0 ≤ a) (hb : 0 ≤ b) : |a - b| ≤ |signed s' a - signed s b| := by
  by_cases hs : s' = s
  · rw [hs, signed_sub]
  · rw [signed_cross hs ha hb, abs_sub_le_iff]; constructor <;> linarith

/-- A magnitude `m₀` at least as near to `x ≥ 0` as `m'` is, under `x`'s sign, at least as near as `m'` under any. -/
theorem signed_near (s s' : Bool) {x m₀ m' : ℚ} (hx : 0 ≤ x) (hm' : 0 ≤ m') (hle : |m₀ - x| ≤ |m' - x|) :
    |signed s m₀ - signed s x| ≤ |signed s' m' - signed s x| := by
  rw [signed_sub]; exact hle.trans (signed_le s s' hm' hx)

/-- When every other magnitude is strictly farther from `x > 0` than `m₀`, no signed value other than `m₀`'s under
`x`'s sign is as near as it. -/
theorem signed_tie (s s' : Bool) {x m₀ m' : ℚ} (hx : 0 < x) (hm₀ : 0 ≤ m₀) (hm' : 0 ≤ m')
    (hlt : m' ≠ m₀ → |m₀ - x| < |m' - x|) (hne : signed s' m' ≠ signed s m₀)
    (heq : |signed s' m' - signed s x| = |signed s m₀ - signed s x|) : False := by
  rw [signed_sub s m₀] at heq
  by_cases hm : m' = m₀
  · rw [hm] at hne heq
    have hs : s' ≠ s := fun hs => hne (by rw [hs])
    rw [signed_cross hs hm₀ hx.le] at heq
    have hpos : 0 < m₀ := lt_of_le_of_ne hm₀ fun h0 => hne (by rw [← h0]; simp [signed])
    have := abs_sub_lt_iff.2 (⟨by linarith, by linarith⟩ : m₀ - x < m₀ + x ∧ x - m₀ < m₀ + x)
    linarith
  · have h1 := hlt hm
    have h2 := signed_le s s' hm' hx.le
    linarith

end Verity.Round
