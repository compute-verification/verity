import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import Definitions.Pouw.Fp8Atom.Pinned
import Proofs.Pouw.Fp8Atom.Checks.Add

/-!
# Proofs of the FP32 add statements

`binade` brackets a positive rational (`binade_spec`, from `Nat.log2` of its numerator and denominator), so a nonzero
`x` has its significand `|x| / ulp x` in `[2^23, 2^24)`. `rn` is the sign times `rp |x|`, where `rp` rounds a
magnitude; `rp` is monotone within a binade (`roundEven_mono`) and across binades (it maps a binade into
`[2^k, 2^(k+1)]`). Exactness is `roundEven` of an integer; absorption and movement follow from `roundEven_near`,
monotonicity and exactness at the neighbours `r ± ulp r`.
-/

namespace Pouw.Fp8Atom.Proofs

open Pouw.Fp8Atom

/-! ## Conformance -/

theorem add_length : Vectors.add.length = 1090 := by decide +kernel

/-- **FP32 add conformance** on the 1,090 vectors. -/
theorem addConformance : AddConformance := by
  refine ⟨by rw [add_length], fun v hv => ?_⟩
  have h := List.all_eq_true.mp Checks.add v hv
  unfold AddVec.holds at h
  split at h
  next x y z ha hb hc => exact ⟨x, y, z, ha, hb, hc, of_decide_eq_true h⟩
  next => exact absurd h Bool.false_ne_true

/-! ## Binades and significands -/

theorem two_zpow_pos (k : ℤ) : (0 : ℚ) < 2 ^ k := zpow_pos (by norm_num) k

theorem two_pow_23 (k : ℤ) : (2 : ℚ) ^ 23 * 2 ^ (k - 23) = 2 ^ k := by
  rw [← zpow_natCast, ← zpow_add₀ two_ne_zero]; congr 1; push_cast; ring

theorem two_pow_24 (k : ℤ) : (2 : ℚ) ^ 24 * 2 ^ (k - 23) = 2 ^ (k + 1) := by
  rw [← zpow_natCast, ← zpow_add₀ two_ne_zero]; congr 1; push_cast; ring

theorem binade_spec {x : ℚ} (hx : 0 < x) : 2 ^ binade x ≤ x ∧ x < 2 ^ (binade x + 1) := by
  simp only [binade]
  have hn0 : x.num.natAbs ≠ 0 := by simp [hx.ne']
  have hxnd : x = (x.num.natAbs : ℚ) / x.den := by
    have : (x.num : ℚ) = (x.num.natAbs : ℚ) := by
      rw [Nat.cast_natAbs, abs_of_nonneg (Rat.num_pos.2 hx).le]
    rw [← this]; exact (Rat.num_div_den x).symm
  have hn1 := Nat.log2_self_le hn0
  have hn2 := @Nat.lt_log2_self x.num.natAbs
  have hd1 := Nat.log2_self_le x.den_nz
  have hd2 := @Nat.lt_log2_self x.den
  generalize x.num.natAbs = n at *
  generalize x.den = d at *
  generalize n.log2 = a at *
  generalize d.log2 = b at *
  have hd : (0 : ℚ) < d := by exact_mod_cast (show 0 < d by omega)
  have hhi : x < 2 ^ ((a : ℤ) - b + 1) := by
    rw [show (a : ℤ) - b + 1 = ((a + 1 : ℕ) : ℤ) - (b : ℕ) by push_cast; ring, zpow_sub₀ two_ne_zero,
      zpow_natCast, zpow_natCast, hxnd, div_lt_div_iff₀ hd (by positivity)]
    have : n * 2 ^ b < 2 ^ (a + 1) * d :=
      calc n * 2 ^ b < 2 ^ (a + 1) * 2 ^ b := Nat.mul_lt_mul_of_pos_right hn2 (by positivity)
        _ ≤ 2 ^ (a + 1) * d := Nat.mul_le_mul_left _ hd1
    exact_mod_cast this
  have hlo : 2 ^ ((a : ℤ) - b - 1) < x := by
    rw [show (a : ℤ) - b - 1 = ((a : ℕ) : ℤ) - (b + 1 : ℕ) by push_cast; ring, zpow_sub₀ two_ne_zero,
      zpow_natCast, zpow_natCast, hxnd, div_lt_div_iff₀ (by positivity) hd]
    have : 2 ^ a * d < n * 2 ^ (b + 1) :=
      calc 2 ^ a * d < 2 ^ a * 2 ^ (b + 1) := Nat.mul_lt_mul_of_pos_left hd2 (by positivity)
        _ ≤ n * 2 ^ (b + 1) := Nat.mul_le_mul_right _ hn1
    exact_mod_cast this
  split_ifs with h
  · exact ⟨hlo.le, by rwa [sub_add_cancel]⟩
  · exact ⟨not_lt.1 h, hhi⟩

theorem binade_unique {x : ℚ} {k : ℤ} (hx : 0 < x) (h1 : 2 ^ k ≤ x) (h2 : x < 2 ^ (k + 1)) : binade x = k := by
  obtain ⟨l1, l2⟩ := binade_spec hx
  have e1 := (zpow_lt_zpow_iff_right₀ (one_lt_two (α := ℚ))).1 (lt_of_le_of_lt l1 h2)
  have e2 := (zpow_lt_zpow_iff_right₀ (one_lt_two (α := ℚ))).1 (lt_of_le_of_lt h1 l2)
  omega

theorem binade_mono {s t : ℚ} (hs : 0 < s) (h : s ≤ t) : binade s ≤ binade t := by
  have := (zpow_lt_zpow_iff_right₀ (one_lt_two (α := ℚ))).1
    (lt_of_le_of_lt (binade_spec hs).1 (lt_of_le_of_lt h (binade_spec (lt_of_lt_of_le hs h)).2))
  omega

theorem ulp_pos (x : ℚ) : 0 < ulp x := two_zpow_pos _

theorem ulp_abs (x : ℚ) : ulp |x| = ulp x := by simp [ulp]

/-- The significand of `x ≠ 0` lies in `[2^23, 2^24)`. -/
theorem sig_bounds {x : ℚ} (hx : x ≠ 0) : 2 ^ 23 ≤ |x| / ulp x ∧ |x| / ulp x < 2 ^ 24 := by
  have hx' : 0 < |x| := abs_pos.2 hx
  obtain ⟨h1, h2⟩ := binade_spec hx'
  rw [le_div_iff₀ (ulp_pos x), div_lt_iff₀ (ulp_pos x), ulp, two_pow_23, two_pow_24]
  exact ⟨h1, h2⟩

/-! ### Rounding to an integer -/

theorem floor_le_roundEven (m : ℚ) : ⌊m⌋ ≤ roundEven m := by
  unfold roundEven; dsimp only; split_ifs <;> omega

theorem roundEven_le (m : ℚ) : roundEven m ≤ ⌊m⌋ + 1 := by
  unfold roundEven; dsimp only; split_ifs <;> omega

theorem roundEven_intCast (n : ℤ) : roundEven n = n := by
  unfold roundEven; simp [Int.floor_intCast]

theorem roundEven_near {m : ℚ} {n : ℤ} (h : |m - n| < 1 / 2) : roundEven m = n := by
  obtain ⟨h1, h2⟩ := abs_lt.1 h
  unfold roundEven; dsimp only
  rcases le_or_gt (n : ℚ) m with hm | hm
  · have hf : ⌊m⌋ = n := Int.floor_eq_iff.2 ⟨hm, by linarith⟩
    rw [hf, ite_eq_left h2]
  · have hf : ⌊m⌋ = n - 1 := Int.floor_eq_iff.2 ⟨by push_cast; linarith, by push_cast; linarith⟩
    rw [hf, ite_eq_right (by push_cast; linarith), ite_eq_left (by push_cast; linarith)]; ring

theorem roundEven_mono {m m' : ℚ} (h : m ≤ m') : roundEven m ≤ roundEven m' := by
  have hq : ⌊m⌋ ≤ ⌊m'⌋ := Int.floor_mono h
  have a1 := floor_le_roundEven m'
  have a2 := roundEven_le m
  rcases hq.lt_or_eq with hlt | heq
  · omega
  · unfold roundEven; dsimp only
    rw [← heq]
    split_ifs <;> first | omega | (exfalso; linarith)

/-! ### Rounding a magnitude -/

/-- `rn` on magnitudes. -/
def rp (t : ℚ) : ℚ := (roundEven (t / ulp t) : ℚ) * ulp t

theorem rn_eq (x : ℚ) : rn x = (if x < 0 then -1 else 1) * rp |x| := by
  unfold rn rp; rw [ulp_abs]; ring

theorem rp_zero : rp 0 = 0 := by
  have h : roundEven 0 = 0 := by simpa using roundEven_intCast 0
  simp [rp, h]

theorem rp_nonneg {t : ℚ} (ht : 0 ≤ t) : 0 ≤ rp t := by
  unfold rp
  have : (0 : ℤ) ≤ roundEven (t / ulp t) :=
    le_trans (Int.floor_nonneg.2 (div_nonneg ht (ulp_pos t).le)) (floor_le_roundEven _)
  exact mul_nonneg (by exact_mod_cast this) (ulp_pos t).le

theorem rp_ge {t : ℚ} (ht : 0 < t) : 2 ^ binade t ≤ rp t := by
  have hb := sig_bounds ht.ne'
  rw [abs_of_pos ht] at hb
  have : (2 ^ 23 : ℤ) ≤ roundEven (t / ulp t) :=
    le_trans (Int.le_floor.2 (by exact_mod_cast hb.1)) (floor_le_roundEven _)
  unfold rp
  calc (2 : ℚ) ^ binade t = 2 ^ 23 * ulp t := by rw [ulp, abs_of_pos ht, two_pow_23]
    _ ≤ _ := mul_le_mul_of_nonneg_right (by exact_mod_cast this) (ulp_pos t).le

theorem rp_le {t : ℚ} (ht : 0 < t) : rp t ≤ 2 ^ (binade t + 1) := by
  have hb := sig_bounds ht.ne'
  rw [abs_of_pos ht] at hb
  have hf : ⌊t / ulp t⌋ < 2 ^ 24 := Int.floor_lt.2 (by exact_mod_cast hb.2)
  have : roundEven (t / ulp t) ≤ 2 ^ 24 := by have := roundEven_le (t / ulp t); omega
  unfold rp
  calc _ ≤ (2 : ℚ) ^ 24 * ulp t := mul_le_mul_of_nonneg_right (by exact_mod_cast this) (ulp_pos t).le
    _ = _ := by rw [ulp, abs_of_pos ht, two_pow_24]

theorem rp_mono {s t : ℚ} (hs : 0 ≤ s) (h : s ≤ t) : rp s ≤ rp t := by
  rcases hs.lt_or_eq with hs | hs
  · have ht := lt_of_lt_of_le hs h
    rcases (binade_mono hs h).lt_or_eq with hlt | heq
    · calc rp s ≤ 2 ^ (binade s + 1) := rp_le hs
        _ ≤ 2 ^ binade t := zpow_le_zpow_right₀ one_le_two (by omega)
        _ ≤ rp t := rp_ge ht
    · have hu : ulp s = ulp t := by rw [ulp, ulp, abs_of_pos hs, abs_of_pos ht, heq]
      unfold rp; rw [hu]
      exact mul_le_mul_of_nonneg_right
        (by exact_mod_cast roundEven_mono (div_le_div_of_nonneg_right h (ulp_pos t).le)) (ulp_pos t).le
  · rw [← hs, rp_zero]; exact rp_nonneg (hs ▸ h)

theorem rn_mono : Monotone rn := by
  intro x y h
  rw [rn_eq, rn_eq]
  rcases le_or_gt 0 x with hx | hx
  · have hy := le_trans hx h
    rw [ite_eq_right (not_lt.2 hx), ite_eq_right (not_lt.2 hy), abs_of_nonneg hx, abs_of_nonneg hy, one_mul, one_mul]
    exact rp_mono hx h
  · rw [ite_eq_left hx, abs_of_neg hx]
    rcases le_or_gt 0 y with hy | hy
    · rw [ite_eq_right (not_lt.2 hy)]
      nlinarith [rp_nonneg (neg_nonneg.2 hx.le), rp_nonneg (abs_nonneg y)]
    · rw [ite_eq_left hy, abs_of_neg hy]
      nlinarith [rp_mono (neg_nonneg.2 hy.le) (neg_le_neg h)]

theorem rn_neg (x : ℚ) : rn (-x) = -rn x := by
  rw [rn_eq, rn_eq, abs_neg]
  rcases lt_trichotomy x 0 with hx | hx | hx
  · rw [ite_eq_right (by linarith), ite_eq_left hx]; ring
  · subst hx; simp [rp_zero]
  · rw [ite_eq_left (by linarith), ite_eq_right (by linarith)]; ring

/-- `rp` fixes a magnitude whose significand is an integer. -/
theorem rp_of_int {t : ℚ} {n : ℤ} (h : t / ulp t = n) : rp t = t := by
  unfold rp; rw [h, roundEven_intCast, ← h, div_mul_cancel₀ _ (ulp_pos t).ne']

/-- A representable `x ≠ 0` has an integer significand. -/
theorem sig_int {x : ℚ} (hr : Representable x) (hx : x ≠ 0) : ∃ n : ℤ, |x| / ulp x = n := by
  obtain ⟨m, e, hm, rfl⟩ := hr
  have hx' : 0 < |(m : ℚ) * 2 ^ e| := abs_pos.2 hx
  have habs : |(m : ℚ) * 2 ^ e| = |(m : ℚ)| * 2 ^ e := by rw [abs_mul, abs_of_pos (two_zpow_pos e)]
  have h1 := (binade_spec hx').1
  unfold ulp
  generalize binade |(m : ℚ) * 2 ^ e| = k at h1 ⊢
  have hk : k ≤ e + 23 := by
    have h2 : |(m : ℚ) * 2 ^ e| < 2 ^ (e + 24) := by
      rw [habs, zpow_add₀ two_ne_zero, mul_comm (2 ^ e)]
      exact mul_lt_mul_of_pos_right (by exact_mod_cast hm) (two_zpow_pos e)
    have := (zpow_lt_zpow_iff_right₀ (one_lt_two (α := ℚ))).1 (lt_of_le_of_lt h1 h2)
    omega
  refine ⟨|m| * 2 ^ (e + 23 - k).toNat, ?_⟩
  rw [habs, mul_div_assoc, ← zpow_sub₀ two_ne_zero]
  push_cast
  rw [← zpow_natCast, Int.toNat_of_nonneg (by omega)]
  congr 2; ring

theorem rn_exact {x : ℚ} (hr : Representable x) : rn x = x := by
  rcases eq_or_ne x 0 with hx | hx
  · subst hx; simp [rn_eq, rp_zero]
  obtain ⟨n, hn⟩ := sig_int hr hx
  rw [rn_eq, rp_of_int (by rw [ulp_abs]; exact hn)]
  split_ifs with h
  · rw [abs_of_neg h]; ring
  · rw [abs_of_nonneg (not_lt.1 h)]; ring

theorem representable_neg {x : ℚ} (h : Representable x) : Representable (-x) := by
  obtain ⟨m, e, hm, rfl⟩ := h
  exact ⟨-m, e, by rwa [abs_neg], by push_cast; ring⟩

/-- A positive representable `r` is `n · ulp r` with `2^23 ≤ n < 2^24`. -/
theorem pos_sig {r : ℚ} (hr : Representable r) (h0 : 0 < r) :
    ∃ n : ℤ, r = n * ulp r ∧ 2 ^ 23 ≤ n ∧ n < 2 ^ 24 := by
  obtain ⟨n, hn⟩ := sig_int hr h0.ne'
  have hb := sig_bounds h0.ne'
  rw [hn] at hb
  rw [abs_of_pos h0] at hn
  refine ⟨n, by rw [← hn, div_mul_cancel₀ _ (ulp_pos r).ne'], by exact_mod_cast hb.1, by exact_mod_cast hb.2⟩

theorem absorb_pos {r i : ℚ} (hr : Representable r) (h0 : 0 < r) (hi : 2 * |i| < ulp r)
    (hb : 2 ^ binade r < r ∨ 0 ≤ i) : rn (r + i) = r := by
  obtain ⟨n, hn, n1, n2⟩ := pos_sig hr h0
  have hu := ulp_pos r
  have hk : (2 : ℚ) ^ binade r = 2 ^ 23 * ulp r := by rw [ulp, abs_of_pos h0, two_pow_23]
  have hk1 : (2 : ℚ) ^ (binade r + 1) = 2 ^ 24 * ulp r := by rw [ulp, abs_of_pos h0, two_pow_24]
  obtain ⟨i1, i2⟩ := abs_lt.1 (show |i| < ulp r / 2 by linarith)
  have n2' : (n : ℚ) + 1 ≤ 2 ^ 24 := by exact_mod_cast (show n + 1 ≤ 2 ^ 24 by omega)
  have hlo : 2 ^ binade r ≤ r + i := by
    rcases hb with hb | hb
    · have : (2 : ℚ) ^ 23 + 1 ≤ n := by
        have : (2 : ℚ) ^ 23 < n :=
          lt_of_mul_lt_mul_right (show (2 : ℚ) ^ 23 * ulp r < n * ulp r by rw [← hk, ← hn]; exact hb) hu.le
        exact_mod_cast (show (2 : ℤ) ^ 23 + 1 ≤ n by exact_mod_cast this)
      rw [hk]; nlinarith
    · linarith [(binade_spec h0).1]
  have hhi : r + i < 2 ^ (binade r + 1) := by rw [hk1]; nlinarith
  have hx : 0 < r + i := lt_of_lt_of_le (two_zpow_pos _) hlo
  have hbx : binade (r + i) = binade r := binade_unique hx hlo hhi
  have hux : ulp (r + i) = ulp r := by rw [ulp, ulp, abs_of_pos hx, abs_of_pos h0, hbx]
  rw [rn_eq, ite_eq_right (not_lt.2 hx.le), one_mul, abs_of_pos hx, rp, hux]
  have : roundEven ((r + i) / ulp r) = n := by
    apply roundEven_near
    rw [add_div, (div_eq_iff hu.ne').2 hn, add_sub_cancel_left, abs_div, abs_of_pos hu, div_lt_iff₀ hu]
    linarith
  rw [this, ← hn]

theorem moves_pos {r i : ℚ} (hr : Representable r) (h0 : 0 < r) (hi : ulp r ≤ |i|) : rn (r + i) ≠ r := by
  obtain ⟨n, hn, n1, n2⟩ := pos_sig hr h0
  have hu := ulp_pos r
  rcases le_or_gt 0 i with h | h
  · rw [abs_of_nonneg h] at hi
    have hrep : Representable (r + ulp r) := by
      rcases (show n + 1 < 2 ^ 24 ∨ n + 1 = 2 ^ 24 by omega) with h1 | h1
      · have e : r + ulp r = ((n + 1 : ℤ) : ℚ) * ulp r := by push_cast; linear_combination hn
        exact ⟨n + 1, binade |r| - 23, by rw [abs_of_nonneg (by omega)]; exact h1, by rw [e, ulp]⟩
      · have h1' : (n : ℚ) + 1 = 2 ^ 24 := by exact_mod_cast h1
        have e : r + ulp r = (2 : ℚ) ^ 24 * ulp r := by rw [← h1']; linear_combination hn
        have hk1 : (2 : ℚ) ^ (binade r + 1) = 2 ^ 24 * ulp r := by rw [ulp, abs_of_pos h0, two_pow_24]
        exact ⟨1, binade r + 1, by norm_num, by rw [e, ← hk1]; push_cast; ring⟩
    have := rn_mono (show r + ulp r ≤ r + i by linarith)
    rw [rn_exact hrep] at this
    intro e; linarith
  · rw [abs_of_neg h] at hi
    have e : r - ulp r = ((n - 1 : ℤ) : ℚ) * ulp r := by push_cast; linear_combination hn
    have hrep : Representable (r - ulp r) :=
      ⟨n - 1, binade |r| - 23, by rw [abs_of_nonneg (by omega)]; omega, by rw [e, ulp]⟩
    have := rn_mono (show r + i ≤ r - ulp r by linarith)
    rw [rn_exact hrep] at this
    intro e; linarith

theorem ulp_neg (x : ℚ) : ulp (-x) = ulp x := by simp [ulp]

theorem rnAdd_exact (x y : ℚ) (h : Representable (x + y)) : rnAdd x y = x + y := rn_exact h

theorem rnAdd_mono (x y x' y' : ℚ) (h : x + y ≤ x' + y') : rnAdd x y ≤ rnAdd x' y' := rn_mono h

theorem rnAdd_absorb (r i : ℚ) (hr : Representable r) (h0 : r ≠ 0) (hi : 2 * |i| < ulp r)
    (hb : 2 ^ binade |r| < |r| ∨ 0 ≤ r * i) : rnAdd r i = r := by
  unfold rnAdd
  rcases h0.lt_or_gt with h | h
  · have hb' : 2 ^ binade (-r) < -r ∨ 0 ≤ -i := by
      rw [abs_of_neg h] at hb
      rcases hb with hb | hb
      · exact Or.inl hb
      · exact Or.inr (by nlinarith)
    have := absorb_pos (representable_neg hr) (neg_pos.2 h) (by rwa [ulp_neg, abs_neg]) hb'
    rw [show -r + -i = -(r + i) by ring, rn_neg] at this
    linarith
  · rw [abs_of_pos h] at hb
    exact absorb_pos hr h hi (hb.imp id fun hb => (mul_nonneg_iff_of_pos_left h).1 hb)

theorem rnAdd_moves (r i : ℚ) (hr : Representable r) (h0 : r ≠ 0) (hi : ulp r ≤ |i|) : rnAdd r i ≠ r := by
  unfold rnAdd
  rcases h0.lt_or_gt with h | h
  · have := moves_pos (representable_neg hr) (neg_pos.2 h) (i := -i) (by rwa [ulp_neg, abs_neg])
    rw [show -r + -i = -(r + i) by ring, rn_neg] at this
    intro e; exact this (by rw [e])
  · exact moves_pos hr h hi

theorem term_representable (t : Term) (h : t.mag < 2 ^ 24) : Representable t.value := by
  refine ⟨(if t.neg then -1 else 1) * t.mag, t.exp - 23, ?_, ?_⟩
  · have : ((t.mag : ℤ)) < 2 ^ 24 := by exact_mod_cast h
    split <;> simpa using this
  · unfold Term.value; split <;> push_cast <;> ring

/-- Every finite FP32 word's value is representable. -/
theorem wordQ_representable {w : Nat} {x : ℚ} (h : wordQ w = some x) : Representable x := by
  unfold wordQ fp32Value at h
  dsimp only at h
  split_ifs at h with h1 h2 h3 h4 <;> simp only [Value.finite?, Option.map_some, Option.map_none,
    reduceCtorEq, Option.some.injEq] at h <;> subst h <;> apply term_representable
  · simp [Term.zero]
  · exact lt_trans (Nat.mod_lt _ (by norm_num)) (by norm_num)
  · exact Nat.or_lt_two_pow (by decide) (lt_trans (Nat.mod_lt _ (by norm_num)) (by norm_num))


/-! ## The pinned statements -/

theorem rnAddExact : RnAddExact := rnAdd_exact

theorem rnAddMono : RnAddMono := rnAdd_mono

theorem rnAddAbsorb : RnAddAbsorb := rnAdd_absorb

theorem rnAddMoves : RnAddMoves := rnAdd_moves

theorem wordsRepresentable : WordsRepresentable := fun _ _ => wordQ_representable

end Pouw.Fp8Atom.Proofs
