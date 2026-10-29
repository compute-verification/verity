import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Basic.Real.Basic
import Mathlib.Data.Finset.Max
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# The mass floor: an integer value with no likely point has a wide spread

A distribution on the integers, given by weights `m` on a finite support `F`, whose every point mass is at most
`p` has second moment at least `(1/p² − 1)/12` about every real centre `c`, so its variance is at least that
(`mass_floor`). The uniform distribution on `1/p` consecutive integers attains it.

The proof needs no integral. `cubic L u` is the antiderivative of `(L² − u²)⁺` with `L = 1/(2p)`, written as
`L²v − v³/3` at the clamp `v` of `u` to `[−L, L]`. Its rise over the unit cell around an integer is at least
`L² − 1/12 − (z − c)²` (`cubic_cell`), the rises telescope over any finite set of integers to at most `4L³/3`
(`sum_cell_le`), and moving the weight onto the points where `(z − c)² + 1/12 < L²` gives the floor.

Per step, the floor reads `1 ≤ p²(12V + 1)`. Its mean over the steps holds at the means (`mass_floor_mean`):
`1/x²` is convex, and summing its tangent line at the mean mass gives `1 ≤ p̄²(12V̄ + 1)`.
-/

namespace Pouw.Dimension.MassFloor

open Finset

/-- `L²u − u³/3`, whose derivative `L² − u²` is nonnegative on `[−L, L]`. -/
noncomputable def poly (L u : ℝ) : ℝ := L ^ 2 * u - u ^ 3 / 3

/-- `poly` at the clamp of `u` to `[−L, L]`: the antiderivative of `(L² − u²)⁺`. -/
noncomputable def cubic (L u : ℝ) : ℝ := poly L (max (-L) (min u L))

lemma clamp_mem {L : ℝ} (hL : 0 ≤ L) (u : ℝ) :
    -L ≤ max (-L) (min u L) ∧ max (-L) (min u L) ≤ L :=
  ⟨le_max_left _ _, max_le (by linarith) (min_le_right _ _)⟩

lemma poly_mono {L v w : ℝ} (hv : -L ≤ v) (hw : w ≤ L) (hvw : v ≤ w) : poly L v ≤ poly L w := by
  unfold poly
  have h1 : 0 ≤ (w - v) * (3 * L ^ 2 - (w ^ 2 + w * v + v ^ 2)) := by
    apply mul_nonneg (by linarith)
    nlinarith [mul_nonneg (by linarith : (0:ℝ) ≤ L - w) (by linarith : (0:ℝ) ≤ L + v),
      mul_nonneg (by linarith : (0:ℝ) ≤ L - v) (by linarith : (0:ℝ) ≤ L + w),
      mul_nonneg (by linarith : (0:ℝ) ≤ L - w) (by linarith : (0:ℝ) ≤ L + w),
      mul_nonneg (by linarith : (0:ℝ) ≤ L - v) (by linarith : (0:ℝ) ≤ L + v)]
  nlinarith [h1]

lemma cubic_mono {L : ℝ} (hL : 0 ≤ L) : Monotone (cubic L) := by
  intro u u' h
  unfold cubic
  have hc : max (-L) (min u L) ≤ max (-L) (min u' L) :=
    max_le_max_left _ (min_le_min_right _ h)
  exact poly_mono (clamp_mem hL u).1 (clamp_mem hL u').2 hc

lemma cubic_le {L : ℝ} (hL : 0 ≤ L) (u : ℝ) : cubic L u ≤ 2 * L ^ 3 / 3 := by
  unfold cubic poly
  obtain ⟨h1, h2⟩ := clamp_mem hL u
  set v := max (-L) (min u L)
  nlinarith [mul_nonneg (sq_nonneg (L - v)) (by linarith : (0:ℝ) ≤ 2 * L + v)]

lemma neg_le_cubic {L : ℝ} (hL : 0 ≤ L) (u : ℝ) : -(2 * L ^ 3 / 3) ≤ cubic L u := by
  unfold cubic poly
  obtain ⟨h1, h2⟩ := clamp_mem hL u
  set v := max (-L) (min u L)
  nlinarith [mul_nonneg (sq_nonneg (L + v)) (by linarith : (0:ℝ) ≤ 2 * L - v)]

/-- Above `−L` the clamp only raises `poly`. -/
lemma poly_le_cubic {L b : ℝ} (hL : 0 ≤ L) (hb : -L ≤ b) : poly L b ≤ cubic L b := by
  unfold cubic
  rcases le_or_gt b L with h | h
  · rw [min_eq_left h, max_eq_right hb]
  · rw [min_eq_right h.le, max_eq_right (by linarith)]
    unfold poly
    nlinarith [mul_nonneg (sq_nonneg (b - L)) (by linarith : (0:ℝ) ≤ 2 * L + b)]

/-- Below `L` the clamp only lowers `poly`. -/
lemma cubic_le_poly {L a : ℝ} (hL : 0 ≤ L) (ha : a ≤ L) : cubic L a ≤ poly L a := by
  unfold cubic
  rw [min_eq_left ha]
  rcases le_or_gt (-L) a with h | h
  · rw [max_eq_right h]
  · rw [max_eq_left h.le]
    unfold poly
    nlinarith [mul_nonneg (sq_nonneg (a + L)) (by linarith : (0:ℝ) ≤ 2 * L - a)]

/-- The rise of `cubic` over the unit cell around `y` is at least `L² − 1/12 − y²`, with equality inside. -/
lemma cubic_cell {L : ℝ} (hL : 0 ≤ L) (y : ℝ) :
    L ^ 2 - 1 / 12 - y ^ 2 ≤ cubic L (y + 1 / 2) - cubic L (y - 1 / 2) := by
  have hpoly : poly L (y + 1 / 2) - poly L (y - 1 / 2) = L ^ 2 - 1 / 12 - y ^ 2 := by
    unfold poly; ring
  rcases lt_or_ge (y + 1 / 2) (-L) with hb | hb
  · have e : cubic L (y + 1 / 2) = cubic L (y - 1 / 2) := by
      unfold cubic
      rw [min_eq_left (by linarith), min_eq_left (by linarith), max_eq_left (by linarith),
        max_eq_left (by linarith)]
    rw [e]; nlinarith
  rcases lt_or_ge L (y - 1 / 2) with ha | ha
  · have e : cubic L (y + 1 / 2) = cubic L (y - 1 / 2) := by
      unfold cubic
      rw [min_eq_right (by linarith), min_eq_right (by linarith)]
    rw [e]; nlinarith
  linarith [poly_le_cubic hL hb, cubic_le_poly hL ha]

/-- The rises over the cells around `z − c`, for `z` in any finite set of integers whose cells end by `t`,
sum to at most `cubic L t + 2L³/3`. -/
lemma sum_cell_le_of {L : ℝ} (hL : 0 ≤ L) (c : ℝ) (s : Finset ℤ) :
    ∀ t : ℝ, (∀ x ∈ s, (x : ℝ) - c + 1 / 2 ≤ t) →
      ∑ z ∈ s, (cubic L ((z : ℝ) - c + 1 / 2) - cubic L ((z : ℝ) - c - 1 / 2))
        ≤ cubic L t + 2 * L ^ 3 / 3 := by
  induction s using Finset.induction_on_max with
  | empty => intro t _; simp; linarith [neg_le_cubic hL t]
  | insert a s hlt ih =>
    intro t ht
    have ha : a ∉ s := fun h => lt_irrefl a (hlt a h)
    rw [sum_insert ha]
    have hs : ∀ x ∈ s, (x : ℝ) - c + 1 / 2 ≤ (a : ℝ) - c - 1 / 2 := by
      intro x hx
      have : (x : ℝ) + 1 ≤ a := by exact_mod_cast hlt x hx
      linarith
    have h1 := ih _ hs
    have h2 := cubic_mono hL (ht a (mem_insert_self a s))
    linarith

/-- Over any finite set of integers the rises sum to at most `4L³/3`, the integral of `(L² − u²)⁺`. -/
lemma sum_cell_le {L : ℝ} (hL : 0 ≤ L) (c : ℝ) (s : Finset ℤ) :
    ∑ z ∈ s, (cubic L ((z : ℝ) - c + 1 / 2) - cubic L ((z : ℝ) - c - 1 / 2)) ≤ 4 * L ^ 3 / 3 := by
  have ht : ∀ x ∈ s, (x : ℝ) - c + 1 / 2 ≤ ∑ y ∈ s, |(y : ℝ) - c + 1 / 2| := fun x hx =>
    (le_abs_self _).trans
      (single_le_sum (f := fun y : ℤ => |(y : ℝ) - c + 1 / 2|) (fun _ _ => abs_nonneg _) hx)
  linarith [sum_cell_le_of hL c s _ ht, cubic_le hL (∑ y ∈ s, |(y : ℝ) - c + 1 / 2|)]

/-- **The mass floor.** Weights `m` on a finite set of integers, summing to one, each at most `p`, have second
moment at least `(1/p² − 1)/12` about every real `c`; at `c` the mean, that is the variance. -/
theorem mass_floor {F : Finset ℤ} {m : ℤ → ℝ} {p : ℝ} (c : ℝ) (hp : 0 < p)
    (hm0 : ∀ z ∈ F, 0 ≤ m z) (hmp : ∀ z ∈ F, m z ≤ p) (h1 : ∑ z ∈ F, m z = 1) :
    (1 / p ^ 2 - 1) / 12 ≤ ∑ z ∈ F, m z * ((z : ℝ) - c) ^ 2 := by
  set L := 1 / (2 * p) with hLdef
  have hL : 0 ≤ L := by positivity
  set d : ℤ → ℝ := fun z => cubic L ((z : ℝ) - c + 1 / 2) - cubic L ((z : ℝ) - c - 1 / 2) with hd
  have hpt : ∀ z ∈ F, -(p * d z) ≤ m z * (((z : ℝ) - c) ^ 2 + 1 / 12 - L ^ 2) := by
    intro z hz
    have hd0 : 0 ≤ d z := sub_nonneg.mpr (cubic_mono hL (by linarith))
    have hdc : L ^ 2 - 1 / 12 - ((z : ℝ) - c) ^ 2 ≤ d z := by
      have := cubic_cell hL ((z : ℝ) - c)
      simpa [hd] using this
    have h0 := hm0 z hz
    have hmp' := hmp z hz
    rcases le_or_gt 0 (((z : ℝ) - c) ^ 2 + 1 / 12 - L ^ 2) with h | h
    · nlinarith [mul_nonneg h0 h, mul_nonneg hp.le hd0]
    · nlinarith [mul_nonneg (sub_nonneg.mpr hmp') (neg_nonneg.mpr h.le), mul_nonneg hp.le (by linarith : (0:ℝ) ≤ d z + (((z : ℝ) - c) ^ 2 + 1 / 12 - L ^ 2))]
  have hsum := sum_le_sum hpt
  rw [sum_neg_distrib, ← mul_sum] at hsum
  have hcell := sum_cell_le hL c F
  have hexp : ∑ z ∈ F, m z * (((z : ℝ) - c) ^ 2 + 1 / 12 - L ^ 2)
      = ∑ z ∈ F, m z * ((z : ℝ) - c) ^ 2 + (1 / 12 - L ^ 2) * ∑ z ∈ F, m z := by
    rw [mul_sum, ← sum_add_distrib]; exact sum_congr rfl fun z _ => by ring
  rw [hexp, h1] at hsum
  have hkey : (1 / p ^ 2 - 1) / 12 = L ^ 2 - 1 / 12 - p * (4 * L ^ 3 / 3) := by
    rw [hLdef]; field_simp; ring
  rw [hkey]
  nlinarith [mul_le_mul_of_nonneg_left hcell hp.le]

/-- Per step the floor reads `1 ≤ q²(12V + 1)`: at the tangent of `1/x²` through any `Q > 0`, a step's term
`Q³(12V + 1)` is at least `3Q − 2q`. -/
lemma step_tangent {q Q V : ℝ} (hq : 0 < q) (hQ : 0 < Q) (h : 1 ≤ q ^ 2 * (12 * V + 1)) :
    3 * Q - 2 * q ≤ Q ^ 3 * (12 * V + 1) := by
  have h1 : Q ^ 3 ≤ Q ^ 3 * (q ^ 2 * (12 * V + 1)) := by
    nlinarith [pow_pos hQ 3]
  have h2 : 0 ≤ (Q - q) ^ 2 * (Q + 2 * q) := by positivity
  have h3 : 0 ≤ q ^ 2 * (Q ^ 3 * (12 * V + 1) - 3 * Q + 2 * q) := by nlinarith
  have hq2 : 0 < q ^ 2 := by positivity
  nlinarith [(mul_nonneg_iff_of_pos_left hq2).mp h3]

/-- **The floor at the means.** If every step `i` of `S` has `1 ≤ q_i²(12V_i + 1)` (the mass floor, with `q_i`
its largest point mass and `V_i` its variance), the mean mass and the mean variance satisfy it too:
`1 ≤ q̄²(12V̄ + 1)`, that is `q̄ ≥ (12V̄ + 1)^(−1/2)`. -/
theorem mass_floor_mean {ι : Type*} {s : Finset ι} (hs : s.Nonempty) {q V : ι → ℝ}
    (hq : ∀ i ∈ s, 0 < q i) (h : ∀ i ∈ s, 1 ≤ q i ^ 2 * (12 * V i + 1)) :
    1 ≤ ((∑ i ∈ s, q i) / s.card) ^ 2 * (12 * ((∑ i ∈ s, V i) / s.card) + 1) := by
  have hS : (0 : ℝ) < s.card := by exact_mod_cast hs.card_pos
  have hsum : 0 < ∑ i ∈ s, q i := sum_pos hq hs
  set Q := (∑ i ∈ s, q i) / s.card with hQdef
  have hQ : 0 < Q := div_pos hsum hS
  have ht := sum_le_sum fun i hi => step_tangent (hq i hi) hQ (h i hi)
  have hl : ∑ i ∈ s, (3 * Q - 2 * q i) = 3 * Q * s.card - 2 * ∑ i ∈ s, q i := by
    rw [sum_sub_distrib, sum_const, nsmul_eq_mul, ← mul_sum]; ring
  have hr : ∑ i ∈ s, Q ^ 3 * (12 * V i + 1) = Q ^ 3 * (12 * ∑ i ∈ s, V i + s.card) := by
    rw [← mul_sum, sum_add_distrib, ← mul_sum, sum_const, nsmul_eq_mul, mul_one]
  rw [hl, hr] at ht
  have hq_sum : ∑ i ∈ s, q i = Q * s.card := by rw [hQdef, div_mul_cancel₀ _ hS.ne']
  set W := (∑ i ∈ s, V i) / s.card with hWdef
  have hV_sum : ∑ i ∈ s, V i = W * s.card := by rw [hWdef, div_mul_cancel₀ _ hS.ne']
  rw [hq_sum, hV_sum] at ht
  have h4 : Q * s.card ≤ Q ^ 3 * (12 * (W * s.card) + s.card) := by linarith
  have h5 : Q * s.card * 1 ≤ Q * s.card * (Q ^ 2 * (12 * W + 1)) := by nlinarith
  exact le_of_mul_le_mul_left h5 (mul_pos hQ hS)

end Pouw.Dimension.MassFloor
