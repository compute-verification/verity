module

public import Properties.Spot.Interval
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# The interval bounds are sound

`Properties.Spot.ExpLowerSound`, `ExpUpperSound`, `LnLowerSound`, `LnUpperSound`, `PowLowerSound`,
`PowUpperSound`, `BoundLnUpperSound` and `BoundLnBoundSound`.

Every rounding is to the bound's side (`floorTo_le`, `le_ceilTo`). The exponential's series stays below
`Σ rⁱ/i!` term by term (`series_false`), so below `e^r` (`Real.sum_le_exp_of_nonneg`), and above it plus
`Real.exp_bound'`'s remainder (`series_true`); rounded squarings keep each side (`squares_false`, `squares_true`), and a
negative argument goes through `e^x = 1/e^{-x}`. A logarithm is whatever candidate its exponential bounds confirm
(`firstAbove_some`, `firstBelow_some`), else `ln y ≤ y − 1` or `1 − 1/y ≤ ln y`, at `|y|` since `ln y = ln |y|`
(`abs_arg`). A power is `e^{θ ln c}`, `Real.rpow`'s value at `c > 0`, and `±e^{θ ln c}` brackets its
`e^{θ ln c} cos(πθ)` at `c < 0`. `Bound`'s series is the partial sum of `ln((1+t)/(1−t)) = 2 Σ t^{2k+1}/(2k+1)`
(`Real.hasSum_log_sub_log_of_abs_lt_one`), below it with every term nonnegative and above it with the tail bounded by
a geometric series (`atanh_partial_le`, `atanh_le_partial`), after scaling by the `2^j` both helpers pick
(`scale_le`), on their ranges (`boundLnUpper_le`, `lnBound_brackets`).

Each bound passes at a deployed precision, one step of `2^-p` wide or less (`_honest`), and each conclusion fails at a
bound on the wrong side (`_nontrivial`).
-/

@[expose] public section

namespace Properties.Spot.SecurityProofs

open Finset

section Interval

open _root_.Spot.Interval

/-! ## Rounding -/

theorem floorTo_le (q : ℚ) (p : ℕ) : floorTo q p ≤ q := by
  unfold floorTo
  rw [div_le_iff₀ (by positivity)]
  exact Rat.floor_le _

theorem le_ceilTo (q : ℚ) (p : ℕ) : q ≤ ceilTo q p := by
  unfold ceilTo
  rw [le_div_iff₀ (by positivity)]
  exact Rat.le_ceil

theorem floorTo_nonneg {q : ℚ} (hq : 0 ≤ q) (p : ℕ) : 0 ≤ floorTo q p := by
  unfold floorTo
  have : (0 : ℤ) ≤ (q * 2 ^ p).floor := Rat.le_floor_iff.2 (by push_cast; positivity)
  have : (0 : ℚ) ≤ ((q * 2 ^ p).floor : ℚ) := by exact_mod_cast this
  positivity

theorem roundTo_false (q : ℚ) (p : ℕ) : roundTo false q p ≤ q := floorTo_le q p

theorem roundTo_true (q : ℚ) (p : ℕ) : q ≤ roundTo true q p := le_ceilTo q p

/-! ## The exponential -/

/-- The series' terms `r^i / i!`, in `ℚ`. -/
theorem term_succ (r : ℚ) (n : ℕ) :
    r ^ n / (n.factorial : ℚ) * r / ((n + 1 : ℕ) : ℚ) = r ^ (n + 1) / ((n + 1).factorial : ℚ) := by
  rw [Nat.factorial_succ]
  push_cast
  have : (n.factorial : ℚ) ≠ 0 := by exact_mod_cast Nat.factorial_ne_zero n
  field_simp
  ring

theorem series_false {r : ℚ} (hr : 0 ≤ r) (w : ℕ) : ∀ n,
    (series false r w n).1 ≤ ∑ i ∈ range n, r ^ i / (i.factorial : ℚ) ∧
      (series false r w n).2 ≤ r ^ n / (n.factorial : ℚ) ∧ 0 ≤ (series false r w n).1 ∧ 0 ≤ (series false r w n).2
  | 0 => by simp [series]
  | n + 1 => by
    obtain ⟨h1, h2, h3, h4⟩ := series_false hr w n
    simp only [series, sum_range_succ]
    refine ⟨by linarith, ?_, by linarith, floorTo_nonneg (by positivity) w⟩
    calc roundTo false ((series false r w n).2 * r / ((n + 1 : ℕ) : ℚ)) w
        ≤ (series false r w n).2 * r / ((n + 1 : ℕ) : ℚ) := roundTo_false _ _
      _ ≤ r ^ n / (n.factorial : ℚ) * r / ((n + 1 : ℕ) : ℚ) := by gcongr
      _ = _ := term_succ r n

theorem series_true {r : ℚ} (hr : 0 ≤ r) (w : ℕ) : ∀ n,
    ∑ i ∈ range n, r ^ i / (i.factorial : ℚ) ≤ (series true r w n).1 ∧
      r ^ n / (n.factorial : ℚ) ≤ (series true r w n).2
  | 0 => by simp [series]
  | n + 1 => by
    obtain ⟨h1, h2⟩ := series_true hr w n
    simp only [series, sum_range_succ]
    refine ⟨by linarith, ?_⟩
    calc r ^ (n + 1) / ((n + 1).factorial : ℚ) = r ^ n / (n.factorial : ℚ) * r / ((n + 1 : ℕ) : ℚ) :=
          (term_succ r n).symm
      _ ≤ (series true r w n).2 * r / ((n + 1 : ℕ) : ℚ) := by gcongr
      _ ≤ _ := roundTo_true _ _

theorem squares_false (w : ℕ) : ∀ (k : ℕ) (v : ℚ) (a : ℝ), 0 ≤ v → (v : ℝ) ≤ Real.exp a →
    ((squares false w k v : ℚ) : ℝ) ≤ Real.exp (a * 2 ^ k)
  | 0, v, a, _, h => by simpa [squares] using h
  | k + 1, v, a, h0, h => by
    simp only [squares]
    have hv : ((roundTo false (v * v) w : ℚ) : ℝ) ≤ Real.exp (2 * a) := by
      calc ((roundTo false (v * v) w : ℚ) : ℝ) ≤ ((v * v : ℚ) : ℝ) := by exact_mod_cast roundTo_false _ _
        _ = (v : ℝ) * v := by push_cast; ring
        _ ≤ Real.exp a * Real.exp a := mul_le_mul h h (by exact_mod_cast h0) (Real.exp_pos a).le
        _ = Real.exp (2 * a) := by rw [← Real.exp_add]; ring_nf
    have := squares_false w k _ (2 * a) (floorTo_nonneg (mul_nonneg h0 h0) w) hv
    rwa [show 2 * a * 2 ^ k = a * 2 ^ (k + 1) by ring] at this

theorem squares_true (w : ℕ) : ∀ (k : ℕ) (v : ℚ) (a : ℝ), Real.exp a ≤ (v : ℝ) →
    Real.exp (a * 2 ^ k) ≤ ((squares true w k v : ℚ) : ℝ)
  | 0, v, a, h => by simpa [squares] using h
  | k + 1, v, a, h => by
    simp only [squares]
    have hv : Real.exp (2 * a) ≤ ((roundTo true (v * v) w : ℚ) : ℝ) := by
      calc Real.exp (2 * a) = Real.exp a * Real.exp a := by rw [← Real.exp_add]; ring_nf
        _ ≤ (v : ℝ) * v := mul_le_mul h h (Real.exp_pos a).le ((Real.exp_pos a).le.trans h)
        _ = ((v * v : ℚ) : ℝ) := by push_cast; ring
        _ ≤ _ := by exact_mod_cast roundTo_true _ _
    have := squares_true w k _ (2 * a) hv
    rwa [show 2 * a * 2 ^ k = a * 2 ^ (k + 1) by ring] at this

theorem le_two_pow_halvings {x : ℚ} (hx : 0 ≤ x) : x ≤ 2 ^ halvings x := by
  have hc : (0 : ℤ) ≤ x.ceil := by
    have : (0 : ℚ) ≤ (x.ceil : ℚ) := hx.trans Rat.le_ceil
    exact_mod_cast this
  calc x ≤ (x.ceil : ℚ) := Rat.le_ceil
    _ = ((x.ceil.toNat : ℕ) : ℚ) := by rw [show (x.ceil.toNat : ℚ) = ((x.ceil.toNat : ℤ) : ℚ) from rfl,
          Int.toNat_of_nonneg hc]
    _ ≤ ((2 ^ halvings x : ℕ) : ℚ) := by
        have : x.ceil.toNat < 2 ^ (x.ceil.toNat.log2 + 1) := Nat.lt_log2_self
        have h2 : 2 ^ (x.ceil.toNat.log2 + 1) ≤ 2 ^ halvings x :=
          Nat.pow_le_pow_right (by norm_num) (by unfold halvings; omega)
        exact_mod_cast (this.trans_le h2).le
    _ = 2 ^ halvings x := by push_cast; rfl

theorem expNonneg_false {x : ℚ} (hx : 0 ≤ x) (w : ℕ) : ((expNonneg false x w : ℚ) : ℝ) ≤ Real.exp x := by
  unfold expNonneg
  simp only [Bool.false_eq_true, ↓reduceIte]
  set s := halvings x
  set r := x / 2 ^ s with hr
  have hr0 : 0 ≤ r := by positivity
  obtain ⟨h1, -, h3, -⟩ := series_false hr0 w (expTerms w)
  have hS : (((series false r w (expTerms w)).1 : ℚ) : ℝ) ≤ Real.exp r := by
    calc (((series false r w (expTerms w)).1 : ℚ) : ℝ)
        ≤ ((∑ i ∈ range (expTerms w), r ^ i / (i.factorial : ℚ) : ℚ) : ℝ) := by exact_mod_cast h1
      _ = ∑ i ∈ range (expTerms w), (r : ℝ) ^ i / (i.factorial : ℝ) := by push_cast; rfl
      _ ≤ Real.exp r := Real.sum_le_exp_of_nonneg (by exact_mod_cast hr0) _
  have := squares_false w s _ r h3 hS
  have hx' : (r : ℝ) * 2 ^ s = x := by
    rw [hr]; push_cast; field_simp
  rwa [hx'] at this

theorem expNonneg_true {x : ℚ} (hx : 0 ≤ x) (w : ℕ) : Real.exp x ≤ ((expNonneg true x w : ℚ) : ℝ) := by
  unfold expNonneg
  simp only [↓reduceIte]
  set s := halvings x
  set r := x / 2 ^ s with hr
  set n := expTerms w
  have hr0 : 0 ≤ r := by positivity
  have hr1 : r ≤ 1 := by
    rw [hr, div_le_one (by positivity)]; exact le_two_pow_halvings hx
  have hn : 0 < n := by show 0 < w / 10 + 4; omega
  obtain ⟨h1, h2⟩ := series_true hr0 w n
  have hS : Real.exp r ≤ (((series true r w n).1 + (series true r w n).2 * ((n + 1 : ℕ) : ℚ) / (n : ℚ) : ℚ) : ℝ) := by
    have hb := Real.exp_bound' (x := (r : ℝ)) (by exact_mod_cast hr0) (by exact_mod_cast hr1) hn
    have hq : ∑ i ∈ range n, r ^ i / (i.factorial : ℚ) + r ^ n / (n.factorial : ℚ) * ((n + 1 : ℕ) : ℚ) / (n : ℚ) ≤
        (series true r w n).1 + (series true r w n).2 * ((n + 1 : ℕ) : ℚ) / (n : ℚ) := by
      gcongr
    have hR : ((∑ i ∈ range n, r ^ i / (i.factorial : ℚ) + r ^ n / (n.factorial : ℚ) * ((n + 1 : ℕ) : ℚ) / (n : ℚ) :
        ℚ) : ℝ) =
        ∑ m ∈ range n, (r : ℝ) ^ m / (m.factorial : ℝ) + (r : ℝ) ^ n * (n + 1) / (n.factorial * n) := by
      push_cast; ring
    calc Real.exp r ≤ _ := hb
      _ = _ := hR.symm
      _ ≤ _ := by exact_mod_cast hq
  have := squares_true w s _ r hS
  have hx' : (r : ℝ) * 2 ^ s = x := by
    rw [hr]; push_cast; field_simp
  rwa [hx'] at this

theorem expUpperSound : ExpUpperSound := by
  intro x p
  unfold expUpper
  simp only
  split_ifs with hx
  · exact (expNonneg_true hx _).trans (by exact_mod_cast le_ceilTo _ _)
  · have hx' : 0 ≤ -x := by linarith
    have hL := expNonneg_false hx' (expPrec x p)
    set L := expNonneg false (-x) (expPrec x p)
    have hm : (max 1 (L : ℝ)) ≤ Real.exp (-x : ℚ) :=
      max_le (by rw [Real.one_le_exp_iff]; exact_mod_cast hx') hL
    have h1 : Real.exp x ≤ ((1 / max 1 L : ℚ) : ℝ) := by
      push_cast
      rw [le_div_iff₀ (lt_of_lt_of_le one_pos (le_max_left _ _))]
      calc Real.exp x * max 1 (L : ℝ) ≤ Real.exp x * Real.exp (-x : ℚ) :=
            mul_le_mul_of_nonneg_left hm (Real.exp_pos _).le
        _ = 1 := by push_cast; rw [← Real.exp_add]; simp
    exact h1.trans (by exact_mod_cast le_ceilTo _ _)

theorem expLowerSound : ExpLowerSound := by
  intro x p
  unfold expLower
  simp only
  split_ifs with hx
  · exact (by exact_mod_cast floorTo_le _ _ : ((floorTo (expNonneg false x (expPrec x p)) p : ℚ) : ℝ) ≤ _).trans
      (expNonneg_false hx _)
  · have hx' : 0 ≤ -x := by linarith
    have hU := expNonneg_true hx' (expPrec x p)
    set U := expNonneg true (-x) (expPrec x p)
    have hU0 : (0 : ℝ) < U := (Real.exp_pos _).trans_le hU
    have h1 : ((1 / U : ℚ) : ℝ) ≤ Real.exp x := by
      push_cast
      rw [div_le_iff₀ hU0]
      calc (1 : ℝ) = Real.exp x * Real.exp (-x : ℚ) := by push_cast; rw [← Real.exp_add]; simp
        _ ≤ Real.exp x * U := mul_le_mul_of_nonneg_left hU (Real.exp_pos _).le
    exact (by exact_mod_cast floorTo_le _ _ : ((floorTo (1 / U) p : ℚ) : ℝ) ≤ _).trans h1

/-! ## The logarithm -/

theorem firstAbove_some (y : ℚ) (p w : ℕ) : ∀ (k : ℕ) (u v : ℚ), firstAbove y p w k u = some v →
    y ≤ expLower v w
  | 0, _, _, h => by simp [firstAbove] at h
  | k + 1, u, v, h => by
    simp only [firstAbove] at h
    split_ifs at h with hy
    · obtain rfl := Option.some.inj h; exact hy
    · exact firstAbove_some y p w k _ v h

theorem firstBelow_some (y : ℚ) (p w : ℕ) : ∀ (k : ℕ) (l v : ℚ), firstBelow y p w k l = some v →
    expUpper v w ≤ y
  | 0, _, _, h => by simp [firstBelow] at h
  | k + 1, l, v, h => by
    simp only [firstBelow] at h
    split_ifs at h with hy
    · obtain rfl := Option.some.inj h; exact hy
    · exact firstBelow_some y p w k _ v h

/-- `lnUpper`'s branch at `y > 0`. -/
theorem log_le_lnUpperPos {y : ℚ} (hy : 0 < y) (p : ℕ) :
    Real.log y ≤ (((firstAbove y p (lnPrec y p) lnTries (ceilTo (lnApprox y (lnPrec y p)) p)).getD
      (ceilTo (y - 1) p) : ℚ) : ℝ) := by
  have hy' : (0 : ℝ) < y := by exact_mod_cast hy
  cases h : firstAbove y p (lnPrec y p) lnTries (ceilTo (lnApprox y (lnPrec y p)) p) with
  | none =>
    simp only [Option.getD_none]
    calc Real.log y ≤ (y : ℝ) - 1 := Real.log_le_sub_one_of_pos hy'
      _ = ((y - 1 : ℚ) : ℝ) := by push_cast; rfl
      _ ≤ _ := by exact_mod_cast le_ceilTo _ _
  | some v =>
    simp only [Option.getD_some]
    rw [Real.log_le_iff_le_exp hy']
    have h1 : (y : ℝ) ≤ ((expLower v (lnPrec y p) : ℚ) : ℝ) := by exact_mod_cast firstAbove_some y p _ _ _ _ h
    exact h1.trans (expLowerSound v _)

/-- `lnLower`'s branch at `y > 0`. -/
theorem lnLowerPos_le_log {y : ℚ} (hy : 0 < y) (p : ℕ) :
    (((firstBelow y p (lnPrec y p) lnTries (floorTo (lnApprox y (lnPrec y p)) p)).getD
      (floorTo (1 - 1 / y) p) : ℚ) : ℝ) ≤ Real.log y := by
  have hy' : (0 : ℝ) < y := by exact_mod_cast hy
  cases h : firstBelow y p (lnPrec y p) lnTries (floorTo (lnApprox y (lnPrec y p)) p) with
  | none =>
    simp only [Option.getD_none]
    calc (((floorTo (1 - 1 / y) p : ℚ)) : ℝ) ≤ ((1 - 1 / y : ℚ) : ℝ) := by exact_mod_cast floorTo_le _ _
      _ = 1 - (y : ℝ)⁻¹ := by push_cast; ring
      _ ≤ Real.log y := Real.one_sub_inv_le_log_of_pos hy'
  | some v =>
    simp only [Option.getD_some]
    rw [Real.le_log_iff_exp_le hy']
    have h1 : ((expUpper v (lnPrec y p) : ℚ) : ℝ) ≤ y := by exact_mod_cast firstBelow_some y p _ _ _ _ h
    exact (expUpperSound v _).trans h1

/-- The magnitude `lnUpper`, `lnLower` and the powers read at `y ≠ 0`: positive, with `ln y = ln |y|`. -/
theorem abs_arg {y : ℚ} (h0 : y ≠ 0) :
    0 < (if 0 < y then y else -y) ∧ Real.log (y : ℝ) = Real.log (((if 0 < y then y else -y : ℚ)) : ℝ) := by
  split_ifs with h
  · exact ⟨h, rfl⟩
  · have hn : y < 0 := lt_of_le_of_ne (not_lt.1 h) h0
    refine ⟨neg_pos.2 hn, ?_⟩
    push_cast
    exact (Real.log_neg_eq_log _).symm

theorem lnUpperSound : LnUpperSound := by
  intro y p
  unfold lnUpper
  by_cases h0 : y = 0
  · subst h0; simp
  rw [ite_eq_right h0]
  obtain ⟨ha, hlog⟩ := abs_arg h0
  dsimp only
  rw [hlog]
  exact log_le_lnUpperPos ha p

theorem lnLowerSound : LnLowerSound := by
  intro y p
  unfold lnLower
  by_cases h0 : y = 0
  · subst h0; simp
  rw [ite_eq_right h0]
  obtain ⟨ha, hlog⟩ := abs_arg h0
  dsimp only
  rw [hlog]
  exact lnLowerPos_le_log ha p

/-! ## Powers -/

/-- `powUpper c θ p` is at least `|c|^θ = e^{θ ln c}` at `θ ≠ 0` and `c ≠ 0`. -/
theorem exp_le_powUpper {c θ : ℚ} (hθ : θ ≠ 0) (hc : c ≠ 0) (p : ℕ) :
    Real.exp (Real.log c * θ) ≤ ((powUpper c θ p : ℚ) : ℝ) := by
  unfold powUpper
  rw [ite_eq_right hθ, ite_eq_right hc]
  obtain ⟨ha, hlog⟩ := abs_arg hc
  dsimp only
  rw [hlog]
  generalize (if 0 < c then c else -c) = a at ha ⊢
  refine le_trans ?_ (expUpperSound _ p)
  apply Real.exp_le_exp.2
  split_ifs with hθ0
  · have := lnUpperSound a (powPrec a θ p)
    have hθ' : (0 : ℝ) ≤ θ := by exact_mod_cast hθ0
    push_cast
    nlinarith
  · have := lnLowerSound a (powPrec a θ p)
    have hθ' : (θ : ℝ) ≤ 0 := by exact_mod_cast (lt_of_not_ge hθ0).le
    push_cast
    nlinarith

/-- `powLower c θ p` is at most `c^θ = e^{θ ln c}` at `θ ≠ 0` and `c > 0`. -/
theorem powLowerPos_le {c θ : ℚ} (hθ : θ ≠ 0) (hc : 0 < c) (p : ℕ) :
    ((powLower c θ p : ℚ) : ℝ) ≤ Real.exp (Real.log c * θ) := by
  unfold powLower
  rw [ite_eq_right hθ, ite_eq_right hc.ne', ite_eq_right (not_lt.2 hc.le)]
  dsimp only
  refine le_trans (expLowerSound _ p) ?_
  apply Real.exp_le_exp.2
  split_ifs with hθ0
  · have := lnLowerSound c (powPrec c θ p)
    have hθ' : (0 : ℝ) ≤ θ := by exact_mod_cast hθ0
    push_cast
    nlinarith
  · have := lnUpperSound c (powPrec c θ p)
    have hθ' : (θ : ℝ) ≤ 0 := by exact_mod_cast (lt_of_not_ge hθ0).le
    push_cast
    nlinarith

theorem powUpperSound : PowUpperSound := by
  intro c θ p
  by_cases hθ : θ = 0
  · subst hθ; simp [powUpper]
  by_cases hc : c = 0
  · subst hc
    have : powUpper 0 θ p = 0 := by unfold powUpper; rw [ite_eq_right hθ, ite_eq_left rfl]
    rw [this]; push_cast; rw [Real.zero_rpow (by exact_mod_cast hθ)]
  refine le_trans ?_ (exp_le_powUpper hθ hc p)
  rcases lt_or_gt_of_ne hc with hn | hp
  · rw [Real.rpow_def_of_neg (by exact_mod_cast hn)]
    exact mul_le_of_le_one_right (Real.exp_pos _).le (Real.cos_le_one _)
  · rw [Real.rpow_def_of_pos (by exact_mod_cast hp)]

theorem powLowerSound : PowLowerSound := by
  intro c θ p
  by_cases hθ : θ = 0
  · subst hθ; simp [powLower]
  by_cases hc : c = 0
  · subst hc
    have : powLower 0 θ p = 0 := by unfold powLower; rw [ite_eq_right hθ, ite_eq_left rfl]
    rw [this]; push_cast; rw [Real.zero_rpow (by exact_mod_cast hθ)]
  rcases lt_or_gt_of_ne hc with hn | hp
  · have : powLower c θ p = -powUpper c θ p := by unfold powLower; rw [ite_eq_right hθ, ite_eq_right hc, ite_eq_left hn]
    rw [this, Real.rpow_def_of_neg (by exact_mod_cast hn)]
    have h1 := exp_le_powUpper hθ hc p
    have h2 := Real.neg_one_le_cos ((θ : ℝ) * Real.pi)
    have h3 := (Real.exp_pos (Real.log c * θ)).le
    push_cast
    nlinarith [mul_nonneg h3 (by linarith : (0 : ℝ) ≤ Real.cos (θ * Real.pi) + 1)]
  · rw [Real.rpow_def_of_pos (by exact_mod_cast hp)]
    exact powLowerPos_le hθ hp p

end Interval

section Bound

open _root_.Spot.Bound (atanhSum twoAtanhUpper twoAtanhLower ln2Lower ln2Upper lnTerms lnBits lnPrec)

/-! ## The atanh series (`Bound.twoAtanhUpper`, `twoAtanhLower`) -/

theorem atanhSum_eq (t2 : ℚ) : ∀ (n i : ℕ) (s p : ℚ),
    atanhSum t2 n i s p = (s + ∑ k ∈ range n, p * t2 ^ k / ((2 * (i + k) + 1 : ℕ) : ℚ), p * t2 ^ n)
  | 0, i, s, p => by simp [atanhSum]
  | n + 1, i, s, p => by
    rw [atanhSum, atanhSum_eq t2 n (i + 1), sum_range_succ']
    refine Prod.ext ?_ (by simp only; ring)
    simp only
    have : ∀ k ∈ range n, p * t2 * t2 ^ k / ((2 * (i + 1 + k) + 1 : ℕ) : ℚ) =
        p * t2 ^ (k + 1) / ((2 * (i + (k + 1)) + 1 : ℕ) : ℚ) := fun k _ => by
      rw [show i + 1 + k = i + (k + 1) by omega, pow_succ]; ring
    rw [sum_congr rfl this]
    simp only [add_zero, pow_zero, mul_one]
    ring

theorem atanhSum_zero (t : ℚ) (N : ℕ) :
    atanhSum (t * t) N 0 0 t = (∑ k ∈ range N, t ^ (2 * k + 1) / ((2 * k + 1 : ℕ) : ℚ), t ^ (2 * N + 1)) := by
  rw [atanhSum_eq]
  refine Prod.ext ?_ ?_
  · simp only [zero_add]
    exact sum_congr rfl fun k _ => by rw [pow_succ, pow_mul, sq]; ring
  · simp only; rw [pow_succ, pow_mul, sq]; ring

theorem log_sub_log_hasSum {t : ℝ} (h0 : 0 ≤ t) (h1 : t < 1) :
    HasSum (fun k : ℕ => 2 * (t ^ (2 * k + 1) / (2 * k + 1))) (Real.log ((1 + t) / (1 - t))) := by
  rw [Real.log_div (by linarith) (by linarith)]
  convert Real.hasSum_log_sub_log_of_abs_lt_one (x := t) (by rw [abs_of_nonneg h0]; exact h1) using 2 with k
  ring

theorem atanh_partial_le {t : ℝ} (h0 : 0 ≤ t) (h1 : t < 1) (N : ℕ) :
    2 * ∑ k ∈ range N, t ^ (2 * k + 1) / (2 * k + 1) ≤ Real.log ((1 + t) / (1 - t)) := by
  rw [mul_sum]
  exact sum_le_hasSum (range N) (fun k _ => by positivity) (log_sub_log_hasSum h0 h1)

theorem atanh_le_partial {t : ℝ} (h0 : 0 ≤ t) (h1 : t < 1) (N : ℕ) :
    Real.log ((1 + t) / (1 - t)) ≤
      2 * (∑ k ∈ range N, t ^ (2 * k + 1) / (2 * k + 1) + t ^ (2 * N + 1) / ((2 * N + 1) * (1 - t * t))) := by
  set f : ℕ → ℝ := fun k => 2 * (t ^ (2 * k + 1) / (2 * k + 1))
  have htail : HasSum (fun k => f (k + N)) (Real.log ((1 + t) / (1 - t)) - ∑ k ∈ range N, f k) :=
    (hasSum_nat_add_iff' N).2 (log_sub_log_hasSum h0 h1)
  have htt : 0 ≤ t * t := mul_nonneg h0 h0
  have htt1 : t * t < 1 := by nlinarith
  have hg : HasSum (fun k : ℕ => 2 * (t ^ (2 * N + 1) / (2 * N + 1)) * (t * t) ^ k)
      (2 * (t ^ (2 * N + 1) / (2 * N + 1)) * (1 - t * t)⁻¹) :=
    (hasSum_geometric_of_lt_one htt htt1).mul_left _
  have hle := hasSum_le (fun k => ?_) htail hg
  · have hsum : ∑ k ∈ range N, f k = 2 * ∑ k ∈ range N, t ^ (2 * k + 1) / (2 * k + 1) := by
      rw [mul_sum]
    have h2 : 2 * (t ^ (2 * N + 1) / (2 * N + 1)) * (1 - t * t)⁻¹ =
        2 * (t ^ (2 * N + 1) / ((2 * N + 1) * (1 - t * t))) := by
      field_simp
    rw [hsum] at hle
    linarith
  · simp only [f]
    have hpow : t ^ (2 * (k + N) + 1) = t ^ (2 * N + 1) * (t * t) ^ k := by
      rw [show 2 * (k + N) + 1 = (2 * N + 1) + 2 * k by ring, pow_add, pow_mul, sq]
    rw [hpow]
    have hN : (0 : ℝ) < 2 * N + 1 := by positivity
    have hkN : (2 * N + 1 : ℝ) ≤ 2 * ((k + N : ℕ) : ℝ) + 1 := by push_cast; linarith [(Nat.cast_nonneg k : (0 : ℝ) ≤ k)]
    have hp : 0 ≤ t ^ (2 * N + 1) * (t * t) ^ k := by positivity
    calc 2 * (t ^ (2 * N + 1) * (t * t) ^ k / (2 * ((k + N : ℕ) : ℝ) + 1))
        ≤ 2 * (t ^ (2 * N + 1) * (t * t) ^ k / (2 * N + 1)) := by gcongr
      _ = 2 * (t ^ (2 * N + 1) / (2 * N + 1)) * (t * t) ^ k := by ring

theorem twoAtanhLower_le {t : ℚ} (h0 : 0 ≤ t) (h1 : t < 1) :
    ((twoAtanhLower t : ℚ) : ℝ) ≤ Real.log ((1 + (t : ℝ)) / (1 - t)) := by
  have := atanh_partial_le (t := (t : ℝ)) (by exact_mod_cast h0) (by exact_mod_cast h1) lnTerms
  unfold twoAtanhLower
  rw [atanhSum_zero]
  push_cast
  exact this

theorem le_twoAtanhUpper {t : ℚ} (h0 : 0 ≤ t) (h1 : t < 1) :
    Real.log ((1 + (t : ℝ)) / (1 - t)) ≤ ((twoAtanhUpper t : ℚ) : ℝ) := by
  have := atanh_le_partial (t := (t : ℝ)) (by exact_mod_cast h0) (by exact_mod_cast h1) lnTerms
  unfold twoAtanhUpper
  simp only
  rw [atanhSum_zero]
  push_cast
  exact this

/-- At `t = (z − 1)/(z + 1)`, `(1 + t)/(1 − t) = z`. -/
theorem atanh_arg {z : ℚ} (hz : 0 < z) :
    (1 + (((z - 1) / (z + 1) : ℚ) : ℝ)) / (1 - (((z - 1) / (z + 1) : ℚ) : ℝ)) = z := by
  have hz' : (0 : ℝ) < z := by exact_mod_cast hz
  have h1 : (z : ℝ) + 1 ≠ 0 := by linarith
  push_cast
  rw [show (1 : ℝ) + (z - 1) / (z + 1) = 2 * z / (z + 1) by field_simp; ring,
    show (1 : ℝ) - (z - 1) / (z + 1) = 2 / (z + 1) by field_simp; ring]
  field_simp

theorem atanh_arg_range {z : ℚ} (hz : 1 ≤ z) : 0 ≤ (z - 1) / (z + 1) ∧ (z - 1) / (z + 1) < 1 := by
  have hz0 : (0 : ℚ) < z + 1 := by linarith
  exact ⟨div_nonneg (by linarith) hz0.le, by rw [div_lt_one hz0]; linarith⟩

theorem log_le_twoAtanhUpper {z : ℚ} (hz : 1 ≤ z) :
    Real.log z ≤ ((twoAtanhUpper ((z - 1) / (z + 1)) : ℚ) : ℝ) := by
  obtain ⟨h0, h1⟩ := atanh_arg_range hz
  have := le_twoAtanhUpper h0 h1
  rwa [atanh_arg (by linarith)] at this

theorem twoAtanhLower_le_log {z : ℚ} (hz : 1 ≤ z) :
    ((twoAtanhLower ((z - 1) / (z + 1)) : ℚ) : ℝ) ≤ Real.log z := by
  obtain ⟨h0, h1⟩ := atanh_arg_range hz
  have := twoAtanhLower_le h0 h1
  rwa [atanh_arg (by linarith)] at this

theorem log_two_le_ln2Upper : Real.log 2 ≤ ((ln2Upper : ℚ) : ℝ) := by
  have := log_le_twoAtanhUpper (z := 2) (by norm_num)
  unfold ln2Upper
  norm_num at this ⊢
  exact this

theorem ln2Lower_le_log_two : ((ln2Lower : ℚ) : ℝ) ≤ Real.log 2 := by
  have := twoAtanhLower_le_log (z := 2) (by norm_num)
  unfold ln2Lower
  norm_num at this ⊢
  exact this

/-! ## The scale `2^j` (`Bound.lnUpper`, `lnBound`) -/

/-- `b · 2^j ≤ a` at the exponent both pick: `j0 = log₂ a − log₂ b`, one less when `a < b · 2^j0`. -/
theorem scale_le {a b : ℕ} (hb : 0 < b) (hab : b ≤ a) :
    b * 2 ^ (if a < b * 2 ^ (a.log2 - b.log2) then a.log2 - b.log2 - 1 else a.log2 - b.log2) ≤ a := by
  split_ifs with h
  · rcases Nat.eq_zero_or_pos (a.log2 - b.log2) with h0 | h0
    · rw [h0]; simpa using hab
    · have hb2 : b < 2 ^ (b.log2 + 1) := Nat.lt_log2_self
      have ha2 : 2 ^ a.log2 ≤ a := Nat.log2_self_le (by omega)
      have he : b.log2 + 1 + (a.log2 - b.log2 - 1) = a.log2 := by omega
      calc b * 2 ^ (a.log2 - b.log2 - 1) ≤ 2 ^ (b.log2 + 1) * 2 ^ (a.log2 - b.log2 - 1) :=
            Nat.mul_le_mul_right _ hb2.le
        _ = 2 ^ a.log2 := by rw [← Nat.pow_add, he]
        _ ≤ a := ha2
  · omega

theorem le_ceil_div (v : ℚ) {q : ℕ} (hq : 0 < q) : v ≤ ((v * q).ceil : ℚ) / q := by
  rw [le_div_iff₀ (by exact_mod_cast hq)]; exact Rat.le_ceil

theorem floor_div_le (v : ℚ) {q : ℕ} (hq : 0 < q) : ((v * q).floor : ℚ) / q ≤ v := by
  rw [div_le_iff₀ (by exact_mod_cast hq)]; exact Rat.floor_le _

/-- `ln(2^j z) = j ln 2 + ln z`. -/
theorem log_scale {z : ℝ} (hz : 0 < z) (j : ℕ) : Real.log ((2 : ℝ) ^ j * z) = j * Real.log 2 + Real.log z := by
  rw [Real.log_mul (by positivity) hz.ne', Real.log_pow]

/-- `y = N / D` compares with `2^k` as `N` does with `D · 2^k`. -/
theorem ratio_lt_two_pow_iff {y : ℚ} {N D : ℕ} (hD : 0 < D) (h : (N : ℚ) / D = y) (k : ℕ) :
    y < ((2 ^ k : ℕ) : ℚ) ↔ N < D * 2 ^ k := by
  subst h
  rw [div_lt_iff₀ (by exact_mod_cast hD), ← Nat.cast_mul, Nat.cast_lt, mul_comm]

/-- `y = N / D` is at least `2^k` when `D · 2^k ≤ N`. -/
theorem two_pow_le_ratio {y : ℚ} {N D : ℕ} (hD : 0 < D) (h : (N : ℚ) / D = y) {k : ℕ} (hk : D * 2 ^ k ≤ N) :
    ((2 ^ k : ℕ) : ℚ) ≤ y := by
  subst h
  rw [le_div_iff₀ (by exact_mod_cast hD), ← Nat.cast_mul, Nat.cast_le, mul_comm]
  exact hk

theorem boundLnUpper_le {δ : ℚ} (hδ : 0 < δ) (hδ1 : δ < 1) :
    Real.log (1 / (δ : ℝ)) ≤ ((_root_.Spot.Bound.lnUpper δ : ℚ) : ℝ) := by
  have hy1 : 1 < 1 / δ := by rw [lt_div_iff₀ hδ]; linarith
  have hnum : 0 < (1 / δ).num := Rat.num_pos.2 (by linarith)
  have hND : (((1 / δ).num.toNat : ℕ) : ℚ) / ((1 / δ).den : ℚ) = 1 / δ := by
    rw [show (((1 / δ).num.toNat : ℕ) : ℚ) = (((1 / δ).num : ℤ) : ℚ) by
      rw [← Int.toNat_of_nonneg hnum.le]; rfl]
    exact Rat.num_div_den _
  have hD : 0 < (1 / δ).den := (1 / δ).den_pos
  have hDN : (1 / δ).den ≤ (1 / δ).num.toNat := by
    have h := hy1.le
    rw [← hND, le_div_iff₀ (by exact_mod_cast hD), one_mul] at h
    exact_mod_cast h
  unfold _root_.Spot.Bound.lnUpper
  simp only [if_congr (ratio_lt_two_pow_iff hD hND _) rfl rfl]
  have hj := scale_le hD hDN
  generalize (if (1 / δ).num.toNat < (1 / δ).den * 2 ^ ((1 / δ).num.toNat.log2 - (1 / δ).den.log2) then
      (1 / δ).num.toNat.log2 - (1 / δ).den.log2 - 1 else (1 / δ).num.toNat.log2 - (1 / δ).den.log2) = j at hj ⊢
  have h2j : (0 : ℚ) < ((2 ^ j : ℕ) : ℚ) := by positivity
  have hz1 : 1 ≤ 1 / δ / ((2 ^ j : ℕ) : ℚ) := by
    rw [le_div_iff₀ h2j, one_mul]; exact two_pow_le_ratio hD hND hj
  have hz0 : (0 : ℝ) < ((1 / δ / ((2 ^ j : ℕ) : ℚ) : ℚ) : ℝ) := by
    have : (0 : ℚ) < 1 / δ / ((2 ^ j : ℕ) : ℚ) := by linarith
    exact_mod_cast this
  have hlog : Real.log (1 / (δ : ℝ)) = j * Real.log 2 + Real.log ((1 / δ / ((2 ^ j : ℕ) : ℚ) : ℚ) : ℝ) := by
    rw [← log_scale hz0 j]
    congr 1; push_cast; field_simp
  rw [hlog]
  refine le_trans ?_ (Rat.cast_le.2 (le_ceil_div _ (Nat.two_pow_pos lnBits)))
  have h2 := log_two_le_ln2Upper
  unfold ln2Upper at h2
  have hz := log_le_twoAtanhUpper hz1
  have hj0 : (0 : ℝ) ≤ j := Nat.cast_nonneg j
  have hmul := mul_le_mul_of_nonneg_left h2 hj0
  push_cast at hz hmul ⊢
  linarith

theorem boundLnUpperSound : BoundLnUpperSound := by
  intro n k
  have hd : (0 : ℚ) < ((n + k + 2 : ℕ) : ℚ) := by positivity
  refine boundLnUpper_le (by positivity) ?_
  rw [div_lt_one hd]
  push_cast
  linarith

/-- `(m − q)/(m + q) = (z − 1)/(z + 1)` at `z = m/q`. -/
theorem nat_ratio {m q : ℕ} (hq : 0 < q) (hm : q ≤ m) :
    ((m - q : ℕ) : ℚ) / ((m + q : ℕ) : ℚ) = ((m : ℚ) / q - 1) / ((m : ℚ) / q + 1) := by
  have hq' : (0 : ℚ) < q := by exact_mod_cast hq
  rw [Nat.cast_sub hm, Nat.cast_add]
  rw [show (m : ℚ) / q - 1 = (m - q) / q by field_simp, show (m : ℚ) / q + 1 = (m + q) / q by field_simp,
    div_div_div_cancel_right₀ hq'.ne']

/-- `ln(a/b) = j ln 2 + ln(a/(b 2^j))`. -/
theorem log_div_scale {a b : ℕ} (ha : 0 < a) (hb : 0 < b) (j : ℕ) :
    Real.log ((a : ℝ) / b) = j * Real.log 2 + Real.log ((((a : ℚ) / ((b * 2 ^ j : ℕ) : ℚ)) : ℚ) : ℝ) := by
  have h0 : (0 : ℝ) < (((a : ℚ) / ((b * 2 ^ j : ℕ) : ℚ)) : ℚ) := by
    have : (0 : ℚ) < (a : ℚ) / ((b * 2 ^ j : ℕ) : ℚ) := by positivity
    exact_mod_cast this
  rw [← log_scale h0 j]
  congr 1
  have hb' : (b : ℝ) ≠ 0 := by positivity
  push_cast
  field_simp

theorem ceil_div_mul {x d : ℕ} (hd : 0 < d) : x ≤ (x + d - 1) / d * d := by
  have h1 := Nat.div_add_mod (x + d - 1) d
  have h2 := Nat.mod_lt (x + d - 1) hd
  rw [Nat.mul_comm] at h1
  omega

/-- The rounded series value at `m/q` bounds `ln(a/d)` from below, and the one at `m'/q` from above. -/
theorem lnBound_core {a d m q j : ℕ} (ha : 0 < a) (hq : 0 < q) (hd : 0 < d) (hm1 : q ≤ m) :
    (m * d ≤ a * q →
      (((j * ln2Lower + twoAtanhLower (((m - q : ℕ) : ℚ) / ((m + q : ℕ) : ℚ)) : ℚ) : ℝ) ≤
        j * Real.log 2 + Real.log (((a : ℚ) / d : ℚ) : ℝ))) ∧
    (a * q ≤ m * d →
      j * Real.log 2 + Real.log (((a : ℚ) / d : ℚ) : ℝ) ≤
        (((j * ln2Upper + twoAtanhUpper (((m - q : ℕ) : ℚ) / ((m + q : ℕ) : ℚ)) : ℚ) : ℝ))) := by
  have hq' : (0 : ℚ) < q := by exact_mod_cast hq
  have hd' : (0 : ℚ) < d := by exact_mod_cast hd
  have hz1 : (1 : ℚ) ≤ (m : ℚ) / q := by rw [le_div_iff₀ hq', one_mul]; exact_mod_cast hm1
  have hz0 : (0 : ℝ) < (((m : ℚ) / q : ℚ) : ℝ) := by
    have : (0 : ℚ) < (m : ℚ) / q := by linarith
    exact_mod_cast this
  have hj0 : (0 : ℝ) ≤ j := Nat.cast_nonneg j
  rw [nat_ratio hq hm1]
  constructor
  · intro hm2
    have hl := twoAtanhLower_le_log hz1
    have hmono : Real.log (((m : ℚ) / q : ℚ) : ℝ) ≤ Real.log (((a : ℚ) / d : ℚ) : ℝ) := by
      apply Real.log_le_log hz0
      apply Rat.cast_le.2
      rw [div_le_div_iff₀ hq' hd']
      exact_mod_cast hm2
    have hmul := mul_le_mul_of_nonneg_left ln2Lower_le_log_two hj0
    push_cast at hl hmono hmul ⊢
    linarith
  · intro hm2
    have hl := log_le_twoAtanhUpper hz1
    have hmono : Real.log (((a : ℚ) / d : ℚ) : ℝ) ≤ Real.log (((m : ℚ) / q : ℚ) : ℝ) := by
      have ha0 : (0 : ℚ) < (a : ℚ) / d := by positivity
      apply Real.log_le_log (by exact_mod_cast ha0)
      apply Rat.cast_le.2
      rw [div_le_div_iff₀ hd' hq']
      exact_mod_cast hm2
    have hmul := mul_le_mul_of_nonneg_left log_two_le_ln2Upper hj0
    push_cast at hl hmono hmul ⊢
    linarith

theorem lnBound_brackets {a b : ℕ} (hb : 0 < b) (hab : b ≤ a) :
    ((_root_.Spot.Bound.lnBound a b false : ℚ) : ℝ) ≤ Real.log ((a : ℝ) / b) ∧
      Real.log ((a : ℝ) / b) ≤ ((_root_.Spot.Bound.lnBound a b true : ℚ) : ℝ) := by
  have ha : 0 < a := lt_of_lt_of_le hb hab
  have hj := scale_le hb hab
  unfold _root_.Spot.Bound.lnBound
  simp only [Bool.false_eq_true, ↓reduceIte]
  generalize (if a < b * 2 ^ (a.log2 - b.log2) then a.log2 - b.log2 - 1 else a.log2 - b.log2) = j at hj ⊢
  have hq : 0 < 2 ^ lnPrec := Nat.two_pow_pos _
  generalize 2 ^ lnPrec = q at hq ⊢
  have hd : 0 < b * 2 ^ j := by positivity
  rw [log_div_scale ha hb j]
  constructor
  · refine le_trans (Rat.cast_le.2 (floor_div_le _ hq)) ?_
    have hm2 : a * q / (b * 2 ^ j) * (b * 2 ^ j) ≤ a * q := Nat.div_mul_le_self _ _
    have hm1 : q ≤ a * q / (b * 2 ^ j) := by
      rw [Nat.le_div_iff_mul_le hd, Nat.mul_comm]; exact Nat.mul_le_mul_right q hj
    exact (lnBound_core ha hq hd hm1).1 hm2
  · refine le_trans ?_ (Rat.cast_le.2 (le_ceil_div _ hq))
    have hm2 : a * q ≤ (a * q + b * 2 ^ j - 1) / (b * 2 ^ j) * (b * 2 ^ j) := ceil_div_mul hd
    have hm1 : q ≤ (a * q + b * 2 ^ j - 1) / (b * 2 ^ j) :=
      Nat.le_of_mul_le_mul_right ((Nat.mul_le_mul_left q hj).trans (by rw [Nat.mul_comm q a]; exact hm2)) hd
    exact (lnBound_core ha hq hd hm1).2 hm2

theorem boundLnBoundSound : BoundLnBoundSound := fun m k => lnBound_brackets (Nat.succ_pos m) (by omega)

end Bound

section Witnesses

open _root_.Spot.Interval (expLower expUpper lnLower lnUpper powLower powUpper)

/-! ## Witnesses -/

/-- At `p = 64`, `e^{-1}`'s bounds are one step of `2^-64` apart or less. -/
theorem expLowerSound_honest : expUpper (-1) 64 - expLower (-1) 64 ≤ 1 / 2 ^ 63 := by decide +kernel

/-- No rational at least `3` is below `e`. -/
theorem expLowerSound_nontrivial : ¬ ((3 : ℚ) : ℝ) ≤ Real.exp ((1 : ℚ) : ℝ) := by
  have := Real.exp_one_lt_d9
  push_cast
  intro h
  linarith

/-- At `p = 64`, `e`'s bounds are one step of `2^-64` apart or less. -/
theorem expUpperSound_honest : expUpper 1 64 - expLower 1 64 ≤ 1 / 2 ^ 63 := by decide +kernel

/-- `2` is not above `e`. -/
theorem expUpperSound_nontrivial : ¬ Real.exp ((1 : ℚ) : ℝ) ≤ ((2 : ℚ) : ℝ) := by
  have := Real.exp_one_gt_d9
  push_cast
  intro h
  linarith

/-- At `p = 64`, `ln(10^-30)`'s bounds are within `2^-62`. -/
theorem lnLowerSound_honest : lnUpper (1 / 10 ^ 30) 64 - lnLower (1 / 10 ^ 30) 64 ≤ 1 / 2 ^ 62 := by decide +kernel

/-- `1` is not below `ln 2`. -/
theorem lnLowerSound_nontrivial : ¬ ((1 : ℚ) : ℝ) ≤ Real.log ((2 : ℚ) : ℝ) := by
  have := Real.log_two_lt_d9
  push_cast
  intro h
  linarith

/-- At `p = 64`, `ln 2`'s bounds are within `2^-62`. -/
theorem lnUpperSound_honest : lnUpper 2 64 - lnLower 2 64 ≤ 1 / 2 ^ 62 := by decide +kernel

/-- `0` is not above `ln 2`. -/
theorem lnUpperSound_nontrivial : ¬ Real.log ((2 : ℚ) : ℝ) ≤ ((0 : ℚ) : ℝ) := by
  have := Real.log_two_gt_d9
  push_cast
  intro h
  linarith

/-- At `p = 64`, `2^-1`'s bounds are within `2^-62`. -/
theorem powLowerSound_honest : powUpper 2 (-1) 64 - powLower 2 (-1) 64 ≤ 1 / 2 ^ 62 := by decide +kernel

/-- `1` is not below `(1/2)^1`. -/
theorem powLowerSound_nontrivial : ¬ ((1 : ℚ) : ℝ) ≤ ((1 / 2 : ℚ) : ℝ) ^ ((1 : ℚ) : ℝ) := by
  push_cast
  rw [Real.rpow_one]
  norm_num

/-- At `p = 64`, `(1/2)^(1/2)`'s bounds are within `2^-62`. -/
theorem powUpperSound_honest : powUpper (1 / 2) (1 / 2) 64 - powLower (1 / 2) (1 / 2) 64 ≤ 1 / 2 ^ 62 := by
  decide +kernel

/-- `0` is not above `(1/2)^1`. -/
theorem powUpperSound_nontrivial : ¬ ((1 / 2 : ℚ) : ℝ) ^ ((1 : ℚ) : ℝ) ≤ ((0 : ℚ) : ℝ) := by
  push_cast
  rw [Real.rpow_one]
  norm_num

/-- `Bound.lnUpper (1/100)` is within `2^-63` of `ln 100` from above: `Bound.lnBound 100 1 false` is below it. -/
theorem boundLnUpperSound_honest :
    _root_.Spot.Bound.lnUpper (1 / 100) - _root_.Spot.Bound.lnBound 100 1 false < 1 / 2 ^ 63 := by
  decide +kernel

/-- `0` is not at least `ln 2 = ln(1/(1/2))`. -/
theorem boundLnUpperSound_nontrivial : ¬ Real.log (1 / ((1 / 2 : ℚ) : ℝ)) ≤ ((0 : ℚ) : ℝ) := by
  have := Real.log_two_gt_d9
  push_cast
  norm_num
  linarith

/-- `Bound.lnBound 100 1`'s two sides are within `2^-100`. -/
theorem boundLnBoundSound_honest :
    _root_.Spot.Bound.lnBound 100 1 true - _root_.Spot.Bound.lnBound 100 1 false < 1 / 2 ^ 100 := by
  decide +kernel

/-- `1` is not below `ln(2/1)`. -/
theorem boundLnBoundSound_nontrivial : ¬ ((1 : ℚ) : ℝ) ≤ Real.log (((2 : ℕ) : ℝ) / ((1 : ℕ) : ℝ)) := by
  have := Real.log_two_lt_d9
  push_cast
  norm_num
  linarith

end Witnesses

end Properties.Spot.SecurityProofs
