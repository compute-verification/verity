import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Analysis.Real.Sqrt

/-!
# The main theorem's inequalities after Loomis–Whitney

The proof of the main theorem (updated matrix multiplications are slowed down by `F`) bounds each IU `R` by the counts
`p` (weight entries it uses), `τ` (full positions), `Q` (input entries it uses at its other positions) and `Y` (output
entries it contributes products to there). These are its steps after Loomis–Whitney (`LoomisWhitney.lean`), stated over
plain numbers in the draft's variables: `b_w` and `b_a` the widths of a weight and an activation entry, `d` the smallest
inner dimension, `K = αΓ` the per-IU read cap and `μ` the weight on the two read counts.

* Cauchy–Schwarz over the multiplications `ℓ`: `∑ ℓ √(p_ℓ q_ℓ y_ℓ) ≤ √(p Q Y)` (`sum_sqrt_mul_mul_le`).
* The μ weighting: two lower bounds on the IUs' total reads, combined with weights `μ` and `1 - μ` (`mu_weighting`).
* `1/s + μ/u ≥ (1 + √μ)² / (u + s)` (`sq_one_add_sqrt_div_le`) and AM–GM (`am_gm_step`), which together bound each IU's
  share of the weighted reads by `c₁ p τ + c₂ √(p Q Y)` (`iu_reads_ge`), and so the total by `min c₁ c₂ · G`
  (`min_mul_sum_le_reads`).
* `F`'s closed form: the maximum over `μ` of `min c₁ c₂` is attained, at `μ = 1/4` or where the two terms meet
  (`isGreatest_min_one_add_sqrt`, and `isGreatest_F` in the draft's variables), and a total read cap of `α size(D)` makes
  `size(D) / 2G ≥ F` (`F_le_div_of_reads`).
-/

namespace Nci.SecurityProofs

open Finset

/-- **Cauchy–Schwarz over the multiplications**: an IU that uses at most `P` weight entries of each multiplication `ℓ`
computes at most `∑ ℓ √(p_ℓ q_ℓ y_ℓ) ≤ √(P Q Y)` products at its other positions, where `Q = ∑ q_ℓ`, `Y = ∑ y_ℓ`. -/
theorem sum_sqrt_mul_mul_le {L : Type*} (s : Finset L) {P : ℝ} {p q y : L → ℝ} (hp : ∀ ℓ ∈ s, p ℓ ≤ P)
    (hq : ∀ ℓ ∈ s, 0 ≤ q ℓ) (hy : ∀ ℓ ∈ s, 0 ≤ y ℓ) :
    ∑ ℓ ∈ s, √(p ℓ * q ℓ * y ℓ) ≤ √(P * (∑ ℓ ∈ s, q ℓ) * ∑ ℓ ∈ s, y ℓ) := by
  have cs : ∑ ℓ ∈ s, √(q ℓ * y ℓ) ≤ √((∑ ℓ ∈ s, q ℓ) * ∑ ℓ ∈ s, y ℓ) :=
    Real.le_sqrt_of_sq_le <| sum_sq_le_sum_mul_sum_of_sq_le_mul s hq hy fun ℓ hℓ =>
      (Real.sq_sqrt (mul_nonneg (hq ℓ hℓ) (hy ℓ hℓ))).le
  calc ∑ ℓ ∈ s, √(p ℓ * q ℓ * y ℓ) ≤ ∑ ℓ ∈ s, √P * √(q ℓ * y ℓ) := sum_le_sum fun ℓ hℓ => by
        rw [← Real.sqrt_mul' _ (mul_nonneg (hq ℓ hℓ) (hy ℓ hℓ)), mul_assoc]
        exact Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_right (hp ℓ hℓ) (mul_nonneg (hq ℓ hℓ) (hy ℓ hℓ)))
    _ = √P * ∑ ℓ ∈ s, √(q ℓ * y ℓ) := (mul_sum _ _ _).symm
    _ ≤ √P * √((∑ ℓ ∈ s, q ℓ) * ∑ ℓ ∈ s, y ℓ) := mul_le_mul_of_nonneg_left cs (Real.sqrt_nonneg _)
    _ = √(P * (∑ ℓ ∈ s, q ℓ) * ∑ ℓ ∈ s, y ℓ) := by
        rw [mul_assoc, Real.sqrt_mul' _ (mul_nonneg (sum_nonneg hq) (sum_nonneg hy))]

/-- **The μ weighting**: the IUs' total reads `T` are at least each IU's own reads `u R + a R` summed (weights, then
activations or partial sums), and at least the weights plus the partial sums that split entries pass between IUs,
`u R + b R` summed; the per-IU reads may include some of those partial sums, so the counts combine with weights `μ` and
`1 - μ` rather than adding. In the draft, `u = b_w p`, `a = b_a (d τ + Q)` and `b = (b_a / 2) Y`. -/
theorem mu_weighting {ι : Type*} (s : Finset ι) {T μ : ℝ} {u a b : ι → ℝ} (hμ₀ : 0 ≤ μ) (hμ₁ : μ ≤ 1)
    (h₁ : ∑ R ∈ s, (u R + a R) ≤ T) (h₂ : ∑ R ∈ s, (u R + b R) ≤ T) :
    ∑ R ∈ s, (u R + μ * a R + (1 - μ) * b R) ≤ T := by
  calc ∑ R ∈ s, (u R + μ * a R + (1 - μ) * b R)
        = μ * ∑ R ∈ s, (u R + a R) + (1 - μ) * ∑ R ∈ s, (u R + b R) := by
          rw [mul_sum, mul_sum, ← sum_add_distrib]
          exact sum_congr rfl fun _ _ => by ring
    _ ≤ μ * T + (1 - μ) * T :=
        add_le_add (mul_le_mul_of_nonneg_left h₁ hμ₀) (mul_le_mul_of_nonneg_left h₂ (by linarith))
    _ = T := by ring

/-- The bound on the weight that the μ weighting puts on an IU's reads: `1/s + μ/u ≥ (1 + √μ)² / (u + s)`, where `u` is
the IU's weight reads and `s` its activation reads (Cauchy–Schwarz in Engel's form). -/
theorem sq_one_add_sqrt_div_le {u s μ : ℝ} (hu : 0 < u) (hs : 0 < s) (hμ : 0 ≤ μ) :
    (1 + √μ) ^ 2 / (u + s) ≤ 1 / s + μ / u := by
  obtain ⟨m, hm, rfl⟩ : ∃ m, 0 ≤ m ∧ μ = m ^ 2 := ⟨√μ, Real.sqrt_nonneg _, (Real.sq_sqrt hμ).symm⟩
  rw [Real.sqrt_sq hm, div_add_div _ _ hs.ne' hu.ne', div_le_div_iff₀ (by positivity) (by positivity)]
  nlinarith [sq_nonneg (u - m * s)]

/-- **AM–GM**, as the main theorem's proof uses it: `u v₂/s + μ v₂ + v₃ ≥ 2 √(u v₂ v₃ (1/s + μ/u))`. -/
theorem am_gm_step {u s μ v₂ v₃ : ℝ} (hu : 0 < u) (hs : 0 < s) (hμ : 0 ≤ μ) (hv₂ : 0 ≤ v₂) (hv₃ : 0 ≤ v₃) :
    2 * √(u * v₂ * v₃ * (1 / s + μ / u)) ≤ u * v₂ / s + μ * v₂ + v₃ := by
  have hX : 0 ≤ u * v₂ / s + μ * v₂ := by positivity
  rw [show u * v₂ * v₃ * (1 / s + μ / u) = (u * v₂ / s + μ * v₂) * v₃ by field_simp, Real.sqrt_mul hX]
  nlinarith [sq_nonneg (√(u * v₂ / s + μ * v₂) - √v₃), Real.sq_sqrt hX, Real.sq_sqrt hv₃]

/-- **One IU's share of the weighted reads**: an IU within the read cap `b_w p + b_a (d τ + Q) ≤ K` contributes at least
`c₁ p τ + c₂ √(p Q Y)` to `∑ R (b_w p + μ b_a (d τ + Q) + (1 - μ) (b_a/2) Y)`, where
`c₁ = (1 + √μ)² b_w b_a d / K` and `c₂ = √2 (1 + √μ) √(1 - μ) b_a √(b_w / K)` are the two terms of `F`'s minimum. -/
theorem iu_reads_ge {bw ba d K μ p τ Q Y : ℝ} (hbw : 0 < bw) (hba : 0 < ba) (hd : 0 < d) (hK : 0 < K)
    (hμ₀ : 0 ≤ μ) (hμ₁ : μ ≤ 1) (hp : 0 ≤ p) (hτ : 0 ≤ τ) (hQ : 0 ≤ Q) (hY : 0 ≤ Y)
    (hcap : bw * p + ba * (d * τ + Q) ≤ K) :
    (1 + √μ) ^ 2 * (bw * ba * d / K) * (p * τ) + √2 * (1 + √μ) * √(1 - μ) * (ba * √(bw / K)) * √(p * Q * Y) ≤
      bw * p + μ * (ba * (d * τ + Q)) + (1 - μ) * (ba / 2 * Y) := by
  have h1μ : 0 ≤ 1 - μ := by linarith
  have hv₃ : 0 ≤ (1 - μ) * (ba / 2 * Y) := by positivity
  have hrest : 0 ≤ μ * (ba * (d * τ + Q)) + (1 - μ) * (ba / 2 * Y) := by positivity
  rcases hp.eq_or_lt with rfl | hp
  · simpa using hrest
  rcases (show 0 ≤ d * τ + Q by positivity).eq_or_lt with hs | hs
  · have hτ0 : τ = 0 := by nlinarith
    have hQ0 : Q = 0 := by nlinarith
    subst hτ0 hQ0
    simpa using add_nonneg (by positivity : 0 ≤ bw * p) hrest
  -- u = b_w p, s = b_a (d τ + Q), and k = 1/s + μ/u is the weight the μ weighting puts on the IU's reads
  have hu : 0 < bw * p := by positivity
  have hs' : 0 < ba * (d * τ + Q) := by positivity
  set k := 1 / (ba * (d * τ + Q)) + μ / (bw * p) with hk_def
  have hk : (1 + √μ) ^ 2 / K ≤ k :=
    (div_le_div_of_nonneg_left (sq_nonneg _) (by positivity) hcap).trans (sq_one_add_sqrt_div_le hu hs' hμ₀)
  -- the full positions: `u v₁/s + μ v₁ = b_w b_a d p τ k ≥ c₁ p τ`
  have full : (1 + √μ) ^ 2 * (bw * ba * d / K) * (p * τ) ≤
      bw * p * (ba * (d * τ)) / (ba * (d * τ + Q)) + μ * (ba * (d * τ)) := by
    calc (1 + √μ) ^ 2 * (bw * ba * d / K) * (p * τ) = (1 + √μ) ^ 2 / K * (bw * p * (ba * (d * τ))) := by ring
      _ ≤ k * (bw * p * (ba * (d * τ))) := mul_le_mul_of_nonneg_right hk (by positivity)
      _ = _ := by rw [hk_def]; field_simp
  -- the other positions, by AM–GM: `u v₂/s + μ v₂ + v₃ ≥ 2 √(u v₂ v₃ k) ≥ c₂ √(p Q Y)`
  have other : √2 * (1 + √μ) * √(1 - μ) * (ba * √(bw / K)) * √(p * Q * Y) ≤
      bw * p * (ba * Q) / (ba * (d * τ + Q)) + μ * (ba * Q) + (1 - μ) * (ba / 2 * Y) := by
    refine le_trans ?_ (am_gm_step hu hs' hμ₀ (by positivity : 0 ≤ ba * Q) hv₃)
    have hsq : (√2 * (1 + √μ) * √(1 - μ) * (ba * √(bw / K)) * √(p * Q * Y)) ^ 2 =
        4 * (bw * p * (ba * Q) * ((1 - μ) * (ba / 2 * Y))) * ((1 + √μ) ^ 2 / K) := by
      simp only [mul_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2), Real.sq_sqrt h1μ,
        Real.sq_sqrt (by positivity : 0 ≤ bw / K), Real.sq_sqrt (by positivity : 0 ≤ p * Q * Y)]
      ring
    have hle : (√2 * (1 + √μ) * √(1 - μ) * (ba * √(bw / K)) * √(p * Q * Y)) ^ 2 ≤
        (2 * √(bw * p * (ba * Q) * ((1 - μ) * (ba / 2 * Y)) * k)) ^ 2 := by
      rw [hsq, mul_pow, Real.sq_sqrt (mul_nonneg (by positivity) (hk.trans' (by positivity)))]
      nlinarith [mul_le_mul_of_nonneg_left hk (by positivity : 0 ≤ bw * p * (ba * Q) * ((1 - μ) * (ba / 2 * Y)))]
    exact le_of_pow_le_pow_left₀ two_ne_zero (by positivity) hle
  calc _ ≤ (bw * p * (ba * (d * τ)) / (ba * (d * τ + Q)) + μ * (ba * (d * τ))) +
        (bw * p * (ba * Q) / (ba * (d * τ + Q)) + μ * (ba * Q) + (1 - μ) * (ba / 2 * Y)) := add_le_add full other
    _ = _ := by field_simp; ring

/-- **The IUs' weighted reads bound their products**: if each IU `R` computes at most `g R ≤ p τ + √(p Q Y)` products
(Lemma 1, and Loomis–Whitney with Cauchy–Schwarz) within its read cap, and the IUs' total reads `T` are at least both
counts of the μ weighting, then `T ≥ min c₁ c₂ · ∑ R g R`. -/
theorem min_mul_sum_le_reads {ι : Type*} (s : Finset ι) {bw ba d K μ T : ℝ} (hbw : 0 < bw) (hba : 0 < ba)
    (hd : 0 < d) (hK : 0 < K) (hμ₀ : 0 ≤ μ) (hμ₁ : μ ≤ 1) {p τ Q Y g : ι → ℝ} (hp : ∀ R ∈ s, 0 ≤ p R)
    (hτ : ∀ R ∈ s, 0 ≤ τ R) (hQ : ∀ R ∈ s, 0 ≤ Q R) (hY : ∀ R ∈ s, 0 ≤ Y R)
    (hcap : ∀ R ∈ s, bw * p R + ba * (d * τ R + Q R) ≤ K)
    (hg : ∀ R ∈ s, g R ≤ p R * τ R + √(p R * Q R * Y R))
    (h₁ : ∑ R ∈ s, (bw * p R + ba * (d * τ R + Q R)) ≤ T) (h₂ : ∑ R ∈ s, (bw * p R + ba / 2 * Y R) ≤ T) :
    min ((1 + √μ) ^ 2 * (bw * ba * d / K)) (√2 * (1 + √μ) * √(1 - μ) * (ba * √(bw / K))) * ∑ R ∈ s, g R ≤ T := by
  set c₁ := (1 + √μ) ^ 2 * (bw * ba * d / K)
  set c₂ := √2 * (1 + √μ) * √(1 - μ) * (ba * √(bw / K))
  have hc : 0 ≤ min c₁ c₂ := le_min (by positivity) (by positivity)
  refine le_trans ?_ (mu_weighting s hμ₀ hμ₁ h₁ h₂)
  rw [mul_sum]
  refine sum_le_sum fun R hR => ?_
  calc min c₁ c₂ * g R ≤ min c₁ c₂ * (p R * τ R + √(p R * Q R * Y R)) := mul_le_mul_of_nonneg_left (hg R hR) hc
    _ ≤ c₁ * (p R * τ R) + c₂ * √(p R * Q R * Y R) := by
        rw [mul_add]
        exact add_le_add (mul_le_mul_of_nonneg_right (min_le_left _ _) (mul_nonneg (hp R hR) (hτ R hR)))
          (mul_le_mul_of_nonneg_right (min_le_right _ _) (Real.sqrt_nonneg _))
    _ ≤ _ := iu_reads_ge hbw hba hd hK hμ₀ hμ₁ (hp R hR) (hτ R hR) (hQ R hR) (hY R hR) (hcap R hR)

section ClosedForm

/-- `(1 + m)³ (1 - m) ≤ 27/16`, with equality at `m = 1/2`: the second term of `F`'s minimum is largest at `μ = 1/4`. -/
private theorem one_add_pow_three_mul_le (m : ℝ) : (1 + m) ^ 3 * (1 - m) ≤ 27 / 16 := by
  nlinarith [mul_nonneg (sq_nonneg (m - 1 / 2)) (add_nonneg (sq_nonneg (m + 3 / 2)) (by norm_num : (0 : ℝ) ≤ 1 / 2))]

/-- `(1 + m)³ (1 - m)` decreases on `[1/2, 1]`. -/
private theorem one_add_pow_three_mul_anti {a b : ℝ} (ha : 1 / 2 ≤ a) (hab : a ≤ b) :
    (1 + b) ^ 3 * (1 - b) ≤ (1 + a) ^ 3 * (1 - a) := by
  have hb : 1 / 2 ≤ b := ha.trans hab
  have bracket : 0 ≤ 2 * (a ^ 2 + a * b + b ^ 2) + (a + b) * (a ^ 2 + b ^ 2) - 2 := by
    nlinarith [mul_nonneg (sub_nonneg.2 ha) (sub_nonneg.2 hb), mul_nonneg (sub_nonneg.2 ha) (sub_nonneg.2 ha),
      mul_nonneg (sub_nonneg.2 hb) (sub_nonneg.2 hb),
      mul_nonneg (mul_nonneg (sub_nonneg.2 ha) (sub_nonneg.2 hb)) (sub_nonneg.2 ha),
      mul_nonneg (mul_nonneg (sub_nonneg.2 ha) (sub_nonneg.2 hb)) (sub_nonneg.2 hb),
      mul_nonneg (mul_nonneg (sub_nonneg.2 ha) (sub_nonneg.2 ha)) (sub_nonneg.2 ha),
      mul_nonneg (mul_nonneg (sub_nonneg.2 hb) (sub_nonneg.2 hb)) (sub_nonneg.2 hb)]
  nlinarith [mul_nonneg (sub_nonneg.2 hab) bracket]

/-- **The maximum over `μ` in `F`, in closed form**: for `A, B > 0`, `min {A (1 + √μ)², B (1 + √μ) √(1 - μ)}` over
`0 < μ < 1` attains its supremum, `(3√3/4) B` at `μ = 1/4` when `B² ≤ 3A²`, and otherwise `4AB⁴ / (A² + B²)²`, where the
two terms meet, at `√μ = (B² - A²) / (A² + B²)`. -/
theorem isGreatest_min_one_add_sqrt {A B : ℝ} (hA : 0 < A) (hB : 0 < B) :
    IsGreatest ((fun μ => min (A * (1 + √μ) ^ 2) (B * ((1 + √μ) * √(1 - μ)))) '' Set.Ioo 0 1)
      (if B ^ 2 ≤ 3 * A ^ 2 then 3 * √3 / 4 * B else 4 * A * B ^ 4 / (A ^ 2 + B ^ 2) ^ 2) := by
  have sq₂ : ∀ μ ∈ Set.Ioo (0 : ℝ) 1, (B * ((1 + √μ) * √(1 - μ))) ^ 2 = B ^ 2 * ((1 + √μ) ^ 3 * (1 - √μ)) := by
    intro μ hμ
    rw [mul_pow, mul_pow, Real.sq_sqrt (by linarith [hμ.2])]
    obtain ⟨m, hm, rfl⟩ : ∃ m, 0 ≤ m ∧ μ = m ^ 2 := ⟨√μ, Real.sqrt_nonneg _, (Real.sq_sqrt hμ.1.le).symm⟩
    rw [Real.sqrt_sq hm]
    ring
  split_ifs with h
  · have h4 : √(1 / 4 : ℝ) = 1 / 2 := by
      rw [show (1 / 4 : ℝ) = (1 / 2) ^ 2 by norm_num]
      exact Real.sqrt_sq (by norm_num)
    have h34 : √(1 - 1 / 4 : ℝ) = √3 / 2 := by
      rw [show (1 - 1 / 4 : ℝ) = 3 / 2 ^ 2 by norm_num, Real.sqrt_div' _ (by norm_num), Real.sqrt_sq (by norm_num)]
    have h3 : √3 ^ 2 = 3 := Real.sq_sqrt (by norm_num)
    have hroot : √3 * B ≤ 3 * A :=
      le_of_pow_le_pow_left₀ two_ne_zero (by positivity) (by rw [mul_pow, mul_pow, h3]; nlinarith)
    refine ⟨⟨1 / 4, ⟨by norm_num, by norm_num⟩, ?_⟩, ?_⟩
    · simp only [h4, h34]
      rw [min_eq_right (by nlinarith)]
      ring
    · rintro _ ⟨μ, hμ, rfl⟩
      refine (min_le_right _ _).trans (le_of_pow_le_pow_left₀ two_ne_zero (by positivity) ?_)
      rw [sq₂ μ hμ, mul_pow, div_pow, mul_pow, h3]
      nlinarith [mul_le_mul_of_nonneg_left (one_add_pow_three_mul_le √μ) (sq_nonneg B)]
  · rw [not_le] at h
    set D := A ^ 2 + B ^ 2 with hD
    have hD0 : 0 < D := by positivity
    set m₀ := (B ^ 2 - A ^ 2) / D with hm₀
    have hplus : 1 + m₀ = 2 * B ^ 2 / D := by rw [hm₀]; field_simp; ring
    have hminus : 1 - m₀ = 2 * A ^ 2 / D := by rw [hm₀]; field_simp; ring
    have hm₀half : 1 / 2 < m₀ := by rw [hm₀, lt_div_iff₀ hD0]; nlinarith
    have hm₀one : m₀ < 1 := by rw [hm₀, div_lt_one hD0]; nlinarith
    have hM : A * (1 + m₀) ^ 2 = 4 * A * B ^ 4 / D ^ 2 := by rw [hplus]; field_simp; ring
    have hM₂ : B ^ 2 * ((1 + m₀) ^ 3 * (1 - m₀)) = (4 * A * B ^ 4 / D ^ 2) ^ 2 := by
      rw [hplus, hminus]; field_simp; ring
    refine ⟨⟨m₀ ^ 2, ⟨by nlinarith, by nlinarith⟩, ?_⟩, ?_⟩
    · have hs : √(m₀ ^ 2) = m₀ := Real.sqrt_sq (by linarith)
      have hw : √(1 - m₀ ^ 2) = 2 * A * B / D := by
        rw [show 1 - m₀ ^ 2 = (2 * A * B / D) ^ 2 by
          rw [show 1 - m₀ ^ 2 = (1 + m₀) * (1 - m₀) by ring, hplus, hminus]; field_simp]
        exact Real.sqrt_sq (by positivity)
      simp only [hs, hw]
      rw [min_eq_left (le_of_eq ?_), hM]
      rw [hM, hplus]; field_simp; ring
    · rintro _ ⟨μ, hμ, rfl⟩
      rcases le_or_gt √μ m₀ with hm | hm
      · refine (min_le_left _ _).trans ?_
        rw [← hM]
        exact mul_le_mul_of_nonneg_left (pow_le_pow_left₀ (by positivity) (by linarith) 2) hA.le
      · refine (min_le_right _ _).trans (le_of_pow_le_pow_left₀ two_ne_zero (by positivity) ?_)
        rw [sq₂ μ hμ, ← hM₂]
        exact mul_le_mul_of_nonneg_left (one_add_pow_three_mul_anti hm₀half.le hm.le) (sq_nonneg B)

/-- **`F`'s closed form**: `F = (1/(2α)) max_{0<μ<1} min {(1 + √μ)² b_w b_a d / K, √2 (1 + √μ) √(1 - μ) b_a √(b_w/K)}`
is attained, and equals `(3√6/8) b_a √(b_w/K) / α` when `2K ≤ 3 b_w d²` (the second term alone binds, at `μ = 1/4`), and
`8 b_w b_a d K / (α (b_w d² + 2K)²)` otherwise (the two terms meet). -/
theorem isGreatest_F {bw ba d K α : ℝ} (hbw : 0 < bw) (hba : 0 < ba) (hd : 0 < d) (hK : 0 < K) (hα : 0 < α) :
    IsGreatest ((fun μ => 1 / (2 * α) * min ((1 + √μ) ^ 2 * (bw * ba * d / K))
        (√2 * (1 + √μ) * √(1 - μ) * (ba * √(bw / K)))) '' Set.Ioo 0 1)
      (if 2 * K ≤ 3 * bw * d ^ 2 then 3 * √6 / 8 * ba * √(bw / K) / α
        else 8 * bw * ba * d * K / (α * (bw * d ^ 2 + 2 * K) ^ 2)) := by
  have hA : 0 < bw * ba * d / K := by positivity
  have hB : 0 < √2 * (ba * √(bw / K)) := by positivity
  have hB2 : (√2 * (ba * √(bw / K))) ^ 2 = 2 * ba ^ 2 * bw / K := by
    rw [mul_pow, mul_pow, Real.sq_sqrt (by norm_num), Real.sq_sqrt (by positivity)]
    ring
  have hcond : (√2 * (ba * √(bw / K))) ^ 2 ≤ 3 * (bw * ba * d / K) ^ 2 ↔ 2 * K ≤ 3 * bw * d ^ 2 := by
    rw [hB2, div_pow, mul_div_assoc', div_le_div_iff₀ hK (by positivity)]
    have hc : 0 < ba ^ 2 * bw * K := by positivity
    constructor <;> intro h <;> nlinarith
  have hf : Monotone fun x : ℝ => 1 / (2 * α) * x := fun x y hxy => mul_le_mul_of_nonneg_left hxy (by positivity)
  have key := hf.map_isGreatest (isGreatest_min_one_add_sqrt hA hB)
  rw [Set.image_image] at key
  convert key using 1
  · refine Set.image_congr fun μ _ => ?_
    rw [mul_comm ((1 + √μ) ^ 2)]
    congr 2
    ring
  · by_cases h : 2 * K ≤ 3 * bw * d ^ 2
    · rw [ite_eq_left h, ite_eq_left (hcond.2 h), show (6 : ℝ) = 3 * 2 by norm_num, Real.sqrt_mul (by norm_num)]
      field_simp
      norm_num
    · rw [ite_eq_right h, ite_eq_right (mt hcond.1 h), show (√2 * (ba * √(bw / K))) ^ 4 = ((√2 * (ba * √(bw / K))) ^ 2) ^ 2 by
        ring, hB2]
      field_simp
      ring

/-- **The theorem's last step**: if the IUs' reads bound `min c₁ c₂ · G ≤ α S` for every `μ`, where `G` is the number of
products and `S` is `size(D)`, then `D` is larger than the multiplications' `2G` gates by at least `F`. -/
theorem F_le_div_of_reads {bw ba d K α S G : ℝ} (hbw : 0 < bw) (hba : 0 < ba) (hd : 0 < d) (hK : 0 < K)
    (hα : 0 < α) (hG : 0 < G)
    (h : ∀ μ ∈ Set.Ioo (0 : ℝ) 1,
      min ((1 + √μ) ^ 2 * (bw * ba * d / K)) (√2 * (1 + √μ) * √(1 - μ) * (ba * √(bw / K))) * G ≤ α * S) :
    (if 2 * K ≤ 3 * bw * d ^ 2 then 3 * √6 / 8 * ba * √(bw / K) / α
      else 8 * bw * ba * d * K / (α * (bw * d ^ 2 + 2 * K) ^ 2)) ≤ S / (2 * G) := by
  obtain ⟨⟨μ, hμ, hF⟩, -⟩ := isGreatest_F hbw hba hd hK hα
  beta_reduce at hF
  rw [← hF, one_div, ← div_eq_inv_mul, div_le_div_iff₀ (by positivity) (by positivity)]
  linarith [h μ hμ]

end ClosedForm

end Nci.SecurityProofs
