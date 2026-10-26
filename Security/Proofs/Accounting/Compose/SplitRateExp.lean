import Definitions.Accounting.Guarantees.Compose

/-!
# The sharp rate: proof

`Accounting.Guarantees.Compose.SplitRateExp`. Write `x = W_F − L_F` and `y = W_0 − L_0` for the unlucky work and
`c = Kρw_max`. As `(1 − t)^K ≤ e^{−Kt}` (`one_sub_pow_le`), the left side is at most the exponential of
`(a/Λ − Kρ)·x + (a/Λ)·L_F + (b/Λ − Kρ)·y + (b/Λ)·L_0`, a linear form on the simplex
`x, L_F, y, L_0 ≥ 0`, `x + L_F + y + L_0 ≤ w_max`. Jensen for `exp` at the simplex's vertices (`exp_four_le`) bounds it by
the right side's affine interpolation of the vertex values:

* at `x = w_max`, `e^{a·w_max/Λ − c} − 1 ≤ −ν·p·w_max` is `e^{a·w_max/Λ}·(p + (1 − p)e^{−c}) ≤ 1`, which is
  `Λ ≥ a·w_max/−ln(p + (1 − p)e^{−c})` (`exp_mul_le_one`);
* at `L_F = w_max`, `e^{a·w_max/Λ} − 1 = ν·(1 − p)·w_max` is `ν`'s definition;

and the same for `y`, `L_0` with `b`, `p0`, `ν₀`.
-/

namespace Accounting.SecurityProofs.Compose.Rates

open Accounting.Compose

/-- Jensen for `exp` with the rest of the weight at `0`. -/
theorem exp_sum_le {ι : Type*} [Fintype ι] (s u : ι → ℝ) (hs : ∀ i, 0 ≤ s i) (h1 : ∑ i, s i ≤ 1) :
    Real.exp (∑ i, s i * u i) ≤ 1 + ∑ i, s i * (Real.exp (u i) - 1) := by
  have h := convexOn_exp.map_sum_le (t := (Finset.univ : Finset (Option ι)))
    (w := fun o => o.elim (1 - ∑ i, s i) s) (p := fun o => o.elim 0 u)
    (fun o _ => by cases o with
      | none => exact sub_nonneg.mpr h1
      | some i => exact hs i)
    (by simp [Fintype.sum_option]) (fun _ _ => Set.mem_univ _)
  simp only [Fintype.sum_option, Option.elim, smul_eq_mul, mul_zero, zero_add, Real.exp_zero, mul_one] at h
  calc Real.exp (∑ i, s i * u i) ≤ 1 - ∑ i, s i + ∑ i, s i * Real.exp (u i) := h
    _ = 1 + ∑ i, s i * (Real.exp (u i) - 1) := by
      simp only [mul_sub, mul_one, Finset.sum_sub_distrib]; ring

/-- `exp_sum_le` at four points. -/
theorem exp_four_le {s₁ s₂ s₃ s₄ : ℝ} (u₁ u₂ u₃ u₄ : ℝ) (h₁ : 0 ≤ s₁) (h₂ : 0 ≤ s₂) (h₃ : 0 ≤ s₃) (h₄ : 0 ≤ s₄)
    (hs : s₁ + s₂ + s₃ + s₄ ≤ 1) :
    Real.exp (s₁ * u₁ + s₂ * u₂ + s₃ * u₃ + s₄ * u₄) ≤
      1 + s₁ * (Real.exp u₁ - 1) + s₂ * (Real.exp u₂ - 1) + s₃ * (Real.exp u₃ - 1) + s₄ * (Real.exp u₄ - 1) := by
  have h := exp_sum_le ![s₁, s₂, s₃, s₄] ![u₁, u₂, u₃, u₄]
    (fun i => by fin_cases i <;> assumption) (by simpa [Fin.sum_univ_four] using hs)
  simp only [Fin.sum_univ_four] at h
  simpa [add_assoc] using h

/-- `(1 − t)^K ≤ e^{−Kt}` for `t ≤ 1`. -/
theorem one_sub_pow_le {t : ℝ} (ht : t ≤ 1) (K : ℕ) : (1 - t) ^ K ≤ Real.exp (-(K * t)) := by
  calc (1 - t) ^ K ≤ Real.exp (-t) ^ K := pow_le_pow_left₀ (by linarith) (Real.one_sub_le_exp_neg t) K
    _ = Real.exp (-(K * t)) := by rw [← Real.exp_nat_mul]; ring_nf

theorem mix_pos {p c : ℝ} (hp0 : 0 ≤ p) (hp1 : p < 1) : 0 < p + (1 - p) * Real.exp (-c) := by
  have := mul_pos (sub_pos.mpr hp1) (Real.exp_pos (-c))
  linarith

theorem mix_lt_one {p c : ℝ} (hp1 : p < 1) (hc : 0 < c) : p + (1 - p) * Real.exp (-c) < 1 := by
  have he : Real.exp (-c) < 1 := by rw [← Real.exp_zero]; exact Real.exp_lt_exp.mpr (by linarith)
  have := mul_lt_mul_of_pos_left he (sub_pos.mpr hp1)
  linarith

/-- A rate at least `A/−ln q` for `q ∈ (0, 1)` is positive, and `e^{A/Λ}·q ≤ 1`. -/
theorem exp_mul_le_one {q A Λ : ℝ} (hq0 : 0 < q) (hq1 : q < 1) (hA : 0 < A) (hΛ : A / -Real.log q ≤ Λ) :
    0 < Λ ∧ Real.exp (A / Λ) * q ≤ 1 := by
  have hL : 0 < -Real.log q := neg_pos.mpr (Real.log_neg hq0 hq1)
  have hΛ0 : 0 < Λ := lt_of_lt_of_le (div_pos hA hL) hΛ
  refine ⟨hΛ0, ?_⟩
  have h1 : A / Λ ≤ -Real.log q := by
    rw [div_le_iff₀ hΛ0]
    rw [div_le_iff₀ hL] at hΛ
    linarith [mul_comm Λ (-Real.log q)]
  calc Real.exp (A / Λ) * q ≤ Real.exp (-Real.log q) * q :=
        mul_le_mul_of_nonneg_right (Real.exp_le_exp.mpr h1) hq0.le
    _ = 1 := by rw [Real.exp_neg, Real.exp_log hq0]; field_simp

/-- One branch's vertex: at `e^{A}·(p + (1 − p)e^{−c}) ≤ 1`, `(1 − p)(e^{A − c} − 1) ≤ −p(e^A − 1)`. -/
theorem vertex_le {p c A : ℝ} (h : Real.exp A * (p + (1 - p) * Real.exp (-c)) ≤ 1) :
    (1 - p) * (Real.exp (A - c) - 1) ≤ -p * (Real.exp A - 1) := by
  have he : Real.exp (A - c) = Real.exp A * Real.exp (-c) := by rw [sub_eq_add_neg, Real.exp_add]
  rw [he]
  nlinarith

/-- The unlucky vertex's term: `(x/w)(e^{A − c} − 1) ≤ ν·(−p·x)` at `ν = (e^A − 1)/((1 − p)w)`. -/
theorem unlucky_le {p c A w x : ℝ} (hp1 : p < 1) (hw : 0 < w) (hx : 0 ≤ x)
    (h : Real.exp A * (p + (1 - p) * Real.exp (-c)) ≤ 1) :
    x / w * (Real.exp (A - c) - 1) ≤ (Real.exp A - 1) / ((1 - p) * w) * (-p * x) := by
  have hp : 0 < 1 - p := sub_pos.mpr hp1
  have key := vertex_le h
  have hid : (Real.exp A - 1) / ((1 - p) * w) * (-p * x) - x / w * (Real.exp A - Real.exp A + (Real.exp (A - c) - 1))
      = x / ((1 - p) * w) * (-p * (Real.exp A - 1) - (1 - p) * (Real.exp (A - c) - 1)) := by
    field_simp
    ring
  have hnn : 0 ≤ x / ((1 - p) * w) * (-p * (Real.exp A - 1) - (1 - p) * (Real.exp (A - c) - 1)) :=
    mul_nonneg (div_nonneg hx (mul_pos hp hw).le) (by linarith)
  have : Real.exp A - Real.exp A + (Real.exp (A - c) - 1) = Real.exp (A - c) - 1 := by ring
  rw [this] at hid
  linarith

/-- The lucky vertex's term: `(L/w)(e^A − 1) = ν·(1 − p)·L` at `ν = (e^A − 1)/((1 − p)w)`. -/
theorem lucky_eq {p A w L : ℝ} (hp1 : p < 1) (hw : 0 < w) :
    L / w * (Real.exp A - 1) = (Real.exp A - 1) / ((1 - p) * w) * ((1 - p) * L) := by
  have hp : (1 - p) ≠ 0 := (sub_pos.mpr hp1).ne'
  field_simp

theorem splitRateExp : Accounting.Guarantees.Compose.SplitRateExp := by
  intro K ρ p p0 a b wmax Λ hK hρ hw hρw hp0 hp1 hq0 hq1 ha hb hΛ
  have hK' : (1 : ℝ) ≤ K := by exact_mod_cast hK
  have hc : 0 < (K : ℝ) * ρ * wmax := by positivity
  have hΛF : a * wmax / -Real.log (p + (1 - p) * Real.exp (-(K * ρ * wmax))) ≤ Λ := by
    rw [hΛ]; exact le_max_left _ _
  have hΛ0 : b * wmax / -Real.log (p0 + (1 - p0) * Real.exp (-(K * ρ * wmax))) ≤ Λ := by
    rw [hΛ]; exact le_max_right _ _
  obtain ⟨hΛpos, hEF⟩ := exp_mul_le_one (mix_pos hp0 hp1) (mix_lt_one hp1 hc) (mul_pos ha hw) hΛF
  obtain ⟨-, hE0⟩ := exp_mul_le_one (mix_pos hq0 hq1) (mix_lt_one hq1 hc) (mul_pos hb hw) hΛ0
  have hEF1 : 1 ≤ Real.exp (a * wmax / Λ) := Real.one_le_exp (div_nonneg (mul_pos ha hw).le hΛpos.le)
  have hE01 : 1 ≤ Real.exp (b * wmax / Λ) := Real.one_le_exp (div_nonneg (mul_pos hb hw).le hΛpos.le)
  refine ⟨div_nonneg (sub_nonneg.mpr hEF1) (mul_pos (sub_pos.mpr hp1) hw).le,
    div_nonneg (sub_nonneg.mpr hE01) (mul_pos (sub_pos.mpr hq1) hw).le, ?_⟩
  intro WF W0 LF L0 hLF hLFW hL0 hL0W hsum
  have ht : ρ * ((WF - LF) + (W0 - L0)) ≤ 1 :=
    (mul_le_mul_of_nonneg_left (show (WF - LF) + (W0 - L0) ≤ wmax by linarith) hρ.le).trans hρw
  have hjensen := exp_four_le (a * wmax / Λ - K * ρ * wmax) (a * wmax / Λ) (b * wmax / Λ - K * ρ * wmax)
    (b * wmax / Λ) (div_nonneg (sub_nonneg.mpr hLFW) hw.le) (div_nonneg hLF hw.le)
    (div_nonneg (sub_nonneg.mpr hL0W) hw.le) (div_nonneg hL0 hw.le)
    (by rw [← add_div, ← add_div, ← add_div, div_le_one hw]; linarith)
  have harg : (WF - LF) / wmax * (a * wmax / Λ - K * ρ * wmax) + LF / wmax * (a * wmax / Λ) +
      (W0 - L0) / wmax * (b * wmax / Λ - K * ρ * wmax) + L0 / wmax * (b * wmax / Λ) =
      (a * WF + b * W0) / Λ + -(K * (ρ * ((WF - LF) + (W0 - L0)))) := by
    field_simp
    ring
  rw [harg] at hjensen
  have hF := unlucky_le (x := WF - LF) hp1 hw (sub_nonneg.mpr hLFW) hEF
  have h0 := unlucky_le (x := W0 - L0) hq1 hw (sub_nonneg.mpr hL0W) hE0
  have hLFeq := lucky_eq (L := LF) (A := a * wmax / Λ) hp1 hw
  have hL0eq := lucky_eq (L := L0) (A := b * wmax / Λ) hq1 hw
  calc Real.exp ((a * WF + b * W0) / Λ) * (1 - ρ * ((WF - LF) + (W0 - L0))) ^ K
      ≤ Real.exp ((a * WF + b * W0) / Λ) * Real.exp (-(K * (ρ * ((WF - LF) + (W0 - L0))))) :=
        mul_le_mul_of_nonneg_left (one_sub_pow_le ht K) (Real.exp_pos _).le
    _ = Real.exp ((a * WF + b * W0) / Λ + -(K * (ρ * ((WF - LF) + (W0 - L0))))) := (Real.exp_add _ _).symm
    _ ≤ _ := hjensen
    _ ≤ _ := by
      rw [hLFeq, hL0eq]
      linarith

end Accounting.SecurityProofs.Compose.Rates
