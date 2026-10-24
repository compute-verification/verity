import Definitions.Nci.Guarantees.Dampener
import Proofs.Nci.Threads

/-!
# Proofs of the compute dampener

The arithmetic core is `sources_le`: the four sources' bounds, summed, with the memory cap taken over the residual and
the declared computation together, give `η ≤ s + (1 − s) min(r, μ) + ν + χ`; the guarantee drops the factor `1 − s`.
The window optimization is `window_quadratic` (and `window_quadratic_any` at every window length, with its rounding term
exact) and `window_linear` (`Λ = e`, one window); `windows_cap` turns the servers' windows into the memory cap's
hypothesis. `DampenerThreadedSlots` discharges the declared computation's bound with `ThreadedThrottledSlots`, and
`DampenerThreaded` is its instance at one slot per weight gate.
-/

namespace Nci.SecurityProofs

open Verity Nci.Dampener

/-- **The four sources, summed** (the dampener's arithmetic core, its tight form): with the residual and the declared
computation under the memory cap together, `η ≤ s + (1 − s) min(r, μ) + ν + χ`. -/
theorem sources_le {κ C U W ρ μ χ M Mu Mr Md Mx : ℝ} (hκ : 0 < κ) (hC : 0 < C) (hU : 0 ≤ U) (hMd : 0 ≤ Md)
    (hM : M ≤ Mu + Mr + Md + Mx) (hu : Mu ≤ κ * U / 2) (hr : Mr ≤ κ * (C - W) / 2) (hd : Md ≤ C / ρ)
    (hx : Mx ≤ κ * χ * (C + U) / 2) (hμ : Mr + Md ≤ κ * μ * C / 2) :
    efficiency κ M C U ≤
      unmonitoredShare C U + (1 - unmonitoredShare C U) * min (residualShare W C) μ + declaredShare κ ρ C U + χ := by
  have hS : 0 < C + U := by linarith
  have hκi : κ * κ⁻¹ = 1 := mul_inv_cancel₀ hκ.ne'
  have hSi : (C + U) * (C + U)⁻¹ = 1 := mul_inv_cancel₀ hS.ne'
  have hCi : C * C⁻¹ = 1 := mul_inv_cancel₀ hC.ne'
  have hCρ : 0 ≤ C / ρ := hMd.trans hd
  have hmon : Mr + Md ≤ κ * C / 2 * min (residualShare W C) μ + C / ρ := by
    rcases le_total (residualShare W C) μ with h | h
    · rw [min_eq_left h]
      have e : κ * C / 2 * residualShare W C = κ * (C - W) / 2 := by
        unfold residualShare
        linear_combination (-(κ * W / 2)) * hCi
      linarith
    · rw [min_eq_right h]
      linarith
  have hB : 0 ≤ κ * U / 2 + κ * C / 2 * min (residualShare W C) μ + C / ρ + κ * χ * (C + U) / 2 - M := by
    linarith
  have key : unmonitoredShare C U + (1 - unmonitoredShare C U) * min (residualShare W C) μ + declaredShare κ ρ C U + χ -
      efficiency κ M C U =
      2 / κ / (C + U) * (κ * U / 2 + κ * C / 2 * min (residualShare W C) μ + C / ρ + κ * χ * (C + U) / 2 - M) := by
    simp only [declaredShare, unmonitoredShare, efficiency]
    linear_combination (-(U * (C + U)⁻¹ + C * min (residualShare W C) μ * (C + U)⁻¹ + χ * (C + U) * (C + U)⁻¹)) * hκi +
      (-(min (residualShare W C) μ + 2 * κ⁻¹ * ρ⁻¹ + χ)) * hSi
  have hfac : 0 ≤ 2 / κ / (C + U) := div_nonneg (div_nonneg zero_le_two hκ.le) hS.le
  linarith [mul_nonneg hfac hB]

/-- **The four sources, summed**, in the guarantee's form: `η ≤ s + min(r, μ) + ν + χ`. Both `r` and `μ` are
nonnegative here, since the residual and the declared computation hold no negative count. -/
theorem dampener_core {κ C U W ρ μ χ M Mu Mr Md Mx : ℝ} (hκ : 0 < κ) (hC : 0 < C) (hU : 0 ≤ U) (hMr : 0 ≤ Mr)
    (hMd : 0 ≤ Md) (hM : M ≤ Mu + Mr + Md + Mx) (hu : Mu ≤ κ * U / 2) (hr : Mr ≤ κ * (C - W) / 2)
    (hd : Md ≤ C / ρ) (hx : Mx ≤ κ * χ * (C + U) / 2) (hμ : Mr + Md ≤ κ * μ * C / 2) :
    efficiency κ M C U ≤ unmonitoredShare C U + min (residualShare W C) μ + declaredShare κ ρ C U + χ := by
  have h := sources_le hκ hC hU hMd hM hu hr hd hx hμ
  have hS : 0 < C + U := by linarith
  have hCi : C * C⁻¹ = 1 := mul_inv_cancel₀ hC.ne'
  have hs0 : 0 ≤ unmonitoredShare C U := div_nonneg hU hS.le
  have hr0 : 0 ≤ residualShare W C := by
    have hCW : 0 ≤ C - W := by
      by_contra hneg
      have h1 : κ * (C - W) < 0 := mul_neg_of_pos_of_neg hκ (not_le.1 hneg)
      linarith
    have e : residualShare W C = (C - W) * C⁻¹ := by
      unfold residualShare
      linear_combination (-1 : ℝ) * hCi
    rw [e]
    exact mul_nonneg hCW (inv_nonneg.2 hC.le)
  have hμ0 : 0 ≤ μ := by
    by_contra hneg
    have h1 : κ * μ * C < 0 := mul_neg_of_neg_of_pos (mul_neg_of_pos_of_neg hκ (not_le.1 hneg)) hC
    linarith
  have hm0 : 0 ≤ min (residualShare W C) μ := le_min hr0 hμ0
  have h2 : (1 - unmonitoredShare C U) * min (residualShare W C) μ ≤ min (residualShare W C) μ := by
    have := mul_nonneg hs0 hm0
    linarith
  linarith

theorem Dampener : Guarantees.Dampener := by
  intro κ C U W ρ μ χ M Mu Mr Md Mx hκ hC hU hM hu hr hd hx hμ
  exact dampener_core hκ hC hU (Nat.cast_nonneg Mr) (Nat.cast_nonneg Md) (by exact_mod_cast hM) hu hr hd hx hμ

/-- **The window optimization, quadratic regime** (the dampener's memory cap): a server's horizon `H ≥ 0` cut into
windows of length `L = R/λ`, each reading `R + λL = 2R` bits and holding at most `(2R)²/(2 d b_w b_a)` counted
multiplications (`K Λ(K)/b_a` with `Λ(K) = K/(2 d b_w)`, `D = d b_w b_a`), holds at most `2 R λ H/D` of them in all, plus
the rounding term `2 R²/D`: one window more, the last one, cut at `H`. -/
theorem window_quadratic {R l H D : ℝ} (hR : 0 < R) (hl : 0 < l) (hH : 0 ≤ H) (hD : 0 < D) :
    (⌈H / (R / l)⌉₊ : ℝ) * (R + l * (R / l)) ^ 2 / (2 * D) ≤ 2 * R * l * H / D + 2 * R ^ 2 / D := by
  have hli : l * l⁻¹ = 1 := mul_inv_cancel₀ hl.ne'
  have hRi : R * R⁻¹ = 1 := mul_inv_cancel₀ hR.ne'
  have hK : R + l * (R / l) = 2 * R := by linear_combination R * hli
  rw [hK, div_div_eq_mul_div]
  have hc : (⌈H * l / R⌉₊ : ℝ) < H * l / R + 1 := Nat.ceil_lt_add_one (div_nonneg (mul_nonneg hH hl.le) hR.le)
  calc (⌈H * l / R⌉₊ : ℝ) * (2 * R) ^ 2 / (2 * D)
      = (⌈H * l / R⌉₊ : ℝ) * (2 * R ^ 2 / D) := by ring
    _ ≤ (H * l / R + 1) * (2 * R ^ 2 / D) :=
        mul_le_mul_of_nonneg_right hc.le (div_nonneg (by positivity) hD.le)
    _ = 2 * R * l * H / D + 2 * R ^ 2 / D := by linear_combination (2 * H * l * R / D) * hRi

/-- **The window bound at every window length**, quadratic regime: windows of length `L > 0` over a horizon `H ≥ 0`,
each reading `R + λL` bits, hold at most `2 R λ H/D` counted multiplications plus `((R − λL)² H/L + (R + λL)²)/(2D)`.
The first part of that term is the cost of `L ≠ R/λ`, which `L = R/λ` makes zero; the second is the last window. -/
theorem window_quadratic_any {R l H L D : ℝ} (hL : 0 < L) (hH : 0 ≤ H) (hD : 0 < D) :
    (⌈H / L⌉₊ : ℝ) * (R + l * L) ^ 2 / (2 * D) ≤
      2 * R * l * H / D + ((R - l * L) ^ 2 * H / L + (R + l * L) ^ 2) / (2 * D) := by
  have hLi : L * L⁻¹ = 1 := mul_inv_cancel₀ hL.ne'
  have hc : (⌈H / L⌉₊ : ℝ) < H / L + 1 := Nat.ceil_lt_add_one (div_nonneg hH hL.le)
  calc (⌈H / L⌉₊ : ℝ) * (R + l * L) ^ 2 / (2 * D)
      = (⌈H / L⌉₊ : ℝ) * ((R + l * L) ^ 2 / (2 * D)) := by ring
    _ ≤ (H / L + 1) * ((R + l * L) ^ 2 / (2 * D)) :=
        mul_le_mul_of_nonneg_right hc.le (div_nonneg (sq_nonneg _) (by linarith))
    _ = 2 * R * l * H / D + ((R - l * L) ^ 2 * H / L + (R + l * L) ^ 2) / (2 * D) := by
        linear_combination (2 * R * l * H / D) * hLi

/-- **The window optimization, linear regime** (`Λ = e`): one window over the whole horizon `H > 0` reads `R + λH` bits
and holds at most `(R + λH) e/b_a` counted multiplications, the bound itself: the starting state counts once. -/
theorem window_linear {R l H e b : ℝ} (hH : 0 < H) :
    (⌈H / H⌉₊ : ℝ) * ((R + l * H) * e / b) = (R + l * H) * e / b := by
  rw [div_self hH.ne', Nat.ceil_one, Nat.cast_one, one_mul]

/-- **The window bound at every window length**, linear regime (`Λ = e`): windows of length `L > 0` over a horizon
`H ≥ 0` hold at most `(R + λH + RH/L + λL) e/b_a` counted multiplications, the starting state once a window: no window
length beats one window (`window_linear`) by more than the state's share. -/
theorem window_linear_any {R l H L e b : ℝ} (hL : 0 < L) (hH : 0 ≤ H) (hR : 0 ≤ R) (hl : 0 ≤ l) (he : 0 ≤ e)
    (hb : 0 < b) :
    (⌈H / L⌉₊ : ℝ) * ((R + l * L) * e / b) ≤ (R + l * H + R * H / L + l * L) * e / b := by
  have hLi : L * L⁻¹ = 1 := mul_inv_cancel₀ hL.ne'
  have hc : (⌈H / L⌉₊ : ℝ) < H / L + 1 := Nat.ceil_lt_add_one (div_nonneg hH hL.le)
  calc (⌈H / L⌉₊ : ℝ) * ((R + l * L) * e / b)
      ≤ (H / L + 1) * ((R + l * L) * e / b) :=
        mul_le_mul_of_nonneg_right hc.le (div_nonneg (mul_nonneg (by positivity) he) hb.le)
    _ = (R + l * H + R * H / L + l * L) * e / b := by linear_combination (l * H * e / b) * hLi

/-- **The memory cap from windows**: `n` servers' windows of `L` ticks, with `α F L = R` so that each reads `2R` bits,
in the quadratic regime of `Λ(2R)`, hold at most `μ (1 + L/H) κ C/2` counted multiplications when `n F H ≤ C`. -/
theorem windows_cap {κ C n F H L R α reuse e d b_w b_a X : ℝ} (hκ : 0 < κ) (hn : 0 ≤ n) (hH : 0 < H) (hL : 0 < L)
    (hR : 0 ≤ R) (hα : 0 ≤ α) (hreuse : 0 ≤ reuse) (hd : 0 < d) (hbw : 0 < b_w) (hba : 0 < b_a)
    (hC : n * F * H ≤ C) (hRL : α * F * L = R) (he : e ≤ reuse * (2 * R) / (4 * d * b_w))
    (hX : b_a * X ≤ n * ⌈H / L⌉₊ * ((R + α * F * L) * Lambda reuse e d b_w (R + α * F * L))) :
    X ≤ κ * (memoryCap reuse κ R α d b_w b_a * (1 + L / H)) * C / 2 := by
  have hκi : κ * κ⁻¹ = 1 := mul_inv_cancel₀ hκ.ne'
  have hLi : L * L⁻¹ = 1 := mul_inv_cancel₀ hL.ne'
  have hHi : H * H⁻¹ = 1 := mul_inv_cancel₀ hH.ne'
  have hK : R + α * F * L = 2 * R := by rw [hRL]; ring
  have hΛ : Lambda reuse e d b_w (2 * R) = reuse * (2 * R) / (4 * d * b_w) := max_eq_right he
  rw [hK, hΛ] at hX
  have hdb : 0 < d * b_w := mul_pos hd hbw
  have hQ : 0 ≤ reuse * R ^ 2 / (d * b_w) := div_nonneg (mul_nonneg hreuse (sq_nonneg R)) hdb.le
  have hP : 0 ≤ reuse * R * α / (d * b_w) := div_nonneg (mul_nonneg (mul_nonneg hreuse hR) hα) hdb.le
  have hLH : 0 ≤ 1 + L / H := by
    have := div_nonneg hL.le hH.le
    linarith
  have hceil : (⌈H / L⌉₊ : ℝ) ≤ (H + L) / L := by
    have hc : (⌈H / L⌉₊ : ℝ) < H / L + 1 := Nat.ceil_lt_add_one (div_nonneg hH.le hL.le)
    have e1 : H / L + 1 = (H + L) / L := by linear_combination (-1 : ℝ) * hLi
    linarith
  have h1 : b_a * X ≤ n * F * (H + L) * (reuse * R * α / (d * b_w)) := by
    calc b_a * X ≤ n * ⌈H / L⌉₊ * (2 * R * (reuse * (2 * R) / (4 * d * b_w))) := hX
      _ = n * ⌈H / L⌉₊ * (reuse * R ^ 2 / (d * b_w)) := by ring
      _ ≤ n * ((H + L) / L) * (reuse * R ^ 2 / (d * b_w)) :=
          mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hceil hn) hQ
      _ = n * F * (H + L) * (reuse * R * α / (d * b_w)) := by
          linear_combination (-(n * (H + L) * reuse * R / (d * b_w) / L)) * hRL +
            (n * (H + L) * reuse * R / (d * b_w) * (α * F)) * hLi
  have h2 : n * F * (H + L) ≤ C * (1 + L / H) := by
    have e2 : n * F * (H + L) = n * F * H * (1 + L / H) := by linear_combination (-(n * F * L)) * hHi
    rw [e2]
    exact mul_le_mul_of_nonneg_right hC hLH
  have h3 : b_a * X ≤ C * (1 + L / H) * (reuse * R * α / (d * b_w)) :=
    h1.trans (mul_le_mul_of_nonneg_right h2 hP)
  have e3 : κ * (memoryCap reuse κ R α d b_w b_a * (1 + L / H)) * C / 2 =
      C * (1 + L / H) * (reuse * R * α / (d * b_w)) / b_a := by
    unfold memoryCap
    linear_combination (reuse * R * α * C * (1 + L / H) / (d * b_w * b_a)) * hκi
  rw [e3, le_div_iff₀ hba]
  calc X * b_a = b_a * X := mul_comm _ _
    _ ≤ _ := h3

theorem DampenerWindows : Guarantees.DampenerWindows := by
  intro κ C U W ρ χ M Mu Mr Md Mx n F H L R α reuse e d b_w b_a hκ hC hU hH hL hR hα hreuse hd hbw hba hM hu hr hdecl
    hx hnC hRL he hcap
  exact Dampener κ C U W ρ (memoryCap reuse κ R α d b_w b_a * (1 + L / H)) χ M Mu Mr Md Mx hκ hC hU hM hu hr hdecl hx
    (windows_cap hκ (Nat.cast_nonneg n) hH hL hR hα hreuse hd hbw hba hnC hRL he hcap)

/-- `Lambda` is `Threading.bound` with real arguments. -/
theorem bound_eq_Lambda {c : Declaration} (T : Threading c) (S b_w : ℝ) (d e : ℕ) :
    T.bound S b_w d e = Lambda T.reuse e d b_w S := rfl

/-- **The declared computation at NCI's thread bound, per parameter slot**: `ThreadedThrottledSlots`'s
`b_a |M| ≤ α Λ size(d)`, with `size(d) ≤ C`, is the dampener's `M_d ≤ C/ρ_T`. -/
theorem declared_le_slots {Γ : ℕ} {α b_w b_a C : ℝ} {c d : Declaration} {T : Threading c} {σ : T.Slots} {dd e : ℕ}
    {Mc : Containment c.D d.D} (hα : 0 ≤ α) (hbw : 0 < b_w) (hba : 0 < b_a) (hdd : 0 < dd) (hcomp : Complies Γ α d)
    (hB : Assumptions.BaselineRegistered c d (PartialContainment.ofContainment Mc))
    (hT : T.Threaded Γ (α * Γ) dd e) (hW : Assumptions.WeightsIncompressibleSlots c T σ Γ (α * Γ) b_w b_a)
    (hsize : (d.D.size d.D.computed : ℝ) ≤ C) :
    (T.M.card : ℝ) ≤ C / rhoT α b_a (T.bound (α * Γ) b_w dd e) := by
  have hΛ : 0 ≤ T.bound (α * Γ) b_w dd e := le_max_of_le_left (Nat.cast_nonneg _)
  have hth := ThreadedThrottledSlots Γ α b_w b_a c d T σ dd e Mc hα hbw hba hdd hcomp hB hT hW
  unfold rhoT
  rw [div_div_eq_mul_div, le_div_iff₀ hba]
  calc (T.M.card : ℝ) * b_a = b_a * T.M.card := mul_comm _ _
    _ ≤ α * T.bound (α * Γ) b_w dd e * d.D.size d.D.computed := hth
    _ ≤ α * T.bound (α * Γ) b_w dd e * C := mul_le_mul_of_nonneg_left hsize (mul_nonneg hα hΛ)
    _ = C * (α * T.bound (α * Γ) b_w dd e) := mul_comm _ _

/-- **The declared computation at NCI's thread bound**: `declared_le_slots` at one slot per weight gate. -/
theorem declared_le {Γ : ℕ} {α b_w b_a C : ℝ} {c d : Declaration} {T : Threading c} {dd e : ℕ}
    {Mc : Containment c.D d.D} (hα : 0 ≤ α) (hbw : 0 < b_w) (hba : 0 < b_a) (hdd : 0 < dd) (hcomp : Complies Γ α d)
    (hB : Assumptions.BaselineRegistered c d (PartialContainment.ofContainment Mc)) (hT : T.Threaded Γ (α * Γ) dd e) (hW : Assumptions.WeightsIncompressible c T Γ (α * Γ) b_w b_a)
    (hsize : (d.D.size d.D.computed : ℝ) ≤ C) :
    (T.M.card : ℝ) ≤ C / rhoT α b_a (T.bound (α * Γ) b_w dd e) :=
  declared_le_slots hα hbw hba hdd hcomp hB hT (weightsIncompressibleSlots_of (Fit.idSlots T) hbw.le hW) hsize

theorem DampenerThreadedSlots : Guarantees.DampenerThreadedSlots := by
  intro κ C U W χ M Mu Mr Mx n Γ F H L R α b_w b_a c d T σ dd e Mc hκ hC hU hH hL hR hα hbw hba hdd hcomp hB hT hW
    hsize hM hu hr hx hnC hRL he hcap
  exact DampenerWindows κ C U W (rhoT α b_a (T.bound (α * Γ) b_w dd e)) χ M Mu Mr T.M.card Mx n F H L R α T.reuse e dd
    b_w b_a hκ hC hU hH hL hR hα (Nat.cast_nonneg _) (Nat.cast_pos.2 hdd) hbw hba hM hu hr
    (declared_le_slots hα hbw hba hdd hcomp hB hT hW hsize) hx hnC hRL he hcap

/-- `DampenerThreadedSlots` at one slot per weight gate. -/
theorem DampenerThreaded : Guarantees.DampenerThreaded := by
  intro κ C U W χ M Mu Mr Mx n Γ F H L R α b_w b_a c d T dd e Mc hκ hC hU hH hL hR hα hbw hba hdd hcomp hB hT hW
    hsize hM hu hr hx hnC hRL he hcap
  exact DampenerThreadedSlots κ C U W χ M Mu Mr Mx n Γ F H L R α b_w b_a c d T (Fit.idSlots T) dd e Mc hκ hC hU hH hL
    hR hα hbw hba hdd hcomp hB hT (weightsIncompressibleSlots_of (Fit.idSlots T) hbw.le hW) hsize hM hu hr hx hnC hRL he
    hcap

/-- **The bound is attained** (completeness, not a claim): at `κ = 2`, `C = 4`, `U = 0`, `W_cert = 3`, `ρ = 4`, `μ = 1`
and `χ = 1/4`, with one counted multiplication in each of the residual, the declared computation and a faulty proof
block, every hypothesis of `Nci.Guarantees.Dampener` holds and `η = 3/4 = s + min(r, μ) + ν + χ`: the hypotheses can
hold together, and the inequality is not loose there. -/
theorem dampener_attained :
    ((0 : ℕ) : ℝ) ≤ 2 * 0 / 2 ∧ ((1 : ℕ) : ℝ) ≤ 2 * (4 - 3) / 2 ∧ ((1 : ℕ) : ℝ) ≤ 4 / 4 ∧
      ((1 : ℕ) : ℝ) ≤ 2 * (1 / 4) * (4 + 0) / 2 ∧ ((1 : ℕ) : ℝ) + ((1 : ℕ) : ℝ) ≤ 2 * 1 * 4 / 2 ∧
      efficiency 2 ((3 : ℕ) : ℝ) 4 0 =
        unmonitoredShare 4 0 + min (residualShare 3 4) 1 + declaredShare 2 4 4 0 + 1 / 4 := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
  norm_num [efficiency, unmonitoredShare, residualShare, declaredShare, min_def]

end Nci.SecurityProofs
