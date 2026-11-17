module

public import Definitions.Slowdown.Guarantees.Slowdown
public import Proofs.Slowdown.Threads

/-! Proofs of the slowdown. -/

@[expose] public section

open InferenceOnly

namespace Slowdown.SecurityProofs

open Verity Slowdown.Slowdown

theorem sources_le {κ C U W ρ μ χ M Mu Mr Md Mx : ℝ} (hκ : 0 < κ) (hC : 0 < C) (hU : 0 ≤ U) (hMd : 0 ≤ Md)
    (hM : M ≤ Mu + Mr + Md + Mx) (hu : Mu ≤ κ * U / 2) (hr : Mr ≤ κ * (C - W) / 2) (hd : Md ≤ C / ρ)
    (hx : Mx ≤ κ * χ * (C + U) / 2) (hμ : Mr + Md ≤ κ * μ * C / 2) :
    efficiency κ M C U ≤
      unregisteredShare C U + (1 - unregisteredShare C U) * min (residualShare W C) μ + declaredShare κ ρ C U + χ := by
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
  have key : unregisteredShare C U + (1 - unregisteredShare C U) * min (residualShare W C) μ + declaredShare κ ρ C U + χ -
      efficiency κ M C U =
      2 / κ / (C + U) * (κ * U / 2 + κ * C / 2 * min (residualShare W C) μ + C / ρ + κ * χ * (C + U) / 2 - M) := by
    simp only [declaredShare, unregisteredShare, efficiency]
    linear_combination (-(U * (C + U)⁻¹ + C * min (residualShare W C) μ * (C + U)⁻¹ + χ * (C + U) * (C + U)⁻¹)) * hκi +
      (-(min (residualShare W C) μ + 2 * κ⁻¹ * ρ⁻¹ + χ)) * hSi
  have hfac : 0 ≤ 2 / κ / (C + U) := div_nonneg (div_nonneg zero_le_two hκ.le) hS.le
  linarith [mul_nonneg hfac hB]

theorem slowdown_core {κ C U W ρ μ χ M Mu Mr Md Mx : ℝ} (hκ : 0 < κ) (hC : 0 < C) (hU : 0 ≤ U) (hMr : 0 ≤ Mr)
    (hMd : 0 ≤ Md) (hM : M ≤ Mu + Mr + Md + Mx) (hu : Mu ≤ κ * U / 2) (hr : Mr ≤ κ * (C - W) / 2)
    (hd : Md ≤ C / ρ) (hx : Mx ≤ κ * χ * (C + U) / 2) (hμ : Mr + Md ≤ κ * μ * C / 2) :
    efficiency κ M C U ≤ unregisteredShare C U + min (residualShare W C) μ + declaredShare κ ρ C U + χ := by
  have h := sources_le hκ hC hU hMd hM hu hr hd hx hμ
  have hS : 0 < C + U := by linarith
  have hCi : C * C⁻¹ = 1 := mul_inv_cancel₀ hC.ne'
  have hs0 : 0 ≤ unregisteredShare C U := div_nonneg hU hS.le
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
  have h2 : (1 - unregisteredShare C U) * min (residualShare W C) μ ≤ min (residualShare W C) μ := by
    have := mul_nonneg hs0 hm0
    linarith
  linarith

theorem Slowdown : Guarantees.Slowdown := by
  intro κ C U W ρ μ χ M Mu Mr Md Mx hκ hC hU hM hu hr hd hx hμ
  exact slowdown_core hκ hC hU (Nat.cast_nonneg Mr) (Nat.cast_nonneg Md) (by exact_mod_cast hM) hu hr hd hx hμ

theorem executionPhase_quadratic {R l H D : ℝ} (hR : 0 < R) (hl : 0 < l) (hH : 0 ≤ H) (hD : 0 < D) :
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

theorem executionPhase_quadratic_any {R l H L D : ℝ} (hL : 0 < L) (hH : 0 ≤ H) (hD : 0 < D) :
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

theorem executionPhase_linear {R l H e b : ℝ} (hH : 0 < H) :
    (⌈H / H⌉₊ : ℝ) * ((R + l * H) * e / b) = (R + l * H) * e / b := by
  rw [div_self hH.ne', Nat.ceil_one, Nat.cast_one, one_mul]

theorem executionPhase_linear_any {R l H L e b : ℝ} (hL : 0 < L) (hH : 0 ≤ H) (hR : 0 ≤ R) (hl : 0 ≤ l) (he : 0 ≤ e)
    (hb : 0 < b) :
    (⌈H / L⌉₊ : ℝ) * ((R + l * L) * e / b) ≤ (R + l * H + R * H / L + l * L) * e / b := by
  have hLi : L * L⁻¹ = 1 := mul_inv_cancel₀ hL.ne'
  have hc : (⌈H / L⌉₊ : ℝ) < H / L + 1 := Nat.ceil_lt_add_one (div_nonneg hH hL.le)
  calc (⌈H / L⌉₊ : ℝ) * ((R + l * L) * e / b)
      ≤ (H / L + 1) * ((R + l * L) * e / b) :=
        mul_le_mul_of_nonneg_right hc.le (div_nonneg (mul_nonneg (by positivity) he) hb.le)
    _ = (R + l * H + R * H / L + l * L) * e / b := by linear_combination (l * H * e / b) * hLi

theorem executionPhases_cap {κ C n F H L R α reuse e d b_w b_a X : ℝ} (hκ : 0 < κ) (hn : 0 ≤ n) (hH : 0 < H) (hL : 0 < L)
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

theorem SlowdownExecutionPhases : Guarantees.SlowdownExecutionPhases := by
  intro κ C U W ρ χ M Mu Mr Md Mx n F H L R α reuse e d b_w b_a hκ hC hU hH hL hR hα hreuse hd hbw hba hM hu hr hdecl
    hx hnC hRL he hcap
  exact Slowdown κ C U W ρ (memoryCap reuse κ R α d b_w b_a * (1 + L / H)) χ M Mu Mr Md Mx hκ hC hU hM hu hr hdecl hx
    (executionPhases_cap hκ (Nat.cast_nonneg n) hH hL hR hα hreuse hd hbw hba hnC hRL he hcap)

theorem bound_eq_Lambda {c : RegisteredCircuit} (T : Threading c) (S b_w : ℝ) (d e : ℕ) :
    T.bound S b_w d e = Lambda T.reuse e d b_w S := rfl

theorem declared_le_slots {Γ : ℕ} {α b_w b_a C : ℝ} {c d : RegisteredCircuit} {T : Threading c} {σ : T.Slots} {dd e : ℕ}
    {Mc : Containment c.D d.D} (hα : 0 ≤ α) (hbw : 0 < b_w) (hba : 0 < b_a) (hdd : 0 < dd) (hcomp : CompliesRate Γ α d)
    (hB : Assumptions.ThetaRegistered c d (PartialContainment.ofContainment Mc))
    (hT : T.Threaded Γ (α * Γ) dd e) (hW : Assumptions.WeightsIncompressibleSlots c T σ Γ (α * Γ) b_w b_a)
    (hsize : (d.D.size d.D.computed : ℝ) ≤ C) :
    (T.M.card : ℝ) ≤ C / rhoT α b_a (T.bound (α * Γ) b_w dd e) := by
  have hΛ : 0 ≤ T.bound (α * Γ) b_w dd e := le_max_of_le_left (Nat.cast_nonneg _)
  have hth := ThreadedThrottledSlots_rate Γ α b_w b_a c d T σ dd e Mc hα hbw hba hdd hcomp hB hT hW
  unfold rhoT
  rw [div_div_eq_mul_div, le_div_iff₀ hba]
  calc (T.M.card : ℝ) * b_a = b_a * T.M.card := mul_comm _ _
    _ ≤ α * T.bound (α * Γ) b_w dd e * d.D.size d.D.computed := hth
    _ ≤ α * T.bound (α * Γ) b_w dd e * C := mul_le_mul_of_nonneg_left hsize (mul_nonneg hα hΛ)
    _ = C * (α * T.bound (α * Γ) b_w dd e) := mul_comm _ _

theorem SlowdownThreadedSlots_rate :
    ∀ (κ C U W χ : ℝ) (M Mu Mr Mx n Γ : ℕ) (F H L R α b_w b_a : ℝ) (c d : RegisteredCircuit) (T : Threading c)
      (σ : T.Slots) (dd e : ℕ) (Mc : Containment c.D d.D),
      0 < κ → 0 < C → 0 ≤ U → 0 < H → 0 < L → 0 ≤ R → 0 ≤ α → 0 < b_w → 0 < b_a → 0 < dd →
      CompliesRate Γ α d → Assumptions.ThetaRegistered c d (PartialContainment.ofContainment Mc) →
      T.Threaded Γ (α * Γ) dd e →
      Assumptions.WeightsIncompressibleSlots c T σ Γ (α * Γ) b_w b_a → (d.D.size d.D.computed : ℝ) ≤ C →
      M ≤ Mu + Mr + T.M.card + Mx →
      (Mu : ℝ) ≤ κ * U / 2 → (Mr : ℝ) ≤ κ * (C - W) / 2 → (Mx : ℝ) ≤ κ * χ * (C + U) / 2 →
      n * F * H ≤ C → α * F * L = R → (e : ℝ) ≤ T.reuse * (2 * R) / (4 * dd * b_w) →
      b_a * ((Mr : ℝ) + T.M.card) ≤ n * ⌈H / L⌉₊ * ((R + α * F * L) * T.bound (R + α * F * L) b_w dd e) →
        efficiency κ M C U ≤
          unregisteredShare C U + min (residualShare W C) (memoryCap T.reuse κ R α dd b_w b_a * (1 + L / H)) +
            declaredShare κ (rhoT α b_a (T.bound (α * Γ) b_w dd e)) C U + χ := by
  intro κ C U W χ M Mu Mr Mx n Γ F H L R α b_w b_a c d T σ dd e Mc hκ hC hU hH hL hR hα hbw hba hdd hcomp hB hT hW
    hsize hM hu hr hx hnC hRL he hcap
  exact SlowdownExecutionPhases κ C U W (rhoT α b_a (T.bound (α * Γ) b_w dd e)) χ M Mu Mr T.M.card Mx n F H L R α T.reuse e dd
    b_w b_a hκ hC hU hH hL hR hα (Nat.cast_nonneg _) (Nat.cast_pos.2 hdd) hbw hba hM hu hr
    (declared_le_slots hα hbw hba hdd hcomp hB hT hW hsize) hx hnC hRL he hcap

theorem SlowdownThreadedSlots : Guarantees.SlowdownThreadedSlots :=
  SlowdownThreadedSlots_rate

/-- `SlowdownThreadedSlots` at one slot per weight gate. -/
theorem SlowdownThreaded : Guarantees.SlowdownThreaded := by
  intro κ C U W χ M Mu Mr Mx n Γ F H L R α b_w b_a r d T dd e Mc hκ hC hU hH hL hR hα hbw hba hdd hcomp hB hT hW
    hsize hM hu hr hx hnC hRL he hcap
  exact SlowdownThreadedSlots κ C U W χ M Mu Mr Mx n Γ F H L R α b_w b_a r d T (Fit.idSlots T) dd e Mc hκ hC hU hH hL
    hR hα hbw hba hdd hcomp hB hT (weightsIncompressibleSlots_of (Fit.idSlots T) hbw.le hW) hsize hM hu hr hx hnC hRL he
    hcap

/-- Completeness, not a claim: at `κ = 2`, `C = 4`, `U = 0`, `W_cert = 3`, `ρ = 4`, `μ = 1` and `χ = 1/4` every
hypothesis holds and the inequality is an equality. -/
theorem slowdown_attained :
    ((0 : ℕ) : ℝ) ≤ 2 * 0 / 2 ∧ ((1 : ℕ) : ℝ) ≤ 2 * (4 - 3) / 2 ∧ ((1 : ℕ) : ℝ) ≤ 4 / 4 ∧
      ((1 : ℕ) : ℝ) ≤ 2 * (1 / 4) * (4 + 0) / 2 ∧ ((1 : ℕ) : ℝ) + ((1 : ℕ) : ℝ) ≤ 2 * 1 * 4 / 2 ∧
      efficiency 2 ((3 : ℕ) : ℝ) 4 0 =
        unregisteredShare 4 0 + min (residualShare 3 4) 1 + declaredShare 2 4 4 0 + 1 / 4 := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
  norm_num [efficiency, unregisteredShare, residualShare, declaredShare, min_def]

end Slowdown.SecurityProofs
