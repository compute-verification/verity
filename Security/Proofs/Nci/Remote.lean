import Definitions.Nci.Guarantees.Remote
import Proofs.Nci.Dampener

/-!
# Proofs of the compute dampener, remote protocol

`CertificateSlack` is Markov's inequality on a finite outcome space (`markov`), applied to the wrong windows' capacity,
whose expectation `expected_wrong` swaps into the windows' marginals. `VerdictLag` sums the skipped work by identity
(`Finset.sum_fiberwise_of_maps_to`): under `TerminalDetection` one identity's faulty blocks all lie within `Δ` of its
first, so they skip at most `Φ Δ`. `DampenerRemote` reads the residual's hypothesis outside `CertificateSlack`'s event
as `DampenerThreaded`'s at `W_cert − ε' C − (JΔ/H) C`, and sums the sources with no memory cap (`sources_le_at`,
unmonitored compute at efficiency `MFU₀`) into both the efficiency and the hardware-multiple forms (`remote_core`); the
event's complement has probability at least `1 − p/ε'` (`prob_ge_of_not`), `μ` summing to 1 (`prob_add_not`).
-/

namespace Nci.SecurityProofs

open Verity Nci.Dampener

/-- `prob` is monotone in the event. -/
theorem prob_mono {Ω : Type} [Fintype Ω] (μ : PMF Ω) {E F : Ω → Prop} (h : ∀ ω, E ω → F ω) :
    prob μ E ≤ prob μ F := by
  unfold prob
  refine Finset.sum_le_sum fun ω _ => ?_
  by_cases hE : E ω
  · simp [hE, h ω hE]
  · by_cases hF : F ω <;> simp [hE, hF]

/-- An event's probability and its complement's sum to 1. -/
theorem prob_add_not {Ω : Type} [Fintype Ω] (μ : PMF Ω) (E : Ω → Prop) :
    prob μ E + prob μ (fun ω => ¬ E ω) = 1 := by
  have h0 : ∑ ω, μ ω = 1 := by
    first
      | exact (tsum_fintype (L := SummationFilter.unconditional Ω) (fun ω => μ ω)).symm.trans μ.tsum_coe
      | simpa using μ.tsum_coe
  have h1 : ∑ ω, (μ ω).toReal = 1 := by
    have h := congrArg ENNReal.toReal h0
    rwa [ENNReal.toReal_sum fun ω _ => μ.apply_ne_top ω, ENNReal.toReal_one] at h
  unfold prob
  rw [← Finset.sum_add_distrib, ← h1]
  refine Finset.sum_congr rfl fun ω _ => ?_
  by_cases h : E ω <;> simp [h]

/-- An event whose complement lies inside an event of probability at most `δ` has probability at least `1 − δ`. -/
theorem prob_ge_of_not {Ω : Type} [Fintype Ω] (μ : PMF Ω) {G F : Ω → Prop} {δ : ℝ} (hGF : ∀ ω, ¬ G ω → F ω)
    (hF : prob μ F ≤ δ) : 1 - δ ≤ prob μ G := by
  have h1 := prob_add_not μ G
  have h2 := prob_mono μ hGF
  linarith

/-- **Markov's inequality** on a finite outcome space: for `Z ≥ 0`, `a · P(a < Z) ≤ E[Z]`. -/
theorem markov {Ω : Type} [Fintype Ω] (μ : PMF Ω) (Z : Ω → ℝ) (hZ : ∀ ω, 0 ≤ Z ω) (a : ℝ) :
    a * prob μ (fun ω => a < Z ω) ≤ ∑ ω, (μ ω).toReal * Z ω := by
  unfold prob
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun ω _ => ?_
  have hq : 0 ≤ (μ ω).toReal := ENNReal.toReal_nonneg
  by_cases h : a < Z ω
  · rw [if_pos h]
    exact (mul_comm a _).trans_le (mul_le_mul_of_nonneg_left h.le hq)
  · rw [if_neg h, mul_zero]
    exact mul_nonneg hq (hZ ω)

/-- The expected capacity of the wrong windows is the sum of each window's capacity times its probability of being
wrong. -/
theorem expected_wrong {Ω : Type} [Fintype Ω] (μ : PMF Ω) {N : ℕ} (c : Fin N → ℝ) (wrong : Ω → Fin N → Prop) :
    ∑ ω, (μ ω).toReal * wrongCapacity c (wrong ω) = ∑ i, c i * prob μ (fun ω => wrong ω i) := by
  unfold wrongCapacity prob
  simp only [Finset.mul_sum]
  refine Finset.sum_comm.trans (Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun ω _ => ?_)
  by_cases h : wrong ω i <;> simp [h, mul_comm]

theorem CertificateSlack : Guarantees.CertificateSlack := by
  intro Ω _ μ N c wrong p ε C hp hε hC hc hcC hsound
  have hZ : ∀ ω, 0 ≤ wrongCapacity c (wrong ω) := by
    intro ω
    unfold wrongCapacity
    refine Finset.sum_nonneg fun i _ => ?_
    by_cases h : wrong ω i <;> simp [h, hc i]
  have hEp : ∑ i, c i * prob μ (fun ω => wrong ω i) ≤ p * C :=
    calc ∑ i, c i * prob μ (fun ω => wrong ω i) ≤ ∑ i, c i * p :=
          Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (hsound i) (hc i)
      _ = (∑ i, c i) * p := (Finset.sum_mul _ _ _).symm
      _ ≤ C * p := mul_le_mul_of_nonneg_right hcC hp
      _ = p * C := mul_comm _ _
  have hm : ε * C * prob μ (fun ω => ε * C < wrongCapacity c (wrong ω)) ≤ p * C :=
    ((markov μ (fun ω => wrongCapacity c (wrong ω)) hZ (ε * C)).trans_eq (expected_wrong μ c wrong)).trans hEp
  rw [le_div_iff₀ hε]
  exact le_of_mul_le_mul_right (by linarith) hC

theorem VerdictLag : Guarantees.VerdictLag := by
  intro β ι _ bad I ident τ w Φ Δ H C hH hΔ hΦ hI hT hcap
  have hfib : ∀ j ∈ I, ∑ b ∈ bad with ident b = j, w b ≤ Φ * Δ := by
    intro j _
    by_cases hne : (bad.filter (fun b => ident b = j)).Nonempty
    · obtain ⟨b₀, hb₀, hmin⟩ := Finset.exists_min_image _ τ hne
      have hb₀' := Finset.mem_filter.1 hb₀
      have e : bad.filter (fun b => ident b = j) =
          bad.filter (fun b => ident b = j ∧ τ b₀ ≤ τ b ∧ τ b ≤ τ b₀ + Δ) := by
        refine Finset.filter_congr fun b hb => ⟨fun hj => ?_, fun h => h.1⟩
        exact ⟨hj, hmin b (Finset.mem_filter.2 ⟨hb, hj⟩), hT b₀ hb₀'.1 b hb (hb₀'.2.trans hj.symm)⟩
      rw [e]
      exact hcap j (τ b₀)
    · rw [Finset.not_nonempty_iff_eq_empty] at hne
      have e : bad.filter (fun b => ident b = j ∧ 0 ≤ τ b ∧ τ b ≤ 0 + Δ) = ∅ := by
        rw [Finset.eq_empty_iff_forall_notMem]
        intro b hb
        have hb' := Finset.mem_filter.1 hb
        have hmem : b ∈ bad.filter (fun b => ident b = j) := Finset.mem_filter.2 ⟨hb'.1, hb'.2.1⟩
        rw [hne] at hmem
        exact Finset.notMem_empty b hmem
      have h0 := hcap j 0
      rw [e, Finset.sum_empty] at h0
      rw [hne, Finset.sum_empty]
      exact h0
  have hΦC : Φ ≤ C / H := by
    rw [le_div_iff₀ hH]
    exact hΦ
  have hJ : (0 : ℝ) ≤ I.card * Δ := mul_nonneg (Nat.cast_nonneg _) hΔ
  calc ∑ b ∈ bad, w b = ∑ j ∈ I, ∑ b ∈ bad with ident b = j, w b := (Finset.sum_fiberwise_of_maps_to hI w).symm
    _ ≤ ∑ _j ∈ I, Φ * Δ := Finset.sum_le_sum hfib
    _ = I.card * Δ * Φ := by
        rw [Finset.sum_const, nsmul_eq_mul]
        ring
    _ ≤ I.card * Δ * (C / H) := mul_le_mul_of_nonneg_left hΦC hJ
    _ = lagShare I.card Δ H * C := by
        unfold lagShare
        ring

/-- **The sources, summed with no memory cap**, unmonitored compute at efficiency `a` (`M_U ≤ a κ U/2`):
`η ≤ a s + (1 − s) r + ν + χ`. At `a = 1` it is `sources_le` with a memory cap that never binds. -/
theorem sources_le_at {κ C U W ρ χ a M Mu Mr Md Mx : ℝ} (hκ : 0 < κ) (hC : 0 < C) (hU : 0 ≤ U)
    (hM : M ≤ Mu + Mr + Md + Mx) (hu : Mu ≤ a * κ * U / 2) (hr : Mr ≤ κ * (C - W) / 2) (hd : Md ≤ C / ρ)
    (hx : Mx ≤ κ * χ * (C + U) / 2) :
    efficiency κ M C U ≤
      a * unmonitoredShare C U + (1 - unmonitoredShare C U) * residualShare W C + declaredShare κ ρ C U + χ := by
  have hS : 0 < C + U := by linarith
  have hκi : κ * κ⁻¹ = 1 := mul_inv_cancel₀ hκ.ne'
  have hSi : (C + U) * (C + U)⁻¹ = 1 := mul_inv_cancel₀ hS.ne'
  have hCi : C * C⁻¹ = 1 := mul_inv_cancel₀ hC.ne'
  have e : κ * C / 2 * residualShare W C = κ * (C - W) / 2 := by
    unfold residualShare
    linear_combination (-(κ * W / 2)) * hCi
  have hB : 0 ≤ a * κ * U / 2 + κ * C / 2 * residualShare W C + C / ρ + κ * χ * (C + U) / 2 - M := by
    linarith
  have key : a * unmonitoredShare C U + (1 - unmonitoredShare C U) * residualShare W C + declaredShare κ ρ C U + χ -
      efficiency κ M C U =
      2 / κ / (C + U) * (a * κ * U / 2 + κ * C / 2 * residualShare W C + C / ρ + κ * χ * (C + U) / 2 - M) := by
    simp only [declaredShare, unmonitoredShare, efficiency]
    linear_combination
      (-(a * U * (C + U)⁻¹ + C * residualShare W C * (C + U)⁻¹ + χ * (C + U) * (C + U)⁻¹)) * hκi +
      (-(residualShare W C + 2 * κ⁻¹ * ρ⁻¹ + χ)) * hSi
  have hfac : 0 ≤ 2 / κ / (C + U) := div_nonneg (div_nonneg zero_le_two hκ.le) hS.le
  linarith [mul_nonneg hfac hB]

/-- **From an efficiency bound to the hardware multiple**: `η ≤ MFU₀ s + x` gives
`max(1, 1/(s + x/MFU₀)) ≤ max(1, MFU₀/η)`. -/
theorem multiple_le {a s x η : ℝ} (ha : 0 < a) (hη : 0 < η) (h : η ≤ a * s + x) :
    max 1 (1 / (s + x / a)) ≤ max 1 (a / η) := by
  refine max_le_max le_rfl ?_
  rcases le_or_gt (s + x / a) 0 with h0 | h0
  · have h1 : 1 / (s + x / a) ≤ 0 := by
      rw [one_div]
      exact inv_nonpos.2 h0
    exact h1.trans (div_pos ha hη).le
  · have hai : a * a⁻¹ = 1 := mul_inv_cancel₀ ha.ne'
    have e : a * (s + x / a) = a * s + x := by linear_combination x * hai
    rw [div_le_div_iff₀ h0 hη, one_mul, e]
    exact h

/-- **The remote dampener's arithmetic core**: the sources summed with no memory cap and unmonitored compute at
`MFU₀ ≤ 1` bound both the efficiency, `η ≤ s + (1 − s) r + ν + χ`, and the hardware multiple,
`max(1, 1/(s + ((1 − s) r + ν + χ)/MFU₀)) ≤ max(1, MFU₀/η)`. -/
theorem remote_core {κ C U W ρ χ a M Mu Mr Md Mx : ℝ} (hκ : 0 < κ) (hC : 0 < C) (hU : 0 ≤ U) (ha : 0 < a)
    (ha1 : a ≤ 1) (hM0 : 0 < M) (hM : M ≤ Mu + Mr + Md + Mx) (hu : Mu ≤ a * κ * U / 2)
    (hr : Mr ≤ κ * (C - W) / 2) (hd : Md ≤ C / ρ) (hx : Mx ≤ κ * χ * (C + U) / 2) :
    efficiency κ M C U ≤
        unmonitoredShare C U + (1 - unmonitoredShare C U) * residualShare W C + declaredShare κ ρ C U + χ ∧
      remoteDampener a (unmonitoredShare C U)
          ((1 - unmonitoredShare C U) * residualShare W C + declaredShare κ ρ C U + χ) ≤
        hardwareMultiple a κ M C U := by
  have h := sources_le_at hκ hC hU hM hu hr hd hx
  have hS : 0 < C + U := by linarith
  have hs0 : 0 ≤ unmonitoredShare C U := div_nonneg hU hS.le
  have h2 : a * unmonitoredShare C U ≤ unmonitoredShare C U := by
    have := mul_le_mul_of_nonneg_right ha1 hs0
    linarith
  have hη : 0 < efficiency κ M C U := by
    unfold efficiency
    exact div_pos (mul_pos (div_pos two_pos hκ) hM0) hS
  exact ⟨by linarith, multiple_le ha hη (by linarith)⟩

/-- The residual read at the work certified correctly and run, `W_cert − ε' C − (JΔ/H) C`, is `r̂ + ε' + JΔ/H`. -/
theorem residualShare_sub {W C ε L : ℝ} (hC : 0 < C) :
    residualShare (W - ε * C - L * C) C = residualShare W C + ε + L := by
  have hCi : C * C⁻¹ = 1 := mul_inv_cancel₀ hC.ne'
  unfold residualShare
  linear_combination (ε + L) * hCi

theorem DampenerRemote : Guarantees.DampenerRemote := by
  intro Ω _ μ N c wrong p ε β ι _ bad I ident τ w Φ Δ H κ C U W χ mfu M Mu Mr Mx Γ α b_w b_a r d T dd e Mc
    hp hε hc hcC hsound hH hΔ hΦ hI hT hcap hκ hC hU hmfu hmfu1 hα hbw hba hdd hcomp hB hthr hW hsize hM0 hM hu hres
    hx
  refine prob_ge_of_not μ ?_ (CertificateSlack Ω μ N c wrong p ε C hp hε hC hc hcC hsound)
  intro ω hbad
  by_contra hgood
  apply hbad
  have hwrong : wrongCapacity c (wrong ω) ≤ ε * C := not_lt.1 hgood
  have hlag := VerdictLag β ι (bad ω) I ident τ w Φ Δ H C hH hΔ hΦ (hI ω) (hT ω) (hcap ω)
  have hr : (Mr ω : ℝ) ≤ κ * (C - (W - ε * C - lagShare I.card Δ H * C)) / 2 := by
    have h1 := hres ω
    have h2 : κ * (C - W + wrongCapacity c (wrong ω) + ∑ b ∈ bad ω, w b) ≤
        κ * (C - (W - ε * C - lagShare I.card Δ H * C)) :=
      mul_le_mul_of_nonneg_left (by linarith) hκ.le
    linarith
  obtain ⟨h1, h2⟩ := remote_core hκ hC hU hmfu hmfu1 (Nat.cast_pos.2 (hM0 ω)) (by exact_mod_cast hM ω) (hu ω) hr
    (declared_le hα hbw hba hdd hcomp hB hthr hW hsize) (hx ω)
  rw [residualShare_sub hC] at h1 h2
  exact ⟨h1, h2⟩

end Nci.SecurityProofs
