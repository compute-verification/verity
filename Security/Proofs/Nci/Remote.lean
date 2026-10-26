import Definitions.Nci.Guarantees.Remote
import Proofs.Nci.Dampener
import Proofs.Nci.Escape

/-!
# Proofs of the compute dampener, remote protocol

`CertificateSlack` is Markov's inequality on a finite outcome space (`markov`), applied to the wrong windows' capacity,
whose expectation `expected_wrong` swaps into the windows' marginals. `VerdictLag` splits off the escaped blocks
(`Finset.sum_sdiff`) and sums the rest by identity (`Finset.sum_fiberwise_of_maps_to`): under `TerminalDetection` one
identity's faulty blocks at or after its first detected one all lie within `Δ` of it, so they skip at most `Φ Δ`.
`escape_slack` is Markov's inequality again, on the escaped blocks' skipped work, whose expectation `expected_escaped`
sums the per-identity bound `sampled_escape` derives from `SampledDetection`. `DampenerRemote` reads the residual's
hypothesis outside both events as `DampenerThreaded`'s at `(1 − ε_W) W_cert − ε' C − (JΔ/H) C − θ C`, and sums the
sources with no memory cap (`sources_le_at`, unmonitored compute at efficiency `MFU₀`) into both the efficiency and the
hardware-multiple forms (`remote_core`); the union of the two events has probability at most the sum of theirs
(`prob_or_le`), so the complement has probability at least `1 − p/ε' − J ((1 − q)/q) B/(θ C)` (`prob_ge_of_not`), `μ`
summing to 1 (`prob_add_not`).
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

/-- The union bound: `P(E ∨ F) ≤ P(E) + P(F)`. -/
theorem prob_or_le {Ω : Type} [Fintype Ω] (μ : PMF Ω) (E F : Ω → Prop) :
    prob μ (fun ω => E ω ∨ F ω) ≤ prob μ E + prob μ F := by
  unfold prob
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun ω _ => ?_
  have hq : 0 ≤ (μ ω).toReal := ENNReal.toReal_nonneg
  by_cases hE : E ω <;> by_cases hF : F ω <;> simp [hE, hF] <;> linarith

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

/-- A block is escaped when it is faulty and no detected faulty block of its identity runs at or before it. -/
theorem mem_escaped {β ι : Type} {bad detected : Finset β} {ident : β → ι} {τ : β → ℝ} {b : β} :
    b ∈ escaped bad detected ident τ ↔ b ∈ bad ∧ ∀ b' ∈ bad, b' ∈ detected → ident b' = ident b → τ b < τ b' := by
  simp only [escaped, Finset.mem_filter]

theorem VerdictLag : Guarantees.VerdictLag := by
  intro β ι _ bad detected I ident τ w Φ Δ H C hH hΔ hΦ hI hT hcap
  classical
  have hEsub : escaped bad detected ident τ ⊆ bad := fun b hb => (mem_escaped.1 hb).1
  have hfib : ∀ j ∈ I, ∑ b ∈ bad \ escaped bad detected ident τ with ident b = j, w b ≤ Φ * Δ := by
    intro j _
    by_cases hne : (bad.filter (fun b => b ∈ detected ∧ ident b = j)).Nonempty
    · obtain ⟨b₀, hb₀, hmin⟩ := Finset.exists_min_image _ τ hne
      obtain ⟨hb₀bad, hb₀det, hb₀j⟩ := Finset.mem_filter.1 hb₀
      have e : (bad \ escaped bad detected ident τ).filter (fun b => ident b = j) =
          bad.filter (fun b => ident b = j ∧ τ b₀ ≤ τ b ∧ τ b ≤ τ b₀ + Δ) := by
        ext b
        simp only [Finset.mem_filter, Finset.mem_sdiff]
        constructor
        · rintro ⟨⟨hb, hbE⟩, hj⟩
          have hn : ¬ ∀ b' ∈ bad, b' ∈ detected → ident b' = ident b → τ b < τ b' :=
            fun h => hbE (mem_escaped.2 ⟨hb, h⟩)
          push_neg at hn
          obtain ⟨b', hb', hdet, hid, hle⟩ := hn
          have h0 : τ b₀ ≤ τ b := (hmin b' (Finset.mem_filter.2 ⟨hb', hdet, hid.trans hj⟩)).trans hle
          exact ⟨hb, hj, h0, hT b₀ hb₀bad hb₀det b hb (hb₀j.trans hj.symm) h0⟩
        · rintro ⟨hb, hj, h0, -⟩
          refine ⟨⟨hb, fun hbE => ?_⟩, hj⟩
          exact absurd ((mem_escaped.1 hbE).2 b₀ hb₀bad hb₀det (hb₀j.trans hj.symm)) (not_lt.2 h0)
      rw [e]
      exact hcap j (τ b₀)
    · have e : (bad \ escaped bad detected ident τ).filter (fun b => ident b = j) = ∅ := by
        rw [Finset.filter_eq_empty_iff]
        intro b hb hj
        refine (Finset.mem_sdiff.1 hb).2 (mem_escaped.2 ⟨(Finset.mem_sdiff.1 hb).1, fun b' hb' hdet hid => ?_⟩)
        exact (hne ⟨b', Finset.mem_filter.2 ⟨hb', hdet, hid.trans hj⟩⟩).elim
      rw [e, Finset.sum_empty]
      obtain ⟨t, ht⟩ : ∃ t, ∀ b ∈ bad, τ b < t := ⟨∑ b ∈ bad, |τ b| + 1, fun b hb =>
        lt_of_le_of_lt ((le_abs_self (τ b)).trans (Finset.single_le_sum (fun b _ => abs_nonneg (τ b)) hb))
          (lt_add_one _)⟩
      have e' : bad.filter (fun b => ident b = j ∧ t ≤ τ b ∧ τ b ≤ t + Δ) = ∅ := by
        rw [Finset.filter_eq_empty_iff]
        intro b hb h
        exact absurd h.2.1 (not_le.2 (ht b hb))
      have h0 := hcap j t
      rw [e', Finset.sum_empty] at h0
      exact h0
  have hΦC : Φ ≤ C / H := by
    rw [le_div_iff₀ hH]
    exact hΦ
  have hJ : (0 : ℝ) ≤ I.card * Δ := mul_nonneg (Nat.cast_nonneg _) hΔ
  have hmaps : ∀ b ∈ bad \ escaped bad detected ident τ, ident b ∈ I := fun b hb => hI b (Finset.mem_sdiff.1 hb).1
  have hrest : ∑ b ∈ bad \ escaped bad detected ident τ, w b ≤ lagShare I.card Δ H * C :=
    calc ∑ b ∈ bad \ escaped bad detected ident τ, w b
        = ∑ j ∈ I, ∑ b ∈ bad \ escaped bad detected ident τ with ident b = j, w b :=
          (Finset.sum_fiberwise_of_maps_to hmaps w).symm
      _ ≤ ∑ _j ∈ I, Φ * Δ := Finset.sum_le_sum hfib
      _ = I.card * Δ * Φ := by
          rw [Finset.sum_const, nsmul_eq_mul]
          ring
      _ ≤ I.card * Δ * (C / H) := mul_le_mul_of_nonneg_left hΦC hJ
      _ = lagShare I.card Δ H * C := by
          unfold lagShare
          ring
  have hsplit := Finset.sum_sdiff hEsub (f := w)
  linarith

/-- The expected work the escaped blocks skip is at most `J ((1 − q)/q) B`: `sampled_escape`'s bound, summed over the
identities. -/
theorem expected_escaped {Ω β ι : Type} [Fintype Ω] [DecidableEq ι] (μ : PMF Ω) (bad detected : Ω → Finset β)
    (I : Finset ι) (ident : β → ι) (τ w : β → ℝ) (q B : ℝ) (hI : ∀ ω, ∀ b ∈ bad ω, ident b ∈ I) (hq : 0 < q)
    (hq1 : q ≤ 1) (hB : 0 ≤ B) (hw : ∀ b, 0 ≤ w b) (hwB : ∀ ω, ∀ b ∈ bad ω, w b ≤ B)
    (hS : Assumptions.SampledDetection μ bad detected ident τ q) :
    ∑ ω, (μ ω).toReal * ∑ b ∈ escaped (bad ω) (detected ω) ident τ, w b ≤ I.card * ((1 - q) / q * B) := by
  have hmaps : ∀ ω, ∀ b ∈ escaped (bad ω) (detected ω) ident τ, ident b ∈ I :=
    fun ω b hb => hI ω b (mem_escaped.1 hb).1
  calc ∑ ω, (μ ω).toReal * ∑ b ∈ escaped (bad ω) (detected ω) ident τ, w b
      = ∑ j ∈ I, ∑ ω, (μ ω).toReal * ∑ b ∈ escaped (bad ω) (detected ω) ident τ with ident b = j, w b := by
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun ω _ => ?_
        rw [← Finset.mul_sum, Finset.sum_fiberwise_of_maps_to (hmaps ω) w]
    _ ≤ ∑ _j ∈ I, (1 - q) / q * B :=
        Finset.sum_le_sum fun j _ => sampled_escape μ bad detected ident τ w q B hq hq1 hB hw hwB hS j
    _ = I.card * ((1 - q) / q * B) := by rw [Finset.sum_const, nsmul_eq_mul]

/-- **The escape slack**: under `SampledDetection`, the faulty blocks accepted before their identity's first detection
skip more than `θ C` with probability at most `J ((1 − q)/q) B/(θ C)`, by Markov's inequality. -/
theorem escape_slack {Ω β ι : Type} [Fintype Ω] [DecidableEq ι] (μ : PMF Ω) (bad detected : Ω → Finset β)
    (I : Finset ι) (ident : β → ι) (τ w : β → ℝ) (q B θ C : ℝ) (hθ : 0 < θ) (hC : 0 < C) (hw : ∀ b, 0 ≤ w b)
    (hI : ∀ ω, ∀ b ∈ bad ω, ident b ∈ I) (hq : 0 < q) (hq1 : q ≤ 1) (hB : 0 ≤ B) (hwB : ∀ ω, ∀ b ∈ bad ω, w b ≤ B)
    (hS : Assumptions.SampledDetection μ bad detected ident τ q) :
    prob μ (fun ω => θ * C < ∑ b ∈ escaped (bad ω) (detected ω) ident τ, w b) ≤
      I.card * ((1 - q) / q * B) / (θ * C) := by
  have hZ : ∀ ω, 0 ≤ ∑ b ∈ escaped (bad ω) (detected ω) ident τ, w b :=
    fun ω => Finset.sum_nonneg fun b _ => hw b
  have hm : θ * C * prob μ (fun ω => θ * C < ∑ b ∈ escaped (bad ω) (detected ω) ident τ, w b) ≤
      ∑ ω, (μ ω).toReal * ∑ b ∈ escaped (bad ω) (detected ω) ident τ, w b :=
    markov μ (fun ω => ∑ b ∈ escaped (bad ω) (detected ω) ident τ, w b) hZ (θ * C)
  have hE := expected_escaped μ bad detected I ident τ w q B hI hq hq1 hB hw hwB hS
  rw [le_div_iff₀ (mul_pos hθ hC)]
  linarith

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

/-- The residual read at the work certified correctly and run, `(1 − ε_W) W_cert − ε' C − (JΔ/H) C − θ C`, is
`r̂ + ε_W W_cert/C + ε' + JΔ/H + θ`. -/
theorem residualShare_sub {W C εW ε L θ : ℝ} (hC : 0 < C) :
    residualShare ((1 - εW) * W - ε * C - L * C - θ * C) C = residualShare W C + εW * W / C + ε + L + θ := by
  have hCi : C * C⁻¹ = 1 := mul_inv_cancel₀ hC.ne'
  unfold residualShare
  linear_combination (ε + L + θ) * hCi

theorem DampenerRemote : Guarantees.DampenerRemote := by
  intro Ω _ μ N c wrong p ε β ι _ bad detected I ident τ w Φ Δ H q B θ εW κ C U W χ mfu M Mu Mr Mx Γ α b_w b_a r d T
    dd e Mc hp hε hc hcC hsound hH hΔ hΦ hI hT hcap hθ hw hq hq1 hB0 hwB hS hκ hC hU hmfu hmfu1 hα hbw hba hdd hcomp hB
    hthr hW hsize hM0 hM hu hres hx
  refine prob_ge_of_not μ ?_ ((prob_or_le μ (fun ω => ε * C < wrongCapacity c (wrong ω))
    (fun ω => θ * C < ∑ b ∈ escaped (bad ω) (detected ω) ident τ, w b)).trans
      (add_le_add (CertificateSlack Ω μ N c wrong p ε C hp hε hC hc hcC hsound)
        (escape_slack μ bad detected I ident τ w q B θ C hθ hC hw hI hq hq1 hB0 hwB hS)))
  intro ω hbad
  by_contra hgood
  apply hbad
  have hwrong : wrongCapacity c (wrong ω) ≤ ε * C := not_lt.1 fun h => hgood (Or.inl h)
  have hesc : ∑ b ∈ escaped (bad ω) (detected ω) ident τ, w b ≤ θ * C := not_lt.1 fun h => hgood (Or.inr h)
  have hlag := VerdictLag β ι (bad ω) (detected ω) I ident τ w Φ Δ H C hH hΔ hΦ (hI ω) (hT ω) (hcap ω)
  have hr : (Mr ω : ℝ) ≤ κ * (C - ((1 - εW) * W - ε * C - lagShare I.card Δ H * C - θ * C)) / 2 := by
    have h1 := hres ω
    have h2 : κ * (C - (1 - εW) * W + wrongCapacity c (wrong ω) + ∑ b ∈ bad ω, w b) ≤
        κ * (C - ((1 - εW) * W - ε * C - lagShare I.card Δ H * C - θ * C)) :=
      mul_le_mul_of_nonneg_left (by linarith) hκ.le
    linarith
  obtain ⟨h1, h2⟩ := remote_core hκ hC hU hmfu hmfu1 (Nat.cast_pos.2 (hM0 ω)) (by exact_mod_cast hM ω) (hu ω) hr
    (declared_le hα hbw hba hdd hcomp hB hthr hW hsize) (hx ω)
  rw [residualShare_sub hC] at h1 h2
  exact ⟨h1, h2⟩

end Nci.SecurityProofs
