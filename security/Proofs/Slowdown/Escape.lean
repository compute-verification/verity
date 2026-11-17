module

public import Definitions.Slowdown.Assumptions.Remote

/-! The escaped blocks under SPOT (`sampled_escape`). -/

@[expose] public section

open Verity.Game renaming prPMFReal → prob

namespace Slowdown.SecurityProofs

open Slowdown.Slowdown

open scoped Classical in
theorem prob_eq_sum {Ω : Type} [Fintype Ω] (μ : PMF Ω) (E : Ω → Prop) :
    prob μ E = ∑ ω, if E ω then (μ ω).toReal else 0 := by
  rw [Verity.Game.prPMFReal, Verity.Game.prPMF, PMF.toOuterMeasure_apply, tsum_fintype,
    ENNReal.toReal_sum fun ω _ => ne_top_of_le_ne_top (μ.apply_ne_top ω) (Set.indicator_le_self _ _ ω)]
  refine Finset.sum_congr rfl fun ω _ => ?_
  by_cases h : E ω <;> simp [Set.indicator, h]

theorem sum_toReal {Ω : Type} [Fintype Ω] (μ : PMF Ω) : ∑ ω, (μ ω).toReal = 1 := by
  have h0 : ∑ ω, μ ω = 1 :=
    (tsum_fintype (L := SummationFilter.unconditional Ω) (fun ω => μ ω)).symm.trans μ.tsum_coe
  have h := congrArg ENNReal.toReal h0
  rwa [ENNReal.toReal_sum fun ω _ => μ.apply_ne_top ω, ENNReal.toReal_one] at h

theorem expected_filter {Ω β : Type} [Fintype Ω] (μ : PMF Ω) (bad : Ω → Finset β) (U : Finset β)
    (hU : ∀ ω, bad ω ⊆ U) (P : Ω → β → Prop) [∀ ω, DecidablePred (P ω)] (w : β → ℝ) :
    ∑ ω, (μ ω).toReal * ∑ b ∈ (bad ω).filter (P ω), w b =
      ∑ b ∈ U, w b * prob μ (fun ω => b ∈ bad ω ∧ P ω b) := by
  classical
  have e : ∀ ω, ∑ b ∈ (bad ω).filter (P ω), w b = ∑ b ∈ U, if b ∈ bad ω ∧ P ω b then w b else 0 := by
    intro ω
    rw [← Finset.sum_filter]
    refine Finset.sum_congr ?_ fun _ _ => rfl
    ext b
    simp only [Finset.mem_filter]
    constructor
    · rintro ⟨hb, hP⟩
      exact ⟨hU ω hb, hb, hP⟩
    · rintro ⟨-, hb, hP⟩
      exact ⟨hb, hP⟩
  simp only [e, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun b _ => ?_
  simp only [prob_eq_sum]
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun ω _ => ?_
  by_cases h : b ∈ bad ω ∧ P ω b
  · simp only [if_pos h]
    ring
  · simp only [if_neg h, mul_zero]

theorem mem_escaped_iff {β ι : Type} {bad detected : Finset β} {ident : β → ι} {τ : β → ℝ} {b : β} :
    b ∈ escaped bad detected ident τ ↔ b ∈ bad ∧ b ∉ detected ∧ noDetectionBefore bad detected ident τ b := by
  simp only [escaped, Finset.mem_filter, noDetectionBefore]
  constructor
  · rintro ⟨hb, hall⟩
    exact ⟨hb, fun hd => lt_irrefl _ (hall b hb hd rfl), fun b' hb' hd _ hid => hall b' hb' hd hid⟩
  · rintro ⟨hb, hnd, hC⟩
    refine ⟨hb, fun b' hb' hd hid => ?_⟩
    by_cases e : b' = b
    · exact (hnd (e ▸ hd)).elim
    · exact hC b' hb' hd e hid

theorem sampled_escape {Ω β ι : Type} [Fintype Ω] [DecidableEq ι] (μ : PMF Ω) (bad detected : Ω → Finset β)
    (ident : β → ι) (τ w : β → ℝ) (q B : ℝ) (hq : 0 < q) (hq1 : q ≤ 1) (hB : 0 ≤ B) (hw : ∀ b, 0 ≤ w b)
    (hwB : ∀ ω, ∀ b ∈ bad ω, w b ≤ B) (hS : Assumptions.SampledDetection μ bad detected ident τ q) (j : ι) :
    ∑ ω, (μ ω).toReal * ∑ b ∈ escaped (bad ω) (detected ω) ident τ with ident b = j, w b ≤ (1 - q) / q * B := by
  classical
  let U : Finset β := Finset.univ.biUnion bad
  have hU : ∀ ω, bad ω ⊆ U := fun ω b hb => Finset.mem_biUnion.2 ⟨ω, Finset.mem_univ _, hb⟩
  let X : Ω → ℝ := fun ω =>
    ∑ b ∈ (bad ω).filter (fun b => ident b = j ∧ b ∉ detected ω ∧ noDetectionBefore (bad ω) (detected ω) ident τ b), w b
  let Z : Ω → ℝ := fun ω =>
    ∑ b ∈ (bad ω).filter (fun b => ident b = j ∧ b ∈ detected ω ∧ noDetectionBefore (bad ω) (detected ω) ident τ b), w b
  let Y : Ω → ℝ := fun ω =>
    ∑ b ∈ (bad ω).filter (fun b => ident b = j ∧ noDetectionBefore (bad ω) (detected ω) ident τ b), w b
  have hesc : ∀ ω, ∑ b ∈ escaped (bad ω) (detected ω) ident τ with ident b = j, w b = X ω := by
    intro ω
    refine Finset.sum_congr ?_ fun _ _ => rfl
    ext b
    simp only [Finset.mem_filter, mem_escaped_iff]
    constructor
    · rintro ⟨⟨hb, hnd, hC⟩, hj⟩
      exact ⟨hb, hj, hnd, hC⟩
    · rintro ⟨hb, hj, hnd, hC⟩
      exact ⟨⟨hb, hnd, hC⟩, hj⟩
  have hYXZ : ∀ ω, Y ω = X ω + Z ω := by
    intro ω
    simp only [Y, X, Z, Finset.sum_filter, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun b _ => ?_
    by_cases h1 : ident b = j <;> by_cases h2 : b ∈ detected ω <;>
      by_cases h3 : noDetectionBefore (bad ω) (detected ω) ident τ b <;> simp [h1, h2, h3]
  have hZ : ∀ ω, Z ω ≤ B := by
    intro ω
    let S := (bad ω).filter (fun b => ident b = j ∧ b ∈ detected ω ∧ noDetectionBefore (bad ω) (detected ω) ident τ b)
    have hcard : S.card ≤ 1 := by
      refine Finset.card_le_one.2 fun a ha b hb => ?_
      obtain ⟨-, haj, had, haN⟩ := Finset.mem_filter.1 ha
      obtain ⟨hbbad, hbj, hbd, hbN⟩ := Finset.mem_filter.1 hb
      obtain ⟨habad, -⟩ := Finset.mem_filter.1 ha
      by_contra hne
      have h1 : τ a < τ b := haN b hbbad hbd (Ne.symm hne) (hbj.trans haj.symm)
      have h2 : τ b < τ a := hbN a habad had hne (haj.trans hbj.symm)
      exact lt_asymm h1 h2
    have hle := Finset.sum_le_card_nsmul S w B fun b hb => hwB ω b (Finset.mem_filter.1 hb).1
    rw [nsmul_eq_mul] at hle
    have hc : (S.card : ℝ) ≤ 1 := by exact_mod_cast hcard
    exact hle.trans (mul_le_of_le_one_left hB hc)
  have hEX : ∑ ω, (μ ω).toReal * X ω ≤ (1 - q) * ∑ ω, (μ ω).toReal * Y ω := by
    simp only [X, Y]
    rw [expected_filter μ bad U hU, expected_filter μ bad U hU, Finset.mul_sum]
    refine Finset.sum_le_sum fun b _ => ?_
    by_cases hj : ident b = j
    · simp only [hj, true_and]
      calc w b * prob μ (fun ω => b ∈ bad ω ∧ b ∉ detected ω ∧ noDetectionBefore (bad ω) (detected ω) ident τ b)
          ≤ w b * ((1 - q) * prob μ (fun ω => b ∈ bad ω ∧ noDetectionBefore (bad ω) (detected ω) ident τ b)) :=
            mul_le_mul_of_nonneg_left (hS b) (hw b)
        _ = (1 - q) * (w b * prob μ (fun ω => b ∈ bad ω ∧ noDetectionBefore (bad ω) (detected ω) ident τ b)) := by
            ring
    · simp [hj, prob_eq_sum]
  have hEY : ∑ ω, (μ ω).toReal * Y ω = ∑ ω, (μ ω).toReal * X ω + ∑ ω, (μ ω).toReal * Z ω := by
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun ω _ => ?_
    rw [hYXZ ω, mul_add]
  have hEZ : ∑ ω, (μ ω).toReal * Z ω ≤ B :=
    calc ∑ ω, (μ ω).toReal * Z ω ≤ ∑ ω, (μ ω).toReal * B :=
          Finset.sum_le_sum fun ω _ => mul_le_mul_of_nonneg_left (hZ ω) ENNReal.toReal_nonneg
      _ = B := by rw [← Finset.sum_mul, sum_toReal, one_mul]
  have hq0 : 0 ≤ 1 - q := by linarith
  have h3 : ∑ ω, (μ ω).toReal * X ω ≤ (1 - q) * ∑ ω, (μ ω).toReal * X ω + (1 - q) * B := by
    have := mul_le_mul_of_nonneg_left hEZ hq0
    rw [hEY, mul_add] at hEX
    linarith
  simp only [hesc]
  rw [div_mul_eq_mul_div, le_div_iff₀ hq]
  nlinarith [h3]

end Slowdown.SecurityProofs
