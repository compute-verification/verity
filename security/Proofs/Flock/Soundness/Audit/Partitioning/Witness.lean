module

public import Proofs.Flock.Soundness.Audit.Partitioning.Model
public import Proofs.Flock.Soundness.Audit.Partitioning.SecurityProofs.Exfil.Witnesses
public import Proofs.Flock.Soundness.Audit.Partitioning.SecurityProofs.Unequal.Witnesses
import Proofs.Flock.Soundness.Audit.SamplingDesign
import Proofs.Flock.Soundness.Audit.Partitioning.SecurityProofs.TwoStage

/-! The partitioning guarantees are not vacuous: for each, an instance whose hypotheses all hold and whose bound is positive. -/

@[expose] public section

namespace FlockSoundness.Audit.Partitioning.Witness

open Finset

theorem converse :
    ((SamplingDesign.subset 4 2 (by norm_num)).miss 2).toReal ≤ 1 / 6 ∧
    1 < ((4 : ℝ) - 2 + 1) * (1 - (1 / 6 : ℝ) ^ ((2 : ℝ)⁻¹)) ∧
    0 < ((4 : ℝ) - 2 + 1) / 2 * Real.log (1 / 6 : ℝ)⁻¹ * (1 - Real.log (1 / 6 : ℝ)⁻¹ / (2 * 2)) := by
  refine ⟨?_, ?_, ?_⟩
  · rw [SamplingDesign.subset_miss _ (by norm_num), ENNReal.toReal_div, ENNReal.toReal_natCast, ENNReal.toReal_natCast]
    norm_num [Nat.choose]
  · have h : (1 / 6 : ℝ) ^ ((2 : ℝ)⁻¹) < 2 / 3 := by
      have := Real.rpow_lt_rpow (by norm_num : (0 : ℝ) ≤ 1 / 6) (by norm_num : (1 / 6 : ℝ) < (2 / 3) ^ 2)
        (by norm_num : (0 : ℝ) < (2 : ℝ)⁻¹)
      rwa [show ((2 : ℝ)⁻¹) = ((2 : ℕ) : ℝ)⁻¹ by norm_num, Real.pow_rpow_inv_natCast (by norm_num) (by norm_num)]
        at this
    linarith
  · have h6 : Real.log (1 / 6 : ℝ)⁻¹ = Real.log 6 := by norm_num
    rw [h6]
    have hpos : 0 < Real.log 6 := Real.log_pos (by norm_num)
    have hlt : Real.log 6 < 4 := by
      rw [Real.log_lt_iff_lt_exp (by norm_num)]
      have := Real.quadratic_le_exp_of_nonneg (by norm_num : (0 : ℝ) ≤ 4)
      linarith
    have : 0 < 1 - Real.log 6 / (2 * 2) := by linarith
    positivity

theorem achievability : min 1 (Real.log (1 / 2 : ℝ)⁻¹ / 2 * 1) + 1 / 4 < 1 :=
  SecurityProofs.Unequal.achievability_witness

theorem granularity :
    MeetsTarget (SamplingDesign.subset 1000 1000 le_rfl) (fun _ => 1) 10 (1 / 2) ∧
    MeetsTarget (SamplingDesign.subset 500 500 le_rfl) (fun _ => 2) 10 (1 / 2) ∧
    0 < 1 - 1 / 10 - 10 / (1000 * 1) - 1 * Real.log (1 / 2 : ℝ)⁻¹ / (2 * 10) ∧
    0 < 1 - 2 / 10 - 10 / (500 * 2) - 2 * Real.log (1 / 2 : ℝ)⁻¹ / (2 * 10) :=
  SecurityProofs.Unequal.granularity_witness

theorem heavyUnit : MeetsTarget (SamplingDesign.subset 1 1 le_rfl) (fun _ => 1) 1 (1 / 2) := by
  intro B hB
  have h1 : (1 : ℝ) ≤ B.card := by simpa using hB
  have h2 : B.card ≤ 1 := by simpa using Finset.card_le_univ B
  have h3 : 1 ≤ B.card := by exact_mod_cast h1
  rw [SamplingDesign.subset_escape, show B.card = 1 by omega]
  norm_num

theorem unequal :
    (∀ u : Fin 100, 1 ≤ (if u.val < 50 then (1 : ℝ) else 2) ∧ (if u.val < 50 then (1 : ℝ) else 2) ≤ 2) ∧
    2 * Real.log (1 / 2 : ℝ)⁻¹ ≤ 10 ∧
    MeetsTarget (SamplingDesign.subset 100 100 le_rfl) (fun u => if u.val < 50 then 1 else 2) 10 (1 / 2) ∧
    0 < Real.log (1 / 2 : ℝ)⁻¹ * (1 - 2 * Real.log (1 / 2 : ℝ)⁻¹ / (2 * 10)) *
      ((∑ u : Fin 100, if u.val < 50 then (1 : ℝ) else 2) / (2 * 10) * (1 - 2 / 10) -
        (⌊Real.logb 2 (2 / 1)⌋₊ + 1)) ∧
    0 < Real.log (1 / 2 : ℝ)⁻¹ * (1 - 2 * Real.log (1 / 2 : ℝ)⁻¹ / (2 * 10)) *
      ((∑ u : Fin 100, (if u.val < 50 then (1 : ℝ) else 2) * unitCost 0 1 (if u.val < 50 then 1 else 2)) /
          (2 ^ 2 * 10) * (1 - 2 / 10) - (⌊Real.logb 2 (2 / 1)⌋₊ + 1) * unitCost 0 1 2) :=
  SecurityProofs.Unequal.unequal_witness

theorem exfiltration :
    (1 : ℝ) * 1 + locBits 4 1 ≤ 4 ∧ (4 : ℝ) < 1 * (1 + 1) + locBits 4 (1 + 1) ∧
    CertifiesLeak (SamplingDesign.subset 4 4 le_rfl) (fun _ => 1) (1 / 2) 4 ∧
    0 < ((4 : ℝ) - 1) * (1 - (1 / 2 : ℝ) ^ ((1 + 1 : ℝ)⁻¹)) ∧
    ((4 : ℝ) - 1) * (1 - (1 / 2 : ℝ) ^ ((1 + 1 : ℝ)⁻¹)) < 4 * Real.log (1 / 2 : ℝ)⁻¹ / (1 + 1) :=
  SecurityProofs.Exfil.witness_exfiltration

theorem refinement :
    2 * (1 + 1) ≤ 4 ∧ (16 : ℝ) * 1 + locBits 4 1 ≤ 20 ∧ (20 : ℝ) < 16 * (1 + 1) + locBits 4 (1 + 1) ∧
    (16 : ℝ) + Real.logb 2 (1 + 2) / (1 + 1) ≤ 16 + Real.logb 2 2 ∧
    CertifiesLeak (SamplingDesign.subset (2 * 4) (2 * 4) le_rfl) (fun _ => 16) (1 / 2) 20 ∧
    0 < (1 - 1 / 4) * (1 - Real.log (1 / 2 : ℝ)⁻¹ / (2 * (1 + 1))) *
      (unitCost 0 1 1 * (4 * Real.log (1 / 2 : ℝ)⁻¹ / (1 + 1))) :=
  SecurityProofs.Exfil.witness_refinement

theorem riskConcave_needs_k_two :
    detectionRisk 2 1 (1 / 2) 1 - detectionRisk 2 1 (1 / 2) 0 <
      detectionRisk 2 1 (1 / 2) 2 - detectionRisk 2 1 (1 / 2) 1 := by
  have e0 : detectionRisk 2 1 (1 / 2) 0 = 0 := by norm_num [detectionRisk]
  have e1 : detectionRisk 2 1 (1 / 2) 1 = -Real.log (3 / 4) := by norm_num [detectionRisk]
  have e2 : detectionRisk 2 1 (1 / 2) 2 = -Real.log (1 / 2) := by norm_num [detectionRisk]
  rw [e0, e1, e2]
  have h := Real.log_lt_log (by norm_num) (show (1 / 2 : ℝ) < (3 / 4) ^ 2 by norm_num)
  rw [Real.log_pow] at h
  push_cast at h
  linarith

theorem riskConcave_needs_small_p :
    detectionRisk 3 2 (4 / 5) 1 - detectionRisk 3 2 (4 / 5) 0 <
      detectionRisk 3 2 (4 / 5) 2 - detectionRisk 3 2 (4 / 5) 1 := by
  have e0 : detectionRisk 3 2 (4 / 5) 0 = 0 := by norm_num [detectionRisk, Nat.choose]
  have e1 : detectionRisk 3 2 (4 / 5) 1 = -Real.log (7 / 15) := by norm_num [detectionRisk, Nat.choose]
  have e2 : detectionRisk 3 2 (4 / 5) 2 = -Real.log (1 / 5) := by norm_num [detectionRisk, Nat.choose]
  rw [e0, e1, e2]
  have h := Real.log_lt_log (by norm_num) (show (1 / 5 : ℝ) < (7 / 15) ^ 2 by norm_num)
  rw [Real.log_pow] at h
  push_cast at h
  linarith

theorem concentration :
    detectionRisk 2 2 (1 / 2) 2 + detectionRisk 2 2 (1 / 2) 0 <
      detectionRisk 2 2 (1 / 2) 1 + detectionRisk 2 2 (1 / 2) 1 := by
  have e0 : detectionRisk 2 2 (1 / 2) 0 = 0 := by norm_num [detectionRisk]
  have e1 : detectionRisk 2 2 (1 / 2) 1 = -Real.log (1 / 2) := by norm_num [detectionRisk]
  have e2 : detectionRisk 2 2 (1 / 2) 2 = -Real.log (1 / 2) := by norm_num [detectionRisk]
  rw [e0, e1, e2]
  have h : Real.log (1 / 2 : ℝ) < 0 := Real.log_neg (by norm_num) (by norm_num)
  linarith

theorem nestedEscape :
    ((nested 2 2 1 1 2 (by norm_num)).missProb {finProdFinEquiv (0, 0)}).toReal = 3 / 4 := by
  have h0 : incorrectIn (R := 2) (N := 2) {finProdFinEquiv (0, 0)} 0 = 1 := by decide
  have h1 : incorrectIn (R := 2) (N := 2) {finProdFinEquiv (0, 0)} 1 = 0 := by decide
  rw [(SecurityProofs.NestedEscape 2 2 1 1 2 (by norm_num) (by norm_num) _).1, Fin.prod_univ_two, h0, h1]
  norm_num [Nat.choose]

end FlockSoundness.Audit.Partitioning.Witness
