import Proofs.Flock.Soundness.Audit.Partitioning.Guarantees
import Proofs.Flock.Soundness.Audit.Partitioning.SecurityProofs.TwoStage
import Proofs.Flock.Soundness.Audit.Partitioning.SecurityProofs.Exfil.Witnesses
import Proofs.Flock.Soundness.Audit.Partitioning.SecurityProofs.Unequal.Witnesses

/-!
# The partitioning guarantees are not vacuous

For each guarantee, an instance whose hypotheses all hold and whose bound is positive. `Law.subset n n` draws every unit,
so it meets every target. Two more show that `RiskConcave`'s hypotheses can't be dropped.

* `converse`: the uniform 2-subset of 4 units has `miss 2 = 1/6`, and at `δ = 1/6` both converse bounds are positive,
  the exact one above 1 (the law draws 2).
* `achievability`: two units of work 1, `C = 2`, `δ = 1/2`, `ε = 1/4`: each unit is drawn with probability below 1.
* `granularity`: 1,000 units of work 1, or 500 of work 2, with `C = 10` and `δ = 1/2`: the target is met, and both
  granularities' factors are positive.
* `heavyUnit`: one unit doing all of `C = 1`, under a law meeting the target.
* `unequal`: 50 units of work 1 and 50 of work 2, `ρ = 2`, `C = 10`, `δ = 1/2`, `f = 0`, `o = 1`: the target is met and
  both unequal bounds are positive.
* `exfiltration`: 4 units 1 bit wide, `K = 1`, `ℓ = 4`, `δ = 1/2`: `ℓ` is in `K`'s window, a law certifies it, and the
  lower bound is positive and below the upper one.
* `refinement`: 4 units 16 bits wide split in 2 of 16 bits, `K = 1`, `ℓ = 20`, `δ = 1/2`, `s = 1`, `f = 0`, `o = 1`: every
  hypothesis holds, a law on the split partition certifies `ℓ`, and the factor is positive.
* `riskConcave_needs_k_two`: at `k = 1` (`N = 2`, `p = 1/2`) the detection risk's increments rise.
* `riskConcave_needs_small_p`: at `k = 2`, `p = 4/5` (`N = 3`) they rise too.
* `concentration`: at `N = k = 2`, `p = 1/2`, two wrong proof units in one replay unit are strictly less detectable than
  one in each of two.
* `nestedEscape`: under `nested` with two replay units of two proof units, `k = 1` and `p = 1/2`, one wrong proof unit
  escapes with probability `3/4`, strictly between `1 − p` and 1.
-/

namespace FlockSoundness.Audit.Partitioning.Witness

open Finset

theorem converse :
    ((Law.subset 4 2 (by norm_num)).miss 2).toReal ≤ 1 / 6 ∧
    1 < ((4 : ℝ) - 2 + 1) * (1 - (1 / 6 : ℝ) ^ ((2 : ℝ)⁻¹)) ∧
    0 < ((4 : ℝ) - 2 + 1) / 2 * Real.log (1 / 6 : ℝ)⁻¹ * (1 - Real.log (1 / 6 : ℝ)⁻¹ / (2 * 2)) := by
  refine ⟨?_, ?_, ?_⟩
  · rw [Law.subset_miss _ (by norm_num), ENNReal.toReal_div, ENNReal.toReal_natCast, ENNReal.toReal_natCast]
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
    MeetsTarget (Law.subset 1000 1000 le_rfl) (fun _ => 1) 10 (1 / 2) ∧
    MeetsTarget (Law.subset 500 500 le_rfl) (fun _ => 2) 10 (1 / 2) ∧
    0 < 1 - 1 / 10 - 10 / (1000 * 1) - 1 * Real.log (1 / 2 : ℝ)⁻¹ / (2 * 10) ∧
    0 < 1 - 2 / 10 - 10 / (500 * 2) - 2 * Real.log (1 / 2 : ℝ)⁻¹ / (2 * 10) :=
  SecurityProofs.Unequal.granularity_witness

theorem heavyUnit : MeetsTarget (Law.subset 1 1 le_rfl) (fun _ => 1) 1 (1 / 2) := by
  intro B hB
  have h1 : (1 : ℝ) ≤ B.card := by simpa using hB
  have h2 : B.card ≤ 1 := by simpa using Finset.card_le_univ B
  have h3 : 1 ≤ B.card := by exact_mod_cast h1
  rw [Law.subset_escape, show B.card = 1 by omega]
  norm_num

theorem unequal :
    (∀ u : Fin 100, 1 ≤ (if u.val < 50 then (1 : ℝ) else 2) ∧ (if u.val < 50 then (1 : ℝ) else 2) ≤ 2) ∧
    2 * Real.log (1 / 2 : ℝ)⁻¹ ≤ 10 ∧
    MeetsTarget (Law.subset 100 100 le_rfl) (fun u => if u.val < 50 then 1 else 2) 10 (1 / 2) ∧
    0 < Real.log (1 / 2 : ℝ)⁻¹ * (1 - 2 * Real.log (1 / 2 : ℝ)⁻¹ / (2 * 10)) *
      ((∑ u : Fin 100, if u.val < 50 then (1 : ℝ) else 2) / (2 * 10) * (1 - 2 / 10) -
        (⌊Real.logb 2 (2 / 1)⌋₊ + 1)) ∧
    0 < Real.log (1 / 2 : ℝ)⁻¹ * (1 - 2 * Real.log (1 / 2 : ℝ)⁻¹ / (2 * 10)) *
      ((∑ u : Fin 100, (if u.val < 50 then (1 : ℝ) else 2) * unitCost 0 1 (if u.val < 50 then 1 else 2)) /
          (2 ^ 2 * 10) * (1 - 2 / 10) - (⌊Real.logb 2 (2 / 1)⌋₊ + 1) * unitCost 0 1 2) :=
  SecurityProofs.Unequal.unequal_witness

theorem exfiltration :
    (1 : ℝ) * 1 + locBits 4 1 ≤ 4 ∧ (4 : ℝ) < 1 * (1 + 1) + locBits 4 (1 + 1) ∧
    CertifiesLeak (Law.subset 4 4 le_rfl) (fun _ => 1) (1 / 2) 4 ∧
    0 < ((4 : ℝ) - 1) * (1 - (1 / 2 : ℝ) ^ ((1 + 1 : ℝ)⁻¹)) ∧
    ((4 : ℝ) - 1) * (1 - (1 / 2 : ℝ) ^ ((1 + 1 : ℝ)⁻¹)) < 4 * Real.log (1 / 2 : ℝ)⁻¹ / (1 + 1) :=
  SecurityProofs.Exfil.witness_exfiltration

theorem refinement :
    2 * (1 + 1) ≤ 4 ∧ (16 : ℝ) * 1 + locBits 4 1 ≤ 20 ∧ (20 : ℝ) < 16 * (1 + 1) + locBits 4 (1 + 1) ∧
    (16 : ℝ) + Real.logb 2 (1 + 2) / (1 + 1) ≤ 16 + Real.logb 2 2 ∧
    CertifiesLeak (Law.subset (2 * 4) (2 * 4) le_rfl) (fun _ => 16) (1 / 2) 20 ∧
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
    ((nested 2 2 1 1 2 (by norm_num)).escape {finProdFinEquiv (0, 0)}).toReal = 3 / 4 := by
  have h0 : wrongIn (R := 2) (N := 2) {finProdFinEquiv (0, 0)} 0 = 1 := by decide
  have h1 : wrongIn (R := 2) (N := 2) {finProdFinEquiv (0, 0)} 1 = 0 := by decide
  rw [(SecurityProofs.NestedEscape 2 2 1 1 2 (by norm_num) (by norm_num) _).1, Fin.prod_univ_two, h0, h1]
  norm_num [Nat.choose]

end FlockSoundness.Audit.Partitioning.Witness
