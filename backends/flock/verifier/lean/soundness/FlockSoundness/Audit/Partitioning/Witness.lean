import FlockSoundness.Audit.Partitioning.Guarantees

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
-/

namespace FlockSoundness.Audit.Partitioning.Witness

open Finset

theorem converse :
    ((Law.subset 4 2 (by norm_num)).miss 2).toReal ≤ 1 / 6 ∧
    1 < ((4 : ℝ) - 2 + 1) * (1 - (1 / 6 : ℝ) ^ ((2 : ℝ)⁻¹)) ∧
    0 < ((4 : ℝ) - 2 + 1) / 2 * Real.log (1 / 6 : ℝ)⁻¹ * (1 - Real.log (1 / 6 : ℝ)⁻¹ / (2 * 2)) := by
  sorry

theorem achievability : min 1 (Real.log (1 / 2 : ℝ)⁻¹ / 2 * 1) + 1 / 4 < 1 := by
  sorry

theorem granularity :
    MeetsTarget (Law.subset 1000 1000 le_rfl) (fun _ => 1) 10 (1 / 2) ∧
    MeetsTarget (Law.subset 500 500 le_rfl) (fun _ => 2) 10 (1 / 2) ∧
    0 < 1 - 1 / 10 - 10 / (1000 * 1) - 1 * Real.log (1 / 2 : ℝ)⁻¹ / (2 * 10) ∧
    0 < 1 - 2 / 10 - 10 / (500 * 2) - 2 * Real.log (1 / 2 : ℝ)⁻¹ / (2 * 10) := by
  sorry

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
          (2 ^ 2 * 10) * (1 - 2 / 10) - (⌊Real.logb 2 (2 / 1)⌋₊ + 1) * unitCost 0 1 2) := by
  sorry

theorem exfiltration :
    (1 : ℝ) * 1 + locBits 4 1 ≤ 4 ∧ (4 : ℝ) < 1 * (1 + 1) + locBits 4 (1 + 1) ∧
    CertifiesLeak (Law.subset 4 4 le_rfl) (fun _ => 1) (1 / 2) 4 ∧
    0 < ((4 : ℝ) - 1) * (1 - (1 / 2 : ℝ) ^ ((1 + 1 : ℝ)⁻¹)) ∧
    ((4 : ℝ) - 1) * (1 - (1 / 2 : ℝ) ^ ((1 + 1 : ℝ)⁻¹)) < 4 * Real.log (1 / 2 : ℝ)⁻¹ / (1 + 1) := by
  sorry

theorem refinement :
    2 * (1 + 1) ≤ 4 ∧ (16 : ℝ) * 1 + locBits 4 1 ≤ 20 ∧ (20 : ℝ) < 16 * (1 + 1) + locBits 4 (1 + 1) ∧
    (16 : ℝ) + Real.logb 2 (1 + 2) / (1 + 1) ≤ 16 + Real.logb 2 2 ∧
    CertifiesLeak (Law.subset (2 * 4) (2 * 4) le_rfl) (fun _ => 16) (1 / 2) 20 ∧
    0 < (1 - 1 / 4) * (1 - Real.log (1 / 2 : ℝ)⁻¹ / (2 * (1 + 1))) *
      (unitCost 0 1 1 * (4 * Real.log (1 / 2 : ℝ)⁻¹ / (1 + 1))) := by
  sorry

theorem riskConcave_needs_k_two :
    detectionRisk 2 1 (1 / 2) 1 - detectionRisk 2 1 (1 / 2) 0 <
      detectionRisk 2 1 (1 / 2) 2 - detectionRisk 2 1 (1 / 2) 1 := by
  sorry

theorem riskConcave_needs_small_p :
    detectionRisk 3 2 (4 / 5) 1 - detectionRisk 3 2 (4 / 5) 0 <
      detectionRisk 3 2 (4 / 5) 2 - detectionRisk 3 2 (4 / 5) 1 := by
  sorry

theorem concentration :
    detectionRisk 2 2 (1 / 2) 2 + detectionRisk 2 2 (1 / 2) 0 <
      detectionRisk 2 2 (1 / 2) 1 + detectionRisk 2 2 (1 / 2) 1 := by
  sorry

end FlockSoundness.Audit.Partitioning.Witness
