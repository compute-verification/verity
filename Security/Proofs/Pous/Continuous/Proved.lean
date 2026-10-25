import Proofs.Pous.Continuous.PerAnswer
import Models.Pous.Guarantees.Proved

/-!
# The continuous audit at a proved verdict: `Pous.Guarantees.ContinuousAuditProved`

`pr_proved_le`: over the audit's outcome and the proof service verifier's coins, the verifier accepts and `bad` holds
with probability at most the clear verdict's `Pr[clear ∧ bad]` plus the proof layer's error, at every audit, verdict
and event (`ProofSound`). `continuous_audit_proved` is it at the continuous audit, from `continuous_audit`.
-/

namespace PousContinuous

open Pous
open scoped ENNReal

open Classical in
/-- **A proved verdict adds the proof layer's error**: if the verifier accepts a claim that `clear` refutes with
probability at most `εp` on every outcome, then `Pr[acc ∧ bad] ≤ Pr[clear ∧ bad] + εp` over the outcome and the
verifier's coins. -/
theorem pr_proved_le {X Ω : Type} [Fintype X] [Fintype Ω] (acc : X → Ω → Prop) (clear bad : X → Prop)
    (εp : ℝ≥0∞) (h : ProofSound acc clear εp) :
    pr (fun y : X × Ω => acc y.1 y.2 ∧ bad y.1) ≤ pr (fun x => clear x ∧ bad x) + εp := by
  rcases isEmpty_or_nonempty Ω with hΩ | hΩ
  · simp [pr]
  rcases isEmpty_or_nonempty X with hX | hX
  · simp [pr]
  have hK0 : (Fintype.card Ω : ℝ≥0∞) ≠ 0 := by simp
  have hKt : (Fintype.card Ω : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
  have hN0 : (Fintype.card X : ℝ≥0∞) ≠ 0 := by simp
  have hNt : (Fintype.card X : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
  have hx : ∀ x, ((Finset.univ.filter fun c => acc x c ∧ ¬ clear x).card : ℝ≥0∞) ≤ εp * Fintype.card Ω :=
    fun x => by
      have h' := h x
      unfold pr at h'
      rw [ENNReal.div_le_iff hK0 hKt] at h'
      convert h'
  have hsplit : (Finset.univ.filter fun y : X × Ω => acc y.1 y.2 ∧ ¬ clear y.1).card =
      ∑ x, (Finset.univ.filter fun c => acc x c ∧ ¬ clear x).card := by
    rw [Finset.card_filter, Fintype.sum_prod_type]
    simp only [Finset.card_filter]
  have h2 : ((Finset.univ.filter fun y : X × Ω => acc y.1 y.2 ∧ ¬ clear y.1).card : ℝ≥0∞) ≤
      Fintype.card X * (εp * Fintype.card Ω) := by
    rw [hsplit, Nat.cast_sum]
    calc ∑ x, ((Finset.univ.filter fun c => acc x c ∧ ¬ clear x).card : ℝ≥0∞)
        ≤ ∑ _x : X, εp * Fintype.card Ω := Finset.sum_le_sum fun x _ => hx x
      _ = Fintype.card X * (εp * Fintype.card Ω) := by
        rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  have hsub : (Finset.univ.filter fun y : X × Ω => acc y.1 y.2 ∧ bad y.1) ⊆
      (Finset.univ.filter fun x => clear x ∧ bad x) ×ˢ (Finset.univ : Finset Ω) ∪
        Finset.univ.filter fun y : X × Ω => acc y.1 y.2 ∧ ¬ clear y.1 := by
    intro y hy
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hy
    by_cases hc : clear y.1
    · exact Finset.mem_union_left _ (Finset.mem_product.2 ⟨by simp [hc, hy.2], Finset.mem_univ _⟩)
    · exact Finset.mem_union_right _ (by simp [hy.1, hc])
  have hA : ((Finset.univ.filter fun y : X × Ω => acc y.1 y.2 ∧ bad y.1).card : ℝ≥0∞) ≤
      (Finset.univ.filter fun x => clear x ∧ bad x).card * Fintype.card Ω +
        Fintype.card X * (εp * Fintype.card Ω) := by
    refine (Nat.cast_le.2 ((Finset.card_le_card hsub).trans (Finset.card_union_le _ _))).trans ?_
    rw [Nat.cast_add, Finset.card_product, Finset.card_univ, Nat.cast_mul]
    exact add_le_add le_rfl h2
  have hc : ∀ a : ℝ≥0∞, a / Fintype.card X * (Fintype.card X * Fintype.card Ω) = a * Fintype.card Ω :=
    fun a => by rw [← mul_assoc, ENNReal.div_mul_cancel hN0 hNt]
  have hfin : ((Finset.univ.filter fun y : X × Ω => acc y.1 y.2 ∧ bad y.1).card : ℝ≥0∞) ≤
      ((Finset.univ.filter fun x => clear x ∧ bad x).card / (Fintype.card X : ℝ≥0∞) + εp) *
        ((Fintype.card X : ℝ≥0∞) * Fintype.card Ω) := by
    refine hA.trans (le_of_eq ?_)
    rw [add_mul, hc]
    ring
  unfold pr
  rw [Fintype.card_prod, Nat.cast_mul, ENNReal.div_le_iff (mul_ne_zero hN0 hK0) (ENNReal.mul_ne_top hNt hKt)]
  convert hfin

/-- **`Pous.Guarantees.ContinuousAuditProved`**: `pr_proved_le` at the memory challenger's verdict and the shortfall,
with `continuous_audit` bounding the clear verdict's part. -/
theorem continuous_audit_proved : Pous.Guarantees.ContinuousAuditProved := by
  intro M sch D Q D_late Q_late β n π hinc W mach arr f βe hβe Ω _ acc εp hp
  let bad : Outcome sch n → Prop := fun x =>
    let pp := (sch.enc W x.1 x.2.1).1
    let C := (sch.enc W x.1 x.2.1).2
    let rs := contRecords mach D Q D_late Q_late W pp C x.1 arr .start (List.ofFn x.2.2)
    (⨅ θ : Set.Ioi (0 : ℝ), ((θ : ℝ) * f + Real.log (1 / βe)) / (1 - Real.exp (-(θ : ℝ)))) <
      (rs.map fun r => max (contFreed sch r.held - epsProof sch β) 0).sum
  exact (pr_proved_le acc _ bad εp hp).trans
    (add_le_add (continuous_audit sch D Q D_late Q_late β n π hinc W mach arr f βe hβe) le_rfl)

end PousContinuous
