module

public import Proofs.Certifier.JointBridge.Guarantees.IntegrityOfIntegrityMonitoringPeriod
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.EndToEnd.Period
import Proofs.Pouw.Prob

/-! The integrity audit of a period, proved from Flock's integrity audit over a monitoring period. -/

@[expose] public section

namespace ResourceAccounting.SecurityProofs.Bridge

open scoped ENNReal

/-- **The integrity audit of a period** (`ResourceAccounting.Guarantees.Bridge.IntegrityOfIntegrityMonitoringPeriod`). -/
theorem IntegrityOfIntegrityMonitoringPeriod : ResourceAccounting.Guarantees.Bridge.IntegrityOfIntegrityMonitoringPeriod := by
  intro ι d α _ C I Ω _ _ r W T q δ hq hδ hqt hA3 R hT hL c p hcert hfin ξ
  have hsub : ∀ x : Ω (r.own .sampled),
      (r.accepts .sampled (Function.update ξ (r.own .sampled) x) ∧
        ¬ r.Holds c p .sampled (Function.update ξ (r.own .sampled) x)) →
      (∀ w ∈ R.record (Function.update ξ (r.own .sampled) x), w.accepted) ∧
        Real.log (1 / δ) / q ≤
          (((R.record (Function.update ξ (r.own .sampled) x)).map Flock.Guarantees.SamplingBlockRec.incorrect).sum : ℝ) := by
    rintro x ⟨hacc, hfail⟩
    refine ⟨(R.accepts_iff _).1 hacc, ?_⟩
    have hfail' : ¬ (((r.parts.incorrect (r.trace (Function.update ξ (r.own .sampled) x))).card : ℝ) ≤
        c.ε * r.units) := hfail
    have hlt := not_le.mp hfail'
    rw [R.wrong_eq] at hlt
    exact hcert.trans hlt.le
  have hlaw : ENNReal.ofReal (Pouw.pr fun x : Ω (r.own .sampled) =>
        (∀ w ∈ R.record (Function.update ξ (r.own .sampled) x), w.accepted) ∧
          Real.log (1 / δ) / q ≤
            (((R.record (Function.update ξ (r.own .sampled) x)).map Flock.Guarantees.SamplingBlockRec.incorrect).sum : ℝ)) =
      FlockSoundness.Game.value (fun l => (∀ w ∈ l, w.accepted) ∧
          Real.log (1 / δ) / q ≤ ((l.map Flock.Guarantees.SamplingBlockRec.incorrect).sum : ℝ))
        (Flock.Guarantees.Period.run W T (R.strat ξ)) :=
    R.samplingDesign ξ fun l => (∀ w ∈ l, w.accepted) ∧
      Real.log (1 / δ) / q ≤ ((l.map Flock.Guarantees.SamplingBlockRec.incorrect).sum : ℝ)
  have hsrc := Flock.SecurityProofs.IntegrityPooled W T q δ hq hδ hqt hA3 (R.strat ξ) (hT ξ) (hL ξ)
  refine (Pouw.pr_mono hsub).trans ?_
  rw [← ENNReal.toReal_ofReal (Pouw.pr_nonneg _), hlaw]
  exact ENNReal.toReal_mono hfin hsrc

end ResourceAccounting.SecurityProofs.Bridge
