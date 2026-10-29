import Proofs.Accounting.JointBridge.Guarantees.IntegrityOfIntegrityPeriod
import Proofs.Flock.EndToEnd.Period
import Proofs.Pouw.Prob

/-!
# Proof of the integrity audit of a period from C-Flock's integrity audit over a monitoring period

At each value of the other coins, a draw of the sampled audit's own coins on which it accepts while more than `ε` of
the units are wrong is one on which every sampling block is accepted and the faulty proof blocks total more than
`ε·units ≥ ln(1/δ)/q`, the event `Flock.SecurityProofs.IntegrityHorizon` bounds; the realization's `law` carries the
bound from the period's game to the uniform coins.
-/

namespace Accounting.SecurityProofs.Bridge

open scoped ENNReal

/-- **The integrity audit of a period** (`Accounting.Guarantees.Bridge.IntegrityOfIntegrityPeriod`). -/
theorem IntegrityOfIntegrityPeriod : Accounting.Guarantees.Bridge.IntegrityOfIntegrityPeriod := by
  intro ι d α _ C I Ω _ _ r W T q δ hq hδ hqt hA3 R hT hL c p hcert hfin ξ
  have hsub : ∀ x : Ω (r.own .sampled),
      (r.accepts .sampled (Function.update ξ (r.own .sampled) x) ∧
        ¬ r.Holds c p .sampled (Function.update ξ (r.own .sampled) x)) →
      (∀ w ∈ R.record (Function.update ξ (r.own .sampled) x), w.accepted) ∧
        Real.log (1 / δ) / q ≤
          (((R.record (Function.update ξ (r.own .sampled) x)).map Flock.Guarantees.WindowRec.wrong).sum : ℝ) := by
    rintro x ⟨hacc, hfail⟩
    refine ⟨(R.accepts_iff _).1 hacc, ?_⟩
    have hfail' : ¬ (((r.parts.wrong (r.transcript (Function.update ξ (r.own .sampled) x))).card : ℝ) ≤
        c.ε * r.units) := hfail
    have hlt := not_le.mp hfail'
    rw [R.wrong_eq] at hlt
    exact hcert.trans hlt.le
  have hlaw : ENNReal.ofReal (Pouw.pr fun x : Ω (r.own .sampled) =>
        (∀ w ∈ R.record (Function.update ξ (r.own .sampled) x), w.accepted) ∧
          Real.log (1 / δ) / q ≤
            (((R.record (Function.update ξ (r.own .sampled) x)).map Flock.Guarantees.WindowRec.wrong).sum : ℝ)) =
      FlockSoundness.Game.value (fun l => (∀ w ∈ l, w.accepted) ∧
          Real.log (1 / δ) / q ≤ ((l.map Flock.Guarantees.WindowRec.wrong).sum : ℝ))
        (Flock.Guarantees.Period.run W T (R.strat ξ)) :=
    R.law ξ fun l => (∀ w ∈ l, w.accepted) ∧
      Real.log (1 / δ) / q ≤ ((l.map Flock.Guarantees.WindowRec.wrong).sum : ℝ)
  have hsrc := Flock.SecurityProofs.IntegrityHorizon W T q δ hq hδ hqt hA3 (R.strat ξ) (hT ξ) (hL ξ)
  refine (Pouw.pr_mono hsub).trans ?_
  rw [← ENNReal.toReal_ofReal (Pouw.pr_nonneg _), hlaw]
  exact ENNReal.toReal_mono hfin hsrc

end Accounting.SecurityProofs.Bridge
