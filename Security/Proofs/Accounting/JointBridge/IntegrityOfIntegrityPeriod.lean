import Proofs.Accounting.JointBridge.Statement.IntegrityOfIntegrityPeriod
import Proofs.Flock.EndToEnd.Period
import Proofs.Pouw.Prob

/-!
# Proof of the integrity audit of a period from C-Flock's integrity audit over a horizon

At each value of the other coins, a draw of the sampled audit's own coins on which it accepts while more than `ε` of
the units are wrong is one on which every window is accepted and the wrong units total more than `ε·units ≥ ln(1/δ)/q`,
the event `Flock.SecurityProofs.IntegrityHorizon` bounds; the realization's `law` carries the bound from the period's
game to the uniform coins.
-/

namespace Accounting.SecurityProofs.Bridge

open FlockSoundness FlockSoundness.Game Flock.Guarantees
open scoped ENNReal

/-- **The integrity audit of a period** (`Accounting.Guarantees.Bridge.IntegrityOfIntegrityPeriod`). -/
theorem IntegrityOfIntegrityPeriod : Accounting.Guarantees.Bridge.IntegrityOfIntegrityPeriod := by
  intro ι d α _ C I Ω _ _ r W T q δ hq hδ hqt hA3 R hCR c p hcert hfin ξ
  set E : List WindowRec → Prop :=
    fun l => (∀ w ∈ l, w.accepted) ∧ Real.log (1 / δ) / q ≤ ((l.map WindowRec.wrong).sum : ℝ)
  have hsub : ∀ x : Ω (r.own .sampled),
      (r.accepts .sampled (Function.update ξ (r.own .sampled) x) ∧
        ¬ r.Holds c p .sampled (Function.update ξ (r.own .sampled) x)) →
      E (R.record (Function.update ξ (r.own .sampled) x)) := by
    rintro x ⟨hacc, hfail⟩
    refine ⟨(R.accepts_iff _).1 hacc, ?_⟩
    have hlt : c.ε * r.units < ((r.parts.wrong (r.transcript (Function.update ξ (r.own .sampled) x))).card : ℝ) :=
      lt_of_not_ge hfail
    rw [R.wrong_eq] at hlt
    exact hcert.trans hlt.le
  have hlaw := R.law ξ E
  have hsrc := Flock.SecurityProofs.IntegrityHorizon W T q δ hq hδ hqt hA3 (R.strat ξ) (hCR ξ)
  calc Pouw.pr (fun x : Ω (r.own .sampled) =>
        r.accepts .sampled (Function.update ξ (r.own .sampled) x) ∧
          ¬ r.Holds c p .sampled (Function.update ξ (r.own .sampled) x))
      ≤ Pouw.pr (fun x : Ω (r.own .sampled) => E (R.record (Function.update ξ (r.own .sampled) x))) :=
        Pouw.pr_mono hsub
    _ = (value E (Period.run W T (R.strat ξ))).toReal := by
        rw [← hlaw, ENNReal.toReal_ofReal (Pouw.pr_nonneg _)]
    _ ≤ (ENNReal.ofReal δ + ∑ t ∈ Finset.range T, ((W t).ks + (W t).link)).toReal :=
        ENNReal.toReal_mono hfin hsrc

end Accounting.SecurityProofs.Bridge
