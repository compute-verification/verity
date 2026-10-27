import Proofs.Accounting.JointBridge.Guarantees.IntegrityOfIntegrityBlock
import Proofs.Flock.EndToEnd.Blocks
import Proofs.Pouw.Prob

/-!
# Proof of the integrity audit of a server-time block from C-Flock's integrity audit of that block

At each value of the other coins, a draw of the period's sampled coins on which its audit accepts while more than `ε` of
its units are wrong is one on which every sampling block of the server-time block is accepted and their wrong units
total more than `ε·units ≥ ln(1/δ)/q`. Read on the horizon's records, that is the block escaping
(`ServerBlocks.BlockEscapes`), the event `Flock.SecurityProofs.IntegrityBlock` bounds; the realization's `law` carries
the bound from the horizon's game to the uniform coins.
-/

namespace Accounting.SecurityProofs.Bridge

open scoped ENNReal

/-- **The integrity audit of a server-time block** (`Accounting.Guarantees.Bridge.IntegrityOfIntegrityBlock`). -/
theorem IntegrityOfIntegrityBlock : Accounting.Guarantees.Bridge.IntegrityOfIntegrityBlock := by
  intro ι d α _ C I Ω _ _ r S b q δ hq hδ hqt hA3 R hT hL c p hcert hfin ξ
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
  have hlaw := R.law ξ fun l => (∀ w ∈ l, w.accepted) ∧
    Real.log (1 / δ) / q ≤ ((l.map Flock.Guarantees.WindowRec.wrong).sum : ℝ)
  have hev : FlockSoundness.Game.value (fun l => (∀ w ∈ l.filter (fun w => w.idx ∈ S.sessions b), w.accepted) ∧
        Real.log (1 / δ) / q ≤
          (((l.filter fun w => w.idx ∈ S.sessions b).map Flock.Guarantees.WindowRec.wrong).sum : ℝ))
        (S.run (R.strat ξ)) =
      FlockSoundness.Game.value (S.BlockEscapes q δ b) (S.run (R.strat ξ)) :=
    FlockSoundness.Game.value_congr (fun l => by
      simp only [Flock.Guarantees.ServerBlocks.BlockEscapes, Flock.Guarantees.ServerBlocks.blockWrong,
        List.mem_filter, decide_eq_true_eq, and_imp]) _
  have hsrc := Flock.SecurityProofs.IntegrityBlock S b q δ hq hδ hqt hA3 (R.strat ξ) (hT ξ) (hL ξ)
  refine (Pouw.pr_mono hsub).trans ?_
  rw [← ENNReal.toReal_ofReal (Pouw.pr_nonneg _), hlaw, hev]
  exact ENNReal.toReal_mono hfin hsrc

end Accounting.SecurityProofs.Bridge
