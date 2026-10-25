import Definitions.Pous.Model.Continuous
import Definitions.Pous.Model.Inc
import Definitions.Pous.Model.Proved

/-!
# The continuous audit with its verdict proved through the proof service

`ContinuousAudit` bounds the clear verdict: the memory challenger accepts its own records (`contAccepts`) and the
freed memory exceeds `G(f)`. Under the proof service the auditor accepts on the proof service's verifier instead, run
on a claim of that verdict after the audit (`Definitions.Pous.Model.Proved`). `ContinuousAuditProved` is the same bound
at that acceptance, plus the proof layer's error, from the proof layer's soundness as a hypothesis (`ProofSound`), the
way the joint audit takes each audit's guarantee. It is proved from `ContinuousAudit`
(`Pous.SecurityProofs.ContinuousAuditProved`). It rests on what `ContinuousAudit` rests on and on `ProofSound`, which
sampling's knowledge soundness at the claims' deployment is to give (open).
-/

namespace Pous.Guarantees

open scoped ENNReal

/-- **The continuous audit's guarantee, at a verdict proved through the proof service.** Under `ContinuousAudit`'s
hypotheses, for every verifier of the verdict's claim over coins `Ω` drawn after the audit, sound at `εp` against the
memory challenger's verdict on its records (`ProofSound`): over the audit's outcome and the verifier's coins, the
verifier accepts and `Σ_j max(ε_j − ε_proof, 0)` exceeds `G(f)` with probability at most `βe + n·(β + 1)·π + εp`. -/
def ContinuousAuditProved : Prop :=
  ∀ {M : IdealModel} (sch : Scheme M) (D Q D_late Q_late β n : ℕ) (π : ℝ≥0∞), TimedINC sch D Q β π →
    ∀ (W : Bits sch.n) (mach : ContMachine sch) (arr : ℕ → ℕ) (f βe : ℝ), 0 < βe →
      ∀ (Ω : Type) [Fintype Ω] (acc : Outcome sch n → Ω → Prop) (εp : ℝ≥0∞),
        ProofSound acc (fun x : Outcome sch n =>
            contAccepts f (contRecords mach D Q D_late Q_late W (sch.enc W x.1 x.2.1).1 (sch.enc W x.1 x.2.1).2 x.1
              arr .start (List.ofFn x.2.2))) εp →
        pr (fun y : Outcome sch n × Ω =>
            let pp := (sch.enc W y.1.1 y.1.2.1).1
            let C := (sch.enc W y.1.1 y.1.2.1).2
            let rs := contRecords mach D Q D_late Q_late W pp C y.1.1 arr .start (List.ofFn y.1.2.2)
            acc y.1 y.2 ∧
              (⨅ θ : Set.Ioi (0 : ℝ), ((θ : ℝ) * f + Real.log (1 / βe)) / (1 - Real.exp (-(θ : ℝ)))) <
                (rs.map fun r => max (contFreed sch r.held - epsProof sch β) 0).sum)
          ≤ ENNReal.ofReal βe + n * (β + 1) * π + εp

end Pous.Guarantees
