module

public import Proofs.Spot.Sampling.RegDraw
public import Proofs.Spot.Sampling.Scheme

/-!
# A design the prover registers, and what a scheme reads of it

`Spot.Sampling.GivesAuditBound` is at one sampling design fixed before the prover's strategy. In
`Pouw.Window.auditReg L session` the design is `L R`, a function of the registration `R` the prover sends before the
coin, so a prover that proposes its design with its registration is a strategy of that game.

* `atRegOf σ`: `atReg` over any units, the strategy of `audit (L σ.1)` that registers `σ`'s `R` and plays its
  continuation; `prob_auditRegOf` is `prob_auditReg` there (`rfl`).
* `givesAuditBound_auditReg`: **the audit bound at the registered design.** For every strategy `σ` of
  `auditReg L session`, `Verity.AuditBound` holds of its outcomes at the design `L σ.1` it registered, with the
  incorrect set that of its committed transcript under the analysis at that design, and the error `εks + εbind` there.
  It is `GivesAuditBound` at `L σ.1`, strategy by strategy: the registration is the strategy's first move, fixed before
  the coin, and a randomized prover is a mixture of strategies.

The analysis and its two error bounds are taken at every design `L R₀` the prover can register, each for every
registration played under it. Only the one at `R₀ = σ.1` is read.
-/

@[expose] public section

namespace Spot.Sampling

open Verity Verity.Game
open scoped ENNReal

variable {U Reg : Type} {L : Reg → Verity.SamplingDesign U} {session : Finset U → Reg → Game Bool}

/-- The strategy of the fixed-design game at `σ`'s registration that plays `σ`, over any units. -/
def atRegOf (σ : Strategy (Pouw.Window.auditReg L session)) : Strategy (audit (L σ.1) Reg session) :=
  ⟨σ.1, σ.2⟩

/-- A strategy of the registered-design audit has the outcome probabilities of the fixed-design audit at the design it
registered. -/
theorem prob_auditRegOf (p : Bool × Finset U → Prop) (σ : Strategy (Pouw.Window.auditReg L session)) :
    prob p (Pouw.Window.auditReg L session) σ = prob p (audit (L σ.1) Reg session) (atRegOf σ) := rfl

/-- **The audit bound at the registered design.** -/
theorem givesAuditBound_auditReg [DecidableEq U] (h : GivesAuditBound)
    (A : (R₀ : Reg) → ExtractionAnalysis (L R₀) Reg session)
    (εks εbind : (R₀ : Reg) → (R : Reg) → Cont (L R₀) Reg session R → ℝ≥0∞)
    (hks : ∀ R₀, (A R₀).KnowledgeSound (εks R₀)) (hbind : ∀ R₀, (A R₀).BindSound (εbind R₀))
    (σ : Strategy (Pouw.Window.auditReg L session)) :
    AuditBound (fun _ => L σ.1) (fun p => prob p (Pouw.Window.auditReg L session) σ) (fun o => o.1 = true)
      (fun _ => (A σ.1).incorrect ((A σ.1).committedOf (atRegOf σ)))
      (εks σ.1 (reg (atRegOf σ)) (cont (atRegOf σ)) + εbind σ.1 (reg (atRegOf σ)) (cont (atRegOf σ))) :=
  h (L σ.1) Reg session (A σ.1) (εks σ.1) (εbind σ.1) (hks σ.1) (hbind σ.1) (atRegOf σ)

end Spot.Sampling
