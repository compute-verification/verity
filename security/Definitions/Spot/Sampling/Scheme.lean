module

public import Definitions.Spot.Sampling.Layer
public import Definitions.Core.Scheme

/-!
# From the sampling layer to a scheme

The link between the proof layer and the accounting layer, stated. `Spot.Sampling.Sound` bounds acceptance with an
event of the committed transcript and the sample; `Verity.AuditBound` is what a scheme reads, and
`Verity.Scheme.AccountingLayer` takes it to `Scheme.GetsBack`.

* `GivesAuditBound`: for every design over the units `Sound` takes, session and analysis as `Sound` takes them, and every
  audit strategy, `AuditBound` holds of the audit game's outcomes (the session's verdict and the sample), with the
  incorrect set that of the strategy's committed transcript and the error `εks + εbind` at the strategy's registration
  and state. It is `Sound` at the event "the sample misses the committed transcript's incorrect set with probability at
  most `δ`", which does not depend on the sample.
-/

@[expose] public section

namespace Spot.Sampling

open Verity Verity.Game
open scoped ENNReal

/-- **The sampling layer gives every scheme its audit bound** (stated). Proved from `Sound` by
`Spot.Sampling.givesAuditBound`. -/
def GivesAuditBound : Prop :=
  ∀ {U : Type} [DecidableEq U] (L : Verity.SamplingDesign U) (Reg : Type) (session : Finset U → Reg → Game Bool)
    (A : ExtractionAnalysis L Reg session) (εks εbind : (R : Reg) → Cont L Reg session R → ℝ≥0∞),
    A.KnowledgeSound εks → A.BindSound εbind → ∀ σ : Strategy (audit L Reg session),
      AuditBound (fun _ => L) (fun p => prob p (audit L Reg session) σ) (fun o => o.1 = true)
        (fun _ => A.incorrect (A.committedOf σ)) (εks (reg σ) (cont σ) + εbind (reg σ) (cont σ))

end Spot.Sampling
