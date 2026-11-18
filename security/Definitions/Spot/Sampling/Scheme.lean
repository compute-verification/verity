module

public import Definitions.Spot.Sampling.Layer
public import Definitions.Core.Scheme

/-!
# From the sampling layer to a scheme

The link between the proof layer and the accounting layer, stated. `Spot.Sampling.Sound` bounds acceptance with an
event of the committed transcript and the sample; `Verity.AuditBound` is what a scheme reads, and
`Verity.Scheme.AccountingLayer` takes it to `Scheme.GetsBack`.

* `GivesAuditBound`: for every design over one partition's units, session and analysis as `Sound` takes them, and every
  audit strategy, `AuditBound` holds of the audit game's outcomes (the session's verdict and the sample), with the
  incorrect set that of the strategy's committed transcript and the error `εks + εbind` at the strategy's registration
  and state. It is `Sound` at the event "the sample misses the committed transcript's incorrect set with probability at
  most `δ`", which does not depend on the sample.

The units are `Fin n`, as `Sound`'s are; a `Verity.Levels` design is not of this form.
-/

@[expose] public section

namespace Spot.Sampling

open Verity Verity.Game
open scoped ENNReal

/-- **The sampling layer gives every scheme its audit bound** (stated). Proved from `Sound` by
`Spot.Sampling.givesAuditBound`. -/
def GivesAuditBound : Prop :=
  ∀ (n : ℕ) (L : SamplingDesign (Fin n)) (Reg : Type) (session : Finset (Fin n) → Reg → Game Bool)
    (Tr : Type) (incorrect : Tr → Finset (Fin n))
    (committed : (R : Reg) → ((ω : L.Ω) → Strategy (session (L.draw ω) R)) → Tr)
    (ks : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → ℝ≥0∞)
    (link : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → Tr → ℝ≥0∞),
    (∀ S R (τ : Strategy (session S R)) u X, u ∈ S → u ∈ incorrect X →
      prob (· = true) (session S R) τ ≤ ks S R τ u + link S R τ u X) →
    ∀ (εks εbind : (R : Reg) → ((ω : L.Ω) → Strategy (session (L.draw ω) R)) → ℝ≥0∞),
    (∀ R (τ : (ω : L.Ω) → Strategy (session (L.draw ω) R)) (tgt : L.Ω → Option (Fin n)),
      avg (fun ω => (tgt ω).elim 0 fun u => if u ∈ L.draw ω then ks (L.draw ω) R (τ ω) u else 0) ≤ εks R τ) →
    (∀ R (τ : (ω : L.Ω) → Strategy (session (L.draw ω) R)) (tgt : L.Ω → Option (Fin n)),
      avg (fun ω => (tgt ω).elim 0 fun u =>
        if u ∈ L.draw ω then link (L.draw ω) R (τ ω) u (committed R τ) else 0) ≤ εbind R τ) →
    ∀ (σ : Strategy (Game.send Reg fun R => Game.coin L.Ω fun ω =>
        (session (L.draw ω) R).map fun ok => (ok, L.draw ω))),
      let cont : (ω : L.Ω) → Strategy (session (L.draw ω) σ.1) := fun ω =>
        Strategy.ofMap (fun ok => (ok, L.draw ω)) (session (L.draw ω) σ.1) (σ.2 ω)
      let X := committed σ.1 cont
      AuditBound L
        (fun p => prob p
          (Game.send Reg fun R => Game.coin L.Ω fun ω => (session (L.draw ω) R).map fun ok => (ok, L.draw ω)) σ)
        (fun o => o.1 = true) (fun _ => incorrect X) (εks σ.1 cont + εbind σ.1 cont)

end Spot.Sampling
