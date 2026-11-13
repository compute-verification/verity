module

public import Proofs.Flock.SpotFlock.Honest.Entry
public import Proofs.Flock.SpotFlock.AddressFacts.WidthOf

/-!
# `WidthFact` at the one-template instance

The wiring map is the identity (`wireMap_held`), so it is idempotent, and every row has one width
(`rowWidths_held`, from `RefWidths R`); the address facts' width at the address map (`widthFact_addressOf`) then
gives `WidthFact`. The hypotheses are `NeededBits` and `RefWidths R`.
-/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy

variable (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code)

/-- **`WidthFact` at the instance.** -/
theorem widthFact_held (hN : NeededBits (held R B desc q c₀)) (hW : RefWidths R) :
    WidthFact (held R B desc q c₀) (addressOf (held R B desc q c₀)) :=
  widthFact_addressOf _ (fun j => by rw [wireMap_held]; rfl) hN (rowWidths_held R B desc q c₀ hW)

end SpotFlock.Honest
