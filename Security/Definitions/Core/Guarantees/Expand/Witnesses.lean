module

public import Definitions.Core.Guarantees.Expand
public import Definitions.Core.Expand.Instances

/-!
# The bit-level expansion's size bounds: honest and nontrivial lemmas

`ExpandSize` and `ExpandAndSize` take no premise, so each has two statements here, at `Verity.Expand.AndPair`
(`Security/Definitions/Core/Expand/Instances.lean`): `<Guarantee>_honest`, its conclusion at that declaration and its
one-unit partition, and `<Guarantee>_nontrivial`, its bound with the right-hand side lowered by one failing there, so
the bound is tight at an explicit circuit.

The proofs are in `Security/Proofs/Core/ExpandWitnesses.lean`.
-/

@[expose] public section

namespace Verity.Guarantees

open Verity.Expand

/-- **The expansion's sizes at `AndPair`**: its computed bits are as many as the width of the computed gates, its unit's
bits as many as the unit's width, and its unit's unregistered reads at most the width of the unit's unregistered
reads. -/
def ExpandSize_honest : Prop :=
  AndPair.d.D.expand.size AndPair.d.D.expand.computed = AndPair.d.D.width AndPair.d.D.computed ∧
    ∀ u, AndPair.d.D.expand.size (AndPair.P.expand.gates u) = AndPair.d.D.width (AndPair.P.gates u) ∧
      (AndPair.d.expand.unregistered (AndPair.P.expand.gates u)).card ≤
        AndPair.d.D.width (AndPair.d.unregistered (AndPair.P.gates u))

/-- **A unit's unregistered reads can grow to the full width of its gate-level ones**: at `AndPair`, the unit's
expansion reads gate 1's two unregistered bits, more than the width of gate 1 less one. -/
def ExpandSize_nontrivial : Prop :=
  ¬ (AndPair.d.expand.unregistered (AndPair.P.expand.gates 0)).card ≤
    AndPair.d.D.width (AndPair.d.unregistered (AndPair.P.gates 0)) - 1

/-- **The expansion's AND counts at `AndPair`**: at most the width of the computed gates, and of the unit's. -/
def ExpandAndSize_honest : Prop :=
  AndPair.d.D.expand.andSize AndPair.d.D.expand.computed ≤ AndPair.d.D.width AndPair.d.D.computed ∧
    ∀ u, AndPair.d.D.expand.andSize (AndPair.P.expand.gates u) ≤ AndPair.d.D.width (AndPair.P.gates u)

/-- **The expansion's AND count can reach the width of the computed gates**: at `AndPair`, both bits of gate 2 are AND
gates, more than its width less one. -/
def ExpandAndSize_nontrivial : Prop :=
  ¬ AndPair.d.D.expand.andSize AndPair.d.D.expand.computed ≤ AndPair.d.D.width AndPair.d.D.computed - 1

end Verity.Guarantees
