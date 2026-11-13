module

public import Proofs.Flock.SpotFlock.Honest.Copy
public import Proofs.Flock.SpotFlock.Honest.Width

/-!
# The five address facts at the one-template instance

`Five` at the instance, each fact from its own file: the width (`widthFact_held`), the copy (`copyFact_held`), the
row (`rowFact_held`), the entry (`entryFact_held`) and one tree a gate (`oneTreeFact_held`); so the address premise
holds there (`addressesConnected_of_held`). The hypotheses:

* on `R`: every written bit's row is one of its ports and none is named like an opening row (`RefPorts`), every bit's
  position is inside its port's row (`RefWidths`), and its circuit copies each bit where the layout reads it
  (`CopiesHeld`);
* `NeededBits`, being proved on its own;
* `AddressRow.RowGoal`, what gives the row fact, at the instance.
-/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy

variable (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code)

/-- **The five address facts at the instance.** -/
theorem five_held
    (hG : AddressRow.RowGoal (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)))
    (hR : RefPorts R) (hW : RefWidths R) (hC : CopiesHeld R B desc q c₀) (hN : NeededBits (held R B desc q c₀)) :
    Five R B desc q c₀ :=
  ⟨widthFact_held R B desc q c₀ hN hW, copyFact_held R B desc q c₀ hG hR hN hC, rowFact_held R B desc q c₀ hG hR hN,
    entryFact_held R B desc q c₀ hR hN, oneTreeFact_held R B desc q c₀ hR hN⟩

/-- **The address premise at the instance, through the five facts.** -/
theorem addressesConnected_of_held
    (hG : AddressRow.RowGoal (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)))
    (hR : RefPorts R) (hW : RefWidths R) (hC : CopiesHeld R B desc q c₀) (hN : NeededBits (held R B desc q c₀)) :
    AddressesConnected (held R B desc q c₀) :=
  addressesConnected_held R B desc q c₀ (five_held R B desc q c₀ hG hR hW hC hN)

end SpotFlock.Honest
